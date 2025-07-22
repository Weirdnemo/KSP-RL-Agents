import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Import our new landing environment
from ksp_landing_env import LandingEnv

# --- Configuration ---
LOG_DIR = Path("landing_logs/")
MODEL_DIR = Path("landing_checkpoints/")
TOTAL_TIMESTEPS = 50_000
SAVE_INTERVAL = 5_000      # Save a model checkpoint every N steps
LOG_ARCHIVE_INTERVAL = 25_000 # Archive the CSV flight log every N steps
EPISODIC_LOG_INTERVAL = 5_000 # Enable detailed logging for one episode every N steps

# PPO hyperparameters - Tuned for better performance
PPO_KWARGS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "policy_kwargs": dict(net_arch=dict(pi=[128, 128], vf=[128, 128])) # Deeper network
}

def find_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    """Finds the latest PPO checkpoint in a directory."""
    try:
        checkpoints = list(model_dir.glob("landing_agent_*.zip"))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    except (ValueError, IndexError):
        return None

class CustomCallback(BaseCallback):
    """
    Saves the model, archives logs, and enables detailed episodic logging at specified intervals.
    """
    def __init__(self, save_freq: int, log_archive_freq: int, episodic_log_freq: int, save_path: Path, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.log_archive_freq = log_archive_freq
        self.episodic_log_freq = episodic_log_freq
        self.save_path = save_path
        self.last_log_enable_step = 0

    def _on_step(self) -> bool:
        # --- Save model checkpoint ---
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            path = self.save_path / f"landing_agent_{self.num_timesteps}.zip"
            self.model.save(path)
            if self.verbose > 0:
                print(f"💾 Saved model checkpoint to {path}")
        
        # --- Archive the main flight log CSV ---
        if self.num_timesteps > 0 and self.num_timesteps % self.log_archive_freq == 0:
            self.training_env.env_method("archive_log_file", self.num_timesteps)
        
        # --- Enable detailed logging for one full episode ---
        if self.num_timesteps - self.last_log_enable_step >= self.episodic_log_freq:
            underlying_env = self.training_env.envs[0]
            if not underlying_env.logging_enabled:
                 print(f"📝 Enabling detailed logging for the next episode (Timestep: {self.num_timesteps})")
                 underlying_env.enable_logging()
                 self.last_log_enable_step = self.num_timesteps

        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        self.training_env.env_method("archive_log_file", self.num_timesteps)

def train_agent():
    """Initializes the environment, model, and starts the training process."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Wrap the environment with VecMonitor to automatically log rewards and episode lengths to TensorBoard
    env = VecMonitor(DummyVecEnv([lambda: LandingEnv()]), str(LOG_DIR))

    model: PPO
    latest_checkpoint = find_latest_checkpoint(MODEL_DIR)

    if latest_checkpoint:
        print(f"✅ Resuming training from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, tensorboard_log=str(LOG_DIR))
        # The number of timesteps is automatically handled by SB3 when loading a model
    else:
        print("🚀 Starting a new training run.")
        model = PPO(env=env, verbose=1, tensorboard_log=str(LOG_DIR), **PPO_KWARGS)

    callback = CustomCallback(
        save_freq=SAVE_INTERVAL,
        log_archive_freq=LOG_ARCHIVE_INTERVAL,
        episodic_log_freq=EPISODIC_LOG_INTERVAL,
        save_path=MODEL_DIR
    )

    try:
        # Note: 'reset_num_timesteps=False' ensures timesteps continue from the loaded model
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            reset_num_timesteps=False 
        )
        print("✅ Training completed.")
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        # Save the final model
        final_model_path = MODEL_DIR / f"landing_agent_final_{model.num_timesteps}.zip"
        model.save(final_model_path)
        print(f"💾 Final model saved at: {final_model_path}")
        env.close()

if __name__ == "__main__":
    train_agent()