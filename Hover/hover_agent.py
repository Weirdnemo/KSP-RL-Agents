import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from ksp_hover_env import HoverEnv

# --- Configuration ---
MODEL_DIR = Path("checkpoints")
TOTAL_TIMESTEPS = 30_000
SAVE_INTERVAL = 2_000      # Save a model checkpoint every 2k steps
LOG_SAVE_INTERVAL = 5_000  # Archive the CSV flight log every 5k steps

# Group PPO hyperparameters in a dictionary for clarity
PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
}

# --- Helper Function to Resume Training ---

def find_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    """Finds the latest PPO checkpoint in a directory."""
    try:
        checkpoints = list(model_dir.glob("ppo_hover_agent_*.zip"))
        if not checkpoints:
            return None
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        return latest_checkpoint
    except (ValueError, IndexError):
        return None

# --- Custom Callback for Saving Model and Logs ---

class CustomCallback(BaseCallback):
    """
    Custom callback to save the model and archive logs at different intervals.
    """
    def __init__(self, save_freq: int, log_save_freq: int, save_path: Path, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.log_save_freq = log_save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        # Save the model checkpoint
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_path / f"ppo_hover_agent_{self.num_timesteps}.zip"
            self.model.save(path)
            if self.verbose > 0:
                print(f"💾 Saved model checkpoint to {path}")
        
        # Archive the flight log
        if self.num_timesteps > 0 and self.num_timesteps % self.log_save_freq == 0:
            # Access the underlying environment to call its archive method
            underlying_env = self.training_env.envs[0]
            if hasattr(underlying_env, 'archive_log_file'):
                underlying_env.archive_log_file(self.num_timesteps)

        return True

# --- Main Training Function ---

def train_agent():
    """Initializes the environment, model, and starts the training process."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Wrap the environment for Stable Baselines3
    env = VecMonitor(DummyVecEnv([lambda: HoverEnv()]))

    model: PPO
    latest_checkpoint = find_latest_checkpoint(MODEL_DIR)

    if latest_checkpoint:
        print(f"✅ Resuming training from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
        initial_timesteps = int(latest_checkpoint.stem.split('_')[-1])
        remaining_timesteps = TOTAL_TIMESTEPS - initial_timesteps
        if remaining_timesteps <= 0:
            print("Target total timesteps already reached. Exiting.")
            return
        print(f"   Training for an additional {remaining_timesteps} timesteps.")
    else:
        print("🚀 Starting a new training run.")
        model = PPO("MlpPolicy", env, verbose=1, **PPO_KWARGS)
        remaining_timesteps = TOTAL_TIMESTEPS

    callback = CustomCallback(
        save_freq=SAVE_INTERVAL,
        log_save_freq=LOG_SAVE_INTERVAL,
        save_path=MODEL_DIR
    )

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callback,
            reset_num_timesteps=not bool(latest_checkpoint)
        )
        print("✅ Training completed.")
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        # Save final model and archive final log
        env.envs[0].archive_log_file(model.num_timesteps) # Final log archive
        final_model_path = MODEL_DIR / f"ppo_hover_agent_final_{model.num_timesteps}.zip"
        model.save(final_model_path)
        print(f"💾 Final model saved at: {final_model_path}")
        env.close()

if __name__ == "__main__":
    train_agent()