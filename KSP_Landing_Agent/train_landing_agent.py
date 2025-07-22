import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Import our new landing environment
from ksp_landing_env import LandingEnv

# --- Configuration ---
MODEL_DIR = Path("landing_checkpoints")
TOTAL_TIMESTEPS = 20_000
SAVE_INTERVAL = 5_000      # Save a model checkpoint every 5k steps
LOG_SAVE_INTERVAL = 10_000 # Archive the CSV flight log every 10k steps

# PPO hyperparameters
PPO_KWARGS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
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
    """Saves the model and archives logs at specified intervals."""
    def __init__(self, save_freq: int, log_save_freq: int, save_path: Path, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.log_save_freq = log_save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            path = self.save_path / f"landing_agent_{self.num_timesteps}.zip"
            self.model.save(path)
            if self.verbose > 0:
                print(f"💾 Saved model checkpoint to {path}")
        
        if self.num_timesteps > 0 and self.num_timesteps % self.log_save_freq == 0:
            underlying_env = self.training_env.envs[0]
            if hasattr(underlying_env, 'archive_log_file'):
                underlying_env.archive_log_file(self.num_timesteps)
        return True

def train_agent():
    """Initializes the environment, model, and starts the training process."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = VecMonitor(DummyVecEnv([lambda: LandingEnv()]))

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
        env.envs[0].archive_log_file(model.num_timesteps) # Final log archive
        final_model_path = MODEL_DIR / f"landing_agent_final_{model.num_timesteps}.zip"
        model.save(final_model_path)
        print(f"💾 Final model saved at: {final_model_path}")
        env.close()

if __name__ == "__main__":
    train_agent()