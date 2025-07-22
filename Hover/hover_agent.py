import time
from pathlib import Path
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Assuming your refactored environment is in ksp_hover_env.py
from ksp_hover_env import HoverEnv

# --- Configuration ---
MODEL_DIR = Path("checkpoints")
TOTAL_TIMESTEPS = 20_000
SAVE_INTERVAL = 2_000  # Save a checkpoint every N steps

# Group PPO hyperparameters in a dictionary for clarity
PPO_KWARGS = {
    "learning_rate": 3e-4,  # 0.0003
    "n_steps": 1024,        # Number of steps to run for each environment per update
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
}

# --- Helper Function to Resume Training ---

def find_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    """Finds the latest PPO checkpoint in a directory based on the timestep in the filename."""
    try:
        checkpoints = list(model_dir.glob("ppo_hover_agent_*.zip"))
        if not checkpoints:
            return None
        # Sort checkpoints by the timestep number in the filename
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        return latest_checkpoint
    except (ValueError, IndexError):
        # In case filenames are not in the expected format
        return None

# --- Custom Callback for Saving ---

class SaveOnStepCallback(BaseCallback):
    """
    A custom callback that saves the model at regular intervals.
    Note: Episode logging is now handled automatically by the VecMonitor wrapper.
    """
    def __init__(self, save_freq: int, save_path: Path, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        # Check if it's time to save
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_path / f"ppo_hover_agent_{self.num_timesteps}.zip"
            self.model.save(path)
            if self.verbose > 0:
                print(f"💾 [MODEL SAVE] Saved model checkpoint to {path}")
        return True

# --- Main Training Function ---

def train_agent():
    """Initializes the environment, model, and starts the training process."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Standard practice: wrap the environment for Stable Baselines3
    # VecMonitor handles episode statistics like reward and length automatically.
    env = VecMonitor(DummyVecEnv([lambda: HoverEnv()]))

    model: PPO
    latest_checkpoint = find_latest_checkpoint(MODEL_DIR)

    if latest_checkpoint:
        print(f"✅ Resuming training from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
        # Reset the total timesteps to continue from the checkpoint's progress
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

    callback = SaveOnStepCallback(save_freq=SAVE_INTERVAL, save_path=MODEL_DIR)

    try:
        # The `reset_num_timesteps=False` is crucial for resuming training
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callback,
            reset_num_timesteps=not bool(latest_checkpoint)
        )
        print("✅ Training completed.")
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    finally:
        # Save the final model regardless of how training ended
        final_model_path = MODEL_DIR / f"ppo_hover_agent_final_{model.num_timesteps}.zip"
        model.save(final_model_path)
        print(f"💾 Final model saved at: {final_model_path}")
        env.close()


if __name__ == "__main__":
    train_agent()