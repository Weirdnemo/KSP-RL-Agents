import os
import time
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from ksp_hover_env import HoverEnv

# -----------------------------
# Config
# -----------------------------
CHECKPOINT_DIR = "checkpoints"
MODEL_NAME = "ppo_hover_agent"
TOTAL_TIMESTEPS = 50_000
SAVE_INTERVAL = 2000  # Save every 2000 steps

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------
# Custom Callback for Saving
# -----------------------------
class SaveCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            checkpoint_file = os.path.join(self.save_path, f"{MODEL_NAME}_{self.num_timesteps}.zip")
            self.model.save(checkpoint_file)
            if self.verbose > 0:
                print(f"[CHECKPOINT] Saved model at {self.num_timesteps} steps -> {checkpoint_file}")
        return True


# -----------------------------
# Training Function
# -----------------------------
def train_hover_agent():
    # Create environment
    env = HoverEnv(step_sleep=0.2)
    env = Monitor(env)

    # PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
    )

    callback = SaveCheckpointCallback(
        save_freq=SAVE_INTERVAL, save_path=CHECKPOINT_DIR, verbose=1
    )

    # Train
    print(f"[INFO] Starting PPO training for {TOTAL_TIMESTEPS} timesteps.")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_final.zip")
    model.save(final_path)
    print(f"[INFO] Training finished. Final model saved as {final_path}")

    env.close()


if __name__ == "__main__":
    train_hover_agent()
