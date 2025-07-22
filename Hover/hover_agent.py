import os
import time
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ksp_hover_env import HoverEnv

# Config
TOTAL_TIMESTEPS = 50_000
CHECKPOINT_INTERVAL = 2_000
EVAL_INTERVAL = 10_000
MODEL_PREFIX = "ppo_hover_agent"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ---------------------------
# CSV Logger for Evaluation
# ---------------------------
def log_hover_episode(model, steps_done):
    csv_path = f"logs/hover_log_{steps_done}.csv"
    env = HoverEnv(step_sleep=0.2)
    obs, _ = env.reset()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "altitude_m", "throttle"])
        writer.writeheader()

        start = time.time()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = time.time() - start
            writer.writerow({
                "time_s": f"{elapsed:.2f}",
                "altitude_m": f"{obs[0]:.2f}",
                "throttle": f"{action[0]:.3f}"
            })
            done = terminated or truncated

    env.close()
    print(f"[LOG] Hover episode logged to {csv_path}")

# ---------------------------
# Custom Callback
# ---------------------------
class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, eval_freq, verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        steps = self.num_timesteps

        # Save checkpoint
        if steps % self.save_freq == 0:
            path = f"checkpoints/{MODEL_PREFIX}_{steps}.zip"
            self.model.save(path)
            if self.verbose:
                print(f"[CHECKPOINT] Saved model at {path}")

        # Log hover performance
        if steps % self.eval_freq == 0:
            log_hover_episode(self.model, steps)

        return True

# ---------------------------
# Training
# ---------------------------
def train_agent(total_steps=TOTAL_TIMESTEPS):
    env = HoverEnv(step_sleep=0.2)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        n_epochs=10,
    )

    callback = CheckpointCallback(
        save_freq=CHECKPOINT_INTERVAL,
        eval_freq=EVAL_INTERVAL,
        verbose=1
    )

    print("[INFO] Starting training...")
    model.learn(total_timesteps=total_steps, callback=callback)
    model.save(f"{MODEL_PREFIX}_final.zip")
    print(f"[INFO] Training finished. Final model saved as {MODEL_PREFIX}_final.zip")
    env.close()

if __name__ == "__main__":
    train_agent()
