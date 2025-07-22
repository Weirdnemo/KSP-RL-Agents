import os
import csv
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ksp_hover_env import HoverEnv

MODEL_PATH = "ppo_hover_agent"
LOG_DIR = "training_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ---------------------------------------------------
# Callback to Save Model Every 2k and Flight Logs Every 10k
# ---------------------------------------------------
class SaveModelAndCSVCallback(BaseCallback):
    def __init__(self, model_save_freq=2_000, log_save_freq=10_000, log_dir=LOG_DIR, verbose=1):
        super(SaveModelAndCSVCallback, self).__init__(verbose)
        self.model_save_freq = model_save_freq
        self.log_save_freq = log_save_freq
        self.log_dir = log_dir
        self.csv_file = None
        self.csv_writer = None
        self.current_log_path = None

    def _on_step(self) -> bool:
        steps = self.num_timesteps

        # Save model every 2k steps
        if steps % self.model_save_freq == 0:
            model_file = os.path.join(self.log_dir, f"ppo_hover_agent_{steps}.zip")
            self.model.save(model_file)
            if self.verbose > 0:
                print(f"[MODEL] Saved model at {model_file}")

        # Start a new CSV log every 10k steps
        if steps % self.log_save_freq == 0:
            if self.csv_file:
                self.csv_file.close()

            log_file = os.path.join(self.log_dir, f"hover_log_{steps // 1000}k.csv")
            self.current_log_path = log_file
            self.csv_file = open(log_file, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(["step", "time_s", "apoapsis_m", "v_speed_mps", "throttle"])

            if self.verbose > 0:
                print(f"[CSV] Starting log file: {log_file}")

        # Append step data to CSV if logging
        if self.csv_writer:
            env = self.training_env.envs[0].env
            obs = env._get_obs()
            step_time = time.time() - env.start_time
            self.csv_writer.writerow([steps, step_time, obs[0], obs[1], env.control.throttle])

        return True

    def _on_training_end(self) -> None:
        if self.csv_file:
            self.csv_file.close()
        if self.verbose > 0:
            print("[INFO] Training finished, final CSV and models saved.")

# ---------------------------------------------------
# Train Agent
# ---------------------------------------------------
def train_agent(total_steps=50_000):
    env = HoverEnv(step_sleep=0.2, max_steps=500)
    model = PPO("MlpPolicy", env, verbose=1)

    callback = SaveModelAndCSVCallback(model_save_freq=2_000, log_save_freq=10_000, log_dir=LOG_DIR)

    print("[INFO] Starting training...")
    model.learn(total_timesteps=total_steps, callback=callback)
    model.save(MODEL_PATH)
    print(f"[INFO] Training complete. Final model saved as {MODEL_PATH}.zip")

    env.close()

if __name__ == "__main__":
    train_agent(total_steps=50_000)
