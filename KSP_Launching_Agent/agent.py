import os
import argparse
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from launching.ksp_env import KSPEnv

# -----------------------------
# Custom Callback for Logging & Checkpoints
# -----------------------------
class EpisodeLogger(BaseCallback):
    def __init__(self, checkpoint_freq=2000, verbose=1):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.episode_rewards = []
        self.timesteps_since_last_checkpoint = 0

    def _on_step(self) -> bool:
        # Check if episode ended
        info = self.locals.get("infos", [{}])[-1]
        if "max_altitude" in info:
            max_alt = info["max_altitude"]
            ep_rew = self.locals["rewards"][-1]
            if self.verbose > 0:
                print(f"[LOG] Timestep {self.num_timesteps} | Reward={ep_rew:.2f} | Max Alt={max_alt:.1f} m")
            self.episode_rewards.append((self.num_timesteps, max_alt))

        # Checkpoint saving
        self.timesteps_since_last_checkpoint += 1
        if self.timesteps_since_last_checkpoint >= self.checkpoint_freq:
            self.timesteps_since_last_checkpoint = 0
            ckpt_path = f"ppo_ksp_checkpoint_{self.num_timesteps}.zip"
            self.model.save(ckpt_path)
            if self.verbose > 0:
                print(f"[CHECKPOINT] Saved model at {ckpt_path}")

        return True


# -----------------------------
# Environment Factory
# -----------------------------
def make_env():
    env = KSPEnv()
    env = Monitor(env)  # Tracks episode rewards/length
    env = DummyVecEnv([lambda: env])  # Vectorized env for SB3
    return env


# -----------------------------
# Training Function
# -----------------------------
def train_agent(total_timesteps=10_000):
    env = make_env()

    # If a saved model exists, load it
    model_path = "ppo_ksp_altitude.zip"
    if os.path.exists(model_path):
        print(f"[INFO] Resuming training from {model_path}")
        model = PPO.load(model_path, env=env)
    else:
        print("[INFO] Starting new training session")
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

    callback = EpisodeLogger(checkpoint_freq=2000, verbose=1)

    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Final save
    model.save(model_path)
    print(f"[INFO] Training finished at {total_timesteps} steps. Model saved as '{model_path}'")
    env.close()


# -----------------------------
# Main Entry
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000, help="Total timesteps for training")
    args = parser.parse_args()

    train_agent(total_timesteps=args.timesteps)
