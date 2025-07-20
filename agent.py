"""
Train a PPO agent in Kerbal Space Program to maximize altitude.

Requirements:
- KSP running with kRPC server active
- Your single-stage rocket is the active vessel on the pad
- ksp_env.py in same folder
- stable-baselines3, shimmy, numpy installed

Usage:
    (ksp_rl) python train_ksp.py --timesteps 20000
"""

import os
import csv
import argparse
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger, KVWriter, HumanOutputFormat

from ksp_env import KSPEnv


# ------------------------------------------------------------
# Episode logger callback
# ------------------------------------------------------------
class EpisodeLoggerCallback(BaseCallback):
    """
    Logs per-episode total reward, length, and max altitude (from env info).
    Works with a single environment (which is required for KSP).
    """
    def __init__(self, log_path="ksp_episodes.csv", verbose=1):
        super().__init__(verbose)
        self.log_path = log_path
        self.ep_count = 0
        self.ep_reward = 0.0
        self.ep_len = 0

        # Write header if new file
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["episode", "total_reward", "length_steps", "max_altitude_m"])

    def _init_callback(self) -> None:
        # Nothing special on init
        return

    def _on_step(self) -> bool:
        # locals["rewards"] is a numpy array of shape (n_envs,)
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        # Single env, so index 0 is fine
        r = float(rewards[0])
        self.ep_reward += r
        self.ep_len += 1

        if dones[0]:
            self.ep_count += 1
            max_alt = float(infos[0].get("max_altitude", 0.0))
            if self.verbose > 0:
                print(f"[Episode {self.ep_count}] steps={self.ep_len} reward={self.ep_reward:.2f} max_alt={max_alt:.2f} m")

            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow([self.ep_count, self.ep_reward, self.ep_len, max_alt])

            # reset counters for next episode
            self.ep_reward = 0.0
            self.ep_len = 0

        return True  # continue training


# ------------------------------------------------------------
# Make and wrap env
# ------------------------------------------------------------
def make_env(step_sleep=0.2, max_steps=1500, auto_launch=True):
    # Create your env instance
    env = KSPEnv(auto_launch=auto_launch, step_sleep=step_sleep, max_steps=max_steps)
    return env


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
def train(total_timesteps: int,
          step_sleep: float = 0.2,
          max_steps: int = 1500,
          learning_rate: float = 3e-4,
          log_path: str = "ksp_episodes.csv",
          model_path: str = "ppo_ksp_maxalt",
          auto_launch: bool = True):

    # IMPORTANT: SB3 will try to vectorize env; we give it the raw single env.
    env = make_env(step_sleep=step_sleep, max_steps=max_steps, auto_launch=auto_launch)

    # Callback for episode logging
    callback = EpisodeLoggerCallback(log_path=log_path, verbose=1)

    # PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=64,              # rollout length before update (tune as desired)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device="cpu",            # change to 'cuda' if you have GPU + want to try (physics is bottleneck though)
    )

    # Train
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # Save trained model
    model.save(model_path)
    print(f"Model saved to: {model_path}")


# ------------------------------------------------------------
# Quick evaluation run (optional)
# ------------------------------------------------------------
def evaluate(model_path: str,
             n_episodes: int = 3,
             step_sleep: float = 0.2,
             max_steps: int = 1500,
             auto_launch: bool = True):

    from stable_baselines3 import PPO

    env = make_env(step_sleep=step_sleep, max_steps=max_steps, auto_launch=auto_launch)
    model = PPO.load(model_path, env=env, device="cpu")

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False
        truncated = False
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
        print(f"[Eval Episode {ep}] Reward={ep_reward:.2f}  MaxAlt={info.get('max_altitude', 0):.2f} m")

    env.close()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=20000, help="Total training timesteps.")
    parser.add_argument("--step_sleep", type=float, default=0.2, help="Seconds to wait between KSP physics steps.")
    parser.add_argument("--max_steps", type=int, default=1500, help="Max env steps per episode.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--no_auto_launch", action="store_true", help="If set, env will NOT auto-launch on reset.")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training.")
    args = parser.parse_args()

    auto_launch = not args.no_auto_launch

    train(
        total_timesteps=args.timesteps,
        step_sleep=args.step_sleep,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        auto_launch=auto_launch,
    )

    if args.eval:
        evaluate(
            model_path="ppo_ksp_maxalt",
            n_episodes=3,
            step_sleep=args.step_sleep,
            max_steps=args.max_steps,
            auto_launch=auto_launch,
        )
