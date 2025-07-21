import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from ksp_env import KSPEnv

# -----------------------------
# Custom Logging Callback
# -----------------------------
class EpisodeLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_max_altitudes = []
        self.episode_durations = []

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [{}])[-1]
        done = self.locals.get("dones", [False])[-1]
        reward = self.locals.get("rewards", [0.0])[-1]

        # Save data if episode ended
        if done:
            ep_info = info if isinstance(info, dict) else {}
            max_alt = ep_info.get("max_altitude", 0.0)
            self.episode_rewards.append(reward)
            self.episode_max_altitudes.append(max_alt)
            duration = time.time() - self.training_env.get_attr("episode_start_time")[0]
            self.episode_durations.append(duration)

            if self.verbose > 0:
                print(
                    f"[LOG] Episode {len(self.episode_rewards)} | "
                    f"Reward={reward:.2f} | Max Alt={max_alt:.1f} m | Duration={duration:.1f} s"
                )
        return True


# -----------------------------
# Create Environment
# -----------------------------
def make_env():
    env = KSPEnv(auto_launch=True, step_sleep=0.2, max_steps=200)
    env = Monitor(env)  # Monitor for stable_baselines
    return env


# -----------------------------
# Training Function
# -----------------------------
def train_agent():
    env = make_env()

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

    callback = EpisodeLogger(verbose=1)
    model.learn(total_timesteps=20_000, callback=callback)

    model.save("ppo_ksp_altitude")
    print("[INFO] Training finished. Model saved as 'ppo_ksp_altitude.zip'")
    env.close()


# -----------------------------
# Test Trained Agent
# -----------------------------
def test_agent():
    env = make_env()
    model = PPO.load("ppo_ksp_altitude", env=env)

    obs, _ = env.reset()
    for step in range(500):
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        env.render()
        if done:
            print(f"[TEST] Episode End | Max Altitude: {info.get('max_altitude', 0):.1f} m")
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    # Train agent (comment this out if you only want to test)
    train_agent()

    # Test trained agent (after training)
    # test_agent()
