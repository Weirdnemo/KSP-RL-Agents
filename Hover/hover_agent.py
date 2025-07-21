import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from ksp_hover_env import HoverEnv  # Import our hover environment


# --------------------------
# Episode Logger
# --------------------------
class HoverLogger(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        done_array = self.locals.get("dones")
        info_array = self.locals.get("infos")

        for done, info in zip(done_array, info_array):
            if done:
                self.episode_count += 1
                max_alt = info.get("max_altitude", 0)
                total_rew = info.get("episode_reward", 0)
                print(f"[LOG] Episode {self.episode_count} | Reward={total_rew:.2f} | Max Alt={max_alt:.1f} m")
        return True


# --------------------------
# Training
# --------------------------
def train_hover_agent(timesteps=50_000):
    env = HoverEnv()
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
    )

    callback = HoverLogger(verbose=1)
    model.learn(total_timesteps=timesteps, callback=callback)

    model.save("ppo_hover_agent")
    print("[INFO] Training complete. Model saved as 'ppo_hover_agent.zip'")
    env.close()


if __name__ == "__main__":
    train_hover_agent(timesteps=50_000)
