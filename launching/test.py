from launching.ksp_env import KSPEnv
import time
from stable_baselines3 import PPO
import numpy as np

env = KSPEnv(auto_launch=True, step_sleep=0.2, max_steps=2000)

model = PPO.load("ppo_ksp_altitude.zip", env=env)
print("[INFO] Model loaded: ppo_ksp_altitude.zip")

obs, _ = env.reset()
done, truncated = False, False
total_reward = 0.0
step_count = 0

print("[INFO] Starting test flight...")
while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1

    # Print telemetry
    print(
        f"Step {step_count} | Alt: {obs[0]:.1f} m | "
        f"Reward: {reward:.2f} | Max Alt: {info.get('max_altitude', 0):.1f} m"
    )

    time.sleep(0.1)

print("[TEST END] Flight completed.")
print(f"Total Steps: {step_count} | Total Reward: {total_reward:.2f}")
if "max_altitude" in info:
    print(f"Max Altitude Reached: {info['max_altitude']:.2f} m")

env.close()