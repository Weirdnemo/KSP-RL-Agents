import time
import csv
import numpy as np
from stable_baselines3 import PPO
from ksp_hover_env import HoverEnv

MODEL_PATH = "ppo_hover_agent_16000"
CSV_LOG = "hover_test_log.csv"

def test_agent(episodes=1):
    # Create environment
    env = HoverEnv(step_sleep=0.2)
    model = PPO.load(MODEL_PATH, env=env)
    print(f"[INFO] Loaded model: {MODEL_PATH}")

    # Prepare CSV logging
    with open(CSV_LOG, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "altitude_m", "thrust_command", "reward"])  # Header

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        steps = 0

        print(f"\n=== TEST EPISODE {ep}/{episodes} ===")
        print(f"Initial Obs: {obs}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated

            # Log altitude and thrust to CSV
            with open(CSV_LOG, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([steps, obs[0], action[0], reward])

            time.sleep(0.1)  # Slow down for visualization

        print(f"[EPISODE END] Steps={steps}, Total Reward={ep_reward:.2f}, Max Altitude={info.get('max_altitude', 0):.2f} m")

    print(f"[INFO] CSV log saved as {CSV_LOG}")
    env.close()

if __name__ == "__main__":
    test_agent(episodes=1)
