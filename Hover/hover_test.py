import time
import csv
import numpy as np
from stable_baselines3 import PPO
from ksp_hover_env import HoverEnv

MODEL_PATH = "ppo_hover_agent_20000"
CSV_LOG = "hover_test_log.csv"


def test_agent(episodes=1):
    # Create environment
    env = HoverEnv(step_sleep=0.2)
    model = PPO.load(MODEL_PATH, env=env)
    print(f"[INFO] Loaded model: {MODEL_PATH}")

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        steps = 0
        start_time = time.time()

        # Prepare CSV for this episode
        log_file = CSV_LOG.replace(".csv", f"_ep{ep}.csv")
        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "time_s", "altitude_m", "thrust_command", "reward"])

            print(f"\n=== TEST EPISODE {ep}/{episodes} ===")
            print(f"Initial Obs: {obs}")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1
                done = terminated or truncated

                # Log to CSV
                elapsed_time = time.time() - start_time
                writer.writerow([steps, round(elapsed_time, 2), obs[0], action[0], reward])

                time.sleep(0.1)  # Slow down for visualization

        avg_reward = ep_reward / steps if steps > 0 else 0.0
        print(
            f"[EPISODE END] Steps={steps}, Total Reward={ep_reward:.2f}, "
            f"Avg Reward/Step={avg_reward:.2f}, Max Altitude={info.get('max_altitude', 0):.2f} m"
        )
        print(f"[CSV LOG] Saved: {log_file}")

    env.close()


if __name__ == "__main__":
    test_agent(episodes=1)
