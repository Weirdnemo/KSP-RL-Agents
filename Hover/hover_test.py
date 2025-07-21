import time
import numpy as np
from Hover.ksp_hover_env import HoverEnv  # Assuming your hover env is named like this

EPISODES = 3
TARGET_ALT = 500.0
ALT_DEADZONE = 5.0  # Hover tolerance
BASE_THROTTLE = 0.20  # Reduced to avoid overshoot
K_ALT = 0.0012
K_VSPD = 0.015

def run_pid_test():
    env = HoverEnv(target_altitude=TARGET_ALT, max_steps=300)
    
    for ep in range(EPISODES):
        print(f"\n=== EPISODE {ep+1}/{EPISODES} (mode=pid) ===")
        obs, _ = env.reset()
        alt, vspd, fuel = obs
        print(f"Initial Obs: {obs}")

        total_reward = 0
        steps = 0
        max_alt = alt
        reached_500 = False
        reason = "limit"

        while True:
            alt, vspd, fuel = obs

            # Check for termination
            if alt >= 600:
                reason = "overshoot"
                break
            if alt <= 350 and reached_500:
                reason = "fall_below"
                break

            # PID-style throttle
            alt_error = TARGET_ALT - alt
            throttle = BASE_THROTTLE + (K_ALT * alt_error) - (K_VSPD * vspd)
            throttle = float(np.clip(throttle, 0.0, 1.0))

            obs, reward, done, _, info = env.step([throttle])
            total_reward += reward
            steps += 1
            max_alt = max(max_alt, alt)

            if alt >= 500:
                reached_500 = True

            if done:
                reason = "crash/limit"
                break
        print(f"[EP DONE] steps={steps} return={total_reward:.2f} max_alt={max_alt:.1f}m reason={reason}\n")
    
    env.close()

if __name__ == "__main__":
    run_pid_test()
