import os
from stable_baselines3 import PPO
from ksp_hover_env import HoverEnv

MODEL_PATH = "ppo_hover_agent_16000.zip"

def continue_training(total_timesteps=50_000):
    env = HoverEnv(step_sleep=0.2)

    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading existing model: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("[WARN] No previous model found, starting new.")
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=1024)

    # Continue training
    model.learn(total_timesteps=total_timesteps)

    # Save the updated model
    model.save(MODEL_PATH)
    print(f"[INFO] Training extended. Model saved to {MODEL_PATH}")

    env.close()


if __name__ == "__main__":
    continue_training(total_timesteps=34_000)  # Finish remaining steps to reach 50k
