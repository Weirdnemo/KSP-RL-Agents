import csv
from pathlib import Path
from stable_baselines3 import PPO

# Assuming your refactored environment is in ksp_hover_env.py
from ksp_hover_env import HoverEnv

# --- Configuration ---
# Make sure this path points to your trained agent .zip file
MODEL_PATH = Path("checkpoints/ppo_hover_agent_final.zip") 
# The name of the file where flight data will be saved
OUTPUT_DATA_FILE = Path("flight_data.csv")


def test_and_record():
    """Loads a trained PPO agent, runs one episode, and records the data."""
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("Please make sure you have a trained agent .zip file at that location.")
        return

    print("🚀 Initializing environment for testing...")
    env = HoverEnv()

    print(f"🧠 Loading trained model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()
    
    print("📋 Recording flight data to", OUTPUT_DATA_FILE)
    with open(OUTPUT_DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header row for the CSV file
        writer.writerow([
            "time_step", "altitude", "vertical_speed", "fuel", "throttle", "reward"
        ])

        time_step = 0
        while True:
            # Use deterministic=True for testing to get the agent's "best" action
            action, _ = model.predict(obs, deterministic=True)
            
            # Unpack the action to get the throttle value
            throttle = action[0]
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get data from the observation array
            altitude, vertical_speed, fuel = obs
            
            # Write the current step's data to the CSV
            writer.writerow([
                time_step, altitude, vertical_speed, fuel, throttle, reward
            ])
            
            # Print real-time data to the console
            env.render()
            
            time_step += 1
            
            if terminated or truncated:
                break
    
    print("\n✅ Episode finished.")
    print(f"Flight data successfully saved to {OUTPUT_DATA_FILE}")
    env.close()


if __name__ == "__main__":
    test_and_record()