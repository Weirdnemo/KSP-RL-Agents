import csv
from pathlib import Path
from stable_baselines3 import PPO

# Import your custom landing environment
from ksp_landing_env import LandingEnv

# --- Configuration ---
# IMPORTANT: Make sure this path points to your final trained agent .zip file
MODEL_PATH = Path("D:/reinforce/KSP/KSP_Landing_Agent/landing_checkpoints/landing_agent_final_21802.zip") 
# The name of the file where the detailed flight data will be saved
OUTPUT_DATA_FILE = Path("test_flight_data.csv")


def test_and_record():
    """Loads a trained PPO agent, runs one full episode, and records the data to a CSV file."""
    if not MODEL_PATH.exists():
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        print("Please make sure the path is correct and you have a trained agent .zip file.")
        return

    print("🚀 Initializing KSP environment for testing...")
    # We don't need the VecMonitor wrapper for testing a single agent
    env = LandingEnv()

    print(f"🧠 Loading trained model from {MODEL_PATH}...")
    model = PPO.load(MODEL_PATH)

    obs, info = env.reset()
    
    print(f"📋 Recording flight data to '{OUTPUT_DATA_FILE}'...")
    with open(OUTPUT_DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        # Write the header row for the CSV file
        writer.writerow([
            "time_step", "altitude", "vertical_speed", "horizontal_speed", "fuel", "throttle", "reward"
        ])

        time_step = 0
        while True:
            # Use deterministic=True for testing to get the agent's most likely action
            action, _ = model.predict(obs, deterministic=True)
            
            # Unpack the action to get the throttle value
            throttle = action[0]
            
            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Get data from the observation array
            altitude, vertical_speed, horizontal_speed, fuel = obs
            
            # Write the current step's data to the CSV
            writer.writerow([
                time_step, altitude, vertical_speed, horizontal_speed, fuel, throttle, reward
            ])
            
            # Print real-time data to the console
            env.render()
            
            time_step += 1
            
            # End the loop if the episode is over
            if terminated or truncated:
                break
    
    print("\n✅ Test episode finished.")
    print(f"Flight data successfully saved to '{OUTPUT_DATA_FILE}'")
    env.close()


if __name__ == "__main__":
    test_and_record()
