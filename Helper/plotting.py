import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
# The CSV file to read data from
DATA_FILE = Path("D:/reinforce/KSP/Hover/flight_data.csv")

def plot_flight_data():
    """Reads flight data from a CSV and generates performance plots."""
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at {DATA_FILE}")
        print("Please run 'test_agent.py' first to generate the flight data.")
        return

    print(f"📊 Reading data from {DATA_FILE} and generating plots...")
    df = pd.read_csv(DATA_FILE)

    # Create a figure with 3 subplots, sharing the x-axis
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('PPO Hover Agent Performance', fontsize=16)

    # --- Plot 1: Altitude vs. Time ---
    axs[0].plot(df['time_step'], df['altitude'], label='Altitude', color='dodgerblue')
    axs[0].axhline(y=500, color='r', linestyle='--', label='Target Altitude (500m)')
    axs[0].set_ylabel('Altitude (m)')
    axs[0].legend()
    axs[0].grid(True, linestyle=':')

    # --- Plot 2: Throttle vs. Time ---
    axs[1].plot(df['time_step'], df['throttle'], label='Throttle', color='green')
    axs[1].set_ylabel('Throttle (0-1)')
    axs[1].set_ylim(0, 1.1)
    axs[1].legend()
    axs[1].grid(True, linestyle=':')

    # --- Plot 3: Vertical Speed vs. Time ---
    axs[2].plot(df['time_step'], df['vertical_speed'], label='Vertical Speed', color='purple')
    axs[2].axhline(y=0, color='r', linestyle='--', label='Hover (0 m/s)')
    axs[2].set_ylabel('Vertical Speed (m/s)')
    axs[2].set_xlabel('Time Steps')
    axs[2].legend()
    axs[2].grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to make room for suptitle
    plt.show()

if __name__ == "__main__":
    plot_flight_data()