import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
# This must match the output file from the test script
DATA_FILE = Path("test_flight_data.csv")

def plot_flight_data():
    """Reads flight data from a CSV and generates performance plots."""
    if not DATA_FILE.exists():
        print(f"❌ Error: Data file not found at '{DATA_FILE}'")
        print("Please run 'test_landing_agent.py' first to generate the flight data.")
        return

    print(f"📊 Reading data from '{DATA_FILE}' and generating plots...")
    df = pd.read_csv(DATA_FILE)

    # Create a figure with 3 subplots, sharing the x-axis for easy comparison
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Landing Agent Performance Analysis', fontsize=16)

    # --- Plot 1: Altitude vs. Time (The "Suicide Burn" Curve) ---
    axs[0].plot(df['time_step'], df['altitude'], label='Altitude', color='dodgerblue', linewidth=2)
    axs[0].axhline(y=0, color='brown', linestyle='--', label='Ground Level')
    axs[0].set_ylabel('Altitude (m)')
    axs[0].legend()
    axs[0].grid(True, linestyle=':')
    axs[0].set_title('Altitude Profile')

    # --- Plot 2: Throttle vs. Time ---
    axs[1].plot(df['time_step'], df['throttle'], label='Throttle', color='green', linewidth=2)
    axs[1].set_ylabel('Throttle (0-1)')
    axs[1].set_ylim(-0.05, 1.05) # Give a little space
    axs[1].legend()
    axs[1].grid(True, linestyle=':')
    axs[1].set_title('Throttle Control')

    # --- Plot 3: Vertical and Horizontal Speed vs. Time ---
    axs[2].plot(df['time_step'], df['vertical_speed'], label='Vertical Speed', color='purple', linewidth=2)
    axs[2].plot(df['time_step'], df['horizontal_speed'], label='Horizontal Speed', color='orange', linestyle=':')
    axs[2].axhline(y=0, color='r', linestyle='--', label='Zero Speed')
    axs[2].set_ylabel('Speed (m/s)')
    axs[2].set_xlabel('Time Steps')
    axs[2].legend()
    axs[2].grid(True, linestyle=':')
    axs[2].set_title('Velocity Control')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    plt.show()

if __name__ == "__main__":
    plot_flight_data()
