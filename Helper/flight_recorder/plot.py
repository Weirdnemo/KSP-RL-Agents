import pandas as pd
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_FILE = 'D:/reinforce/KSP/flight_recorder/flight_log.csv'
OUTPUT_FILE = 'flight_dashboard.png'

def plot_full_dashboard():
    """Reads the flight log and generates a dashboard of all key telemetry."""
    try:
        # Read the flight log data using pandas
        df = pd.read_csv(INPUT_FILE)
        print(f"Successfully loaded '{INPUT_FILE}'. Generating dashboard...")
    except FileNotFoundError:
        print(f"❌ Error: The file '{INPUT_FILE}' was not found.")
        print("Please run the kRPC flight logging script first to generate the data.")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # Create a figure and a grid of subplots. 3 rows, 2 columns.
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle('Full Flight Telemetry Dashboard', fontsize=20, weight='bold')

    # X-axis data for all plots
    time = df['MET (s)']

    # 1. Altitude Profile (Altitude, Apoapsis, Periapsis)
    ax1 = axs[0, 0]
    ax1.plot(time, df['Altitude (m)'], label='Altitude', color='skyblue')
    ax1.plot(time, df['Apoapsis (m)'], label='Apoapsis', color='green', linestyle='--')
    ax1.set_title('Altitude Profile')
    ax1.set_ylabel('Altitude (m)')
    ax1.grid(True, linestyle=':')
    ax1.legend()

    # 2. Velocity Profile
    ax2 = axs[0, 1]
    ax2.plot(time, df['Velocity (m/s)'], label='Velocity', color='orangered')
    ax2.set_title('Velocity Profile')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True, linestyle=':')
    ax2.legend()

    # 3. Forces (Thrust and G-Force) using a twin axis
    ax3 = axs[1, 0]
    ax3_twin = ax3.twinx() # Create a second y-axis
    
    ax3.plot(time, df['Thrust (N)'], label='Thrust', color='cyan')
    ax3_twin.plot(time, df['G-Force (g)'], label='G-Force', color='magenta')
    
    ax3.set_title('Forces Analysis')
    ax3.set_ylabel('Thrust (N)', color='cyan')
    ax3_twin.set_ylabel('G-Force (g)', color='magenta')
    ax3.grid(True, linestyle=':')
    # Manual legend for twin axes
    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3_twin.legend(lines + lines2, labels + labels2, loc='upper right')


    # 4. Mass and Resources
    ax4 = axs[1, 1]
    ax4_twin = ax4.twinx()

    ax4.plot(time, df['Mass (kg)'], label='Total Mass', color='gold')
    ax4_twin.plot(time, df['Liquid Fuel'], label='Liquid Fuel', color='lime', linestyle='-.')
    ax4_twin.plot(time, df['Oxidizer'], label='Oxidizer', color='deepskyblue', linestyle='-.')

    ax4.set_title('Mass & Resources')
    ax4.set_ylabel('Mass (kg)', color='gold')
    ax4_twin.set_ylabel('Resource Units')
    ax4.grid(True, linestyle=':')
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4_twin.legend(lines + lines2, labels + labels2, loc='upper right')


    # 6. Turn off the unused subplot
    axs[2, 1].axis('off')
    axs[2, 0].axis('off')
    # Save the entire figure to a file
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"✅ Dashboard saved successfully as '{OUTPUT_FILE}'")


# --- Main execution block ---
if __name__ == '__main__':
    plot_full_dashboard()