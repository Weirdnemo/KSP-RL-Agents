import krpc
import time
import csv

# --- Configuration ---
LOG_FREQUENCY_HZ = 1  # How many times per second to log data. 1 = once per second.
OUTPUT_FILE = 'flight_log.csv' # The name of the output file.

def setup_connection():
    """Establishes and returns a connection to the kRPC server."""
    print('Connecting to kRPC server...')
    try:
        # Connect to the server with a specific client name
        conn = krpc.connect(name='Flight Logger')
        print(f'Successfully connected to KSP version {conn.krpc.get_status().version}')
        return conn
    except krpc.error.ConnectionError as e:
        print(f"Error connecting to the server: {e}")
        print("\nPlease make sure KSP is running and the kRPC server is enabled.")
        return None

def flight_logger(conn):
    """Monitors the active vessel and logs its flight data to a CSV file."""
    vessel = conn.space_center.active_vessel
    ref_frame = vessel.orbit.body.reference_frame
    
    print(f"Targeting vessel: {vessel.name}")
    print("Waiting for launch... (Script will start logging when you leave the launchpad)")

    # Wait until the vessel's situation is no longer 'pre_launch'
    while vessel.situation == conn.space_center.VesselSituation.pre_launch:
        time.sleep(1)
    
    print("🚀 Launch detected! Starting flight log...")

    # Set up kRPC data streams for better performance
    met = conn.add_stream(getattr, vessel, 'met')
    altitude = conn.add_stream(getattr, vessel.flight(), 'mean_altitude')
    apoapsis = conn.add_stream(getattr, vessel.orbit, 'apoapsis_altitude')
    periapsis = conn.add_stream(getattr, vessel.orbit, 'periapsis_altitude')
    velocity = conn.add_stream(getattr, vessel.flight(ref_frame), 'speed')
    g_force = conn.add_stream(getattr, vessel.flight(), 'g_force')
    thrust = conn.add_stream(getattr, vessel, 'thrust')
    mass = conn.add_stream(getattr, vessel, 'mass')
    situation = conn.add_stream(getattr, vessel, 'situation')
    
    # Resource streams for the entire vessel
    liquid_fuel = conn.add_stream(vessel.resources.amount, 'LiquidFuel')
    oxidizer = conn.add_stream(vessel.resources.amount, 'Oxidizer')

    # Define the header row for the CSV file
    header = [
        'MET (s)', 'Altitude (m)', 'Apoapsis (m)', 'Periapsis (m)', 
        'Velocity (m/s)', 'G-Force (g)', 'Thrust (N)', 'Mass (kg)', 
        'Liquid Fuel', 'Oxidizer', 'Situation'
    ]
    
    try:
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)

            # Main logging loop: continues until the vessel has landed or splashed down
            while situation() not in [
                conn.space_center.VesselSituation.landed,
                conn.space_center.VesselSituation.splashed
            ]:
                # Collect the current data from all the streams
                log_data = [
                    met(),
                    altitude(),
                    apoapsis(),
                    periapsis(),
                    velocity(),
                    g_force(),
                    thrust(),
                    mass(),
                    liquid_fuel(),
                    oxidizer(),
                    situation().name  # Use .name to get a readable string like 'FLYING'
                ]
                
                # Write the collected data as a new row in the CSV
                writer.writerow(log_data)
                
                # Wait for the next logging interval
                time.sleep(1.0 / LOG_FREQUENCY_HZ)

    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    finally:
        # Important: remove the streams to prevent memory leaks in the kRPC server
        met.remove()
        altitude.remove()
        apoapsis.remove()
        periapsis.remove()
        velocity.remove()
        g_force.remove()
        thrust.remove()
        mass.remove()
        
        situation.remove()
        liquid_fuel.remove()
        oxidizer.remove()
        print(f"✅ Log saved to {OUTPUT_FILE}")

# --- Main execution block ---
if __name__ == '__main__':
    connection = setup_connection()
    if connection:
        flight_logger(connection)
        connection.close()