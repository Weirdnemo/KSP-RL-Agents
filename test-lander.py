import time
import krpc

# --- Config ---
TARGET_START_ALT = 1200.0   # desired start altitude for landing episode
CLIMB_MARGIN     = 100.0    # how high above target we climb before cutting throttle
BURN_FALLBACK_S  = 3.0      # seconds to burn if we can't sense climb properly
STEP              = 0.05    # polling interval


def wait_for_scene(sc, timeout=20.0, poll=0.25):
    """Wait until flight scene is active after revert_to_launch."""
    t0 = time.time()
    while True:
        try:
            v = sc.active_vessel
            sit = v.situation.name.lower()
            if sit in ("pre_launch", "flying"):
                return v
        except Exception:
            pass
        if time.time() - t0 > timeout:
            raise TimeoutError("Timed out waiting for flight scene.")
        time.sleep(poll)


def main():
    print("[INFO] Connecting to kRPC...")
    conn = krpc.connect(name="Spawn Landing POC")
    sc = conn.space_center

    # Revert to launch for clean start
    print("[INFO] Reverting to launch...")
    try:
        sc.revert_to_launch()
    except Exception as e:
        raise RuntimeError("Could not revert to launch. Make sure it's enabled.") from e

    # Let scene reload
    time.sleep(2.0)
    vessel = wait_for_scene(sc)
    control = vessel.control

    # Arm SAS to hold vertical
    control.sas = True
    try:
        control.sas_mode = sc.SASMode.stability_assist
    except Exception:
        pass  # some craft/modes may not support this

    # Throttle full, ignite
    print("[INFO] Igniting...")
    control.throttle = 1.0
    control.activate_next_stage()  # launch
    time.sleep(0.25)

    # Streams
    flight_srf = vessel.flight(vessel.surface_reference_frame)
    alt_s = conn.add_stream(getattr, flight_srf, "surface_altitude")
    vs_s = conn.add_stream(getattr, flight_srf, "vertical_speed")

    target_climb_alt = TARGET_START_ALT + CLIMB_MARGIN
    reached_climb = False
    burn_start = time.time()

    print(f"[INFO] Climbing to ~{target_climb_alt:.0f} m...")
    while True:
        alt = alt_s()
        if alt >= target_climb_alt:
            reached_climb = True
            break
        if time.time() - burn_start > BURN_FALLBACK_S:
            print("[WARN] Didn't reach climb target in fallback burn window; proceeding anyway.")
            break
        time.sleep(STEP)

    # Cut engines and coast
    control.throttle = 0.0
    print("[INFO] Burn complete. Coasting...")

    # Wait until descending through TARGET_START_ALT
    print(f"[INFO] Waiting to pass DOWN through {TARGET_START_ALT:.0f} m...")
    while True:
        alt = alt_s()
        vspd = vs_s()
        if vspd < 0 and alt <= TARGET_START_ALT:
            print("[INFO] Descent window reached. Starting landing episode.")
            break
        # Safety break if rocket crashes/explodes before reaching start altitude
        if not _has_command_module(vessel):
            print("[ABORT] Vessel destroyed before reaching start altitude.")
            conn.close()
            return
        time.sleep(STEP)

    # Freeze control; RL agent can take over now
    # (Leave throttle at 0; SAS still holding attitude.)
    start_alt   = alt_s()
    start_vspd  = vs_s()
    fuel        = _fuel_fraction(vessel)

    print("\n===== LANDING EPISODE START STATE =====")
    print(f" Altitude:        {start_alt:.2f} m (surface)")
    print(f" Vertical speed:  {start_vspd:.2f} m/s (negative=descending)")
    print(f" Fuel fraction:   {fuel:.2f}")
    print("=======================================\n")

    # *** HAND OFF TO RL HERE ***
    # Example: call your landing env, passing conn so it attaches to same session.
    # Or just exit and use this logic in the env.reset() you’ll build.

    conn.close()


def _has_command_module(vessel):
    try:
        return len(vessel.parts.with_module("ModuleCommand")) > 0
    except Exception:
        return False


def _fuel_fraction(vessel):
    try:
        max_f = max(1.0, vessel.resources.max("LiquidFuel"))
        cur_f = vessel.resources.amount("LiquidFuel")
        return cur_f / max_f
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
