import time
import csv
from pathlib import Path
from typing import Tuple, Dict, Any

import gymnasium as gym
import krpc
import numpy as np

class LandingEnv(gym.Env):
    """
    An RL environment for landing a rocket in KSP, updated with advanced reward shaping.
    Goal: Perform a soft, fuel-efficient, and precise landing from a starting altitude.
    """
    metadata = {'render_modes': ['human']}

    # --- Configuration Constants ---
    MAX_IMPACT_VELOCITY = 4.0  # m/s
    MAX_SPEED_LIMIT = 200.0  # m/s, for observation space clipping
    EPISODE_TIME_LIMIT = 120.0 # seconds
    MAX_LANDING_ANGLE = 5.0  # degrees from vertical

    # --- Reward Function Weights (Tunable) ---
    W_SUCCESS = 300.0  # Large reward for a successful landing
    W_CRASH_IMPACT = -15.0 # Penalty multiplier for impact speed on crash
    W_CRASH_ANGLE = -100.0 # Flat penalty for tipping over on crash
    W_FUEL = -0.1     # Penalty per unit of throttle (encourages efficiency)
    W_TIME = -0.05    # Small penalty per step to encourage speed
    W_VELOCITY = -0.01  # Penalty for speed (shaped by altitude)
    W_POSITION = -0.02  # Penalty for distance from the landing target
    W_ANGLE = -0.5      # Penalty for deviation from retrograde hold
    W_PBRS = 1.0        # Weight for Potential-Based Reward Shaping

    def __init__(self, step_sleep: float = 0.2):
        super().__init__()
        self.step_sleep = step_sleep
        self.log_dir = Path("landing_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flight_log_file = self.log_dir / "landing_flight_log.csv"
        
        # --- Action Space (Throttle Control) ---
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32
        )

        # --- Enriched Observation Space ---
        # [alt, v_speed, h_speed, fuel_pct, pitch, angular_v, dist_target, mass]
        obs_low = np.array([
            0.0, -self.MAX_SPEED_LIMIT, 0.0, 0.0, -90.0, -5.0, 0.0, 0.0
        ], dtype=np.float32)
        obs_high = np.array([
            5000.0, self.MAX_SPEED_LIMIT, self.MAX_SPEED_LIMIT, 1.0, 90.0, 5.0, 5000.0, 20000.0
        ], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --- kRPC ---
        self.conn = None
        self.vessel = None
        self.landing_target_coords = None # Will be set on reset
        
        # --- Logging ---
        self.logging_enabled = False
        self._init_flight_log()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment by loading a quicksave to start in mid-air."""
        super().reset(seed=seed)
        if self.vessel:
            print(f"[EPISODE DONE] Steps={self.steps}, Reward={self.episode_reward:.2f}")

        self._reset_episode_state()
        self._connect_to_krpc()

        print("Loading quicksave 'landing_start' to begin landing sequence...")
        self.sc.load("landing_start")
        
        time.sleep(2) # Allow time for the scene to load
        self._wait_for_vessel()
        self._bind_vessel_and_streams()
        
        # --- Define landing target as the spot directly below the starting point ---
        self.landing_target_coords = self.vessel.position(self.vessel.orbit.body.reference_frame)

        # --- Initial SAS Control ---
        self.vessel.control.sas = True
        time.sleep(0.1) 
        self.vessel.control.sas_mode = self.vessel.control.sas_mode.retrograde

        self.prev_potential = self._calculate_potential(self._get_obs())
        obs = self._get_obs()
        return obs, {}

    def _reset_episode_state(self):
        """Resets variables that track episode progress."""
        self.steps = 0
        self.episode_reward = 0.0
        self.start_time = time.time()
        self.prev_potential = 0.0

    def _connect_to_krpc(self):
        """Initializes the kRPC connection."""
        if not self.conn:
            try:
                print("Connecting to kRPC server...")
                self.conn = krpc.connect(name="KSP Landing Env")
                self.sc = self.conn.space_center
            except krpc.error.ConnectionError as e:
                raise RuntimeError("Could not connect to kRPC server.") from e

    def _wait_for_vessel(self, timeout: float = 30.0):
        """Waits for the game to be in a controllable state after loading."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                if self.sc.active_vessel and self.sc.active_vessel.situation.name not in ['docked', 'escaping']:
                    return
            except krpc.error.RPCError:
                pass 
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for a controllable vessel.")
    
    def _bind_vessel_and_streams(self):
        """Binds to the active vessel and creates kRPC data streams for efficiency."""
        self.vessel = self.sc.active_vessel
        body_frame = self.vessel.orbit.body.reference_frame
        flight = self.vessel.flight(body_frame)

        self.alt_s = self.conn.add_stream(getattr, flight, 'surface_altitude')
        self.vspeed_s = self.conn.add_stream(getattr, flight, 'vertical_speed')
        self.hspeed_s = self.conn.add_stream(getattr, flight, 'horizontal_speed')
        self.pitch_s = self.conn.add_stream(getattr, self.vessel.flight(), 'pitch')
        self.angular_v_s = self.conn.add_stream(getattr, self.vessel, 'angular_velocity', body_frame)
        self.mass_s = self.conn.add_stream(getattr, self.vessel, 'mass')
        self.position_s = self.conn.add_stream(self.vessel.position, body_frame)
        
        self.fuel_max = max(1.0, self.vessel.resources.max("LiquidFuel"))
        self.fuel_s = self.conn.add_stream(self.vessel.resources.amount, "LiquidFuel")

    def _get_obs(self) -> np.ndarray:
        """Gets the current observation vector from the kRPC streams."""
        try:
            # Calculate distance to target in 2D (ground plane)
            pos_vec = self.position_s()
            dist_to_target = np.linalg.norm([pos_vec[1] - self.landing_target_coords[1], pos_vec[2] - self.landing_target_coords[2]])
            
            obs = np.array([
                self.alt_s(),
                self.vspeed_s(),
                self.hspeed_s(),
                self.fuel_s() / self.fuel_max,
                self.pitch_s(),
                np.linalg.norm(self.angular_v_s()),
                dist_to_target,
                self.mass_s()
            ], dtype=np.float32)
        except (krpc.error.RPCError, krpc.error.StreamError, ValueError):
            # If stream fails (e.g., vessel destroyed), return a zeroed state
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advances the environment by one time step."""
        throttle = float(np.clip(action[0], 0, 1))
        try:
            self.vessel.control.throttle = throttle
            if self.vessel.control.sas_mode != self.vessel.control.sas_mode.retrograde:
                self.vessel.control.sas_mode = self.vessel.control.sas_mode.retrograde
        except krpc.error.RPCError:
            # Handle cases where the vessel is destroyed between steps
            return self._get_obs(), self.W_CRASH_IMPACT * 50, True, False, {}

        time.sleep(self.step_sleep)
        obs = self._get_obs()
        
        terminated = self._check_termination(obs)
        reward_info = self._compute_reward(obs, throttle, terminated)
        reward = sum(reward_info.values())
        
        self.episode_reward += reward
        self._log_step(obs, throttle, reward_info)
        
        truncated = (time.time() - self.start_time) >= self.EPISODE_TIME_LIMIT
        self.steps += 1

        return obs, reward, terminated, truncated, {}

    def _calculate_potential(self, obs: np.ndarray) -> float:
        """
        Calculates a 'potential' value for PBRS.
        High potential = good state (low, slow, near target).
        Low potential = bad state (high and fast, or low and fast).
        """
        alt, v_speed, h_speed, _, _, _, dist_target, _ = obs
        # Normalize values to prevent huge potential numbers
        norm_alt = np.clip(alt / 1000.0, 0, 1) # Normalize altitude over 1km
        norm_speed = np.clip(np.sqrt(v_speed**2 + h_speed**2) / 100, 0, 1) # Normalize speed over 100m/s
        norm_dist = np.clip(dist_target / 500.0, 0, 1) # Normalize distance over 500m
        
        # We want low alt, low speed, and low distance to be high potential.
        # So we use (1 - value) for each.
        return (1 - norm_alt) + (1 - norm_speed) + (1 - norm_dist)

    def _compute_reward(self, obs: np.ndarray, throttle: float, terminated: bool) -> Dict[str, float]:
        """Calculates all components of the reward for the current state."""
        alt, v_speed, h_speed, _, pitch, angular_v, dist_target, _ = obs
        rewards = {}

        if terminated:
            impact_speed = np.sqrt(v_speed**2 + h_speed**2)
            is_upright = abs(pitch - 90.0) < self.MAX_LANDING_ANGLE

            if impact_speed < self.MAX_IMPACT_VELOCITY and is_upright:
                print(f"✅ SUCCESSFUL LANDING! Speed: {impact_speed:.2f} m/s")
                rewards["r_success"] = self.W_SUCCESS
            else:
                reason = "fast impact" if impact_speed >= self.MAX_IMPACT_VELOCITY else "tipped over"
                print(f"💥 CRASH! Reason: {reason}. Speed: {impact_speed:.2f} m/s")
                rewards["r_crash_impact"] = self.W_CRASH_IMPACT * impact_speed
                if not is_upright:
                    rewards["r_crash_angle"] = self.W_CRASH_ANGLE
            return rewards
        
        # --- Dense / Shaping Rewards ---
        # 1. PBRS: Reward for moving to a more "promising" state
        current_potential = self._calculate_potential(obs)
        rewards["r_pbrs"] = self.W_PBRS * (current_potential - self.prev_potential)
        self.prev_potential = current_potential

        # 2. Velocity Penalty: Penalize speed, especially at low altitudes
        # The penalty scales inversely with altitude (more penalty when closer to ground)
        speed_penalty_scale = 1 / max(0.1, alt / 500) # Scale up penalty within last 500m
        rewards["r_velocity"] = self.W_VELOCITY * (abs(v_speed) + abs(h_speed)) * speed_penalty_scale

        # 3. Positional Penalty: Penalize distance from the landing site
        rewards["r_position"] = self.W_POSITION * dist_target

        # 4. Orientation Penalty: Penalize high angular velocity
        rewards["r_angle"] = self.W_ANGLE * angular_v

        # 5. Efficiency Penalties: Time and Fuel
        rewards["r_fuel"] = self.W_FUEL * throttle
        rewards["r_time"] = self.W_TIME
        
        return rewards

    def _init_flight_log(self):
        """Creates the header for the detailed flight log CSV file."""
        with open(self.flight_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            headers = ["step", "alt", "v_spd", "h_spd", "fuel", "pitch", "ang_v", "dist_tgt", "mass", "throttle", 
                       "r_total", "r_pbrs", "r_vel", "r_pos", "r_ang", "r_fuel", "r_time"]
            writer.writerow(headers)

    def _log_step(self, obs: np.ndarray, throttle: float, reward_info: Dict[str, float]):
        """Logs a single step with detailed obs and reward components."""
        if self.logging_enabled:
            with open(self.flight_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                total_reward = sum(reward_info.values())
                row = [self.steps] + [f"{x:.2f}" for x in obs] + [
                    f"{throttle:.3f}", f"{total_reward:.4f}",
                    f"{reward_info.get('r_pbrs', 0):.4f}", f"{reward_info.get('r_velocity', 0):.4f}",
                    f"{reward_info.get('r_position', 0):.4f}", f"{reward_info.get('r_angle', 0):.4f}",
                    f"{reward_info.get('r_fuel', 0):.4f}", f"{reward_info.get('r_time', 0):.4f}"
                ]
                writer.writerow(row)

    def archive_log_file(self, archive_step_count: int):
        """Renames the current flight log and starts a new one."""
        if self.flight_log_file.exists() and self.flight_log_file.stat().st_size > 100:
            archive_name = self.log_dir / f"landing_log_steps_{archive_step_count}.csv"
            self.flight_log_file.replace(archive_name)
            print(f"🗄️  Archived flight log to {archive_name}")
            self._init_flight_log()

    def enable_logging(self):
        self.logging_enabled = True

    def disable_logging(self):
        self.logging_enabled = False

    def _check_termination(self, obs: np.ndarray) -> bool:
        """Checks if the vessel has landed, crashed, or lost control."""
        try:
            situation = self.vessel.situation
            # Check for landed/splashed states OR if altitude is near zero with very low v_speed
            if situation.name in ['landed', 'splashed']:
                return True
            # Failsafe for rare cases where situation doesn't update instantly
            if obs[0] < 1.0 and abs(obs[1]) < 0.5:
                return True
            if self.vessel.parts.controlling is None:
                print("🛑 Control lost! Vessel likely destroyed.")
                return True
        except krpc.error.RPCError:
            # This will be thrown if the vessel is destroyed and no longer exists
            return True
        return False
        
    def render(self, mode: str = "human"):
        """Prints the current state of the environment to the console."""
        obs = self._get_obs()
        print(
            f"Step: {self.steps:03d} | "
            f"Alt: {obs[0]:6.1f}m | "
            f"VSpd: {obs[1]:+6.1f}m/s | "
            f"HSpd: {obs[2]:5.1f}m/s | "
            f"Dist: {obs[6]:5.1f}m | "
            f"Fuel: {obs[3]:.2f}"
        )

    def close(self):
        """Closes the kRPC connection."""
        if self.conn:
            print("Closing kRPC connection.")
            self.conn.close()
            self.conn = None