import time
import numpy as np
import gymnasium as gym
import krpc
import csv
from typing import Tuple, Dict, Any
from pathlib import Path

class LandingEnv(gym.Env):
    """
    An RL environment for landing a rocket in KSP.
    Goal: Perform a soft, fuel-efficient landing from a starting altitude.
    """
    metadata = {'render_modes': ['human']}

    #--- Configuration Constants ---
    MAX_IMPACT_VELOCITY = 4.0  # m/s
    MAX_SPEED_LIMIT = 200.0    # m/s, used for observation space clipping
    EPISODE_TIME_LIMIT = 120.0 # seconds
    MAX_LANDING_ANGLE = 5.0    # degrees of deviation from vertical

    #--- Reward Constants ---
    REWARD_SUCCESS = 200.0

    def __init__(self, step_sleep: float = 0.2):
        super().__init__()
        self.step_sleep = step_sleep
        self.log_dir = Path("landing_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flight_log_file = self.log_dir / "landing_flight_log.csv"
        
        #--- Action Space (Throttle Control) ---
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32
        )

        #--- Observation Space [altitude, v_speed, h_speed, fuel_percent] ---
        obs_low = np.array([0.0, -self.MAX_SPEED_LIMIT, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([5000.0, self.MAX_SPEED_LIMIT, self.MAX_SPEED_LIMIT, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        #--- kRPC Connection ---
        self.conn = None
        self.vessel = None
        
        #--- Logging Control ---
        self.logging_enabled = False
        self._init_flight_log()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment for a new episode."""
        super().reset(seed=seed)
        if self.vessel:
            print(f"[EPISODE DONE] Steps={self.steps}, Reward={self.episode_reward:.2f}")

        self._reset_episode_state()

        if not self.conn:
            try:
                print("Connecting to kRPC server...")
                self.conn = krpc.connect(name="KSP Landing Env")
            except krpc.error.ConnectionError as e:
                raise RuntimeError("Could not connect to kRPC server.") from e

        self.sc = self.conn.space_center
        print("Loading quicksave to begin landing sequence...")
        self.sc.load("quicksave")
        
        time.sleep(2)
        self._wait_for_vessel()
        self._bind_vessel_and_streams()
        self.vessel.control.sas = True

        obs = self._get_obs()
        # Initialize the altitude tracker for the first step's reward calculation
        self.prev_altitude = obs[0]
        
        return obs, {}

    def _reset_episode_state(self):
        """Resets variables that track episode progress."""
        self.steps = 0
        self.episode_reward = 0.0
        self.start_time = time.time()
        self.prev_altitude = None # This will be set on the first observation

    def _wait_for_vessel(self, timeout: float = 30.0):
        """Waits for the game to be in a controllable state after loading."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.sc.active_vessel and self.sc.active_vessel.situation.name not in ['docked', 'escaping']:
                    return
            except krpc.error.RPCError:
                pass # Ignore errors during scene transitions
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for a controllable vessel.")
    
    def _bind_vessel_and_streams(self):
        """Binds to the active vessel and creates kRPC data streams."""
        self.vessel = self.sc.active_vessel
        ref_frame = self.vessel.orbit.body.reference_frame
        flight = self.vessel.flight(ref_frame)

        self.alt_s = self.conn.add_stream(getattr, flight, 'surface_altitude')
        self.vspeed_s = self.conn.add_stream(getattr, flight, 'vertical_speed')
        self.hspeed_s = self.conn.add_stream(getattr, flight, 'horizontal_speed')
        self.pitch_s = self.conn.add_stream(getattr, flight, 'pitch')
        
        self.fuel_max = max(1.0, self.vessel.resources.max("LiquidFuel"))
        self.fuel_s = self.conn.add_stream(self.vessel.resources.amount, "LiquidFuel")

    def _get_obs(self) -> np.ndarray:
        """Gets the current observation from the kRPC streams, handling errors."""
        try:
            obs = np.array([
                self.alt_s(), self.vspeed_s(), self.hspeed_s(), self.fuel_s() / self.fuel_max
            ], dtype=np.float32)
        except (krpc.error.RPCError, krpc.error.StreamError, ValueError):
            # If vessel is destroyed, streams will fail. Return a zeroed observation.
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advances the environment by one time step."""
        throttle = float(np.clip(action[0], 0, 1))
        try:
            self.vessel.control.throttle = throttle
        except krpc.error.RPCError:
            # If we lose connection while setting throttle, it's a crash
            return self._get_obs(), -200.0, True, False, {}

        time.sleep(self.step_sleep)
        obs = self._get_obs()
        
        terminated = self._check_termination(obs)
        truncated = (time.time() - self.start_time) >= self.EPISODE_TIME_LIMIT
        reward = self._compute_reward(obs, throttle, terminated, truncated)
        
        self.episode_reward += reward
        self._log_step(obs, throttle, reward)
        self.steps += 1

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, obs: np.ndarray, throttle: float, terminated: bool, truncated: bool) -> float:
        """Calculates a direct reward signal for landing behavior."""

        # If the episode is ending due to timeout, apply a large penalty.
        if truncated:
            print("⏰ Episode timed out. Applying penalty.")
            return -100.0 # Make timeout as bad as a failed landing.

        # --- Terminal Rewards (End of Episode) ---
        if terminated:
            try:
                v_speed, h_speed = obs[1], obs[2]
                final_pitch = self.pitch_s()
                impact_speed = np.sqrt(v_speed**2 + h_speed**2)
                angle_off_vertical = abs(final_pitch - 90.0)
                is_upright = angle_off_vertical < self.MAX_LANDING_ANGLE
                is_soft_impact = impact_speed < self.MAX_IMPACT_VELOCITY
                is_destroyed = self.vessel.parts.controlling is None

                if is_destroyed:
                    print("💥 Crash! Vessel destroyed.")
                    return -200.0
                if is_soft_impact and is_upright:
                    print(f"✅ SUCCESSFUL LANDING! Speed: {impact_speed:.2f} m/s, Angle: {angle_off_vertical:.1f}°")
                    return self.REWARD_SUCCESS
                else:
                    print(f"💥 Failed Landing. Speed: {impact_speed:.2f} m/s, Angle: {angle_off_vertical:.1f}°")
                    return -100.0
            except krpc.error.RPCError:
                 print(f"💥 Crash! Vessel connection lost.")
                 return -200.0

        # --- Shaping Rewards (During Flight) ---
        
        # 1. Altitude Change Reward: Directly reward coming down
        altitude = obs[0]
        altitude_reward = (self.prev_altitude - altitude) * 0.1
        self.prev_altitude = altitude
        
        # 2. Velocity Penalty: Penalize high speeds to encourage slowing down
        v_speed, h_speed = obs[1], obs[2]
        total_speed = np.sqrt(v_speed**2 + h_speed**2)
        velocity_penalty = - (total_speed / self.MAX_SPEED_LIMIT) * 0.5
        
        # 3. Proximity Bonus: Small reward for being near the ground
        proximity_bonus = np.exp(-0.01 * altitude) * 0.2

        return altitude_reward + velocity_penalty + proximity_bonus
    
    def _check_termination(self, obs: np.ndarray) -> bool:
        """More robust check for termination conditions."""
        # Direct check for ground impact is the most reliable signal
        if obs[0] <= 0.1: # obs[0] is altitude
            return True

        try:
            # Secondary checks for landed state or loss of control
            situation = self.vessel.situation
            if situation.name in ['landed', 'splashed', 'pre_launch']:
                return True
            if self.vessel.parts.controlling is None:
                print("🛑 Control lost! Vessel likely destroyed.")
                return True
        except krpc.error.RPCError:
            # If we lose connection to the vessel, it's terminated
            return True
        return False
        
    def _init_flight_log(self):
        """Creates the header for the flight log CSV file."""
        with open(self.flight_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "altitude_m", "v_speed", "h_speed", "throttle", "reward"])

    def _log_step(self, obs: np.ndarray, throttle: float, reward: float):
        """Logs a single step to the flight log file if logging is enabled."""
        if self.logging_enabled:
            with open(self.flight_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                altitude, v_speed, h_speed, _ = obs
                writer.writerow([self.steps, f"{altitude:.2f}", f"{v_speed:.2f}", f"{h_speed:.2f}", f"{throttle:.3f}", f"{reward:.4f}"])

    def archive_log_file(self, archive_step_count: int):
        """Renames the current flight log to an archive and starts a new one."""
        if self.flight_log_file.exists() and self.flight_log_file.stat().st_size > 0:
            archive_name = self.flight_log_file.with_name(f"landing_flight_log_steps_{archive_step_count}.csv")
            self.flight_log_file.replace(archive_name)
            print(f"🗄️  Archived flight log to {archive_name}")
            self._init_flight_log()

    def enable_logging(self):
        """Turns on logging for the next episode."""
        self.logging_enabled = True

    def disable_logging(self):
        """Turns off logging."""
        self.logging_enabled = False

    def render(self, mode: str = "human"):
        """Prints the current state of the environment to the console."""
        obs = self._get_obs()
        print(f"Step: {self.steps:03d} | Alt: {obs[0]:6.1f}m | VSpd: {obs[1]:6.1f}m/s | HSpd: {obs[2]:6.1f}m/s | Fuel: {obs[3]:.2f}")

    def close(self):
        """Closes the kRPC connection."""
        if self.conn:
            print("Closing kRPC connection.")
            self.conn.close()
            self.conn = None
