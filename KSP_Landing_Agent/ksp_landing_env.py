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
    # Corrected metadata key from 'render.modes' to 'render_modes'
    metadata = {'render_modes': ['human']}

    #---Configuration Constants---#
    MAX_IMPACT_VELOCITY = 4.0 #m/s
    MAX_SPEED_LIMIT = 200.0 #m/s
    EPISODE_TIME_LIMIT = 120.0 #seconds
    MAX_LANDING_ANGLE = 5.0 #MAX degrees of deviation from vertical.

    #---Reward Constants---#
    PENALTY_POSITIVE_VSPEED = -0.8
    REWARD_SUCCESS = 200.0
    PENALTY_CRASH_MULTIPLIER = 10.0
    PENALTY_FUEL_PER_THROTTLE = -0.05
    PENALTY_TIME = -0.1

    def __init__(self, step_sleep: float = 0.2):
        super().__init__()
        self.step_sleep = step_sleep
        self.log_dir = Path("landing_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flight_log_file = self.log_dir / "landing_flight_log.csv"
        
        #---Action Space---#
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32
        )

        #---Observation Space---#
        obs_low = np.array([0.0, -self.MAX_SPEED_LIMIT, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([5000.0, self.MAX_SPEED_LIMIT, self.MAX_SPEED_LIMIT, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        #---KRPC---#
        self.conn = None
        self.vessel = None
        
        #---Logging Control---#
        self.logging_enabled = False
        self._init_flight_log()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment by loading a quicksave to start in mid-air."""
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
        
        time.sleep(2) # Allow time for the scene to load
        self._wait_for_vessel()
        self._bind_vessel_and_streams()
        self.vessel.control.sas = True

        obs = self._get_obs()
        return obs, {}

    def _reset_episode_state(self):
        """Resets variables that track episode progress."""
        self.steps = 0
        self.episode_reward = 0.0
        self.start_time = time.time()

    def _wait_for_vessel(self, timeout: float = 30.0):
        """Waits for the game to be in a controllable state after loading."""
        start = time.time()
        while time.time() - start < timeout:
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
        
        # Use the non-rotating reference frame of the celestial body for all flight data
        ref_frame = self.vessel.orbit.body.reference_frame
        flight = self.vessel.flight(ref_frame)

        # --- Use modern, direct syntax for creating streams ---
        self.alt_s = self.conn.add_stream(getattr, flight, 'surface_altitude')
        self.vspeed_s = self.conn.add_stream(getattr, flight, 'vertical_speed')
        self.hspeed_s = self.conn.add_stream(getattr, flight, 'horizontal_speed')

        # FIX: Get pitch from the correct flight object (surface-based, not orbital)
        self.pitch_s = self.conn.add_stream(getattr, flight, 'pitch')
        
        # Fuel stream is already using the correct syntax
        self.fuel_max = max(1.0, self.vessel.resources.max("LiquidFuel"))
        self.fuel_s = self.conn.add_stream(self.vessel.resources.amount, "LiquidFuel")
    
    def _get_obs(self) -> np.ndarray:
        """Gets the current observation from the kRPC streams."""
        try:
            obs = np.array([
                self.alt_s(), self.vspeed_s(), self.hspeed_s(), self.fuel_s() / self.fuel_max
            ], dtype=np.float32)
        # FIX: Added ValueError to the list of exceptions to catch
        except (krpc.error.RPCError, krpc.error.StreamError, ValueError):
            # If streams fail (e.g., vessel destroyed), return a zeroed observation
            # The _check_termination method will then handle the end of the episode.
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one time step."""
        throttle = float(np.clip(action[0], 0, 1))
        try:
            self.vessel.control.throttle = throttle
        except krpc.error.RPCError:
            return self._get_obs(), -self.PENALTY_CRASH_MULTIPLIER * 50, True, False, {}

        time.sleep(self.step_sleep)
        obs = self._get_obs()
        
        terminated = self._check_termination()
        reward = self._compute_reward(obs, throttle, terminated)
        self.episode_reward += reward
        self._log_step(obs, throttle, reward)
        
        truncated = (time.time() - self.start_time) >= self.EPISODE_TIME_LIMIT
        self.steps += 1

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, obs: np.ndarray, throttle: float, terminated: bool) -> float:
        """
        Calculates a dense reward for the current state to guide the agent.
        """
        # --- Unpack observations for clarity ---
        altitude, v_speed, h_speed, _ = obs

        # --- Terminal Rewards (End of Episode) ---
        if terminated:
            # Get final state information
            try:
                final_pitch = self.pitch_s()
                situation = self.vessel.situation.name
                is_destroyed = self.vessel.parts.controlling is None
            except krpc.error.RPCError:
                # This happens if the vessel is completely gone
                final_pitch = 0 
                situation = "destroyed"
                is_destroyed = True

            # Case 1: Vessel is destroyed or has lost control
            if is_destroyed or situation not in ['landed', 'splashed', 'pre_launch']:
                print(f"💥 Crash! Vessel lost.")
                return -200.0 # Large, fixed penalty for destruction

            # Case 2: Vessel has landed or splashed down
            impact_speed = np.sqrt(v_speed**2 + h_speed**2)
            angle_off_vertical = abs(final_pitch - 90.0)
            is_upright = angle_off_vertical < self.MAX_LANDING_ANGLE
            is_soft_impact = impact_speed < self.MAX_IMPACT_VELOCITY

            # Successful Landing (soft and upright)
            if is_soft_impact and is_upright:
                print(f"✅ SUCCESSFUL LANDING! Speed: {impact_speed:.2f} m/s, Angle: {angle_off_vertical:.1f}°")
                
                # Start with a base success reward
                reward = self.REWARD_SUCCESS # +200

                # Add a bonus for how slow the impact was (max +50)
                # The closer to 0 m/s, the higher the bonus
                velocity_bonus = 50.0 * (1 - (impact_speed / self.MAX_IMPACT_VELOCITY))
                reward += velocity_bonus

                # Add a bonus for how upright the landing was (max +50)
                # The closer to 0 degrees off vertical, the higher the bonus
                angle_bonus = 50.0 * (1 - (angle_off_vertical / self.MAX_LANDING_ANGLE))
                reward += angle_bonus
                
                return reward
            
            # Failed Landing (hard impact or tipped over)
            else:
                reason = []
                if not is_soft_impact: reason.append("fast impact")
                if not is_upright: reason.append("tipped over")
                print(f"💥 Failed Landing! Reason: {', '.join(reason)}. Speed: {impact_speed:.2f} m/s, Angle: {angle_off_vertical:.1f}°")
                
                # Penalty scales with how hard the crash was
                penalty = 100 + (impact_speed * 10)
                return -penalty

        # --- Shaping Rewards (During Flight) ---
        else:
            # 1. Encourage slowing down as altitude decreases.
            # This is a key heuristic for landing: the lower you are, the slower you should be.
            # We define a "safe" vertical speed based on altitude.
            safe_v_speed_target = -max(1.0, altitude / 20.0) # Target speed gets slower closer to the ground
            
            # Reward is higher when actual v_speed is close to the safe target speed
            speed_error = abs(v_speed - safe_v_speed_target)
            # Use an exponential to reward being very close to the target speed
            speed_reward = 0.5 * np.exp(-0.1 * speed_error)

            # 2. Penalize horizontal speed to encourage vertical descent
            h_speed_penalty = -abs(h_speed) * 0.01

            # 3. Penalize fuel usage and time
            fuel_penalty = self.PENALTY_FUEL_PER_THROTTLE * throttle
            time_penalty = self.PENALTY_TIME

            return speed_reward + h_speed_penalty + fuel_penalty + time_penalty
    
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
            archive_name = self.flight_log_file.with_name(
                f"landing_flight_log_steps_{archive_step_count}.csv"
            )
            self.flight_log_file.replace(archive_name)
            print(f"🗄️  Archived flight log to {archive_name}")
            self._init_flight_log()

    def enable_logging(self):
        """Turns on logging for the next episode."""
        self.logging_enabled = True

    def disable_logging(self):
        """Turns off logging."""
        self.logging_enabled = False

    def _check_termination(self) -> bool:
        """Checks if the vessel has landed, crashed, or lost control."""
        try:
            situation = self.vessel.situation
            if situation.name in ['landed', 'splashed', 'pre_launch']:
                return True
            if self.vessel.parts.controlling is None:
                print("🛑 Control lost! Vessel likely destroyed.")
                return True
        except krpc.error.RPCError:
            return True
        return False
        
    def render(self, mode: str = "human"):
        """Prints the current state of the environment to the console."""
        obs = self._get_obs()
        print(
            f"Step: {self.steps:03d} | "
            f"Alt: {obs[0]:6.1f}m | "
            f"VSpd: {obs[1]:6.1f}m/s | "
            f"HSpd: {obs[2]:6.1f}m/s | "
            f"Fuel: {obs[3]:.2f}"
        )

    def close(self):
        """Closes the kRPC connection."""
        if self.conn:
            print("Closing kRPC connection.")
            self.conn.close()
            self.conn = None
