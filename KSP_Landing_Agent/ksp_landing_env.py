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

    #---Configuration Constants---#
    MAX_IMPACT_VELOCITY = 4.0 #m/s
    MAX_SPEED_LIMIT = 200.0 #m/s
    EPISODE_TIME_LIMIT = 120.0 #seconds
    MAX_LANDING_ANGLE = 5.0 #MAX degrees of deviation from vertical.

    #---Reward Constants---#
    REWARD_SUCCESS = 200.0
    
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
        # Initialize potential for first step
        self.previous_potential = self._get_potential(obs)
        return obs, {}

    def _get_potential(self, obs: np.ndarray) -> float:
        """
        Calculates a 'potential' value where 0 is the goal (landed) and 1 is the worst case.
        This function is now NORMALIZED.
        """
        # Normalize each observation component to a 0-1 range
        norm_alt = obs[0] / self.observation_space.high[0] # Altitude / 5000m
        v_speed, h_speed = obs[1], obs[2]
        # Normalize total speed to a 0-1 range
        total_speed = np.sqrt(v_speed**2 + h_speed**2)
        norm_speed = total_speed / self.MAX_SPEED_LIMIT # Speed / 200 m/s

        # Potential is a weighted sum of normalized altitude and speed.
        # This keeps the potential value in a small, stable range (approx. 0-2).
        return (norm_alt * 1.0) + (norm_speed * 1.0)
    
    def _reset_episode_state(self):
        self.steps = 0
        self.episode_reward = 0.0
        self.start_time = time.time()
        self.previous_potential = None

    def _wait_for_vessel(self, timeout: float = 30.0):
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
        try:
            obs = np.array([
                self.alt_s(), self.vspeed_s(), self.hspeed_s(), self.fuel_s() / self.fuel_max
            ], dtype=np.float32)
        except (krpc.error.RPCError, krpc.error.StreamError, ValueError):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        throttle = float(np.clip(action[0], 0, 1))
        try:
            self.vessel.control.throttle = throttle
        except krpc.error.RPCError:
            return self._get_obs(), -200, True, False, {}

        time.sleep(self.step_sleep)
        obs = self._get_obs()
        
        terminated = self._check_termination(obs)
        reward = self._compute_reward(obs, throttle, terminated)
        self.episode_reward += reward
        self._log_step(obs, throttle, reward)
        
        truncated = (time.time() - self.start_time) >= self.EPISODE_TIME_LIMIT
        self.steps += 1

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, obs: np.ndarray, throttle: float, terminated: bool) -> float:
        """
        Calculates reward based on landing performance and potential shaping.
        """
        # --- Terminal Rewards (End of Episode) ---
        if terminated:
            # Your existing terminal reward logic is great, so we'll reuse it.
            try:
                v_speed, h_speed = obs[1], obs[2]
                final_pitch = self.pitch_s()
                impact_speed = np.sqrt(v_speed**2 + h_speed**2)
                angle_off_vertical = abs(final_pitch - 90.0)
                is_upright = angle_off_vertical < self.MAX_LANDING_ANGLE
                is_soft_impact = impact_speed < self.MAX_IMPACT_VELOCITY
                
                is_destroyed = self.vessel.parts.controlling is None
                if is_destroyed:
                    return -200.0 # Large penalty for destruction

                if is_soft_impact and is_upright:
                    return self.REWARD_SUCCESS # +200 for a perfect landing
                else:
                    # Smaller penalty for a failed but non-destructive landing
                    return -100.0 
            except krpc.error.RPCError:
                 return -200.0 # Vessel connection lost = crash
        
        # --- Potential-Based Shaping Reward (During Flight) ---
        current_potential = self._get_potential(obs)
        reward_shaping = self.previous_potential - current_potential
        self.previous_potential = current_potential

        # Add a small, constant penalty for using fuel/time to encourage efficiency
        throttle_penalty = -throttle * 0.01

        # The final reward is the shaping reward plus the small efficiency penalty.
        # This will be a small number on each step, preventing reward explosion.
        return reward_shaping + throttle_penalty
    
    def _check_termination(self, obs: np.ndarray) -> bool:
        """More robust check for termination conditions."""
        # NEW: Direct check for ground impact is the most reliable signal
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
        with open(self.flight_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "altitude_m", "v_speed", "h_speed", "throttle", "reward"])

    def _log_step(self, obs: np.ndarray, throttle: float, reward: float):
        if self.logging_enabled:
            with open(self.flight_log_file, "a", newline="") as f:
                writer = csv.writer(f)
                altitude, v_speed, h_speed, _ = obs
                writer.writerow([self.steps, f"{altitude:.2f}", f"{v_speed:.2f}", f"{h_speed:.2f}", f"{throttle:.3f}", f"{reward:.4f}"])

    def archive_log_file(self, archive_step_count: int):
        if self.flight_log_file.exists() and self.flight_log_file.stat().st_size > 0:
            archive_name = self.flight_log_file.with_name(f"landing_flight_log_steps_{archive_step_count}.csv")
            self.flight_log_file.replace(archive_name)
            print(f"🗄️  Archived flight log to {archive_name}")
            self._init_flight_log()

    def enable_logging(self):
        self.logging_enabled = True

    def disable_logging(self):
        self.logging_enabled = False

    def render(self, mode: str = "human"):
        obs = self._get_obs()
        print(f"Step: {self.steps:03d} | Alt: {obs[0]:6.1f}m | VSpd: {obs[1]:6.1f}m/s | HSpd: {obs[2]:6.1f}m/s | Fuel: {obs[3]:.2f}")

    def close(self):
        if self.conn:
            print("Closing kRPC connection.")
            self.conn.close()
            self.conn = None