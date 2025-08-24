import time
import csv
import gymnasium as gym
import numpy as np
import krpc
from pathlib import Path
from typing import Tuple, Dict, Any

class HoverEnv(gym.Env):
    """
    A reinforcement learning environment for hovering a rocket in Kerbal Space Program (KSP).

    The goal is to ascend to a target altitude and maintain a stable hover.
    """
    metadata = {"render_modes": ["human"]}

    # --- Configuration Constants ---
    TARGET_ALTITUDE = 500.0  # meters
    MAX_ALTITUDE_LIMIT = 1000.0
    CRASH_ALTITUDE = 1.0
    MAX_SPEED_LIMIT = 200.0
    EPISODE_TIME_LIMIT = 60.0  # seconds

    # --- Reward Shaping Constants ---
    REWARD_ALTITUDE_SIGMA = 50.0  # Std deviation for Gaussian altitude reward
    REWARD_HOVER_BONUS = 2.0
    PENALTY_VELOCITY = 0.02
    REWARD_ASCENT = 0.05       # New: Encourages climbing
    PENALTY_OVERSHOOT = -200.0
    PENALTY_CRASH = -200.0
    PENALTY_NO_FUEL = -100.0

    def __init__(self, step_sleep: float = 0.2, log_dir: str = "logs"):
        super().__init__()
        self.step_sleep = step_sleep
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.flight_log_file = self.log_dir / "hover_flight_log.csv"
        
        # Action space: [throttle]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32
        )
        # Observation space: [altitude, vertical_speed, fuel_fraction]
        obs_low = np.array([0.0, -self.MAX_SPEED_LIMIT, 0.0], dtype=np.float32)
        obs_high = np.array([self.MAX_ALTITUDE_LIMIT, self.MAX_SPEED_LIMIT, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # KRPC and episode state are initialized in reset()
        self.conn = None
        self.vessel = None
        self._reset_episode_state()
        self._init_flight_log()

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to the initial state for a new episode."""
        super().reset(seed=seed)

        if self.vessel: # Print summary of the previous episode
            print(f"[EPISODE DONE] Steps={self.steps}, Reward={self.episode_reward:.2f}, MaxAlt={self.max_altitude:.1f}m")

        self._reset_episode_state()

        if not self.conn:
            try:
                print("Connecting to kRPC server...")
                self.conn = krpc.connect(name="KSP Hover Env")
            except krpc.error.ConnectionError as e:
                raise RuntimeError("Could not connect to kRPC server. Is KSP running with the server active?") from e
        
        self.sc = self.conn.space_center

        try:
            self.sc.revert_to_launch()
        except RuntimeError:
            print("[WARN] Revert to launch failed. Attempting to load quicksave.")
            self.sc.load("quicksave")
        
        time.sleep(3) 
        self._wait_for_prelaunch()

        self._bind_vessel_and_streams()

        self.vessel.control.sas = True
        self.vessel.control.throttle = 0.0
        self.vessel.control.activate_next_stage()

        obs = self._get_obs()
        return obs, {}
    
    def archive_log_file(self, archive_step_count: int):
        """Renames the current flight log to an archive and starts a new one."""
        if self.flight_log_file.exists() and self.flight_log_file.stat().st_size > 0:
            archive_name = self.flight_log_file.with_name(
                f"hover_flight_log_steps_{archive_step_count}.csv"
            )
            self.flight_log_file.replace(archive_name)
            if self.vessel: # A check to see if we are in an active episode
                print(f"🗄️  Archived flight log to {archive_name}")
            # Start a new log file for the next segment of training
            self._init_flight_log()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one time step."""
        throttle = float(np.clip(action[0], 0, 1))
        try:
            self.vessel.control.throttle = throttle
        except krpc.error.RPCError:
            obs = self._get_obs()
            return obs, self.PENALTY_CRASH, True, False, {"reason": "Vessel destroyed"}

        time.sleep(self.step_sleep)

        obs = self._get_obs()
        alt, v_speed, fuel_frac = obs

        terminated, term_reason = self._check_termination(alt, v_speed, fuel_frac)
        reward = self._compute_reward(alt, v_speed, fuel_frac, terminated, term_reason)
        self.episode_reward += reward

        truncated = (time.time() - self.start_time) >= self.EPISODE_TIME_LIMIT

        self._log_step(alt, throttle, reward)
        self.steps += 1
        if alt > self.max_altitude:
            self.max_altitude = alt

        info = {
            "max_altitude": self.max_altitude,
            "altitude_error": abs(alt - self.TARGET_ALTITUDE),
            "termination_reason": term_reason
        }
        
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, alt: float, v_speed: float, fuel_frac: float, terminated: bool, reason: str) -> float:
        """Calculates the reward for the current state."""
        if terminated:
            if reason == "crashed":
                return self.PENALTY_CRASH
            if reason == "overshot":
                return self.PENALTY_OVERSHOOT
            if reason == "out_of_fuel":
                return self.PENALTY_NO_FUEL
            return 0.0

        # Gaussian reward for being near the target altitude
        altitude_error = alt - self.TARGET_ALTITUDE
        altitude_reward = self.REWARD_HOVER_BONUS * np.exp(-0.5 * (altitude_error / self.REWARD_ALTITUDE_SIGMA)**2)
        
        # Penalty for vertical speed to encourage hovering
        velocity_penalty = self.PENALTY_VELOCITY * abs(v_speed)

        # Reward for moving upward to encourage exploration
        ascent_reward = max(0, v_speed * self.REWARD_ASCENT)
        
        # Combine the rewards
        reward = altitude_reward - velocity_penalty + ascent_reward
        return reward

    def _check_termination(self, alt: float, v_speed: float, fuel_frac: float) -> Tuple[bool, str]:
        """Checks for episode termination conditions."""
        if alt <= self.CRASH_ALTITUDE and self.steps > 1:
            return True, "crashed"
        if alt > self.MAX_ALTITUDE_LIMIT:
            return True, "overshot"
        if fuel_frac <= 0 and alt > self.CRASH_ALTITUDE:
            return True, "out_of_fuel"
        return False, ""

    def _reset_episode_state(self):
        """Resets variables that track episode progress."""
        self.steps = 0
        self.episode_reward = 0.0
        self.start_time = time.time()
        self.max_altitude = 0.0

    def _bind_vessel_and_streams(self):
        """Binds to the active vessel and creates kRPC data streams."""
        self.vessel = self.sc.active_vessel
        flight = self.vessel.flight(self.vessel.orbit.body.reference_frame)

        self.altitude_s = self.conn.add_stream(getattr, flight, 'mean_altitude')
        self.vspeed_s = self.conn.add_stream(getattr, flight, 'vertical_speed')
        
        self.fuel_max = max(1.0, self.vessel.resources.max("LiquidFuel"))
        self.fuel_s = self.conn.add_stream(self.vessel.resources.amount, "LiquidFuel")

    def _get_obs(self) -> np.ndarray:
        """Gets the current observation from the kRPC streams."""
        try:
            obs = np.array([
                self.altitude_s(),
                self.vspeed_s(),
                self.fuel_s() / self.fuel_max
            ], dtype=np.float32)
        except (krpc.error.RPCError, krpc.error.StreamError):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _wait_for_prelaunch(self, timeout: float = 20.0):
        """Waits for the game to be in a controllable 'pre_launch' state."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                if self.sc.active_vessel and self.sc.active_vessel.situation.name == 'pre_launch':
                    return
            except krpc.error.RPCError:
                pass 
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for pre-launch state.")
    
    def _init_flight_log(self):
        """Creates the header for the flight log CSV file."""
        with open(self.flight_log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "altitude_m", "throttle", "reward"])

    def _log_step(self, altitude: float, throttle: float, reward: float):
        """Logs a single step to the flight log file."""
        with open(self.flight_log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.steps, f"{altitude:.2f}", f"{throttle:.3f}", f"{reward:.4f}"])

    def render(self, mode: str = "human"):
        """Prints the current state of the environment to the console."""
        obs = self._get_obs()
        print(
            f"Step: {self.steps:04d} | "
            f"Alt: {obs[0]:7.1f}m | "
            f"V-Spd: {obs[1]:6.1f}m/s | "
            f"Fuel: {obs[2]:.2f} | "
            f"Reward: {self.episode_reward:8.2f}"
        )

    def close(self):
        """Closes the kRPC connection."""
        if self.conn:
            print("Closing kRPC connection.")
            self.conn.close()
            self.conn = None