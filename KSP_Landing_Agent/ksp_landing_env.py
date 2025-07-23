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
    Goal: Pass through a 10m 'gate' with low vertical speed.
    """
    metadata = {'render_modes': ['human']}

    #--- Configuration Constants ---
    MAX_GATE_VELOCITY = 10.0   # m/s (Vertical speed at the 10m gate)
    MAX_UPWARD_SPEED = 5.0     # m/s, exceeding this terminates the episode
    MAX_SPEED_LIMIT = 200.0    # m/s, used for observation space clipping
    EPISODE_TIME_LIMIT = 120.0 # seconds

    #--- Reward Constants ---
    REWARD_SUCCESS_GATE = 200.0

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
        self.engines = []
        self.gate_passed = False # This will latch to True on passing the 10m gate
        
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
        self.prev_altitude = obs[0]
        
        return obs, {}

    def _reset_episode_state(self):
        """Resets variables that track episode progress."""
        self.steps = 0
        self.episode_reward = 0.0
        self.start_time = time.time()
        self.prev_altitude = None
        self.engines = []
        self.gate_passed = False # Reset the gate flag for the new episode

    def _wait_for_vessel(self, timeout: float = 30.0):
        """Waits for the game to be in a controllable state after loading."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.sc.active_vessel and self.sc.active_vessel.situation.name not in ['docked', 'escaping']:
                    return
            except krpc.error.RPCError:
                pass
            time.sleep(0.5)
        raise TimeoutError("Timed out waiting for a controllable vessel.")
    
    def _bind_vessel_and_streams(self):
        """Binds to the active vessel and creates kRPC data streams."""
        self.vessel = self.sc.active_vessel
        self.engines = self.vessel.parts.engines
        ref_frame = self.vessel.orbit.body.reference_frame
        flight = self.vessel.flight(ref_frame)

        self.alt_s = self.conn.add_stream(getattr, flight, 'surface_altitude')
        self.vspeed_s = self.conn.add_stream(getattr, flight, 'vertical_speed')
        self.hspeed_s = self.conn.add_stream(getattr, flight, 'horizontal_speed')
        
        self.fuel_max = max(1.0, self.vessel.resources.max("LiquidFuel"))
        self.fuel_s = self.conn.add_stream(self.vessel.resources.amount, "LiquidFuel")

    def _get_obs(self) -> np.ndarray:
        """Gets the current observation from the kRPC streams, handling errors."""
        try:
            obs = np.array([
                self.alt_s(), self.vspeed_s(), self.hspeed_s(), self.fuel_s() / self.fuel_max
            ], dtype=np.float32)
        except (krpc.error.RPCError, krpc.error.StreamError, ValueError):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advances the environment by one time step."""
        throttle = float(np.clip(action[0], 0, 1))
        try:
            self.vessel.control.throttle = throttle
        except krpc.error.RPCError:
            return self._get_obs(), -200.0, True, False, {}

        time.sleep(self.step_sleep)
        obs = self._get_obs()
        
        # NEW: Check for excessive upward velocity
        if obs[1] > self.MAX_UPWARD_SPEED:
            print(f"🚀 Excessive upward velocity! Terminating. V-Speed: {obs[1]:.2f} m/s")
            # Return a large penalty and terminate the episode immediately
            return obs, -150.0, True, False, {}

        terminated = self._check_termination(obs)
        truncated = (time.time() - self.start_time) >= self.EPISODE_TIME_LIMIT
        reward = self._compute_reward(obs, throttle, terminated, truncated)
        
        self.episode_reward += reward
        self._log_step(obs, throttle, reward)
        self.steps += 1

        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, obs: np.ndarray, throttle: float, terminated: bool, truncated: bool) -> float:
        """Calculates reward based on performance at the 10m gate."""
        
        if truncated:
            print("⏰ Episode timed out. Applying penalty.")
            return -100.0
        if terminated:
            try:
                if self.vessel.parts.controlling is None:
                    print("💥 Crash! Vessel destroyed after gate.")
                    return -50.0
            except krpc.error.RPCError:
                return -50.0
            return 0

        altitude = obs[0]
        
        if self.gate_passed:
            return -throttle * 0.5

        if altitude <= 10.0:
            self.gate_passed = True
            print("--- 10m GATE ---")
            v_speed = obs[1]
            is_soft_approach = abs(v_speed) < self.MAX_GATE_VELOCITY
            
            if is_soft_approach:
                reward = self.REWARD_SUCCESS_GATE * (1 - (abs(v_speed) / self.MAX_GATE_VELOCITY))
                print(f"✅ GOOD APPROACH! V-Speed: {abs(v_speed):.2f} m/s. Reward: {reward:.2f}")
                return reward
            else:
                print(f"💥 BAD APPROACH! V-Speed: {abs(v_speed):.2f} m/s")
                return -100.0
        else:
            altitude_reward = (self.prev_altitude - altitude) * 0.1
            self.prev_altitude = altitude
            
            v_speed, h_speed = obs[1], obs[2]
            total_speed = np.sqrt(v_speed**2 + h_speed**2)
            velocity_penalty = - (total_speed / self.MAX_SPEED_LIMIT) * 0.5
            
            proximity_bonus = np.exp(-0.01 * altitude) * 0.2

            return altitude_reward + velocity_penalty + proximity_bonus
    
    def _check_termination(self, obs: np.ndarray) -> bool:
        """Termination is now only ground contact or vessel destruction."""
        if obs[0] <= 0.1:
            return True

        try:
            if self.vessel.parts.controlling is None:
                print("🛑 Control lost! Vessel likely destroyed.")
                return True
            for engine in self.engines:
                if engine.part is None:
                    pass
        except krpc.error.RPCError:
            print("🔥 Engine or critical part destroyed!")
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
