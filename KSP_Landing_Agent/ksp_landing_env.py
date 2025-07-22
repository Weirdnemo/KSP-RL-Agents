import time, numpy as np, gymnasium as gym, krpc
from typing import Tuple, Dict, Any

class LandingEnv(gym.env):
    metadata = {'render.modes': ['human']}

    #---Configuration Constants---#
    MAX_IMPACT_VELOCITY = 4.0 #m/s
    MAX_SPEED_LIMIT = 200.0 #m/s
    EPISODE_TIME_LIMIT = 60.0 #seconds
    MAX_LANDING_ANGLE = 5.0 #MAX degrees of deviation from vertical.

    #---Reward Constants---#
    REWARD_SUCCESS = 200.0
    PENALTY_CRASH_MULTIPLIER = 10.0
    PENALTY_FUEL_PER_THROTTLE = -0.05
    PENALTY_TIME = -0.1

    def __init__(self, step_sleep: float = 0.2):
        super().__init__()
        self.step_sleep = step_sleep

        #---Action Space---#
        self.action_space = gym.spaces.Box(
            low=np.array([0.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        #---Observation Space---#
        obs_low = np.array([0.0, -self.MAX_SPEED_LIMIT, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([5000.0, self.MAX_SPEED_LIMIT, self.MAX_SPEED_LIMIT, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )

        #---KRPC---#
        self.conn = None
        self.vessel = None

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
        # Use a reference frame that follows the vessel's orientation
        ref_frame = self.vessel.orbit.body.reference_frame
        flight = self.vessel.flight(ref_frame)

        # Create streams for performance
        self.alt_s = self.conn.add_stream(getattr, flight, 'surface_altitude')
        self.vspeed_s = self.conn.add_stream(getattr, flight, 'vertical_speed')
        self.hspeed_s = self.conn.add_stream(getattr, flight, 'horizontal_speed')
        self.pitch_s = self.conn.add_stream(getattr, self.vessel.flight(), 'pitch')
        self.fuel_max = max(1.0, self.vessel.resources.max("LiquidFuel"))
        self.fuel_s = self.conn.add_stream(self.vessel.resources.amount, "LiquidFuel")

def _get_obs(self) -> np.ndarray:
    """Gets the current observation from the kRPC streams."""
    try:
        obs = np.array([
            self.alt_s(),
            self.vspeed_s(),
            self.hspeed_s(),
            self.fuel_s() / self.fuel_max
        ], dtype=np.float32)
    except (krpc.error.RPCError, krpc.error.StreamError):
        # If stream fails, return a zeroed state
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
    
    return np.clip(obs, self.observation_space.low, self.observation_space.high)

def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one time step."""
        throttle = float(np.clip(action[0], 0, 1))
        try:
            self.vessel.control.throttle = throttle
        except krpc.error.RPCError: # Handle case where vessel is unexpectedly destroyed
            return self._get_obs(), -self.PENALTY_CRASH_MULTIPLIER * 50, True, False, {}

        time.sleep(self.step_sleep)

        obs = self._get_obs()
        
        terminated = self._check_termination()
        reward = self._compute_reward(obs, throttle, terminated)
        self.episode_reward += reward

        truncated = (time.time() - self.start_time) >= self.EPISODE_TIME_LIMIT
        self.steps += 1

        return obs, reward, terminated, truncated, {}

def _compute_reward(self, obs: np.ndarray, throttle: float, terminated: bool) -> float:
    """Calculates the reward for the current state."""
    _, v_speed, h_speed, _ = obs
# The new version in _compute_reward()
    if terminated:
        # Get final orientation and speed
        final_pitch = self.pitch_s()
        impact_speed = np.sqrt(v_speed**2 + h_speed**2)
    
        # Check if the vessel is upright
        is_upright = abs(final_pitch - 90.0) < self.MAX_LANDING_ANGLE
    
        # A successful landing must be BOTH slow and upright
        if impact_speed < self.MAX_IMPACT_VELOCITY and is_upright:
            print(f"✅ SUCCESSFUL LANDING! Speed: {impact_speed:.2f} m/s, Angle: {90 - final_pitch:.1f}° off vertical")
            return self.REWARD_SUCCESS
        else:
            # If either condition fails, it's a crash
            reason = "fast impact" if not impact_speed < self.MAX_IMPACT_VELOCITY else "tipped over"
            print(f"💥 Crash! Reason: {reason}. Speed: {impact_speed:.2f} m/s, Angle: {90 - final_pitch:.1f}° off vertical")
            # Add an extra penalty if it tipped over
            tip_penalty = 50 if not is_upright else 0
            return -self.PENALTY_CRASH_MULTIPLIER * impact_speed - tip_penalty
    
    # While in-air, give small penalties to encourage efficiency
    shaping_reward = self.PENALTY_TIME + (self.PENALTY_FUEL_PER_THROTTLE * throttle)
    return shaping_reward
def _check_termination(self) -> bool:
    """Checks if the vessel has landed, splashed, or crashed."""
    try:
        situation = self.vessel.situation
        # 'pre_launch' is what a crash on land often registers as
        return situation.name in ['landed', 'splashed', 'pre_launch']
    except krpc.error.RPCError:
        return True
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