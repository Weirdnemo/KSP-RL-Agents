import time
import numpy as np
import gym
import krpc


class HoverEnv(gym.Env):
    """
    RL environment for a rocket hover challenge.
    - Actions: [throttle] in [0, 1].
    - Observations: [altitude, vertical_speed, fuel_frac].
    - Goal: Reach 500m altitude and hover as long as possible.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, target_altitude=500.0, max_steps=300, step_sleep=0.2):
        super().__init__()
        self.target_altitude = target_altitude
        self.max_steps = max_steps
        self.step_sleep = step_sleep

        # Connect to KRPC
        self.conn = krpc.connect(name="KSP Hover Env")
        self.sc = self.conn.space_center
        self.vessel = None
        self.control = None

        # Spaces
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1000.0, 0.0], dtype=np.float32),
            high=np.array([1000.0, 1000.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode variables
        self.max_altitude = 0.0
        self.prev_altitude = 0.0
        self.steps = 0
        self.done = False
        self.reached_hover_alt = False

        # Streams
        self.altitude_s = None
        self.vspeed_s = None
        self.fuel_s = None
        self.fuel_max = 1.0

        # Initialize
        self._bind_vessel()
        self._make_streams()

    # --------------------------
    # RESET
    # --------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.steps = 0
        self.max_altitude = 0.0
        self.reached_hover_alt = False

        # Revert to launch
        try:
            self.sc.revert_to_launch()
        except Exception as e:
            print(f"[WARN] revert_to_launch failed: {e}. Trying quicksave.")
            self.sc.load("quicksave")

        time.sleep(5)  # Wait for vessel load
        self._bind_vessel()
        self._make_streams()

        self.control.throttle = 1.0
        self.control.activate_next_stage()

        obs = self._get_obs()
        self.prev_altitude = obs[0]
        print(f"Initial Obs: {obs}")
        return obs, {}

    # --------------------------
    # STEP
    # --------------------------
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # Apply throttle
        throttle_cmd = float(np.clip(action[0], 0.0, 1.0))
        self.control.throttle = throttle_cmd

        # Sleep for step duration
        time.sleep(self.step_sleep)

        # Observations
        obs = self._get_obs()
        alt, vspd, fuel = obs

        # Track max altitude
        if alt > self.max_altitude:
            self.max_altitude = alt

        # Mark when we reach hover altitude
        if alt >= 500.0:
            self.reached_hover_alt = True

        # Reward shaping
        reward = -abs(alt - 500.0) * 0.05  # closer to 500m is better
        reward -= (1.0 - fuel) * 0.01      # penalty for fuel usage

        # Termination conditions
        crashed = alt <= 1.0 or not self._has_command_module()
        timeout = self.steps >= self.max_steps
        too_high = alt > 600.0
        dropped_below = self.reached_hover_alt and alt < 350.0

        if crashed or timeout or too_high or dropped_below:
            self.done = True
            penalty = -50.0 if (too_high or dropped_below or crashed) else 0.0
            reward += penalty
            print(f"[EPISODE END] Max Altitude: {self.max_altitude:.2f} m")
            return obs, reward, True, False, {"max_altitude": self.max_altitude}

        self.steps += 1
        return obs, reward, False, False, {}

    # --------------------------
    # RENDER
    # --------------------------
    def render(self, mode="human"):
        print(
            f"Step {self.steps} | Alt: {self.prev_altitude:.1f} m | "
            f"Max: {self.max_altitude:.1f} m | Fuel: {self._fuel_frac():.2f}"
        )

    # --------------------------
    # CLOSE
    # --------------------------
    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # --------------------------
    # Helpers
    # --------------------------
    def _bind_vessel(self):
        self.vessel = self.sc.active_vessel
        self.control = self.vessel.control

    def _make_streams(self):
        flight = self.vessel.flight()
        self.altitude_s = self.conn.add_stream(getattr, flight, "mean_altitude")
        self.vspeed_s = self.conn.add_stream(getattr, flight, "vertical_speed")
        self.fuel_max = max(1.0, self.vessel.resources.max("LiquidFuel"))
        self.fuel_s = self.conn.add_stream(self.vessel.resources.amount, "LiquidFuel")

    def _fuel_frac(self):
        try:
            return self.fuel_s() / self.fuel_max
        except Exception:
            return 0.0

    def _get_obs(self):
        try:
            alt = float(self.altitude_s())
            vspd = float(self.vspeed_s())
            fuel = self._fuel_frac()
        except Exception:
            alt, vspd, fuel = 0, 0, 0
            self.done = True
        return np.array([alt, vspd, fuel], dtype=np.float32)

    def _has_command_module(self):
        try:
            return len(self.vessel.parts.with_module("ModuleCommand")) > 0
        except Exception:
            return False
