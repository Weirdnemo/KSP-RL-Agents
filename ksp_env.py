import time
import numpy as np
import gymnasium as gym
import krpc


class KSPEnv(gym.Env):
    """
    RL environment for KSP single-stage rocket.
    - Actions: [pitch, yaw] in [-1, 1].
    - Observations: [altitude, vertical_speed, pitch_deg, heading_deg, fuel_frac].
    - Reward: altitude gain per step, penalty when descending.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, auto_launch=True, step_sleep=0.2, max_steps=1500, max_episode_time=35.0):
        super().__init__()
        self.auto_launch = auto_launch
        self.step_sleep = step_sleep
        self.max_steps = max_steps
        self.max_episode_time = max_episode_time  # Time limit in seconds

        # Connect to KRPC
        self.conn = krpc.connect(name="KSP RL Env")
        self.sc = self.conn.space_center

        # Vessel & controls
        self.vessel = None
        self.control = None

        # Action & observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1000.0, -180.0, -180.0, 0.0], dtype=np.float32),
            high=np.array([100000.0, 1000.0, 180.0, 180.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode variables
        self.max_altitude = 0.0
        self.episode_max_alt = 0.0
        self.prev_altitude = 0.0
        self.steps = 0
        self.done = False
        self.episode_start_time = None

        # Streams
        self.altitude_s = None
        self.vspeed_s = None
        self.pitch_s = None
        self.heading_s = None
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
        self.episode_max_alt = 0.0
        self.prev_altitude = 0.0
        self.episode_start_time = time.time()

        # Revert to launch
        try:
            self.sc.revert_to_launch()
        except Exception as e:
            print(f"[WARN] revert_to_launch failed: {e}. Trying quicksave.")
            self.sc.load("quicksave")

        # Wait for the game to load vessel
        time.sleep(5)
        self._wait_for_prelaunch()

        # Bind vessel & streams
        self._bind_vessel()
        self._make_streams()

        # Auto launch
        if self.auto_launch:
            self.control.throttle = 1.0
            self.control.activate_next_stage()

        obs = self._get_obs()
        self.prev_altitude = obs[0]
        return obs, {}

    # --------------------------
    # STEP
    # --------------------------
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {"max_altitude": self.episode_max_alt}

        # Apply actions
        pitch_cmd = float(np.clip(action[0], -1.0, 1.0))
        yaw_cmd = float(np.clip(action[1], -1.0, 1.0))
        self.control.pitch = pitch_cmd
        self.control.yaw = yaw_cmd

        # Step duration
        time.sleep(self.step_sleep)

        # Observations
        obs = self._get_obs()
        alt, vspd = obs[0], obs[1]

        # Update max altitude
        if alt > self.episode_max_alt:
            self.episode_max_alt = alt

        # Reward: altitude gain, penalize descent
        reward = alt - self.prev_altitude
        if vspd < 0:
            reward -= abs(vspd) * 0.1

        self.prev_altitude = alt
        self.steps += 1

        # Termination conditions
        crashed = self._check_crash()
        timeout_steps = self.steps >= self.max_steps
        timeout_time = (time.time() - self.episode_start_time) >= self.max_episode_time

        if crashed or timeout_steps or timeout_time:
            self.done = True
            print(
                f"[EPISODE END] Max Altitude: {self.episode_max_alt:.2f} m | Duration: {time.time() - self.episode_start_time:.1f} s"
            )
            reward += self.episode_max_alt * 0.001
            return obs, reward, True, False, {"max_altitude": self.episode_max_alt}

        return obs, reward, False, False, {}

    # --------------------------
    # RENDER
    # --------------------------
    def render(self, mode="human"):
        print(
            f"Step {self.steps} | Alt: {self.prev_altitude:.1f} m | "
            f"Max: {self.episode_max_alt:.1f} m | Fuel: {self._fuel_frac():.2f}"
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

    def _wait_for_prelaunch(self, poll=0.5, timeout=20):
        start = time.time()
        while True:
            try:
                v = self.sc.active_vessel
                if v and v.situation.name.lower() in ("pre_launch", "flying"):
                    return
            except Exception:
                pass
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for prelaunch state.")
            time.sleep(poll)

    def _make_streams(self):
        flight = self.vessel.flight()
        self.altitude_s = self.conn.add_stream(getattr, flight, "mean_altitude")
        self.vspeed_s = self.conn.add_stream(getattr, flight, "vertical_speed")
        self.pitch_s = self.conn.add_stream(getattr, flight, "pitch")
        self.heading_s = self.conn.add_stream(getattr, flight, "heading")
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
            pitch = float(self.pitch_s())
            heading = float(self.heading_s())
            fuel = self._fuel_frac()
        except Exception:
            alt, vspd, pitch, heading, fuel = 0, 0, 0, 0, 0
            self.done = True

        if alt > self.max_altitude:
            self.max_altitude = alt

        obs = np.array([alt, vspd, pitch, heading, fuel], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _has_command_module(self):
        try:
            return len(self.vessel.parts.with_module("ModuleCommand")) > 0
        except Exception:
            return False

    def _check_crash(self):
        # Crash condition: No command module or altitude near ground
        try:
            if not self._has_command_module():
                return True
            if self.altitude_s() <= 1.0:  # Close to ground
                return True
        except Exception:
            return True
        return False
