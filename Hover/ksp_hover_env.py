import time
import gymnasium as gym
import numpy as np
import krpc


class HoverEnv(gym.Env):
    """
    Hover environment for KSP.
    Goal: Reach ~500 m altitude and hover as long as possible.
    - Observations: [altitude, vertical_speed, fuel_frac]
    - Actions: [throttle] in [0, 1].
    - Reward: Encourages being near 500 m and keeping vertical_speed near 0.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, step_sleep=0.2, max_steps=500):
        super().__init__()
        self.step_sleep = step_sleep
        self.max_steps = max_steps
        self.episode_time_limit = 40.0  # seconds

        # KRPC setup
        self.conn = krpc.connect(name="KSP Hover Env")
        self.sc = self.conn.space_center
        self.vessel = None
        self.control = None

        # Action and observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -200.0, 0.0], dtype=np.float32),
            high=np.array([1000.0, 200.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Streams
        self.altitude_s = None
        self.vspeed_s = None
        self.fuel_s = None
        self.fuel_max = 1.0

        # Episode state
        self.steps = 0
        self.start_time = 0
        self.max_altitude = 0.0
        self.done = False

        # Bind vessel and streams
        self._bind_vessel()
        self._make_streams()

    # ------------------------
    # RESET
    # ------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.done = False
        self.steps = 0
        self.start_time = time.time()
        self.max_altitude = 0.0

        try:
            self.sc.revert_to_launch()
        except Exception as e:
            print(f"[WARN] revert_to_launch failed: {e}. Trying quicksave.")
            self.sc.load("quicksave")

        time.sleep(5)
        self._wait_for_prelaunch()

        self._bind_vessel()
        self._make_streams()

        self.control.throttle = 0.0
        self.control.activate_next_stage()

        obs = self._get_obs()
        return obs, {}

    # ------------------------
    # STEP
    # ------------------------
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {"max_altitude": self.max_altitude}

        throttle = float(np.clip(action[0], 0.0, 1.0))
        self.control.throttle = throttle
        time.sleep(self.step_sleep)

        obs = self._get_obs()
        alt, vspd = obs[0], obs[1]

        # Update max altitude
        if alt > self.max_altitude:
            self.max_altitude = alt

        # Reward logic
        reward = 0.0

        # Strong positive reward near 500 m
        altitude_error = abs(alt - 500)
        reward -= altitude_error * 0.05  # penalty for being away from 500

        # Extra reward for being close to 500
        if 450 <= alt <= 525:
            reward += 10.0 - altitude_error * 0.1

        # Penalize vertical speed (we want vspd near 0)
        reward -= abs(vspd) * 0.05

        # Predictive penalty if going too fast near 400–500 m
        if 400 < alt < 500:
            reward -= abs(vspd) * 0.1

        # Heavy penalty and termination if >600 m
        if alt > 600:
            reward -= 200.0
            self.done = True

        # Heavy penalty if crashes
        if self._check_crash():
            reward -= 200.0
            self.done = True

        # Time-based termination
        elapsed = time.time() - self.start_time
        if elapsed >= self.episode_time_limit:
            self.done = True

        self.steps += 1
        return obs, reward, self.done, False, {
            "max_altitude": self.max_altitude,
            "elapsed_time": elapsed,
            "altitude_error": altitude_error
        }

    # ------------------------
    # RENDER
    # ------------------------
    def render(self, mode="human"):
        obs = self._get_obs()
        print(
            f"Step {self.steps} | Alt: {obs[0]:.1f} m | Vspd: {obs[1]:.1f} m/s | Fuel: {obs[2]:.2f} | MaxAlt: {self.max_altitude:.1f} m"
        )

    # ------------------------
    # CLOSE
    # ------------------------
    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # ------------------------
    # Helpers
    # ------------------------
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
        obs = np.array([alt, vspd, fuel], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _check_crash(self):
        try:
            if self.altitude_s() <= 1.0:
                return True
        except Exception:
            return True
        return False
