import time
import gymnasium as gym
import numpy as np
import krpc


class HoverEnv(gym.Env):
    """
    Hover environment for KSP using apoapsis control.
    Goal: Keep apoapsis near 500 m and maintain low vertical speed (<5 m/s).
    Observations: [apoapsis, vertical_speed, fuel_frac]
    Actions: [throttle] in [0, 1].
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
            low=np.array([0.0, -100.0, 0.0], dtype=np.float32),
            high=np.array([1000.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # Streams
        self.apoapsis_s = None
        self.vspeed_s = None
        self.fuel_s = None
        self.fuel_max = 1.0

        # Episode state
        self.steps = 0
        self.start_time = 0
        self.max_apoapsis = 0.0
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
        self.max_apoapsis = 0.0

        try:
            self.sc.revert_to_launch()
        except Exception as e:
            print(f"[WARN] revert_to_launch failed: {e}. Trying quicksave.")
            self.sc.load("quicksave")

        time.sleep(5)
        self._wait_for_prelaunch()

        self._bind_vessel()
        self._make_streams()

        self.control.sas = True
        self.control.throttle = 0.0
        self.control.activate_next_stage()

        obs = self._get_obs()
        return obs, {}

    # ------------------------
    # STEP
    # ------------------------
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {"max_apoapsis": self.max_apoapsis}

        throttle = float(np.clip(action[0], 0.0, 1.0))
        self.control.throttle = throttle
        time.sleep(self.step_sleep)

        obs = self._get_obs()
        apo, vspd = obs[0], obs[1]

        # Update max apoapsis
        if apo > self.max_apoapsis:
            self.max_apoapsis = apo

        # Reward logic
        reward = 0.0
        apo_error = abs(apo - 500)
        reward -= apo_error * 0.05  # penalty for being away from 500

        if 450 <= apo <= 525:
            reward += 10.0 - apo_error * 0.1  # strong reward near 500

        # Vertical speed penalty
        if apo < 510:  # only damp vertical speed if close
            reward -= abs(vspd) * 0.1

        # Termination conditions
        if apo > 600 or (apo < 350 and self.vspeed_s() < -5):
            reward -= 200.0
            self.done = True

        elapsed = time.time() - self.start_time
        if elapsed >= self.episode_time_limit:
            self.done = True

        self.steps += 1
        return obs, reward, self.done, False, {
            "max_apoapsis": self.max_apoapsis,
            "elapsed_time": elapsed,
            "apo_error": apo_error
        }

    # ------------------------
    # RENDER
    # ------------------------
    def render(self, mode="human"):
        obs = self._get_obs()
        print(
            f"Step {self.steps} | Apoapsis: {obs[0]:.1f} m | Vspd: {obs[1]:.1f} m/s | Fuel: {obs[2]:.2f} | MaxApo: {self.max_apoapsis:.1f} m"
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
        orbit = self.vessel.orbit
        flight = self.vessel.flight()
        self.apoapsis_s = self.conn.add_stream(getattr, orbit, "apoapsis_altitude")
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
            apo = float(self.apoapsis_s())
            vspd = float(self.vspeed_s())
            fuel = self._fuel_frac()
        except Exception:
            apo, vspd, fuel = 0, 0, 0
            self.done = True
        obs = np.array([apo, vspd, fuel], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)
