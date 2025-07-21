import time
import numpy as np
import gymnasium as gym
import krpc


class HoverEnv(gym.Env):
    """
    RL environment for hovering at ~500m altitude.
    - Action: [throttle] in [0, 1].
    - Observations: [altitude, vertical_speed, fuel_frac].
    - Reward: Max at 500m, penalties outside 450-525m, heavy penalty >600m.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, step_sleep=0.2):
        super().__init__()
        self.step_sleep = step_sleep
        self.max_steps = int(40 / self.step_sleep)  # 40s episode limit

        # Connect to KRPC
        self.conn = krpc.connect(name="KSP Hover Env")
        self.sc = self.conn.space_center

        # Vessel & controls
        self.vessel = None
        self.control = None

        # Action & observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1000.0, 0.0], dtype=np.float32),
            high=np.array([100000.0, 1000.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Episode variables
        self.max_altitude = 0.0
        self.episode_max_alt = 0.0
        self.steps = 0
        self.done = False

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
        self.episode_max_alt = 0.0

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

        # Auto launch with engine on idle
        self.control.throttle = 0.0
        self.control.activate_next_stage()

        obs = self._get_obs()
        return obs, {}

    # --------------------------
    # STEP
    # --------------------------
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {"max_altitude": self.episode_max_alt}

        # Apply throttle action
        throttle_cmd = float(np.clip(action[0], 0.0, 1.0))
        self.control.throttle = throttle_cmd
        time.sleep(self.step_sleep)

        # Get observations
        obs = self._get_obs()
        alt = obs[0]
        vspd = obs[1]

        # Update max altitude
        self.episode_max_alt = max(self.episode_max_alt, alt)

        # Reward calculation
        reward = self._compute_reward(alt)

        # Step counting
        self.steps += 1
        reason = None

        # Termination conditions
        if alt > 600:
            reward -= 100.0
            self.done = True
            reason = "overshoot_>600"
        elif self.steps >= self.max_steps:
            self.done = True
            reason = "time_out"
        elif self._check_crash():
            self.done = True
            reason = "crash"

        if self.done:
            print(f"[EPISODE END] Max Altitude: {self.episode_max_alt:.2f} m | Reason: {reason}")
            return obs, reward, True, False, {"max_altitude": self.episode_max_alt}

        return obs, reward, False, False, {}

    # --------------------------
    # REWARD FUNCTION
    # --------------------------
    def _compute_reward(self, alt):
        target_alt = 500.0
        if 450 <= alt <= 525:
            dist = abs(alt - target_alt)
            reward = 1.0 - (dist / 50.0)  # Max 1.0 at 500m
            if dist < 10:
                reward += 0.5  # Bonus for staying very close
        else:
            dist = abs(alt - target_alt)
            reward = -dist / 100.0
        return reward

    # --------------------------
    # RENDER
    # --------------------------
    def render(self, mode="human"):
        print(
            f"Step {self.steps} | Alt: {self._safe_val(self.altitude_s):.1f} m | "
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
        self.fuel_max = max(1.0, self.vessel.resources.max("LiquidFuel"))
        self.fuel_s = self.conn.add_stream(self.vessel.resources.amount, "LiquidFuel")

    def _fuel_frac(self):
        try:
            return self.fuel_s() / self.fuel_max
        except Exception:
            return 0.0

    def _safe_val(self, stream):
        try:
            return stream()
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

        self.max_altitude = max(self.max_altitude, alt)
        obs = np.array([alt, vspd, fuel], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _check_crash(self):
        try:
            if not len(self.vessel.parts.with_module("ModuleCommand")) > 0:
                return True
            if self.altitude_s() <= 1.0:
                return True
        except Exception:
            return True
        return False
