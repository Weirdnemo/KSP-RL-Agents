import time
import csv
import gymnasium as gym
import numpy as np
import krpc
import os


class HoverEnv(gym.Env):
    """
    Hover environment for KSP.
    Goal: Reach ~500 m altitude and hover as long as possible.
    Observations: [altitude, vertical_speed, fuel_frac]
    Actions: [throttle] in [0, 1].
    Reward: Gaussian reward near 500 m, penalties for overshoot/crash.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, step_sleep=0.2, max_steps=500, log_interval=10_000):
        super().__init__()
        self.step_sleep = step_sleep
        self.max_steps = max_steps
        self.episode_time_limit = 40.0  # seconds
        self.log_interval = log_interval

        # Logging
        self.global_steps = 0
        self.flight_log_file = "hover_flight_log.csv"
        self._init_flight_log()

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
        self.episode_reward = 0.0

        # Bind vessel and streams
        self._bind_vessel()
        self._make_streams()

    # ------------------------
    # RESET
    # ------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.steps > 0:
            print(f"[EP DONE] steps={self.steps}, total_reward={self.episode_reward:.2f}, max_alt={self.max_altitude:.1f}m")

        self.done = False
        self.steps = 0
        self.episode_reward = 0.0
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

        # Activate SAS & engine
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
            return self._get_obs(), 0.0, True, False, {"max_altitude": self.max_altitude}

        throttle = float(np.clip(action[0], 0.0, 1.0))
        self.control.throttle = throttle
        time.sleep(self.step_sleep)

        obs = self._get_obs()
        alt, vspd = obs[0], obs[1]

        # Update max altitude
        if alt > self.max_altitude:
            self.max_altitude = alt

        # Compute reward
        reward, altitude_error = self._compute_reward(alt, vspd)
        self.episode_reward += reward

        # Log step
        self._log_step(alt, throttle, reward)

        # Check time-based termination
        elapsed = time.time() - self.start_time
        if elapsed >= self.episode_time_limit:
            self.done = True

        self.steps += 1
        self.global_steps += 1

        return obs, reward, self.done, False, {
            "max_altitude": self.max_altitude,
            "elapsed_time": elapsed,
            "altitude_error": altitude_error
        }

    # ------------------------
    # REWARD FUNCTION
    # ------------------------
    def _compute_reward(self, alt, vspd):
        altitude_error = abs(alt - 500)
        reward = -0.01 * (altitude_error ** 2)

        if 450 <= alt <= 550:
            reward += 5.0 - 0.1 * altitude_error

        if 450 <= alt <= 550:
            reward -= abs(vspd) * 0.5

        if alt > 400 and vspd > 15:
            reward -= (vspd - 15) * 0.5

        if alt > 600:
            reward -= 200.0
            self.done = True

        if self._check_crash():
            reward -= 200.0
            self.done = True

        return reward, altitude_error

    # ------------------------
    # LOGGING
    # ------------------------
    def _init_flight_log(self):
        with open(self.flight_log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "altitude_m", "throttle", "reward"])

    def _log_step(self, altitude, throttle, reward):
        with open(self.flight_log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.global_steps, altitude, throttle, reward])

        # Save a checkpoint CSV every 10k steps
        if self.global_steps % self.log_interval == 0:
            backup_name = f"hover_flight_log_{self.global_steps//1000}k.csv"
            os.replace(self.flight_log_file, backup_name)
            self._init_flight_log()
            print(f"[LOG] Saved flight log to {backup_name}")

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
