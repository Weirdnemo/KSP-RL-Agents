import time, numpy as np, gymnasium as gym, krpc
from typing import Tuple, Dict, Any

class LandingEnv(gym.env):
    metadata = {'render.modes': ['human']}

    #---Configuration Constants---#
    MAX_IMPACT_VELOCITY = 4.0 #m/s
    MAX_SPEED_LIMIT = 200.0 #m/s
    EPISODE_TIME_LIMIT = 60.0 #seconds

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

        