# ---------------------------------------------------------------- #

import numpy as np

from gymnasium import Env
from gymnasium.spaces import Discrete

from typing import Any, SupportsFloat

# ---------------------------------------------------------------- #

class RecyclingRobot(Env):
    """
    Sutton & Barto "Recycling Robot" page 52
    observation_space:
        low: 0
        high: 1
    action_space:
        wait: 0
        search: 1
        recharge: 2
    """

    def __init__(self, time_step_max, alpha, beta, r_wait, r_search_low, r_search_high):
        """
        Args:
            time_step_max: once reached, step outputs `done = True`
            alpha: probability of staying high when searching high
            beta:  probability of staying low when searching low
            r_wait: reward (function) when waiting
            r_search_low: reward (function) when searching low
            r_search_high: reward (function) when searching high
        """
        super().__init__()

        self._time_step_max = time_step_max
        self.time_step_current = 0

        assert 0 <= alpha <= 1
        assert 0 <= beta <= 1

        self._alpha = alpha
        self._beta = beta

        self._r_wait = r_wait
        self._r_search_low = r_search_low
        self._r_search_high = r_search_high

        self.observation_space = Discrete(2)
        self.action_space = Discrete(3)

        self._step_dict = {
            (0, 0): lambda: ( 0, self.get_reward("wait") ),
            (0, 1): lambda: ( 0, self.get_reward("search_low") )
                if np.random.random() < self._beta
                else (1, -3),
            (0, 2): lambda: (1, 0),
            (1, 0): lambda: ( 1, self.get_reward("wait") ),
            (1, 1): lambda: ( 1, self.get_reward("search_high") )
                if np.random.random() < self._alpha
                else ( 0, self.get_reward("search_high") ),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        self.time_step_current = 0
        self.obs_current = 0

        return self.obs_current, {}

    def get_reward(self, act_str):
        if act_str == "wait":        r_ = self._r_wait
        if act_str == "search_low":  r_ = self._r_search_low
        if act_str == "search_high": r_ = self._r_search_high

        if callable(r_): return r_()
        else: return r_

    def step(self, act: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.time_step_current += 1

        obs_next, reward = self._step_dict[self.obs_current, act]()
        done = self.time_step_current >= self._time_step_max

        self.obs_current = obs_next

        return obs_next, reward, done, False, {}

# ---------------------------------------------------------------- #

mask_array = np.array([
    [True, True, True],
    [True, True, False],
])

def mask_fn(env: Env) -> np.ndarray:
    return mask_array[env.obs_current] # type: ignore

# ---------------------------------------------------------------- #
