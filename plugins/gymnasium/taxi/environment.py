# ---------------------------------------------------------------- #

from gymnasium import Env

from infinite_horizon_off_policy_estimation.taxi.environment import taxi

# ---------------------------------------------------------------- #

class Taxi(Env):
    def __init__(self, length=5):
        super().__init__()
        self.env = taxi(length)

    def reset(self, **kwargs):
        super().reset(**kwargs)

        obs_init = self.env.reset()
        info = {}

        return obs_init, info
    
    def step(self, action):
        obs_next, rew = self.env.step(action)

        terminated = False
        truncated = False
        info = {}

        return obs_next, rew, terminated, truncated, info
    
    @property
    def n_obs(self): return self.env.n_state

    @property
    def n_act(self): return self.env.n_action

# ---------------------------------------------------------------- #
