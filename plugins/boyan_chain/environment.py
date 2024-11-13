# ---------------------------------------------------------------- #

from gymnasium import Env
from gymnasium.spaces import Discrete

from dice_rl_TU_Vienna.environment import MyTFEnvironment
from dice_rl_TU_Vienna.applications.boyan_chain.specs import get_observation_action_spec
from dice_rl_TU_Vienna.wrappers import AbsorbingWrapper, LoopingWrapper

# ---------------------------------------------------------------- #

class BoyanChain(Env):
    reward_range = (-3, 0)
    action_space = Discrete(2)

    def __init__(self, N, seed=None):
        self.N = N

        self.observation_space = Discrete(self.N + 1)

        self.reset(seed=seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        state = self.observation_space.sample()
        info = {}

        self.s = state
        return state, info

    def step(self, act):
        obs_next = self.get_obs_next(self.s, act)
        rew = self.get_rew(self.s, act, obs_next)
        terminated = obs_next == 0
        truncated = False
        info = {}

        self.s = obs_next
        return obs_next, rew, terminated, truncated, info

    def get_obs_next(self, obs, act):
        if obs >= 2:
            assert act in {0, 1}
            obs_next = obs - 1 - act

        elif obs == 1:
            obs_next = 0

        elif obs == 0:
            obs_next = 0

        else:
            raise NotImplementedError

        return obs_next

    def get_rew(self, obs, act, obs_next):
        A = obs >= 2; B = obs == 1; C = obs == 0

        if A:   rew = -3
        elif B: rew = -2
        elif C: rew = 0
        else: raise NotImplementedError

        return rew

# ---------------------------------------------------------------- #

def get_env(seed, N, kind):
    env = BoyanChain(N, seed)
    if kind == "episodic":   env = AbsorbingWrapper(env)
    if kind == "continuing": env = LoopingWrapper(env)
    return env

# ---------------------------------------------------------------- #

class TFPyBoyanChain(MyTFEnvironment):
    def __init__(self, N, kind, seed=None):
        super().__init__(
            BoyanChain(N, kind, seed), # type: ignore
            *get_observation_action_spec(N) ) # type: ignore

# ---------------------------------------------------------------- #
