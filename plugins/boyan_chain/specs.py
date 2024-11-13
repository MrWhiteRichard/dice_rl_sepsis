# ---------------------------------------------------------------- #

import numpy as np

from dice_rl_TU_Vienna.specs import get_observation_action_spec

# ---------------------------------------------------------------- #

def get_observation_action_spec_boyan_chain(N, tabular_continuous):

    if tabular_continuous == "tabular":
        return get_observation_action_spec(
            obs_shape=(), act_shape=(),
            obs_dtype=np.int64, act_dtype=np.int64,
            obs_min=0, obs_max=N, act_min=0, act_max=1,
        )

    if tabular_continuous == "continuous":
        return get_observation_action_spec(
            obs_shape=(N+1,), act_shape=(),
            obs_dtype=np.float32, act_dtype=np.int64,
            obs_min=0, obs_max=1, act_min=0, act_max=1,
        )

    raise NotImplementedError

# ---------------------------------------------------------------- #
