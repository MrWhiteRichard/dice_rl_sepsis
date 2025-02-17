# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.specs import (
    get_observation_action_spec_tabular,
    get_observation_action_spec_continuous, )

# ---------------------------------------------------------------- #

def get_observation_action_spec_boyan_chain_tabular(N):
    return get_observation_action_spec_tabular(
        n_obs=N+1, n_act=2, )


def get_observation_action_spec_boyan_chain_continuous(N):
    return get_observation_action_spec_continuous(
        obs_min=0, obs_max=1, n_act=2, obs_shape=(N+1,), )


get_observation_action_spec_boyan_chain = {
    "tabular": get_observation_action_spec_boyan_chain_tabular,
    "continuous": get_observation_action_spec_boyan_chain_continuous,
}

# ---------------------------------------------------------------- #
