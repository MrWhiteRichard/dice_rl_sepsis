# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.specs import (
    get_observation_action_spec_tabular,
    get_observation_action_spec_continuous, )

# ---------------------------------------------------------------- #

def get_observation_action_spec_boyan_chain_tabular(N):
    bounds = 0, N, 0, 1
    return get_observation_action_spec_tabular(bounds)


def get_observation_action_spec_boyan_chain_continuous(N):
    bounds = 0, 1, 0, 1
    shapes = (N+1,), ()
    return get_observation_action_spec_continuous(bounds, shapes)


get_observation_action_spec_boyan_chain = {
    "tabular": get_observation_action_spec_boyan_chain_tabular,
    "continuous": get_observation_action_spec_boyan_chain_continuous,
}

# ---------------------------------------------------------------- #
