import numpy as np

from dice_rl_TU_Vienna.specs import get_observation_action_spec


def get_observation_action_spec_RRT_mimic_iv_continuous(
        obs_min, obs_max, act_min, act_max):

    return get_observation_action_spec(
        obs_shape=(47,), act_shape=(),
        obs_dtype=np.float32, act_dtype=np.int64,
        obs_min=obs_min, obs_max=obs_max, act_min=act_min, act_max=act_max,
    )

def get_observation_action_spec_RRT_mimic_iv_tabular(
        obs_min, obs_max, act_min, act_max):

    return get_observation_action_spec(
        obs_shape=(), act_shape=(),
        obs_dtype=np.int32, act_dtype=np.int64,
        obs_min=obs_min, obs_max=obs_max, act_min=act_min, act_max=act_max,
    )

get_observation_action_spec_RRT_mimic_iv = {
    "tabular":    get_observation_action_spec_RRT_mimic_iv_tabular,
    "continuous": get_observation_action_spec_RRT_mimic_iv_continuous,
}