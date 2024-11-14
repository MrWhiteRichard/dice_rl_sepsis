import numpy as np

from dice_rl_TU_Vienna.specs import get_observation_action_spec


def get_observation_action_spec_sepsis_amsterdam_continuous(bounds):
    obs_min, obs_max, act_min, act_max = bounds

    return get_observation_action_spec(
        obs_shape=(382,), act_shape=(),
        obs_dtype=np.float32, act_dtype=np.int64,
        obs_min=obs_min, obs_max=obs_max, act_min=act_min, act_max=act_max,
    )
