# ---------------------------------------------------------------- #

import warnings

import numpy as np

from sb3_contrib.common.wrappers import ActionMasker

from plugins.recycling_robot.environment import RecyclingRobot, mask_array, mask_fn
from plugins.recycling_robot.config import *

from dice_rl_TU_Vienna.plugins.stable_baselines3.policy import get_TFPolicyMaskablePPO
from dice_rl_TU_Vienna.plugins.stable_baselines3.policy import load_or_create_model_MaskablePPO

# ---------------------------------------------------------------- #

env = RecyclingRobot(
    time_step_max=1_000,
    alpha=0.5, beta=1.0,
    r_wait=0,
    r_search_low=lambda: np.random.poisson(1.0),
    r_search_high=lambda: np.random.poisson(3.0),
)

env_masked = ActionMasker(env, mask_fn)

model = {
    k: load_or_create_model_MaskablePPO(
        model_dir=model_dir[k],
        env=env_masked,
        total_timesteps=total_timesteps[k],
    )
        for k in K
}

policy = {
    k: get_TFPolicyMaskablePPO(env, model[k], mask_array)
        for k in K
}

# ---------------------------------------------------------------- #

def get_act(obs, model):
    with warnings.catch_warnings(action="ignore", category=UserWarning):
        action_masks = env_masked.action_masks()

    act, _ = model.predict(obs, action_masks=action_masks)
    act = int(act)
    return act


def get_act_behavior(obs):
    return get_act(obs, model["b"])

def get_act_evaluation(obs):
    return get_act(obs, model["e"])

# ---------------------------------------------------------------- #
