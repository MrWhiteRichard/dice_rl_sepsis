# ---------------------------------------------------------------- #

import gymnasium as gym

from plugins.stable_baslines.policy import (
    get_TFPolicyPPO_from_env_model, load_or_create_model_PPO, )

from plugins.gymnasium.cartpole.config import *
from dice_rl_TU_Vienna.dataset import load_or_create_dataset_StepsEpisodes
from dice_rl_TU_Vienna.value import get_get_policy_value_env

from dice_rl_TU_Vienna.wrappers import AbsorbingWrapper

from plugins.gymnasium.cartpole.config import *

# ---------------------------------------------------------------- #

env = gym.make('CartPole-v0')
env_wrapped = AbsorbingWrapper(env, absorbing_rew=-1)

get_act_uniform = lambda obs: env.action_space.sample()
model = {
    k: load_or_create_model_PPO(
        model_dir=model_dir[k],
        env=env,
        total_timesteps=total_timesteps[k],
    )
        for k in K
}

get_act_model_b = lambda obs: int( model["b"].predict(obs)[0] )
get_act_model_e = lambda obs: int( model["e"].predict(obs)[0] )

get_act = { "b": get_act_model_b, "e": get_act_model_e, }

dataset = load_or_create_dataset_StepsEpisodes(
    dataset_dir=dataset_dir,
    env=env_wrapped, get_act=get_act["b"],
    num_trajectory=num_trajectory,
    max_trajectory_length=max_trajectory_length,
    by="episodes",
    seed=seed,
)
get_policy_value = {}
rewards = {}

for k in K:
    x, y = get_get_policy_value_env(
        env=env,
        get_act=get_act[k],
        num_trajectory=num_trajectory,
        pad_rew=-1,
        verbosity=1,
    )

    get_policy_value[k] = x
    rewards[k] = y

target_policy = get_TFPolicyPPO_from_env_model(env=env, model=model["e"])

# ---------------------------------------------------------------- #
