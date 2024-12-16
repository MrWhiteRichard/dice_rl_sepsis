# ---------------------------------------------------------------- #

import gymnasium as gym

from dice_rl_TU_Vienna.dataset import load_or_create_dataset_StepsEpisodes
from dice_rl_TU_Vienna.wrappers import AbsorbingWrapper
from dice_rl_TU_Vienna.plugins.stable_baselines3.policy import (
    get_TFPolicyPPO, load_or_create_model_PPO, )

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

target_policy = get_TFPolicyPPO(env=env, model=model["e"])

# ---------------------------------------------------------------- #
