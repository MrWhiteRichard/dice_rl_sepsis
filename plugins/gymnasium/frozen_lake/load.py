# ---------------------------------------------------------------- #

import gymnasium as gym

from gymnasium.wrappers.time_limit import TimeLimit

from dice_rl_TU_Vienna.wrappers import LoopingWrapper
from dice_rl_TU_Vienna.dataset import load_or_create_dataset_Experience
from dice_rl_TU_Vienna.plugins.stable_baselines3.policy import (
    load_or_create_model_PPO, get_TFPolicyPPO, )

from plugins.gymnasium.frozen_lake.config import *
from plugins.gymnasium.frozen_lake.analytical_solver import AnalyticalSolverFrozenLake
from plugins.gymnasium.frozen_lake.transition_lister import (
    get_transitions_sample_env, get_transitions_exact, )

# ---------------------------------------------------------------- #

print("env")
env = {
    k: gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=is_slippery[k])
        for k in K
}

print("env_wrapped")
env_wrapped = { k: LoopingWrapper(v) for k, v in env.items() }

print("env_wrapped_limited")
env_wrapped_limited = {
    k: TimeLimit(env=v, max_episode_steps=max_trajectory_length)
        for k, v in env_wrapped.items()
}

print("dataset")
dataset = {
    k: load_or_create_dataset_Experience(
        dataset_dir=dataset_dir[k],
        env=env_wrapped[k],
        num_experience=num_experience,
        seed=seed,
    )
        for k in K
}

print("model")
model = {
    k: load_or_create_model_PPO(
        model_dir=model_dir[k],
        env=env[k],
        total_timesteps=total_timesteps,
    )
        for k in K
}

print("target_policy")
target_policy = {
    k: get_TFPolicyPPO(
        env=env_wrapped[k],
        model=model[k],
    )
        for k in K
}

print("get_act_uniform")
get_act_uniform_d = lambda obs: env["d"].action_space.sample()
get_act_uniform_s = lambda obs: env["s"].action_space.sample()
get_act_uniform = {
    "d": get_act_uniform_d, "s": get_act_uniform_s, }

print("get_act_model")
get_act_model_d = lambda obs: int( model["d"].predict(obs)[0] )
get_act_model_s = lambda obs: int( model["s"].predict(obs)[0] )
get_act_model = {
    "d": get_act_model_d, "s": get_act_model_s, }

# ---------------------------------------------------------------- #


transitions_sample_env = {}
transitions_exact_env = {}

transitions_sample_dataset = {}
transitions_exact_dataset = {}

for k in K:
    print(names[k])

    # env

    ts_env = get_transitions_sample_env(env_wrapped[k], num_transitions=1_000)
    te_env = get_transitions_exact(ts_env)

    transitions_sample_env[k] = ts_env
    transitions_exact_env [k] = te_env

    # # dataset

    # ts_dataset = get_transitions_sample_dataset(dataset[k])
    # te_dataset = get_transitions_exact(ts_dataset)

    # transitions_sample_dataset[k] = ts_dataset
    # transitions_exact_dataset [k] = te_dataset

print("analytical_solver")
analytical_solver = {
    k: AnalyticalSolverFrozenLake(
        model=model[k],
        transitions=transitions_exact_env[k],
    )
        for k in K
}

# ---------------------------------------------------------------- #
