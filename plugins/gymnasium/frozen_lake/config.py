# ---------------------------------------------------------------- #

import os

import gymnasium as gym

from gymnasium.wrappers.time_limit import TimeLimit

from dice_rl_TU_Vienna.wrappers import LoopingWrapper
from dice_rl_TU_Vienna.dataset import load_or_create_dataset_Experience
from dice_rl_TU_Vienna.applications.stable_baslines.policy import (
    load_or_create_model_PPO, get_TFPolicyPPO_from_env_model, )

from dice_rl_TU_Vienna.applications.gymnasium.frozen_lake.transition_lister import (
    get_transitions_sample_env, get_transitions_sample_dataset, get_transitions_exact, )
from dice_rl_TU_Vienna.applications.gymnasium.frozen_lake.analytical_solver import AnalyticalSolverFrozenLake

# ---------------------------------------------------------------- #

K = ["d", "s"]
names = { "d": "deterministic", "s": "stochastic", }
is_slippery = { "d": False, "s": True, }

data_dir = os.path.join("data", "dice_rl", "frozenlake")

datasets_dir = os.path.join(data_dir, "datasets")
policies_dir = os.path.join(data_dir, "policies")
outputs_dir  = os.path.join(data_dir, "outputs")

save_dir_images = os.path.join(outputs_dir, "images")

total_timesteps = 100_000
num_experience = 100_000
seed = 0

num_trajectory = 1_000
max_trajectory_length = 200

hparam_str_dataset = {
    k: f"{num_experience=}_{seed=}_{is_slippery=}"
        for k, is_slippery in zip(K, [False, True])
}

model_dir = {
    k: os.path.join(policies_dir, f"{total_timesteps=}_{is_slippery=}")
        for k, is_slippery in zip(K, [False, True])
}

dataset_dir = {
    k: os.path.join(datasets_dir, hparam_str_dataset[k])
        for k in K
}

aux_estimates_dir = {
    k: os.path.join(outputs_dir, f"{is_slippery=}")
        for k, is_slippery in zip(K, [False, True])
}

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
    k: get_TFPolicyPPO_from_env_model(
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
