# ---------------------------------------------------------------- #

import os

from dice_rl_TU_Vienna.dataset import load_or_create_dataset_Experience
from plugins.boyan_chain.policy import TFPolicyBoyanChain
from plugins.boyan_chain.environment import get_env
from plugins.boyan_chain.analytical_solver import AnalyticalSolverBoyanChain

from plugins.boyan_chain.specs import get_observation_action_spec_boyan_chain
from dice_rl_TU_Vienna.dataset import get_dataset_spec

# ---------------------------------------------------------------- #

data_dir = os.path.join("data", "boyan_chain")

datasets_dir = os.path.join(data_dir, "datasets")
outputs_dir  = os.path.join(data_dir, "outputs")

save_dir_images = os.path.join(outputs_dir, "images", "continuous")

# ---------------------------------------------------------------- #

seeds = [0, 1, 2, 3]
num_experience = 100_000
p = 0.1
N = 12

tabular_continuous = "continuous"

by = "experience"

K = ["e", "c"]
kind = { "e": "episodic", "c": "continuing", }

# ---------------------------------------------------------------- #

hparam_str_dataset = {
    seed: {
        k: f"{seed=}_{num_experience=}_{N=}_kind={kind[k]}"
            for k in K
    }
        for seed in seeds
}
hparam_str_policy = f"{p=}_{N=}"


dataset_dir = {
    seed: {
        k: os.path.join(datasets_dir, hparam_str_dataset[seed][k])
            for k in K
    }
        for seed in seeds
}

# ---------------------------------------------------------------- #

observation_spec, action_spec = get_observation_action_spec_boyan_chain(N, tabular_continuous)
dataset_spec = get_dataset_spec(observation_spec, action_spec, step_num_max=2)

env = {
    seed: {
        k: get_env(seed=seed, N=N, kind=kind[k])
            for k in K
    }
        for seed in seeds
}

dataset = {
    seed: {
        k: load_or_create_dataset_Experience(
            dataset_dir=dataset_dir[seed][k],
            env=env[seed][k], num_experience=num_experience, seed=seed,
        )
            for k in K
    }
        for seed in seeds
}

target_policy = TFPolicyBoyanChain(
    N=N, p=p, tabular_continuous=tabular_continuous)

analytical_solver = {
    k: AnalyticalSolverBoyanChain(N=N, p=p, kind=kind[k])
        for k in K
}

# ---------------------------------------------------------------- #
