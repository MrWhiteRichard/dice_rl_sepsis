# ---------------------------------------------------------------- #

import os

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
