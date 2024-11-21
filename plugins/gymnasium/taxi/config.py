# ---------------------------------------------------------------- #

import os

from plugins.dice_rl.dataset import get_hparam_str_dataset

# ---------------------------------------------------------------- #

data_dir = os.path.join("data", "dice_rl", "taxi")

datasets_dir = os.path.join(data_dir, "datasets")
save_dir     = os.path.join(data_dir, "outputs")
policies_dir = os.path.join(data_dir, "policies")

save_dir_images = os.path.join(save_dir, "images")

# ---------------------------------------------------------------- #

env_name = "taxi"
seed = 0
num_trajectory = 1_000
max_trajectory_length = 100
tabular_obs = True

K = ["b", "e"]
names = { "b": "behavior", "e": "evaluation", }
alphas = { "b": 0.0, "e": 1.0, }

hparam_str = {
    k: get_hparam_str_dataset(
        env_name, tabular_obs, alphas[k], seed, num_trajectory, max_trajectory_length,
    )
        for k in K
}

aux_estimates_dir = {
    k: os.path.join(save_dir, hparam_str[k])
        for k in K
}

# ---------------------------------------------------------------- #
