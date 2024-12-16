# ---------------------------------------------------------------- #

import os

# ---------------------------------------------------------------- #

data_dir = os.path.join("data", "dice_rl", "cartpole")

datasets_dir = os.path.join(data_dir, "datasets")
policies_dir = os.path.join(data_dir, "policies")
outputs_dir  = os.path.join(data_dir, "outputs")

save_dir_images = os.path.join(outputs_dir, "images")

by = "episodes"

K = ["b", "e"]
names = { "b": "behavior", "e": "evaluation", }

total_timesteps = { "b": 10_000, "e": 100_000, }
num_trajectory = 500
max_trajectory_length = 200
seed = 0

hparam_str_policy = { k: f"total_timesteps={total_timesteps[k]}" for k in K }

hparam_str_dataset = "_".join([
    hparam_str_policy["b"],
    f"{num_trajectory=}", f"{max_trajectory_length=}", f"{seed=}",
])

model_dir = {
    k: os.path.join(policies_dir, hparam_str_policy[k])
        for k in K
}
dataset_dir = os.path.join(datasets_dir, hparam_str_dataset)
save_dir    = os.path.join(outputs_dir,  hparam_str_policy["e"], hparam_str_dataset)

# ---------------------------------------------------------------- #
