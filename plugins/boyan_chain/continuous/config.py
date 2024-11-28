# ---------------------------------------------------------------- #

import os

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
