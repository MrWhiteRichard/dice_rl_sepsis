# ---------------------------------------------------------------- #

import os

# ---------------------------------------------------------------- #

K = ["b", "e"]

base_dir = os.path.join("data", "recycling_robot")
dataframes_dir = os.path.join(base_dir, "dataframes")
policies_dir = os.path.join(base_dir, "policies")

total_timesteps = { "b": 10_000, "e": 100_000, }

model_dir = {
    k: os.path.join(policies_dir, f"total_timesteps={total_timesteps[k]}")
        for k in K
}

# ---------------------------------------------------------------- #
