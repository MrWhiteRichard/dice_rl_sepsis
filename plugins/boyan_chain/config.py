# ---------------------------------------------------------------- #

import os

# ---------------------------------------------------------------- #

kinds = ["episodic", "continuing"]

seeds = [0, 1, 2, 3]
n_samples = 100_000

n_act = 2

prob = 0.1

# ---------------------------------------------------------------- #

dir_base = os.path.join("data", "boyan_chain")
dir_images = {
    ct: os.path.join(dir_base, "images", ct)
        for ct in ["continuous", "tabular"]
}

# ---------------------------------------------------------------- #
