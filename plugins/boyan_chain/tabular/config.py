# ---------------------------------------------------------------- #

import numpy as np

import os

# ---------------------------------------------------------------- #

data_dir = os.path.join("data", "boyan_chain")

datasets_dir = os.path.join(data_dir, "datasets")
save_dir     = os.path.join(data_dir, "outputs")

save_dir_images = os.path.join(save_dir, "images", "tabular")

# ---------------------------------------------------------------- #

seeds = [0, 1, 2, 3]
num_experience = 100_000
p = 0.1
Ns = [ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ]

tabular_continuous = "tabular"

gammas = np.array([0.9, 0.99, 0.999, 0.9999])

projected = True
modified = True
lam = 1e-6

K = ["e", "c"]
kind = { "e": "episodic", "c": "continuing", }

# ---------------------------------------------------------------- #

get_act = lambda obs: 1 - int( np.random.random() < p )

# ---------------------------------------------------------------- #

std_girth = 0.5
alpha = 0.1

# ---------------------------------------------------------------- #
