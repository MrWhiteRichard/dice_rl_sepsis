# ---------------------------------------------------------------- #

import os
import gymnasium as gym
from dice_rl_TU_Vienna.estimators.get import get_gammas_2
from plugins.gymnasium.taxi.environment import Taxi

# ---------------------------------------------------------------- #

kinds = ["behavior", "evaluation"]
ups = ["uniform", "policy"]

# ---------------------------------------------------------------- #

id_policy = {
    "behavior": "2025-02-12T13:21:21.067254",
    "evaluation": "2025-02-12T13:23:07.487448",
}

# ---------------------------------------------------------------- #

dir_base = os.path.join("data", "gymnasium", "taxi")
dir_images = os.path.join(dir_base, "images")

dir_policy = {
    kind: os.path.join(dir_base, id_policy[kind])
        for kind in kinds
}

# ---------------------------------------------------------------- #

total_timesteps = {
    "behavior": 10_000,
    "evaluation": 100_000,
}
n_trajectories = 500
max_episode_steps = 200
seed = 0

gammas = get_gammas_2()

projected = True
modified = True
lamda = 1e-6

n_obs = Taxi().n_obs
n_act = Taxi().n_act

# ---------------------------------------------------------------- #

env_title = "Taxi"

colors_OnPE = ["lightgrey", "grey"]
colors_VAFE = ["blue"]
colors_DICE = ["orange", "green", "red"]
colors_IS = ["pink", "deeppink"]
colors_OffPE = colors_VAFE + colors_DICE

markers_OnPE = ["v", "^"]
markers_VAFE = ["1"]
markers_DICE = ["2", "3", "4"]
markers_IS = ["+", "x"]
markers_OffPE = markers_VAFE + markers_DICE

colors_lim_OnPE = colors_OnPE
colors_lim_OffPE = ["orange"]
markers_lim_OnPE = markers_OnPE
markers_lim_OffPE = ["2"]

# ---------------------------------------------------------------- #
