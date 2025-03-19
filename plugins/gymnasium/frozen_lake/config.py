# ---------------------------------------------------------------- #

import os
import gymnasium as gym
from dice_rl_TU_Vienna.estimators.get import get_gammas_log10
from dice_rl_TU_Vienna.plugins.stable_baselines3.specs import get_specs_env

# ---------------------------------------------------------------- #

kinds = ["deterministic", "stochastic"]
ups = ["uniform", "policy"]
bys = ["by_samples", "by_episodes"]
is_slippery = { "deterministic": False, "stochastic": True, }

id_env = {
    "deterministic": "2025-02-07T14:39:02.410219",
    "stochastic": "2025-02-07T14:40:35.834973",
}

# ---------------------------------------------------------------- #

dir_data   = os.path.join("data",   "gymnasium", "frozen_lake")
dir_images = os.path.join("images", "gymnasium", "frozen_lake")

dir_env = { kind: os.path.join(dir_data, id_env[kind]) for kind in kinds }

# ---------------------------------------------------------------- #

total_timesteps = 100_000
n_samples = 100_000
n_trajectories = 500
max_episode_steps = 200
seed = 0

specs = get_specs_env( gym.make("FrozenLake-v1", desc=None, map_name="4x4") )
n_obs = specs["obs"]["max"] + 1
n_act = specs["act"]["max"] + 1

gammas = get_gammas_log10()

projected = True
modified = True
lamda = 1e-6

# ---------------------------------------------------------------- #
