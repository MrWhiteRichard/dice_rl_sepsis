# ---------------------------------------------------------------- #

import os

import gymnasium as gym

from dice_rl_TU_Vienna.get_recordings import get_recordings_cos_angle
from dice_rl_TU_Vienna.plugins.stable_baselines3.specs import get_specs_env

from plugins.gymnasium.cartpole.dataset import get_dataset_cartpole

# ---------------------------------------------------------------- #

kinds = [ "behavior", "evaluation", ]

id_policy = {
    "behavior": "2025-02-06T12:51:54.519095",
    "evaluation": "2025-02-06T12:52:22.614772",
}
id_dataset = {
    "behavior": "2025-02-06T14:56:13.508204",
    "evaluation": "2025-02-06T14:56:23.202427",
}

# ---------------------------------------------------------------- #

dir_base = os.path.join("data", "gymnasium", "cartpole")
dir_images = os.path.join(dir_base, "images")

dir_policy = {
    kind: os.path.join(dir_base, id_policy[kind])
        for kind in kinds
}
dir_dataset = {
    kind: os.path.join(dir_policy[kind], id_dataset[kind])
        for kind in kinds
}

# ---------------------------------------------------------------- #
# policy

total_timesteps = { "behavior": 10_000, "evaluation": 100_000, }
n_trajectories = 500
max_episode_steps = 200

# ---------------------------------------------------------------- #
# evaluation

# -------------------------------- #

env = gym.make('CartPole-v0')
specs = get_specs_env(env)
assert specs["act"]["min"] == 0
assert specs["act"]["shape"] == ()

# -------------------------------- #

gammas = [0.1, 0.5, 0.9]
p = 1.5
lamda = 1.0

seed = 0
batch_size = 64

learning_rates = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
hidden_dimensions = [32]

obs_min   = specs["obs"]["min"]
obs_max   = specs["obs"]["max"]
n_act     = specs["act"]["max"] + 1
obs_shape = specs["obs"]["shape"]

dataset = get_dataset_cartpole(dir_dataset["behavior"], dir_policy["evaluation"])
preprocess_obs = None
preprocess_act = None
preprocess_rew = None

dir = dir_dataset["behavior"]
get_recordings = get_recordings_cos_angle
other_hyperparameters = { "id_policy": id_policy["evaluation"], }

n_steps = 100_000
verbosity = 1
pbar_keys = None

# ---------------------------------------------------------------- #
# plotting

names = ["NeuralDualDice", "NeuralGradientDice", "NeuralGenDice"]

hyperparameters_evaluation = {
    (0.1, "NeuralDualDice"):     { "name": "NeuralDualDice",     "gamma": 0.1, "seed": seed, "batch_size": batch_size, "learning_rate": 0.001, "hidden_dimensions": hidden_dimensions, "other": { "id_policy": id_policy["evaluation"], "p": p, }, },
    (0.5, "NeuralDualDice"):     { "name": "NeuralDualDice",     "gamma": 0.5, "seed": seed, "batch_size": batch_size, "learning_rate": 0.001, "hidden_dimensions": hidden_dimensions, "other": { "id_policy": id_policy["evaluation"], "p": p, }, },
    (0.9, "NeuralDualDice"):     { "name": "NeuralDualDice",     "gamma": 0.9, "seed": seed, "batch_size": batch_size, "learning_rate": 0.001, "hidden_dimensions": hidden_dimensions, "other": { "id_policy": id_policy["evaluation"], "p": p, }, },
    (0.1, "NeuralGradientDice"): { "name": "NeuralGradientDice", "gamma": 0.1, "seed": seed, "batch_size": batch_size, "learning_rate": 0.001, "hidden_dimensions": hidden_dimensions, "other": { "id_policy": id_policy["evaluation"], "lamda": lamda, }, },
    (0.5, "NeuralGradientDice"): { "name": "NeuralGradientDice", "gamma": 0.5, "seed": seed, "batch_size": batch_size, "learning_rate": 0.001, "hidden_dimensions": hidden_dimensions, "other": { "id_policy": id_policy["evaluation"], "lamda": lamda, }, },
    (0.9, "NeuralGradientDice"): { "name": "NeuralGradientDice", "gamma": 0.9, "seed": seed, "batch_size": batch_size, "learning_rate": 0.001, "hidden_dimensions": hidden_dimensions, "other": { "id_policy": id_policy["evaluation"], "lamda": lamda, }, },
    (0.1, "NeuralGenDice"):      { "name": "NeuralGenDice",      "gamma": 0.1, "seed": seed, "batch_size": batch_size, "learning_rate": 0.001, "hidden_dimensions": hidden_dimensions, "other": { "id_policy": id_policy["evaluation"], "lamda": lamda, }, },
    (0.5, "NeuralGenDice"):      { "name": "NeuralGenDice",      "gamma": 0.5, "seed": seed, "batch_size": batch_size, "learning_rate": 0.001, "hidden_dimensions": hidden_dimensions, "other": { "id_policy": id_policy["evaluation"], "lamda": lamda, }, },
    (0.9, "NeuralGenDice"): None,
}

alpha = 0.1
n_ma = 16
markevery = 100

colors = ["green", "red", "cyan"]
markers = ["3", "4", "+"]

# ---------------------------------------------------------------- #
