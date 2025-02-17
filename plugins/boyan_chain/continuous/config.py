# ---------------------------------------------------------------- #

import numpy as np
import pandas as pd

from dice_rl_TU_Vienna.get_recordings import get_recordings_cos_angle
from dice_rl_TU_Vienna.preprocess import one_hot_encode_observation
from dice_rl_TU_Vienna.utils.general import merge_dicts

from plugins.boyan_chain.analytical_solver import AnalyticalSolverBoyanChain
from plugins.boyan_chain.dataset import get_dataset
from plugins.boyan_chain.config import *

# ---------------------------------------------------------------- #

# dataset

N = 12

# ---------------------------------------------------------------- #
# evaluation

gamma = { "episodic": [0.1, 0.5, 0.9], "continuing": 1.0, }
p = 1.5
lamda = { "episodic": 1.0, "continuing": [0.0, 0.1, 0.5, 1.0, 2.0], }

batch_size = 64
learning_rates = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
hidden_dimensions = [32]

obs_min = np.zeros(N+1)
obs_max = np.ones(N+1)
obs_shape = (N+1,)

dataset = { seed: {} for seed in seeds }
preprocess_obs = one_hot_encode_observation
preprocess_act = None
preprocess_rew = None

dir = { seed: {} for seed in seeds }
analytical_solver = { kind: AnalyticalSolverBoyanChain(N, prob, kind) for kind in kinds }
def get_get_recordings(gamma, kind):
    def get_recordings(
        estimator,
        obs_init, obs, act, obs_next, probs_init, probs_next,
        values, loss, gradients,
        pv_s, pv_w, ):

        cos_angle = get_recordings_cos_angle(
            estimator,
            obs_init, obs, act, obs_next, probs_init, probs_next,
            values, loss, gradients,
            pv_s, pv_w, )

        errors = analytical_solver[kind].errors(
            gamma=gamma,
            pv_approx={ "s": pv_s, "w": pv_w, },
            sdc_approx_network=estimator.w,
        )

        return merge_dicts(cos_angle, errors)

    return get_recordings

n_steps = 100_000
verbosity = 1
pbar_keys = ["loss", "pv_s", "pv_w", "cos_angle", "pv_error_s", "pv_error_w", "sdc_L2_error"]

# -------------------------------- #

for seed in seeds:
    for kind in kinds:

        dataset_, id_dataset_ = get_dataset(seed, n_samples, N, kind)
        assert id_dataset_ is not None

        dataset[seed][kind] = dataset_
        id_dataset = id_dataset_
        dir[seed][kind] = os.path.join(dir_base, id_dataset_)

# ---------------------------------------------------------------- #
# plotting

i_lamda_best = 3
error_names = [ "pv_error_s", "pv_error_w", "sdc_L2_error", "norm_error", ]

std_girth = 0.5
alpha = 0.1
markevery = 100

colors = {
    "episodic": ["green", "red", "cyan"],
    "continuing": ["red", "cyan"],
}
markers = {
    "episodic": ["3", "4", "+"],
    "continuingc": ["4", "+"],
}

names = {
    "episodic": ["NeuralDualDice", "NeuralGradientDice", "NeuralGenDice"],
    "continuing": ["NeuralGradientDice", "NeuralGenDice"],
}

hyperparameters_evaluation = {
    "episodic": {
        seed: {
            (0.1, "NeuralDualDice"):     { "name": "NeuralDualDice",     "gamma": 0.1, "seed": seed, "batch_size": 64, "learning_rate": 0.01,  "hidden_dimensions": [32], "p": 1.5, },
            (0.5, "NeuralDualDice"):     { "name": "NeuralDualDice",     "gamma": 0.5, "seed": seed, "batch_size": 64, "learning_rate": 0.001, "hidden_dimensions": [32], "p": 1.5, },
            (0.9, "NeuralDualDice"):     { "name": "NeuralDualDice",     "gamma": 0.9, "seed": seed, "batch_size": 64, "learning_rate": 0.001, "hidden_dimensions": [32], "p": 1.5, },
            (0.1, "NeuralGradientDice"): { "name": "NeuralGradientDice", "gamma": 0.1, "seed": seed, "batch_size": 64, "learning_rate": 0.01,  "hidden_dimensions": [32], "lamda": 1.0, },
            (0.5, "NeuralGradientDice"): { "name": "NeuralGradientDice", "gamma": 0.5, "seed": seed, "batch_size": 64, "learning_rate": 0.01,  "hidden_dimensions": [32], "lamda": 1.0, },
            (0.9, "NeuralGradientDice"): { "name": "NeuralGradientDice", "gamma": 0.9, "seed": seed, "batch_size": 64, "learning_rate": 0.01,  "hidden_dimensions": [32], "lamda": 1.0, },
            (0.1, "NeuralGenDice"):      { "name": "NeuralGenDice",      "gamma": 0.1, "seed": seed, "batch_size": 64, "learning_rate": 0.01,  "hidden_dimensions": [32], "lamda": 1.0, },
            (0.5, "NeuralGenDice"):      { "name": "NeuralGenDice",      "gamma": 0.5, "seed": seed, "batch_size": 64, "learning_rate": 0.01,  "hidden_dimensions": [32], "lamda": 1.0, },
            (0.9, "NeuralGenDice"):      { "name": "NeuralGenDice",      "gamma": 0.9, "seed": seed, "batch_size": 64, "learning_rate": 0.01,  "hidden_dimensions": [32], "lamda": 1.0, },
        }
            for seed in seeds
    },
    "continuing": {
        seed: {
            "NeuralGradientDice": { "name": "NeuralGradientDice", "gamma": gamma[kind], "seed": seed, "batch_size": 64, "learning_rate": 0.01,  "hidden_dimensions": [32], "lamda": lamda[kind][i_lamda_best], },
            "NeuralGenDice":      { "name": "NeuralGenDice",      "gamma": gamma[kind], "seed": seed, "batch_size": 64, "learning_rate": 0.01,  "hidden_dimensions": [32], "lamda": lamda[kind][i_lamda_best], },
        }
            for seed in seeds
    }
}

labels = names

# ---------------------------------------------------------------- #
