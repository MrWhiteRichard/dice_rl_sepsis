# ---------------------------------------------------------------- #

from plugins.boyan_chain.config import *

import numpy as np

# ---------------------------------------------------------------- #
# dataset

Ns = [ 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, ]

# ---------------------------------------------------------------- #
# policy

get_act_evaluation = lambda obs: 1 - int( np.random.random() < prob )

# ---------------------------------------------------------------- #
# evaluation

n_trajectories = 1_000

gammas = np.array([0.9, 0.99, 0.999, 0.9999])

projected = True
modified = True
lamda = 1e-6

n_obs = { N: N + 1 for N in Ns }

# ---------------------------------------------------------------- #
# plotting

colors = {
    "OnPE": "grey",
    "TabularVafe": "blue",
    "TabularDice": "orange",
    "TabularDualDice": "green",
    "TabularGradientDice": "red",
    "analytical": "black",
}

markers = {
    "OnPE": "^",
    "TabularVafe": "1",
    "TabularDice": "2",
    "TabularDualDice": "3",
    "TabularGradientDice": "4",
    "analytical": ".",
}

std_girth = 0.5
alpha = 0.1

# ---------------------------------------------------------------- #
