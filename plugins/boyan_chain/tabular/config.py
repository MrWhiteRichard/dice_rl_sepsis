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

labels_OnPE = ["OnPE evaluation"]
labels_VAFE = ["TabularVafe"]
labels_DICE = ["TabularDice", "TabularDualDice", "TabularGradientDice"]
labels_ANAL = ["analytical"]

colors_OnPE = ["grey"]
colors_VAFE = ["blue"]
colors_DICE = ["orange", "green", "red"]
colors_ANAL = ["black"]

markers_OnPE = ["^"]
markers_VAFE = ["1"]
markers_DICE = ["2", "3", "4"]
markers_ANAL = ["."]

labels_approx  = labels_OnPE  + labels_VAFE  + labels_DICE
colors_approx  = colors_OnPE  + colors_VAFE  + colors_DICE
markers_approx = markers_OnPE + markers_VAFE + markers_DICE

std_girth = 0.5
alpha = 0.1

# ---------------------------------------------------------------- #
