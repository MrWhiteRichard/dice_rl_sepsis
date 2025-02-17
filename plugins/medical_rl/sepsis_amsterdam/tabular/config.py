# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.estimators.get import get_gammas_2

from plugins.medical_rl.sepsis_amsterdam.config import *

# ---------------------------------------------------------------- #
# evaluation

gammas = get_gammas_2()

projected = True
modified = True
lamda = 1e-6

# ---------------------------------------------------------------- #
# plotting

env_title = "Sepsis Amsterdam Tabular"

colors_OnPE = ["black", "lightgrey", "grey"]
colors_VAFE = ["blue"]
colors_DICE = ["orange", "green", "red"]
colors_OffPE = colors_VAFE + colors_DICE

markers_OnPE = [".", "v", "^"]
markers_VAFE = ["1"]
markers_DICE = ["2", "3", "4"]
markers_OffPE = markers_VAFE + markers_DICE

colors_lim = colors_OnPE
markers_lim = markers_OnPE

# ---------------------------------------------------------------- #

n_clusters = 256
n_init = 10

n_obs = n_clusters + 2
n_act = 5

# ---------------------------------------------------------------- #
