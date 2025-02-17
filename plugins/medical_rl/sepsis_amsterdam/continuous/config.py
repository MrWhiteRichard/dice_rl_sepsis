# ---------------------------------------------------------------- #

import os

import numpy as np
import pandas as pd

from dice_rl_TU_Vienna.get_recordings import get_recordings_cos_angle

from plugins.medical_rl.sepsis_amsterdam.config import *

# ---------------------------------------------------------------- #
# evaluation

gamma = 0.9
p = 1.5
lamda = 1.0

seed = 42
batch_size = 1024
learning_rates = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
hidden_dimensionss = [ [32], [64], [128], [256], ]

obs_min = np.load( os.path.join(dir_policy["continuous"], "obs_min.npy") )
obs_max = np.load( os.path.join(dir_policy["continuous"], "obs_max.npy") )
n_act = 5
obs_shape = (382,)

dataset = pd.read_parquet(
    os.path.join(dir_dataset["continuous"], "dataset.parquet")
)
preprocess_obs = None
preprocess_act = None
preprocess_rew = None
dir = dir_policy["continuous"]
get_recordings = get_recordings_cos_angle

n_steps = 500_000
verbosity = 1
pbar_keys = None

# ---------------------------------------------------------------- #
