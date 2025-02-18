# -----------------------------------------

import os

import pandas as pd

from stable_baselines3 import PPO

from dice_rl_TU_Vienna.plugins.stable_baselines3.dataset import get_probs

# ---------------------------------------------------------------- #

n_act = 2

def get_dataset_cartpole(dir_dataset, dir_policy):
    dataset = pd.read_parquet( os.path.join(dir_dataset, "dataset.parquet") )
    policy  = PPO.load( os.path.join(dir_policy, "policy.zip") )

    dataset["probs_init"] = list( get_probs(dataset["obs_init"], policy, n_act) )
    dataset["probs_next"] = list( get_probs(dataset["obs_next"], policy, n_act) )

    return dataset

# ---------------------------------------------------------------- #
