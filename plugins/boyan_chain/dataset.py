# ---------------------------------------------------------------- #

import numpy as np

from gymnasium.wrappers.time_limit import TimeLimit

from dice_rl_TU_Vienna.dataset import get_dataset as get_dataset_general

from plugins.boyan_chain.environment import get_env
from plugins.boyan_chain.config import dir_base, prob

# ---------------------------------------------------------------- #

def get_dataset_by_samples(seed, n_samples, N, kind, verbosity=0):

    env = get_env(seed, N, kind)
    get_act = lambda obs: env.action_space.sample()
    hyperparameters = {
        "seed": seed, "n_samples": n_samples, "N": N, "kind": kind, }
    dataset, id_datset = get_dataset_general(
        dir_base, env, get_act, hyperparameters, verbosity, )

    probs = [ np.array([prob, 1-prob]) ] * len(dataset)
    dataset["probs_next"] = probs
    dataset["probs_init"] = probs

    return dataset, id_datset

def get_dataset_by_trajectories(seed, n_trajectories, get_act, N, kind, verbosity=0):

    env = TimeLimit( get_env(seed, N, kind), max_episode_steps=N+1, )
    hyperparameters = {
        "seed": seed, "n_trajectories": n_trajectories, "N": N, "kind": kind, }
    dataset, id_datset = get_dataset_general(
        dir_base, env, get_act, hyperparameters, verbosity, )

    probs = [ np.array([prob, 1-prob]) ] * len(dataset)
    dataset["probs_next"] = probs
    dataset["probs_init"] = probs

    return dataset, id_datset

# ---------------------------------------------------------------- #
