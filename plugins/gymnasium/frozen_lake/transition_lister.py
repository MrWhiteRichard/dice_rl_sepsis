# ---------------------------------------------------------------- #

import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import product
from tqdm import tqdm
from fractions import Fraction

# ---------------------------------------------------------------- #

fraction = lambda x: str( Fraction.from_float(x).limit_denominator(1_000) )
unique_counts = lambda l: list( zip( *np.unique(l, return_counts=True) ) )

def quantize(x):
    u, c = np.unique(x, return_counts=True)
    p = c / len(x)
    q = np.round(p * 3) / 3
    return u, c, p, q

# -------------------------------- #

def get_transitions_sample_env(env, num_transitions):
    obs_next_dict = {}
    rew_dict = {}

    for obs, act in tqdm( product( range(16), range(4), ), total=64, ):
        obs_next_dict[(obs, act)] = []
        rew_dict[(obs, act)] = []

        for _ in range(num_transitions):
            env.reset()
            env.unwrapped.s = obs
            obs_next, rew, _, _, _ = env.step(act)

            obs_next_dict[(obs, act)].append(obs_next)
            rew_dict[(obs, act)].append(rew)

    return {
        "obs_next_dict": obs_next_dict,
        "rew_dict": rew_dict,
    }

def get_transitions_sample_dataset(dataset, by="experience"):
    if by != "experience": raise NotImplementedError

    all_experience, _ = dataset.get_all_episodes()
    num_experience    = dataset.capacity // 3

    obs_next_dict = {}
    rew_dict      = {}

    for obs, act in product( range(16), range(4), ):
        obs_next_dict[(obs, act)] = []
        rew_dict     [(obs, act)] = []

    for i in tqdm( range(num_experience) ):
        experience = tf.nest.map_structure(lambda t: t[i], all_experience)

        obs = int( experience.observation[1] ) # type: ignore
        act = int( experience.action[1] )      # type: ignore

        obs_next = int  ( experience.observation[2] ) # type: ignore
        rew      = float( experience.reward[1] )      # type: ignore

        obs_next_dict[(obs, act)].append(obs_next)
        rew_dict     [(obs, act)].append(rew)

    return {
        "obs_next_dict": obs_next_dict,
        "rew_dict": rew_dict,
    }


def get_transitions_exact(transitions):
    obs_next_dict = {}
    rew_dict = {}

    for obs, act in tqdm( product( range(16), range(4), ), total=64, ):

        # obs_next

        obs_next_dict[(obs, act)] = {}
        u, _, _, q = quantize(
            transitions["obs_next_dict"][(obs, act)] )

        obs_next_dict[(obs, act)]["obs_next"] = u
        obs_next_dict[(obs, act)]["probs"] = q

        # rew

        u, _, _, q = quantize(
            transitions["rew_dict"][(obs, act)] )

        rew_dict[(obs, act)] = np.dot(u, q)

    return {
        "obs_next_dict": obs_next_dict,
        "rew_dict": rew_dict,
    }


def list_transitions_exact(transitions):
    rew_dict      = transitions["rew_dict"]
    obs_next_dict = transitions["obs_next_dict"]

    c_obs = []
    c_act = []
    c_rew = []
    c_obs_next = []
    c_probs = []

    for obs, act in product( range(16), range(4), ):
        c_obs.append(obs)
        c_act.append(act)

        rew = rew_dict[(obs, act)]

        O = obs_next_dict[(obs, act)]["obs_next"]
        P = obs_next_dict[(obs, act)]["probs"]
        obs_next = []
        probs = []
        for o, p in zip(O, P):
            if p != 0:
                obs_next.append(o)
                probs   .append(p)

        rew = fraction(rew)
        probs = [ fraction(p) for p in probs ]

        if len(obs_next) == 1: obs_next = obs_next[0]
        if len(probs)    == 1: probs    = probs   [0]

        c_rew.append(rew)
        c_obs_next.append(obs_next)
        c_probs.append(probs)

    return pd.DataFrame({
        "obs": c_obs,
        "act": c_act,
        "obs_next": c_obs_next,
        "probs": c_probs,
        "rew": c_rew,
    })

def list_transitions_sample(transitions_sample):
    rew_dict      = transitions_sample["rew_dict"]
    obs_next_dict = transitions_sample["obs_next_dict"]

    c_obs = []
    c_act = []
    c_rew = []
    c_obs_next = []

    for obs, act in product( range(16), range(4), ):
        c_obs.append(obs)
        c_act.append(act)

        obs_next = obs_next_dict[(obs, act)]
        rew      = rew_dict     [(obs, act)]

        c_obs_next.append( unique_counts(obs_next) )
        c_rew     .append( unique_counts(rew) )

    return pd.DataFrame({
        "obs": c_obs,
        "act": c_act,
        "obs_next": c_obs_next,
        "rew": c_rew,
    })

# ---------------------------------------------------------------- #
