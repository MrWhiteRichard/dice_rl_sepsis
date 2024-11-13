# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from tqdm import tqdm

# ---------------------------------------------------------------- #

def get_treatement_lengths_offline(dataset, n_pads):
    all_steps = dataset.get_all_steps(include_terminal_steps=True)
    treatement_lengths = []

    for step_num in tqdm(range(dataset.capacity)):
        env_step = tf.nest.map_structure( lambda t: t[step_num, ...], all_steps )

        if env_step.is_first(): # type: ignore
            t_start = step_num

        if env_step.is_last(): # type: ignore
            t_end = step_num - n_pads + 1
            treatement_length = t_end - t_start
            treatement_lengths.append(
                treatement_length if all_steps.reward[t_end-1] == 1 else float("inf") )

    return np.array(treatement_lengths)


# ---------------------------------------------------------------- #

def get_treatement_lengths_online(env, model, num_episodes):

    treatement_lengths = []

    for _ in tqdm( range(num_episodes) ):

        obs, _ = env.reset()

        done = False
        treatement_length = 0

        while not done:
            act, _ = model.predict(obs)
            obs, rew, done, _, _ = env.step(act)

            treatement_length += 1

        treatement_lengths.append(
            treatement_length if rew == 1 else float("inf") )

    return np.array(treatement_lengths)

# ---------------------------------------------------------------- #
