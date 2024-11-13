# ---------------------------------------------------------------- #

import numpy as np
import pandas as pd
import tensorflow as tf

# ---------------------------------------------------------------- #

def display_dataframe_slice(n, dataframe):
    display( dataframe.head(n) ) # type: ignore

def display_episodes_dataframe_dataset(num_episodes, dataframe=None, dataset=None):
    A = dataframe is not None
    B = dataset   is not None

    if A:
        f = dataframe.loc[:, "t"] == 0
        l = dataframe[f].index.tolist()
        indices_dataframe_init = np.array(l)

    if B:
        all_steps = dataset.get_all_steps(include_terminal_steps=True)

        indices_dataset_init, *_ = np.where(all_steps.step_type == 0)
        indices_dataset_term, *_ = np.where(all_steps.step_type == 2)

    for i in range(num_episodes):

        if A:
            k = indices_dataframe_init[i]
            l = indices_dataframe_init[i + 1] \
                if i < num_episodes else None

            display( dataframe.iloc[k:l] ) # type: ignore

        if B:
            k = indices_dataset_init[i]
            l = indices_dataset_term[i] + 1

            dataset_slice = tf.nest.map_structure(
                lambda t: t[k:l, ...], all_steps, )

            # print_namedtuple(dataset_slice)

            display( # type: ignore
                pd.DataFrame({
                    "step_type":   np.array(dataset_slice.step_type),   # type: ignore
                    "step_num":    np.array(dataset_slice.step_num),    # type: ignore
                    "observation": np.array(dataset_slice.observation), # type: ignore
                    "action":      np.array(dataset_slice.action),      # type: ignore
                    "reward":      np.array(dataset_slice.reward),      # type: ignore
                    "discount":    np.array(dataset_slice.discount),    # type: ignore
                })
            )

        if A or B: print("#", "-"*64, "#", "\n")

# ---------------------------------------------------------------- #
