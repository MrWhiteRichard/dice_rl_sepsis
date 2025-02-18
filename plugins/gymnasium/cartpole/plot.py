# ---------------------------------------------------------------- #

import random

from dice_rl_TU_Vienna.plot.continuous import \
    get_logs_and_plot as get_logs_and_plot_general
from dice_rl_TU_Vienna.utils.general import dict_to_str, flatten_dict

from plugins.gymnasium.cartpole.config import *

# ---------------------------------------------------------------- #

def get_logs_and_plot(
        get_policy_value,
        #
        hyperparameters_evaluation,
        hyperparameters_dict,
        #
        ylims=None,
        n_samples_moving_average=None,
        #
        dir_save=None,
        verbosity=0,
    ):

    suptitle = f"Cartpole" + "\n" + (
        dict_to_str( flatten_dict( random.choice(hyperparameters_evaluation), ), blacklist=["gamma", "id_policy"], )
    )
    titles = [
        {
            "pv": f"gamma={dictionary['gamma']}"
        }
            for dictionary in hyperparameters_evaluation
    ]
    hlines = [
        {
            "pv": [
                {
                    "y": get_policy_value["behavior"](dictionary['gamma']),
                    "label": "OnPE behavior",
                    "linestyle": "--",
                },
                {
                    "y": get_policy_value["evaluation"](dictionary['gamma']),
                    "label": "OnPE evaluation",
                    "linestyle": ":",
                },
            ]
        }
            for dictionary in hyperparameters_evaluation
    ]
    append_extras = None

    file_name = suptitle.replace("\n", "; ")

    return get_logs_and_plot_general(
        dir_base,
        #
        hyperparameters_evaluation,
        hyperparameters_dict,
        #
        suptitle,
        titles,
        ylims,
        n_samples_moving_average,
        hlines,
        #
        append_extras,
        #
        dir_save, file_name,
        verbosity,
    )

# ---------------------------------------------------------------- #
