# ---------------------------------------------------------------- #

import random

from dice_rl_TU_Vienna.latex import latex_labels
from dice_rl_TU_Vienna.plot.continuous import \
    get_logs_and_plot as get_logs_and_plot_general
from dice_rl_TU_Vienna.utils.general import dict_to_str
from dice_rl_TU_Vienna.utils.numpy import moving_average

from plugins.boyan_chain.continuous.config import *

# ---------------------------------------------------------------- #

colors = {
    "pv_error_s": "blue",
    "pv_error_w": "orange",
    "sdc_L2_error": "green",
    "norm_error": "red",
}

def append_error(
        error_name, info_row,
        i_log, log,
        n_samples_moving_average):

    ns_ma = n_samples_moving_average[i_log].get(error_name, None)
    use_ma = ns_ma is not None

    x = log["data"][error_name]["steps"]
    y = log["data"][error_name]["values"]
    label = latex_labels[error_name]("")

    info_plot = {
        "x": x, "y": y,
        "label": None if use_ma else label,
        "color": colors[error_name],
        "alpha": 0.1 if use_ma else 1,
    }
    info_row["plots"].append(info_plot)

    if use_ma:
        x_ma = x[ns_ma-1:]
        y_ma = moving_average(y, ns_ma)
        info_plot = {
            "x": x_ma, "y": y_ma,
            "label": label,
            "color": colors[error_name],
            "alpha": 1,
        }
        info_row["plots"].append(info_plot)

def append_errors(
        info_column,
        i_log, log,
        titles, ylims, n_samples_moving_average, hlines):

    info_row = {}

    info_row["plots"] = []

    for error_name in error_names:
        append_error(error_name, info_row, i_log, log, n_samples_moving_average)

    info_row["plot_type"] = "semilogy"
    info_row["ylabel"] = "errors"
    info_row["ylim"] = None if ylims is None else ylims[i_log].get("errors", None)

    info_column.append(info_row)

def get_logs_and_plot(
        analytical_solver,
        #
        hyperparameters_evaluation,
        hyperparameters_dataset,
        #
        ylims=None,
        n_samples_moving_average=None,
        #
        dir_save=None,
        verbosity=0,
    ):

    hyperparameters_policy = None

    suptitle = f"Boyan Chain Continuous - {analytical_solver.kind}" + "\n" + (
        dict_to_str( random.choice(hyperparameters_evaluation), blacklist=["gamma"], )
            if analytical_solver.kind == "episodic" else
        dict_to_str( random.choice(hyperparameters_evaluation), blacklist=["lamda"], )
    )
    titles = [
        {
            "pv":
                f"gamma={dictionary['gamma']}"
                    if analytical_solver.kind == "episodic" else
                f"lamda={dictionary['lamda']}"
        }
            for dictionary in hyperparameters_evaluation
    ]
    hlines = [
        {
            "pv": [
                {
                    "y": analytical_solver.solve(dictionary['gamma'], primal_dual="dual")[0],
                    "label": "analytical",
                    "linestyle": ":",
                }
            ]
        }
            for dictionary in hyperparameters_evaluation
    ]
    append_extras = [ append_errors, ]

    file_name = suptitle.replace("\n", "; ")

    return get_logs_and_plot_general(
        dir_base,
        #
        hyperparameters_evaluation,
        hyperparameters_policy,
        hyperparameters_dataset,
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
