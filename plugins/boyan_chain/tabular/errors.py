# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from dice_rl_TU_Vienna.estimators.tabular.tabular_dice          import TabularDice
from dice_rl_TU_Vienna.estimators.tabular.tabular_dual_dice     import TabularDualDice
from dice_rl_TU_Vienna.estimators.tabular.tabular_gradient_dice import TabularGradientDice

from dice_rl_TU_Vienna.applications.boyan_chain.tabular.config import *

# ---------------------------------------------------------------- #

gammas = np.arange(0.1, 1, 0.1)

estimator_names = ["TabularDice", "TabularDualDice", "TabularGradientDice"]
estimator_types = [TabularDice, TabularDualDice, TabularGradientDice]
solve_kwargs = [
    { "modified": True }, {}, { "lam": 1e-6 }, ]

error_names = [
    "pv_error",
    "sdc_L1_error", "sdc_L2_error",
    "bellman_L1_error", "bellman_L2_error",
    "norm_error",
    "negativity",
]

# -------------------------------- #

def get_error_path(config, estimator_name, error_name=None, **kwargs):
    base_dir = config["save_dir_episodic"]

    hparam_str_sample = "_".join(["by=experience", "obs_act=True"])

    hparam_str_solve = "_".join(["projected=False", "weighted=True"])
    if len(kwargs) > 0:
        hparam_str_solve = "_".join(
            [hparam_str_solve] + [ f"{k}={v}" for k, v in kwargs.items() ] )

    path = os.path.join(base_dir, hparam_str_sample, hparam_str_solve, estimator_name)

    if error_name is not None:
        path = os.path.join(path, f"{error_name}.npy")

    return path

# -------------------------------- #

def create_errors_episodic(config, aux_estimates_episodic):

    errors = {
        estimator_name: { error_name: [] for error_name in error_names }
        for estimator_name in estimator_names
    }

    X = zip(estimator_names, estimator_types, solve_kwargs)
    for x in X:
        estimator_name, estimator_type, kwargs = x

        estimator = estimator_type(
            dataset=config["dataset_episodic"],
            evaluation_policy=config["evaluation_policy"],
            aux_estimates=aux_estimates_episodic,
        )

        for gamma in gammas:
            sdc_approx, _, pv_approx = estimator \
                .solve(gamma=gamma, projected=False, weighted=True, **kwargs)

            error_dict = config["analytical_solver_episodic"] \
                .errors(
                    gamma=gamma,
                    sdc_approx=sdc_approx,
                    pv_approx=pv_approx,
                )

            for error_name in error_names:
                errors[estimator_name][error_name].append( error_dict[error_name] )

    return errors


def save_errors_episodic(config, errors):
    for estimator_name in estimator_names:
        errors[estimator_name] = { # type: ignore
            error_name: np.array(error_dict)
            for error_name, error_dict in errors[estimator_name].items()
        }

        for error_name, error_value in errors[estimator_name].items():
            folder_path = get_error_path(config, estimator_name)
            if not tf.io.gfile.isdir(folder_path):
                tf.io.gfile.makedirs(folder_path)
    
            file_path = os.path.join(folder_path, f"{error_name}.npy")
            np.save(file_path, error_value)
            print(f"saved {file_path}")


def load_errors_episodic(config):

    errors = {
        estimator_name: {}
        for estimator_name in estimator_names
    }

    for estimator_name in estimator_names:
        for error_name in error_names:
            path = get_error_path(config, estimator_name, error_name)
            errors[estimator_name][error_name] = np.load(path)
            print(f"loaded {path}")

            assert len(errors[estimator_name][error_name]) == len(gammas)

    return errors


def get_errors_episodic(config, aux_estimates_episodic=None):
    try:
        errors = load_errors_episodic(config)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        assert False

    except:
        assert aux_estimates_episodic is not None
        errors = create_errors_episodic(config, aux_estimates_episodic)
        save_errors_episodic(config, errors)

    return errors

# ---------------------------------------------------------------- #

def create_errors_continuing(config, aux_estimates_continuing):

    errors = {}

    X = zip(estimator_names, estimator_types, solve_kwargs)
    for x in X:
        estimator_name, estimator_type, kwargs = x

        if estimator_name == "TabularDualDice": continue

        estimator = estimator_type(
            dataset=config["dataset_episodic"],
            evaluation_policy=config["evaluation_policy"],
            aux_estimates=aux_estimates_continuing,
        )

        solution = estimator \
            .solve(gamma=1.0, projected=False, weighted=True, **kwargs)

        if estimator_name == "TabularDice":
            sdc_approx, ev_approx, pv_approx = solution
        else:
            sdc_approx, pv_approx = solution

        error_dict = config["analytical_solver_continuing"] \
            .errors(
                gamma=1.0,
                sdc_approx=sdc_approx,
                pv_approx=pv_approx,
            )

        errors[estimator_name] = error_dict

    return errors, ev_approx

# ---------------------------------------------------------------- #
