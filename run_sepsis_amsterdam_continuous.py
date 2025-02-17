# ---------------------------------------------------------------- #

from itertools import product

from dice_rl_TU_Vienna.estimators.neural.neural_dual_dice     import NeuralDualDice
from dice_rl_TU_Vienna.estimators.neural.neural_gen_dice      import NeuralGenDice
from dice_rl_TU_Vienna.estimators.neural.neural_gradient_dice import NeuralGradientDice

from dice_rl_TU_Vienna.utils.bedtime import computer_sleep

from plugins.medical_rl.sepsis_amsterdam.continuous.config import *

# ---------------------------------------------------------------- #

def run_sepsis_amsterdam_continuous(loops):

    for learning_rate, hidden_dimensions in loops.get("NeuralDualDice", []):
        estimator = NeuralDualDice(
            gamma, p,
            seed, batch_size,
            learning_rate, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings,
        )
        estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

    for learning_rate, hidden_dimensions in loops.get("NeuralGenDice", []):
        estimator = NeuralGenDice(
            gamma, lamda,
            seed, batch_size,
            learning_rate, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings,
        )
        estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

    for learning_rate, hidden_dimensions in loops.get("NeuralGradientDice", []):
        estimator = NeuralGradientDice(
            gamma, lamda,
            seed, batch_size, learning_rate, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings,
        )
        estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

# ---------------------------------------------------------------- #

run_sepsis_amsterdam_continuous(
    loops={
        "NeuralDualDice":     [ ( 1e-4, [128], ), ],
        "NeuralGenDice":      [],
        "NeuralGradientDice": [],
    }
)

computer_sleep()

# ---------------------------------------------------------------- #
