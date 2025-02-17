# ---------------------------------------------------------------- #

from itertools import product

from dice_rl_TU_Vienna.estimators.neural.neural_dual_dice     import NeuralDualDice
from dice_rl_TU_Vienna.estimators.neural.neural_gen_dice      import NeuralGenDice
from dice_rl_TU_Vienna.estimators.neural.neural_gradient_dice import NeuralGradientDice

from dice_rl_TU_Vienna.utils.general import list_safe_zip
from dice_rl_TU_Vienna.utils.bedtime import computer_sleep

from plugins.gymnasium.cartpole.config import *

# ---------------------------------------------------------------- #

def run_cartpole(loops):

    for learning_rate_, gamma_ in loops.get("NeuralDualDice", []):
        estimator = NeuralDualDice(
            gamma_, p,
            seed, batch_size,
            learning_rate_, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings, other_hyperparameters,
        )
        estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

    for learning_rate_, gamma_ in loops.get("NeuralGenDice", []):
        estimator = NeuralGenDice(
            gamma_, lamda,
            seed, batch_size,
            learning_rate_, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings, other_hyperparameters,
        )
        estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

    for learning_rate_, gamma_ in loops.get("NeuralGradientDice", []):
        estimator = NeuralGradientDice(
            gamma_, lamda,
            seed, batch_size,
            learning_rate_, hidden_dimensions,
            obs_min, obs_max, n_act, obs_shape,
            dataset, preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings, other_hyperparameters,
        )
        estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

# ---------------------------------------------------------------- #

# run_cartpole(
#     loops={
#         "NeuralDualDice":     product(learning_rates, gammas),
#         "NeuralGenDice":      product(learning_rates, gammas),
#         "NeuralGradientDice": product(learning_rates, gammas),
#     }
# )

# computer_sleep()

# ---------------------------------------------------------------- #
