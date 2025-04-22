# ---------------------------------------------------------------- #

from itertools import product

from dice_rl_TU_Vienna.estimators.neural.neural_dual_dice     import NeuralDualDice
from dice_rl_TU_Vienna.estimators.neural.neural_gen_dice      import NeuralGenDice
from dice_rl_TU_Vienna.estimators.neural.neural_gradient_dice import NeuralGradientDice

from dice_rl_TU_Vienna.utils.bedtime import computer_sleep

from plugins.sepsis.bologheanu.continuous.config import *

# ---------------------------------------------------------------- #

def run_sepsis_bologheanu_continuous(loops):

    for learning_rate_, hidden_dimensions_ in loops.get("NeuralDualDice", []):
        estimator = NeuralDualDice(
            gamma, p,
            seed, batch_size,
            learning_rate_, hidden_dimensions_,
            obs_min, obs_max, n_act, obs_shape,
            dataset["neural"], preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings,
        )
        estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

    for learning_rate_, hidden_dimensions_ in loops.get("NeuralGenDice", []):
        estimator = NeuralGenDice(
            gamma, lamda,
            seed, batch_size,
            learning_rate_, hidden_dimensions_,
            obs_min, obs_max, n_act, obs_shape,
            dataset["neural"], preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings,
        )
        estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

    for learning_rate_, hidden_dimensions_ in loops.get("NeuralGradientDice", []):
        estimator = NeuralGradientDice(
            gamma, lamda,
            seed, batch_size, learning_rate_, hidden_dimensions_,
            obs_min, obs_max, n_act, obs_shape,
            dataset["neural"], preprocess_obs, preprocess_act, preprocess_rew,
            dir, get_recordings,
        )
        estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

# ---------------------------------------------------------------- #

# run_sepsis_bologheanu_continuous(
#     loops={
#         "NeuralDualDice":     [ ( learning_rate_, [128], ) for learning_rate_ in learning_rates ],
#         "NeuralGenDice":      [ ( learning_rate_, [128], ) for learning_rate_ in learning_rates ],
#         "NeuralGradientDice": [ ( learning_rate_, [128], ) for learning_rate_ in learning_rates ],
#     }
# )

run_sepsis_bologheanu_continuous(
    loops={
        "NeuralGenDice":      [ ( 1e-4, hidden_dimensions_, ) for hidden_dimensions_ in hidden_dimensionss if hidden_dimensions_ != [128] ],
        "NeuralGradientDice": [ ( 1e-3, hidden_dimensions_, ) for hidden_dimensions_ in hidden_dimensionss if hidden_dimensions_ != [128] ],
    }
)

computer_sleep()

# ---------------------------------------------------------------- #
