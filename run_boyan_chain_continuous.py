# ---------------------------------------------------------------- #

from itertools import product

from dice_rl_TU_Vienna.estimators.neural.neural_dual_dice     import NeuralDualDice
from dice_rl_TU_Vienna.estimators.neural.neural_gen_dice      import NeuralGenDice
from dice_rl_TU_Vienna.estimators.neural.neural_gradient_dice import NeuralGradientDice

from dice_rl_TU_Vienna.utils.general import list_safe_zip
from dice_rl_TU_Vienna.utils.bedtime import computer_sleep

from plugins.boyan_chain.continuous.config import *

# ---------------------------------------------------------------- #

def run_boyan_chain_continuous(seeds, loops):

    for seed_ in seeds:

        kind_ = "episodic"

        for learning_rate_, gamma_ in loops.get("NeuralDualDice", {}).get(kind_, []):
            estimator = NeuralDualDice(
                gamma_, p,
                seed_, batch_size,
                learning_rate_, hidden_dimensions,
                obs_min, obs_max, n_act, obs_shape,
                dataset[seed_][kind_], preprocess_obs, preprocess_act, preprocess_rew,
                dir[seed_][kind_], get_get_recordings(gamma_, kind_),
            )
            estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

        for learning_rate_, gamma_ in loops.get("NeuralGenDice", {}).get(kind_, []):
            estimator = NeuralGenDice(
                gamma_, lamda[kind_],
                seed_, batch_size,
                learning_rate_, hidden_dimensions,
                obs_min, obs_max, n_act, obs_shape,
                dataset[seed_][kind_], preprocess_obs, preprocess_act, preprocess_rew,
                dir[seed_][kind_], get_get_recordings(gamma_, kind_),
            )
            estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

        for learning_rate_, gamma_ in loops.get("NeuralGradientDice", {}).get(kind_, []):
            estimator = NeuralGradientDice(
                gamma_, lamda[kind_],
                seed_, batch_size,
                learning_rate_, hidden_dimensions,
                obs_min, obs_max, n_act, obs_shape,
                dataset[seed_][kind_], preprocess_obs, preprocess_act, preprocess_rew,
                dir[seed_][kind_], get_get_recordings(gamma_, kind_),
            )
            estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

        kind_ = "continuing"

        for learning_rate_, lamda_ in loops.get("NeuralGenDice", {}).get(kind_, []):
            estimator = NeuralGradientDice(
                gamma[kind_], lamda_,
                seed_, batch_size,
                learning_rate_, hidden_dimensions,
                obs_min, obs_max, n_act, obs_shape,
                dataset[seed_][kind_], preprocess_obs, preprocess_act, preprocess_rew,
                dir[seed_][kind_], get_get_recordings(gamma_, kind_),
            )
            estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

        for learning_rate_, lamda_ in loops.get("NeuralGradientDice", {}).get(kind_, []):
            estimator = NeuralGradientDice(
                gamma[kind_], lamda_,
                seed_, batch_size,
                learning_rate_, hidden_dimensions,
                obs_min, obs_max, n_act, obs_shape,
                dataset[seed_][kind_], preprocess_obs, preprocess_act, preprocess_rew,
                dir[seed_][kind_], get_get_recordings(gamma_, kind_),
            )
            estimator.evaluate_loop(n_steps, verbosity, pbar_keys)

# ---------------------------------------------------------------- #

# run_boyan_chain_continuous(
#     seeds=[0],
#     loops={
#         "NeuralDualDice":      { "episodic": product(learning_rates, gamma["episodic"]), },
#         "NeuralGenDice":       { "episodic": product(learning_rates, gamma["episodic"]), "continuing": product(learning_rates, lamda["continuing"]) },
#         "NeuralGradientDice":  { "episodic": product(learning_rates, gamma["episodic"]), "continuing": product(learning_rates, lamda["continuing"]) },
#     }
# )

# run_boyan_chain_continuous(
#     seeds=[1, 2, 3],
#     loops={
#         "NeuralDualDice":      { "episodic": list_safe_zip([0.01, 0.001, 0.001], gamma["episodic"]), },
#         "NeuralGenDice":       { "episodic": list_safe_zip([0.01, 0.01,  0.01],  gamma["episodic"]), "continuing": list_safe_zip([0.01], [1.0]) },
#         "NeuralGradientDice":  { "episodic": list_safe_zip([0.01, 0.01,  0.01],  gamma["episodic"]), "continuing": list_safe_zip([0.01], [1.0]) },
#     }
# )

# computer_sleep()

# ---------------------------------------------------------------- #
