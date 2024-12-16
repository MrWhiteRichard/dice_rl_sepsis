# ---------------------------------------------------------------- #

from itertools import product

from dice_rl_TU_Vienna.runners.neural_dual_dice_runner     import NeuralDualDiceRunner
from dice_rl_TU_Vienna.runners.neural_gen_dice_runner      import NeuralGenDiceRunner
from dice_rl_TU_Vienna.runners.neural_gradient_dice_runner import NeuralGradientDiceRunner

from dice_rl_TU_Vienna.runners.aux_recorders import aux_recorder_cos_angle

from plugins.gymnasium.cartpole.load import *

from utils.general import list_safe_zip
from utils.bedtime import computer_sleep

# ---------------------------------------------------------------- #

num_steps = 100_000
batch_size = 64
hidden_dims = (32,)
regularizer_mlp = 0.0

f_exponent = 1.5
lam = 1.0

# ---------------------------------------------------------------- #

learning_rates = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
gammas = [0.1, 0.5, 0.9]

# ---------------------------------------------------------------- #

def run_cartpole(loops):

    for learning_rate, gamma in loops["NeuralDualDice"]:
        NeuralDualDiceRunner(
            gamma=gamma,
            num_steps=num_steps,
            batch_size=batch_size,
            seed=seed,
            v_hidden_dims=hidden_dims,
            w_hidden_dims=hidden_dims,
            v_learning_rate=learning_rate,
            w_learning_rate=learning_rate,
            v_regularizer=regularizer_mlp,
            w_regularizer=regularizer_mlp,
            f_exponent=f_exponent,
            dataset=dataset,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

    for learning_rate, gamma in loops["NeuralGenDice"]:
        NeuralGradientDiceRunner(
            gamma=gamma,
            num_steps=num_steps,
            batch_size=batch_size,
            seed=seed,
            v_hidden_dims=hidden_dims,
            w_hidden_dims=hidden_dims,
            v_learning_rate=learning_rate,
            w_learning_rate=learning_rate,
            u_learning_rate=learning_rate,
            v_regularizer=regularizer_mlp,
            w_regularizer=regularizer_mlp,
            lam=lam,
            dataset=dataset,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

    for learning_rate, gamma in loops["NeuralGradientDice"]:
        NeuralGenDiceRunner(
            gamma=gamma,
            num_steps=num_steps,
            batch_size=batch_size,
            seed=seed,
            v_hidden_dims=hidden_dims,
            w_hidden_dims=hidden_dims,
            v_learning_rate=learning_rate,
            w_learning_rate=learning_rate,
            u_learning_rate=learning_rate,
            v_regularizer=regularizer_mlp,
            w_regularizer=regularizer_mlp,
            lam=lam,
            dataset=dataset,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

# ---------------------------------------------------------------- #

# run_cartpole(
#     loops={
#         "NeuralDualDice":     product(learning_rates, gammas),
#         "NeuralGenDice":      product(learning_rates, gammas),
#         "NeuralGradientDice": product(learning_rates, gammas),
#     }
# )

computer_sleep()

# ---------------------------------------------------------------- #
