# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.runners.neural_dual_dice_runner     import NeuralDualDiceRunner
from dice_rl_TU_Vienna.runners.neural_gen_dice_runner      import NeuralGenDiceRunner
from dice_rl_TU_Vienna.runners.neural_gradient_dice_runner import NeuralGradientDiceRunner

from dice_rl_TU_Vienna.runners.aux_recorders import aux_recorder_cos_angle

from plugins.gymnasium.cartpole.load import *

from utils.bedtime import computer_sleep

# ---------------------------------------------------------------- #

num_steps = 100_000
batch_size = 64
hidden_dims = (32,)
regularizer_mlp = 0.0
f_exponent = 1.5
regularizer_norm = 1.0

by = "episodes"

# ---------------------------------------------------------------- #

learning_rates = []
gammas = []

learning_rates = [ 5e-4, 5e-4, 1e-3, ]
gammas = [0.1, 0.5, 0.9]
for learning_rate, gamma in zip(learning_rates, gammas):
    neural_dual_dice_runner = NeuralDualDiceRunner(
        gamma=gamma,
        num_steps=num_steps,
        batch_size=batch_size,
        seed=seed,
        primal_hidden_dims=hidden_dims,
        dual_hidden_dims=hidden_dims,
        primal_learning_rate=learning_rate,
        dual_learning_rate=learning_rate,
        regularizer_primal=regularizer_mlp,
        regularizer_dual=regularizer_mlp,
        f_exponent=f_exponent,
        dataset=dataset,
        target_policy=target_policy,
        save_dir=save_dir,
        by=by,
        aux_recorder=aux_recorder_cos_angle,
        aux_recorder_pbar=["cos_angle"],
    )

learning_rates = [ 5e-4, 1e-3, 5e-4, ]
gammas = [0.1, 0.5, 0.9]
for learning_rate, gamma in zip(learning_rates, gammas):
    neural_gen_dice_runner = NeuralGradientDiceRunner(
        gamma=gamma,
        num_steps=num_steps,
        batch_size=batch_size,
        seed=seed,
        primal_hidden_dims=hidden_dims,
        dual_hidden_dims=hidden_dims,
        primal_learning_rate=learning_rate,
        dual_learning_rate=learning_rate,
        norm_learning_rate=learning_rate,
        regularizer_primal=regularizer_mlp,
        regularizer_dual=regularizer_mlp,
        regularizer_norm=regularizer_norm,
        dataset=dataset,
        target_policy=target_policy,
        save_dir=save_dir,
        by=by,
        aux_recorder=aux_recorder_cos_angle,
        aux_recorder_pbar=["cos_angle"],
    )

learning_rates = [ 5e-4, 1e-4, ]
gammas = [0.1, 0.5]
for learning_rate, gamma in zip(learning_rates, gammas):
    neural_gen_dice_runner = NeuralGenDiceRunner(
        gamma=gamma,
        num_steps=num_steps,
        batch_size=batch_size,
        seed=seed,
        primal_hidden_dims=hidden_dims,
        dual_hidden_dims=hidden_dims,
        primal_learning_rate=learning_rate,
        dual_learning_rate=learning_rate,
        norm_learning_rate=learning_rate,
        regularizer_primal=regularizer_mlp,
        regularizer_dual=regularizer_mlp,
        regularizer_norm=regularizer_norm,
        dataset=dataset,
        target_policy=target_policy,
        save_dir=save_dir,
        by=by,
        aux_recorder=aux_recorder_cos_angle,
        aux_recorder_pbar=["cos_angle"],
    )

# ---------------------------------------------------------------- #

computer_sleep()

# ---------------------------------------------------------------- #
