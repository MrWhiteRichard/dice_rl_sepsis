# ---------------------------------------------------------------- #

from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay # type: ignore

from dice_rl_TU_Vienna.runners.neural_dual_dice_runner     import NeuralDualDiceRunner
from dice_rl_TU_Vienna.runners.neural_gen_dice_runner      import NeuralGenDiceRunner
from dice_rl_TU_Vienna.runners.neural_gradient_dice_runner import NeuralGradientDiceRunner

from plugins.medical_rl.continuous.config import *

from utils.bedtime import computer_sleep

# ---------------------------------------------------------------- #

gamma = 0.9
num_steps = 500_000
batch_size = 1024
regularizer_mlp = 0.0
regularizer_norm = 1.0
f_exponent = 1.5

target_policy = evaluation_policy
by = "steps"

# ---------------------------------------------------------------- #

for seed in []:
    learning_rates = []
    hidden_dimss = []
    for learning_rate, hidden_dims in zip(learning_rates, hidden_dimss):
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
        )

for seed in [0]:
    learning_rates = [
        # 1e-4,
        # PiecewiseConstantDecay([27_500], [1e-4, 1e-5]),
        # PiecewiseConstantDecay([17_500], [1e-4, 1e-5]),
        # PiecewiseConstantDecay([15_000], [1e-4, 1e-5]),
        # 5e-5,
        # PiecewiseConstantDecay([25_000], [5e-5, 1e-5]),
    ]
    hidden_dimss = [
        # *
        # (32,),
        # (64,),
        # (128,),
        # (256,),
        # (256,),
    ]
    for learning_rate, hidden_dims in zip(learning_rates, hidden_dimss):

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
            )

for seed in [0]:
    learning_rates = [
        # 1e-3,
    ]
    hidden_dimss = [
        # *
    ]
    for learning_rate, hidden_dims in zip(learning_rates, hidden_dimss):

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
        )

# ---------------------------------------------------------------- #

computer_sleep()

# ---------------------------------------------------------------- #
