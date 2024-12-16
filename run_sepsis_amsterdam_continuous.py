# ---------------------------------------------------------------- #

from itertools import product

from tensorflow.keras.optimizers.schedules import ExponentialDecay, PiecewiseConstantDecay # type: ignore

from dice_rl_TU_Vienna.runners.neural_dual_dice_runner     import NeuralDualDiceRunner
from dice_rl_TU_Vienna.runners.neural_gen_dice_runner      import NeuralGenDiceRunner
from dice_rl_TU_Vienna.runners.neural_gradient_dice_runner import NeuralGradientDiceRunner

from dice_rl_TU_Vienna.runners.aux_recorders import aux_recorder_cos_angle

from plugins.medical_rl.sepsis_amsterdam.continuous.load import *

from utils.general import list_safe_zip
from utils.bedtime import computer_sleep

# ---------------------------------------------------------------- #

num_steps = 500_000
batch_size = 1024
regularizer_mlp = 0.0

gamma = 0.9
lam = 1.0
f_exponent = 1.5

# ---------------------------------------------------------------- #

learning_rates = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
hidden_dimss = [ (32,), (64,), (128,), (256,), ]

# ---------------------------------------------------------------- #

def run_sepsis_amsterdam_continuous(loops):

    for learning_rate, hidden_dims in loops["NeuralDualDice"]:
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
            target_policy=evaluation_policy,
            save_dir=save_dir,
            by=by,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

    for learning_rate, hidden_dims in loops["NeuralGenDice"]:
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
            target_policy=evaluation_policy,
            save_dir=save_dir,
            by=by,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

    for learning_rate, hidden_dims in loops["NeuralGradientDice"]:
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
            target_policy=evaluation_policy,
            save_dir=save_dir,
            by=by,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

# ---------------------------------------------------------------- #

run_sepsis_amsterdam_continuous(
    loops={
        "NeuralDualDice":     [],
        "NeuralGenDice":      [],
        "NeuralGradientDice": product([1e-3], [ (128,)]),
    }
)

computer_sleep()

# ---------------------------------------------------------------- #
