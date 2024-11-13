# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.runners.neural_dual_dice_runner     import NeuralDualDiceRunner
from dice_rl_TU_Vienna.runners.neural_gen_dice_runner      import NeuralGenDiceRunner
from dice_rl_TU_Vienna.runners.neural_gradient_dice_runner import NeuralGradientDiceRunner

from dice_rl_TU_Vienna.dataset import one_hot_encode_observation

from dice_rl_TU_Vienna.applications.boyan_chain.continuous.config import *

from utils.bedtime import computer_sleep

# ---------------------------------------------------------------- #

num_steps = 100_000
batch_size = 64
hidden_dims = (32,)
regularizer_mlp = 0.0

# ---------------------------------------------------------------- #

for seed in [1, 2, 3]:

    # episodic
    k = "e"
    save_dir = os.path.join(outputs_dir, hparam_str_policy, hparam_str_dataset[seed][k])

    A = [1.5] * 3
    B = [0.005, 0.001, 0.001]
    C = [0.1, 0.5, 0.9]
    for f_exponent, learning_rate, gamma in zip(A, B, C):
        break
        neural_dual_dice_runner = NeuralDualDiceRunner(
            gamma=gamma,
            num_steps=num_steps,
            batch_size=batch_size,
            primal_hidden_dims=hidden_dims,
            dual_hidden_dims=hidden_dims,
            primal_learning_rate=learning_rate,
            dual_learning_rate=learning_rate,
            regularizer_primal=regularizer_mlp,
            regularizer_dual=regularizer_mlp,
            f_exponent=f_exponent,
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
        )

    A = [1.0] * 3
    B = [0.005, 0.001, 0.001]
    C = [0.1, 0.5, 0.9]
    for regularizer_norm, learning_rate, gamma in zip(A, B, C):
        break
        neural_gen_dice_runner = NeuralGenDiceRunner(
            gamma=gamma,
            num_steps=num_steps,
            batch_size=batch_size,
            primal_hidden_dims=hidden_dims,
            dual_hidden_dims=hidden_dims,
            primal_learning_rate=learning_rate,
            dual_learning_rate=learning_rate,
            norm_learning_rate=learning_rate,
            regularizer_primal=regularizer_mlp,
            regularizer_dual=regularizer_mlp,
            regularizer_norm=regularizer_norm,
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
        )

    A = [1.0] * 3
    B = [0.005, 0.005, 0.005]
    C = [0.1, 0.5, 0.9]

    for regularizer_norm, learning_rate, gamma in zip(A, B, C):
        break
        neural_gen_dice_runner = NeuralGradientDiceRunner(
            gamma=gamma,
            num_steps=num_steps,
            batch_size=batch_size,
            primal_hidden_dims=hidden_dims,
            dual_hidden_dims=hidden_dims,
            primal_learning_rate=learning_rate,
            dual_learning_rate=learning_rate,
            norm_learning_rate=learning_rate,
            regularizer_primal=regularizer_mlp,
            regularizer_dual=regularizer_mlp,
            regularizer_norm=regularizer_norm,
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
        )

    # continuing
    k = "c"
    save_dir = os.path.join(outputs_dir, hparam_str_policy, hparam_str_dataset[seed][k])

    A = [0.01]
    B = [0.5]
    for learning_rate, regularizer_norm in zip(A, B):
        break
        neural_gen_dice_runner = NeuralGenDiceRunner(
            gamma=1.0,
            num_steps=num_steps,
            batch_size=batch_size,
            primal_hidden_dims=hidden_dims,
            dual_hidden_dims=hidden_dims,
            primal_learning_rate=learning_rate,
            dual_learning_rate=learning_rate,
            norm_learning_rate=learning_rate,
            regularizer_primal=regularizer_mlp,
            regularizer_dual=regularizer_mlp,
            regularizer_norm=regularizer_norm,
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
        )

    A = [0.01]
    B = [0.1]
    for learning_rate, regularizer_norm in zip(A, B):
        break
        neural_gradient_dice_runner = NeuralGradientDiceRunner(
            gamma=1.0,
            num_steps=num_steps,
            batch_size=batch_size,
            primal_hidden_dims=hidden_dims,
            dual_hidden_dims=hidden_dims,
            primal_learning_rate=learning_rate,
            dual_learning_rate=learning_rate,
            norm_learning_rate=learning_rate,
            regularizer_primal=regularizer_mlp,
            regularizer_dual=regularizer_mlp,
            regularizer_norm=regularizer_norm,
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
        )

# ---------------------------------------------------------------- #

computer_sleep()

# ---------------------------------------------------------------- #
