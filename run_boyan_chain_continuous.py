# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.runners.neural_dual_dice_runner     import NeuralDualDiceRunner
from dice_rl_TU_Vienna.runners.neural_gen_dice_runner      import NeuralGenDiceRunner
from dice_rl_TU_Vienna.runners.neural_gradient_dice_runner import NeuralGradientDiceRunner

from dice_rl_TU_Vienna.dataset import one_hot_encode_observation
from dice_rl_TU_Vienna.runners.aux_recorders import aux_recorder_cos_angle

from plugins.boyan_chain.continuous.config import *
from plugins.boyan_chain.continuous.load import *

from utils.bedtime import computer_sleep

# ---------------------------------------------------------------- #

num_steps = 100_000
batch_size = 64
hidden_dims = (32,)
regularizer_mlp = 0.0

# ---------------------------------------------------------------- #

for seed in [0, 1, 2, 3]:

    # episodic
    k = "e"
    save_dir = os.path.join(outputs_dir, hparam_str_policy, hparam_str_dataset[seed][k])

    f_exponent = 1.5
    regularizer_norm = 1.0

    A = [0.005, 0.001, 0.001]
    B = [0.1, 0.5, 0.9]
    for learning_rate, gamma in zip(A, B):
        # break
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
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

    A = [0.005, 0.001, 0.001]
    B = [0.1, 0.5, 0.9]
    for learning_rate, gamma in zip(A, B):
        # break
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
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

    A = [0.005, 0.005, 0.005]
    B = [0.1, 0.5, 0.9]

    for learning_rate, gamma in zip(A, B):
        # break
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
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

    # continuing
    k = "c"
    save_dir = os.path.join(outputs_dir, hparam_str_policy, hparam_str_dataset[seed][k])

    A = [0.01]
    B = [0.5]
    for learning_rate, regularizer_norm in zip(A, B):
        # break
        neural_gen_dice_runner = NeuralGenDiceRunner(
            gamma=1.0,
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
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

    A = [0.01]
    B = [0.1]
    for learning_rate, regularizer_norm in zip(A, B):
        # break
        neural_gradient_dice_runner = NeuralGradientDiceRunner(
            gamma=1.0,
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
            dataset=dataset[seed][k],
            dataset_spec=dataset_spec,
            target_policy=target_policy,
            save_dir=save_dir,
            by=by,
            analytical_solver=analytical_solver[k],
            env_step_preprocessing=one_hot_encode_observation,
            aux_recorder=aux_recorder_cos_angle,
            aux_recorder_pbar=["cos_angle"],
        )

# ---------------------------------------------------------------- #

computer_sleep()

# ---------------------------------------------------------------- #
