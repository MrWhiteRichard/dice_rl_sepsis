# ---------------------------------------------------------------- #

from itertools import product

from dice_rl_TU_Vienna.runners.neural_dual_dice_runner     import NeuralDualDiceRunner
from dice_rl_TU_Vienna.runners.neural_gen_dice_runner      import NeuralGenDiceRunner
from dice_rl_TU_Vienna.runners.neural_gradient_dice_runner import NeuralGradientDiceRunner

from dice_rl_TU_Vienna.dataset import one_hot_encode_observation
from dice_rl_TU_Vienna.runners.aux_recorders import aux_recorder_cos_angle

from plugins.boyan_chain.continuous.load import *

from utils.general import list_safe_zip
from utils.bedtime import computer_sleep

# ---------------------------------------------------------------- #

num_steps = 100_000
batch_size = 64
hidden_dims = (32,)
regularizer_mlp = 0.0

# ---------------------------------------------------------------- #

seeds = [0, 1, 2, 3]
learning_rates = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
gammas = [0.1, 0.5, 0.9]
lams = [0.0, 0.1, 0.5, 1.0, 2.0]

# ---------------------------------------------------------------- #

def run_boyan_chain_continuous(seeds, loops):

    for seed in seeds:

        # episodic
        k = "e"
        save_dir = os.path.join(outputs_dir, hparam_str_policy, hparam_str_dataset[seed][k])

        f_exponent = 1.5
        lam = 1.0

        for learning_rate, gamma in loops["NeuralDualDice"]["episodic"]:
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

        for learning_rate, gamma in loops["NeuralGenDice"]["episodic"]:
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

        for learning_rate, gamma in loops["NeuralGradientDice"]["episodic"]:
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

        for learning_rate, lam in loops["NeuralGenDice"]["continuing"]:
            NeuralGenDiceRunner(
                gamma=1.0,
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

        for learning_rate, lam in loops["NeuralGradientDice"]["continuing"]:
            NeuralGradientDiceRunner(
                gamma=1.0,
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

# run_boyan_chain_continuous(
#     seeds=[0],
#     loops={
#         "NeuralDualDice":      { "episodic": product(learning_rates, gammas), },
#         "NeuralGenDice":       { "episodic": product(learning_rates, gammas), "continuing": product(learning_rates, lams) },
#         "NeuralGradientDice":  { "episodic": product(learning_rates, gammas), "continuing": product(learning_rates, lams) },
#     }
# )

# run_boyan_chain_continuous(
#     seeds=[1, 2, 3],
#     loops={
#         "NeuralDualDice":      { "episodic": list_safe_zip([0.01, 0.001, 0.001], gammas), },
#         "NeuralGenDice":       { "episodic": list_safe_zip([0.01, 0.01, 0.01], gammas), "continuing": list_safe_zip([0.01], [1.0]) },
#         "NeuralGradientDice":  { "episodic": list_safe_zip([0.01, 0.01, 0.01], gammas), "continuing": list_safe_zip([0.01], [1.0]) },
#     }
# )

computer_sleep()

# ---------------------------------------------------------------- #
