# ---------------------------------------------------------------- #

import os

from dice_rl_TU_Vienna.applications.medical_rl.dataset import load_or_create_dataset_medical_rl
from dice_rl_TU_Vienna.applications.medical_rl.continuous.policy import (
    TFPolicyMedicalRLContinuous, load_model_medical_rl_continuous, )

from medical_rl.data_formatters.amsterdam import AmsterdamFormatter

# ---------------------------------------------------------------- #

data_dir = os.path.join("data", "medical_rl")

datasets_dir = os.path.join(data_dir, "datasets")
policies_dir = os.path.join(data_dir, "policies")
outputs_dir  = os.path.join(data_dir, "outputs")

train_size = 0.4
valid_size = 0.1
test_size = 0.5
seed = 42

n_pads = 1

policy_n_neurons = 1024
policy_learning_rate = 0.0001
policy_batch_size = 512
policy_gamma = 0.99

hparam_str_dataframe = "_".join([
    f"split={train_size, valid_size, test_size}", f"{seed=}",
])
hparam_str_dataset = "_".join([
    f"{n_pads=}",
])
hparam_str_policy = "_".join([
    f"n_neurons={policy_n_neurons}",
    f"learning_rate={policy_learning_rate}",
    f"batch_size={policy_batch_size}",
    f"gamma={policy_gamma}",
])

x = os.path.join(datasets_dir, hparam_str_dataframe)
y = os.path.join(policies_dir, hparam_str_dataframe)
z = os.path.join(outputs_dir,  hparam_str_dataframe, hparam_str_policy)

dataset_dir = os.path.join(x, hparam_str_dataset)
policy_dir  = os.path.join(y, f"{hparam_str_policy}.h5")
save_dir    = os.path.join(z, hparam_str_dataset)

save_dir_images = os.path.join(outputs_dir, hparam_str_dataframe, "images")

# ---------------------------------------------------------------- #

_, _, test, bounds = AmsterdamFormatter().load_random_split_and_bounds(
    os.path.join(data_dir, "splits"), seed,
    train_size, valid_size, test_size )

obs_min, obs_max, act_min, act_max = bounds

evaluation_model = load_model_medical_rl_continuous(
    policy_n_neurons, policy_learning_rate, policy_batch_size, policy_gamma,
    policy_dir)

# trash_model = load_model_medical_rl_continuous(
    # policy_n_neurons)

# ---------------------------------------------------------------- #

dataset = load_or_create_dataset_medical_rl(
    dataset_dir=dataset_dir,
    data=test,
    bounds=bounds,
    n_pads=n_pads,
)

evaluation_policy = TFPolicyMedicalRLContinuous(
    evaluation_model, bounds, )

# trash_policy = TFPolicyMedicalRLContinuous(
#     trash_model, bounds, )

# ---------------------------------------------------------------- #
