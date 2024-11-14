# ---------------------------------------------------------------- #

import os

from dice_rl_TU_Vienna.dataset import load_or_create_dataset_Dataframe

from plugins.medical_rl.sepsis_amsterdam.continuous.config import *
from plugins.medical_rl.sepsis_amsterdam.continuous.policy import (
    TFPolicyMedicalRLContinuous, load_model_medical_rl_continuous, )

from medical_rl.data_formatters.amsterdam import AmsterdamFormatter

from plugins.medical_rl.sepsis_amsterdam.continuous.specs import \
    get_observation_action_spec_sepsis_amsterdam_continuous

# ---------------------------------------------------------------- #

print("loading split and bounds")
_, _, test, bounds = AmsterdamFormatter().load_random_split_and_bounds(
    os.path.join(data_dir, "splits"), seed,
    train_size, valid_size, test_size )

print("getting observation_action_spec")
observation_action_spec = \
    get_observation_action_spec_sepsis_amsterdam_continuous(bounds)

print("getting model")
evaluation_model = load_model_medical_rl_continuous(
    policy_n_neurons, policy_learning_rate, policy_batch_size, policy_gamma,
    policy_dir)
# trash_model = load_model_medical_rl_continuous(
    # policy_n_neurons)

# ---------------------------------------------------------------- #

print("getting dataset")
dataset = load_or_create_dataset_Dataframe(
    dataset_dir=dataset_dir,
    df=test,
    get_split=lambda df: AmsterdamFormatter().RL_split(df),
    observation_action_spec=observation_action_spec,
    n_pads=n_pads,
    verbosity=1,
)

print("getting evaluation_policy")
evaluation_policy = TFPolicyMedicalRLContinuous(
    evaluation_model, bounds, )
# trash_policy = TFPolicyMedicalRLContinuous(
#     trash_model, bounds, )

# ---------------------------------------------------------------- #
