# ---------------------------------------------------------------- #

import os
import warnings

from plugins.medical_rl.sepsis_amsterdam.tabular.policy import TFPolicyMedicalRLTabular
from plugins.medical_rl.sepsis_amsterdam.column_labels import get_column_labels
from dice_rl_TU_Vienna.dataset import (
    load_or_create_dataset_StepsEpisodes,
    load_or_create_dataset_Dataframe, )

from dice_rl_TU_Vienna.wrappers import AbsorbingWrapper, LoopingWrapper
from plugins.stable_baslines.policy import load_or_create_model_MaskablePPO

from medical_rl.data_formatters.amsterdam import AmsterdamFormatter
from medical_rl.libs.cluster import get_data_clustered_completed
from medical_rl.libs.envs import get_env

from plugins.medical_rl.sepsis_amsterdam.tabular.config import *
from plugins.medical_rl.sepsis_amsterdam.tabular.specs import \
    get_observation_action_spec_sepsis_amsterdam_tabular

# ---------------------------------------------------------------- #

print("loading split")
train, _, test = AmsterdamFormatter().load_random_split(
    os.path.join(data_dir, "splits"), seed,
    train_size, valid_size, test_size)

# -------------------------------- #

column_labels = get_column_labels(id=column_labels_id)

print("clustering train")
train_clustered = get_data_clustered_completed(
    data=train, column_labels=column_labels,
    n_clusters=n_clusters, n_init=n_init, seed=seed, path=kmeans_dir,
)

print("clustering test")
test_clustered = get_data_clustered_completed(
    data=test, column_labels=column_labels,
    n_clusters=n_clusters, n_init=n_init, seed=seed, path=kmeans_dir,
)

print("getting bounds")
bounds = AmsterdamFormatter().RL_bounds(n_clusters=n_clusters)
obs_min, obs_max, act_min, act_max = bounds

print("getting observation_action_spec")
observation_action_spec = \
    get_observation_action_spec_sepsis_amsterdam_tabular(bounds)

# -------------------------------- #

print("getting env_train")
env_train, action_masks_train = get_env(
    n_clusters=n_clusters,
    data_clustered=train_clustered, )

print("getting env_test")
env_test, action_masks_test = get_env(
    n_clusters=n_clusters,
    data_clustered=test_clustered, )

env_test_absorbing = AbsorbingWrapper(env_test)
env_test_looping   = LoopingWrapper  (env_test)

print("getting models")
model = {
    k: load_or_create_model_MaskablePPO(
        model_dir=model_dir[k],
        env=env_train,
        total_timesteps=total_timesteps_model[k],
    )
        for k in ["ex", "ev"]
}

# -------------------------------- #

def get_act_model_exploratory(obs):
    with warnings.catch_warnings(action="ignore", category=UserWarning):
        action_masks = env_test.action_masks()

    act, _ = model["ex"].predict(obs, action_masks=action_masks)
    return act

def get_act_model_evaluation(obs):
    with warnings.catch_warnings(action="ignore", category=UserWarning):
        action_masks = env_test.action_masks()

    act, _ = model["ev"].predict(obs, action_masks=action_masks)
    return act

get_act_model = {
    "ex": get_act_model_exploratory,
    "ev": get_act_model_evaluation,
}

# -------------------------------- #

print("getting datasets")

dataset = {}

dataset[""] = load_or_create_dataset_Dataframe(
    dataset_dir=dataset_dir[""],
    df=test_clustered,
    get_split=lambda df: AmsterdamFormatter().RL_split(df),
    observation_action_spec=observation_action_spec,
    n_pads=n_pads,
    verbosity=1,
)

for k in ["ev", "ex"]:
    dataset[k] = load_or_create_dataset_StepsEpisodes(
        dataset_dir=dataset_dir[k],
        env=env_test,
        get_act=get_act_model_exploratory,
        num_trajectory=num_trajectory,
        max_trajectory_length=max_trajectory_length,
        by="steps",
        seed=seed,
        n_pads=n_pads,
        verbosity=1,
    )

evaluation_policy = TFPolicyMedicalRLTabular(
    model=model["ev"], bounds=bounds, action_masks=action_masks_test)

# ---------------------------------------------------------------- #
