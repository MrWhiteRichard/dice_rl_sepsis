# ---------------------------------------------------------------- #

import os
import warnings

from dice_rl_TU_Vienna.applications.medical_rl.tabular.policy import TFPolicyMedicalRLTabular
from dice_rl_TU_Vienna.applications.medical_rl.column_labels import get_column_labels
from dice_rl_TU_Vienna.applications.medical_rl.dataset import (
    load_or_create_dataset_medical_rl, load_or_create_dataset_medical_rl_simulator, )

from dice_rl_TU_Vienna.wrappers import AbsorbingWrapper, LoopingWrapper
from dice_rl_TU_Vienna.applications.stable_baslines.policy import load_or_create_model_MaskablePPO

from medical_rl.data_formatters.amsterdam import AmsterdamFormatter
from medical_rl.libs.cluster import get_data_clustered_completed
from medical_rl.libs.envs import get_env

# ---------------------------------------------------------------- #

data_dir = os.path.join("data", "medical_rl")

datasets_dir = os.path.join(data_dir, "datasets")
policies_dir = os.path.join(data_dir, "policies")
outputs_dir  = os.path.join(data_dir, "outputs")

# -------------------------------- #

train_size = 0.4
valid_size = 0.1
test_size = 0.5
seed = 42

column_labels_id = 2

n_clusters = 256
n_init = 10

n_pads = 1

total_timesteps_model = { "ex": 10_000, "ev": 100_000, }

num_trajectory = 10_000
max_trajectory_length = None

K = ["", "ex", "ev"]
names = { "": "original", "ex": "exploratory", "ev": "evaluation", }

# -------------------------------- #

train, _, test = AmsterdamFormatter().load_random_split(
    os.path.join(data_dir, "splits"), seed,
    train_size, valid_size, test_size)

# ---------------------------------------------------------------- #

hparam_str_model = {
    k: f"total_timesteps={total_timesteps_model[k]}"
        for k in ["ex", "ev"]
}

hparam_str_split = "_".join([
    f"split={train_size, valid_size, test_size}", f"{seed=}",
])
hparam_str_clustering = "_".join([
    f"{column_labels_id=}",
    f"{n_clusters=}", f"{n_init=}",
])
hparam_str_dataframe = "_".join([
    hparam_str_split, hparam_str_clustering,
])

hparam_str_dataset = {}    
hparam_str_dataset[""] = "_".join([
    f"{n_pads=}",
])
for k in ["ex", "ev"]:
    hparam_str_dataset[k] = "_".join([
        hparam_str_model[k],
        f"{num_trajectory=}",
        f"{max_trajectory_length=}",
        f"{seed=}",
        f"{n_pads=}"
    ])

x  = os.path.join(datasets_dir, hparam_str_dataframe)
y  = os.path.join(policies_dir, hparam_str_dataframe)
z1 = os.path.join(outputs_dir,  hparam_str_dataframe)
z2 = os.path.join(z1, hparam_str_model["ev"])

dataset_dir = {
    k: os.path.join(x, hparam_str_dataset[k])
        for k in K
}

model_dir = {
    k: os.path.join(y, hparam_str_model[k])
        for k in ["ex", "ev"]
}

save_dir_images = os.path.join(z1, "images")

save_dir = {
    k: os.path.join(z2, hparam_str_dataset[k])
        for k in K
}

kmeans_dir = x

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

dataset = {}

dataset[""] = load_or_create_dataset_medical_rl(
    dataset_dir=dataset_dir[""],
    data=test_clustered, bounds=bounds, n_pads=n_pads, )

for k in ["ev", "ex"]:
    dataset[k] = load_or_create_dataset_medical_rl_simulator(
        dataset_dir=dataset_dir[k],
        env=env_test, get_act=get_act_model_exploratory,
        num_trajectory=num_trajectory, max_trajectory_length=max_trajectory_length,
        seed=seed,
        by="steps",
        n_pads=n_pads,
    )

evaluation_policy = TFPolicyMedicalRLTabular(
    model=model["ev"], bounds=bounds, action_masks=action_masks_test)

# ---------------------------------------------------------------- #
