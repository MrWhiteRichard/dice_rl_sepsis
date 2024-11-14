# ---------------------------------------------------------------- #

import os

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

# ---------------------------------------------------------------- #
