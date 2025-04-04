# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.utils.os import os_path_join

# ---------------------------------------------------------------- #

names = ["original", "exploratory", "evaluation"]
labels = ["train", "test"]

# ---------------------------------------------------------------- #

id_split = "2025-01-30T11:42:45.694984"
id_clustering = "2025-01-30T12:01:58.273325"
id_policy = {
    "continuous": "2025-01-30T11:50:56.552678",
}
id_dataset = {}

# ---------------------------------------------------------------- #

dir_data   = os_path_join("data",   "medical_rl", "sepsis_amsterdam")
dir_images = os_path_join("images", "medical_rl", "sepsis_amsterdam")

dir_split = os_path_join(dir_data, id_split)
dir_clustering = os_path_join(dir_split, id_clustering)
dir_policy = {
    "continuous": os_path_join(dir_split, id_policy["continuous"]),
}
dir_dataset = {}

# ---------------------------------------------------------------- #
