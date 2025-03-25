# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.utils.os import os_path_join

# ---------------------------------------------------------------- #

names = ["original", "exploratory", "evaluation"]

# ---------------------------------------------------------------- #

id_split = "2025-01-30T11:42:45.694984"
id_clustering = "2025-01-30T12:01:58.273325"
id_policy = {
    "continuous": "2025-01-30T11:50:56.552678",
    "tabular": {
        "original": None,
        "exploratory": "2025-01-31T10:25:32.802016",
        "evaluation":  "2025-01-31T10:27:51.724072",
    }
}
id_dataset = {
    "exploratory": "2025-03-22T18:46:50.582092",
    "evaluation":  "2025-03-22T18:47:53.614328",
    "original":    "2025-03-22T18:29:07.873471",
}

# ---------------------------------------------------------------- #

dir_data   = os_path_join("data",   "medical_rl", "sepsis_amsterdam")
dir_images = os_path_join("images", "medical_rl", "sepsis_amsterdam")

dir_split = os_path_join(dir_data, id_split)
dir_clustering = os_path_join(dir_split, id_clustering)
dir_policy = {
    "continuous": os_path_join(dir_split, id_policy["continuous"]),
    "tabular": {
        name: os_path_join(dir_clustering, id_policy["tabular"][name])
            for name in names
    }
}
dir_dataset = {
    "continuous": dir_policy["continuous"],
    "tabular": {
        name: os_path_join(
            dir_policy["tabular"][name], id_dataset[name],
        )
            for name in names
    },
}

# ---------------------------------------------------------------- #
