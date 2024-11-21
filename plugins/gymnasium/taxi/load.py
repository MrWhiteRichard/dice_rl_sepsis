# ---------------------------------------------------------------- #

from dice_rl.environments.env_policies import get_env_and_policy
from plugins.dice_rl.dataset import load_or_create_dataset

from plugins.gymnasium.taxi.config import *

# ---------------------------------------------------------------- #


env, target_policy = get_env_and_policy(
    load_dir=policies_dir,
    env_name="taxi",
    alpha=1.0,
    tabular_obs=True)

dataset = {
    k: load_or_create_dataset(
        datasets_dir, policies_dir,
        env_name, seed, num_trajectory, max_trajectory_length, alphas[k], tabular_obs,
    )
        for k in K
}

# ---------------------------------------------------------------- #
