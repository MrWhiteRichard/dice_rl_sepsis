# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.dataset import get_dataset_spec
from dice_rl_TU_Vienna.dataset import load_or_create_dataset_Experience

from plugins.boyan_chain.specs import get_observation_action_spec_boyan_chain_continuous
from plugins.boyan_chain.environment import get_env
from plugins.boyan_chain.policy import TFPolicyBoyanChain
from plugins.boyan_chain.analytical_solver import AnalyticalSolverBoyanChain
from plugins.boyan_chain.continuous.config import *

# ---------------------------------------------------------------- #

observation_spec, action_spec = get_observation_action_spec_boyan_chain_continuous(N)
dataset_spec = get_dataset_spec(observation_spec, action_spec, step_num_max=2)

env = {
    seed: {
        k: get_env(seed=seed, N=N, kind=kind[k])
            for k in K
    }
        for seed in seeds
}

dataset = {
    seed: {
        k: load_or_create_dataset_Experience(
            dataset_dir=dataset_dir[seed][k],
            env=env[seed][k], num_experience=num_experience, seed=seed,
        )
            for k in K
    }
        for seed in seeds
}

target_policy = TFPolicyBoyanChain(
    N=N, p=p, tabular_continuous=tabular_continuous)

analytical_solver = {
    k: AnalyticalSolverBoyanChain(N=N, p=p, kind=kind[k])
        for k in K
}

# ---------------------------------------------------------------- #