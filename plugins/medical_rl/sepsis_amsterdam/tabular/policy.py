# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.plugins.stable_baselines3.policy import TFPolicyMaskablePPO
from dice_rl_TU_Vienna.specs import get_observation_action_spec_tabular

# ---------------------------------------------------------------- #

class TFPolicyMedicalRLTabular(TFPolicyMaskablePPO):
    def __init__(self, model, bounds, action_masks):
        super().__init__(
            model,
            action_masks,
            *get_observation_action_spec_tabular(bounds),
        )

# ---------------------------------------------------------------- #
