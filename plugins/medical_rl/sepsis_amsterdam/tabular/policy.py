# ---------------------------------------------------------------- #

from dice_rl_TU_Vienna.applications.stable_baslines.policy import TFPolicyMaskablePPO
from dice_rl_TU_Vienna.applications.medical_rl.tabular.specs import get_observation_action_spec_medical_rl_tabular

# ---------------------------------------------------------------- #

class TFPolicyMedicalRLTabular(TFPolicyMaskablePPO):
    def __init__(self, model, bounds, action_masks):
        super().__init__(
            model,
            action_masks,
            *get_observation_action_spec_medical_rl_tabular(bounds),
        )

# ---------------------------------------------------------------- #
