# ---------------------------------------------------------------- #

import numpy as np

from RRT_mimic_iv.data_formatter import MimicIVFormatter

from dice_rl_TU_Vienna.policy import MyTFPolicy
from plugins.RRT_mimic_iv.specs import get_observation_action_spec_RRT_mimic_iv

# ---------------------------------------------------------------- #

class TFPolicySepsisMimicIV(MyTFPolicy):
    def __init__(self, space_type="tabular", uniform=False):
        self.space_type = space_type
        self.uniform = uniform

        data_formatter = MimicIVFormatter(space_type=space_type)
        bounds = data_formatter.RL_bounds()

        super().__init__(
            *get_observation_action_spec_RRT_mimic_iv[space_type](*bounds), )

        self.probs_all = data_formatter.additional_vars["model_probs"]

    def _probs(self, time_step):
        index = time_step.other_info["indices"]

        probs = self.probs_all[index]
        assert abs(np.sum(probs) - 1) < 1e-9

        if self.uniform:
            probs = np.ones_like(probs) / len(probs)

        return probs

# ---------------------------------------------------------------- #
