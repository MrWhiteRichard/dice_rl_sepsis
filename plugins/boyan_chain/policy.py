# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from dice_rl_TU_Vienna.policy import MyTFPolicy
from plugins.boyan_chain.specs import get_observation_action_spec_boyan_chain_continuous

# ---------------------------------------------------------------- #

class TFPolicyBoyanChain(MyTFPolicy):
    def __init__(self, N, p, tabular_continuous):
        self.N = N
        self.p = p
        self.tabular_continuous = tabular_continuous

        super().__init__( *get_observation_action_spec_boyan_chain_continuous(self.N) )

    def _probs(self, time_step):
        try:
            observation = time_step.observation

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            assert False

        except:
            observation = time_step

        probs = np.array([ self.p, 1-self.p ])

        if tf.rank(observation) > ( 1 if self.tabular_continuous else 0 ): # type: ignore
            l = len(observation)
            probs = np.tile(probs, l)
            probs = np.reshape(probs, [l, -1])

        return probs

# ---------------------------------------------------------------- #
