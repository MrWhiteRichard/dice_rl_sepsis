# ---------------------------------------------------------------- #

from medical_rl.libs.RL import Model

from dice_rl_TU_Vienna.policy import MyTFPolicy
from plugins.medical_rl.continuous.specs import get_observation_action_spec_medical_rl_continuous

# ---------------------------------------------------------------- #

class TFPolicyMedicalRLContinuous(MyTFPolicy):
    def __init__(self, model, bounds):
        self.model = model

        super().__init__(
            *get_observation_action_spec_medical_rl_continuous(bounds) )

    def _probs(self, time_step):
        pass

    def _logits(self, time_step):
        observation = time_step.observation
        logits, _ = self.model(observation)
        return logits


def load_model_medical_rl_continuous(
        n_neurons, learning_rate=None, batch_size=None, gamma=None,
        model_dir=None):

    model = Model(5, n_neurons)

    A = learning_rate is None
    B = batch_size is None
    C = gamma is None
    D = model_dir is None

    if not (A or B or C):
        assert not D
        print("Loading model", model_dir)
        model.load_weights(model_dir)

    return model

# ---------------------------------------------------------------- #
