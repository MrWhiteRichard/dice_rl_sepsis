from dice_rl_TU_Vienna.applications.medical_rl.continuous.specs import get_observation_action_spec_medical_rl_continuous
from dice_rl_TU_Vienna.applications.medical_rl.tabular.specs import get_observation_action_spec_medical_rl_tabular

get_observation_action_spec_medical_rl = {
    "tabular":    get_observation_action_spec_medical_rl_tabular,
    "continuous": get_observation_action_spec_medical_rl_continuous,
}
