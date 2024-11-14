from plugins.medical_rl.sepsis_amsterdam.continuous.specs import get_observation_action_spec_sepsis_amsterdam_continuous
from plugins.medical_rl.sepsis_amsterdam.tabular.specs import get_observation_action_spec_sepsis_amsterdam_tabular

get_observation_action_spec_sepsis_amsterdam = {
    "tabular":    get_observation_action_spec_sepsis_amsterdam_tabular,
    "continuous": get_observation_action_spec_sepsis_amsterdam_continuous,
}
