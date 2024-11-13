import numpy as np
import tensorflow as tf

from tf_agents.specs import TensorSpec

from RRT_mimic_iv.data_formatter import MimicIVFormatter

from dice_rl_TU_Vienna.applications.medical_rl.dataset import TFOffpolicyDatasetGenerator
from dice_rl_TU_Vienna.applications.RRT_mimic_iv.specs import get_observation_action_spec_RRT_mimic_iv


class TFOffpolicyDatasetGeneratorSepsisMimicIV(TFOffpolicyDatasetGenerator):
    obs_index = False
 
    def __init__(self, space_type="continuous", n_pads=1):
        data_formatter = MimicIVFormatter(space_type=space_type)
        _, _, data, _ = data_formatter.load_random_split()
        obs_min, obs_max, act_min, act_max = data_formatter.RL_bounds()

        super().__init__(
            data_formatter,
            data,
            obs_min, obs_max, act_min, act_max,
            n_pads)

    def get_observation_action_spec(self, obs_min, obs_max, act_min, act_max):
        return get_observation_action_spec_RRT_mimic_iv[self.space_type](
            obs_min, obs_max, act_min, act_max )

    def get_info_specs(self):
        policy_info = {}
        env_info = {}
        other_info = {
            "target_policy_probs": TensorSpec(
                shape=(20,),
                dtype=np.float32,
                name="target_policy_probs",
            )
        }

        return policy_info, env_info, other_info

    def get_other_info(self, id):
        data_filtered = self.data[self.data.ID == id]
        indices = np.array(data_filtered.index)
        indices = np.pad(indices, (0, self.n_pads), mode="edge")
        target_policy_probs = self.data_formatter.model_probs[indices]
        target_policy_probs = tf.convert_to_tensor(
            target_policy_probs,
            dtype=tf.float32,
            name="target_policy_probs",
        )
        other_info = { "target_policy_probs": target_policy_probs }
        return other_info
