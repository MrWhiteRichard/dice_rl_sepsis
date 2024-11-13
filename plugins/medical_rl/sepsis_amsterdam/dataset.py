# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from tf_agents.trajectories.time_step import time_step_spec as get_time_step_spec

from tqdm import tqdm

from dice_rl.data.dataset import EnvStep
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

from dice_rl_TU_Vienna.specs import get_step_num_spec
from dice_rl_TU_Vienna.applications.dice_rl.create_dataset import add_episodes_to_dataset
from dice_rl_TU_Vienna.applications.medical_rl.specs import get_observation_action_spec_medical_rl

from medical_rl.data_formatters.amsterdam import AmsterdamFormatter

from dice_rl_TU_Vienna.dataset import TFOffpolicyDatasetGenerator_StepsEpisodes
from dice_rl_TU_Vienna.dataset import load_or_create_dataset

# ---------------------------------------------------------------- #

class TFOffpolicyDatasetGenerator:
    def __init__(self,
        data_formatter,
        data,
        bounds,
        n_pads=1):

        self.data_formatter = data_formatter
        self.data = data
        self.n_pads = n_pads

        ids, _, observations, _, _ = self.data_formatter.RL_split(data)
        ids_unique, ids_counts  = np.unique(ids, return_counts=True)
        episode_lengths         = ids_counts

        self.num_episodes         = len(ids_unique)
        self.max_episode_length = np.max(episode_lengths)

        self.space_type = "tabular" if len(observations.shape) == 1 else "continuous"
        self.n_features = 1 if self.space_type == "tabular" else observations.shape[-1]
        self.n_samples  = len(data)

        self.spec     = self.get_spec(bounds)
        self.capacity = self.get_capacity()
        self.dataset  = TFOffpolicyDataset(self.spec, self.capacity)

        self.obs_index = False

    def get_spec(self, bounds):
        observation_spec, action_spec = self.get_observation_action_spec(bounds)

        time_step_spec = get_time_step_spec(observation_spec)
        step_num_spec = get_step_num_spec(step_num_max=self.max_episode_length)

        policy_info, env_info, other_info = self.get_info_specs()

        return EnvStep(
            step_type  =time_step_spec.step_type,
            step_num   =step_num_spec,
            observation=time_step_spec.observation,
            action     =action_spec,
            reward     =time_step_spec.reward,
            discount   =time_step_spec.discount,
            policy_info=policy_info,
            env_info   =env_info,
            other_info =other_info,
        )

    def get_observation_action_spec(self, bounds):
        return None, None

    def get_info_specs(self):
        return {}, {}, {}

    def get_capacity(self):
        return self.n_samples + self.n_pads * self.num_episodes

    def add_episodes_to_dataset(self, verbosity=0, max_episodes=float("inf")):
        pbar = enumerate(self.data["id"].unique())
        if verbosity > 0:
            print("adding episodes to dataset")
            pbar = tqdm( pbar, total=int(min( len(self.data["id"].unique()), max_episodes )) )

        for i, id in pbar:
            if i >= max_episodes: break
            episode, valid_ids = self.get_episode(id)
            add_episodes_to_dataset(episode, valid_ids, self.dataset)

    def get_episode(self, id):
        episode_length = np.sum(self.data["id"] == id)
        episode_length_padded = episode_length + self.n_pads

        step_type = self.get_step_type(episode_length_padded)
        step_num  = self.get_step_num (episode_length_padded)

        observation, action, reward = self.get_observation_action_reward(id)
        discount = self.get_discount(episode_length_padded)
        policy_info, env_info, other_info = self.get_info(id)

        episode =  EnvStep(
            step_type, step_num,
            observation, action, reward,
            discount,
            policy_info, env_info, other_info)

        valid_ids = self.get_valid_ids(episode_length_padded)

        return episode, valid_ids

    def get_step_type(self, episode_length_padded):
        step_type_np = np.ones(episode_length_padded)
        step_type_np[0] = 0
        step_type_np[-1] = 2
        step_type = tf.convert_to_tensor(
            step_type_np, dtype=self.spec.step_type.dtype)

        return step_type

    def get_step_num(self, episode_length_padded):
        step_num_np = np.arange(episode_length_padded)
        step_num = tf.convert_to_tensor(
            step_num_np, dtype=self.spec.step_num.dtype)

        return step_num

    def get_observation_action_reward(self, id):
        data_filtered = self.data[self.data["id"] == id]
        _, _, obs, act, rew = self.data_formatter.RL_split(data_filtered)

        obs = np.array(obs)
        act = np.array(act)
        rew = np.array(rew)

        if self.obs_index:
            indices = np.array(data_filtered.index)
            indices = np.expand_dims(indices, axis=1)
            if self.space_type == "tabular": obs = np.expand_dims(obs, axis=1)
            obs = np.concatenate([indices, obs], axis=1)

        obs_term = np.expand_dims(obs[-1], axis=0)
        act_term = np.expand_dims(0, axis=0)
        rew_term = np.expand_dims(0, axis=0)

        obs = np.concatenate([obs] + [obs_term] * self.n_pads, axis=0)
        act = np.concatenate([act] + [act_term] * self.n_pads, axis=0)
        rew = np.concatenate([rew] + [rew_term] * self.n_pads, axis=0)

        obs = tf.convert_to_tensor(obs, dtype=self.spec.observation.dtype)
        act = tf.convert_to_tensor(act, dtype=self.spec.action     .dtype)
        rew = tf.convert_to_tensor(rew, dtype=self.spec.reward     .dtype)

        return obs, act, rew

    def get_discount(self, episode_length_padded):
        gamma = 1
        discount_np = gamma ** np.arange(episode_length_padded)
        discount_np[-1] = 0
        discount = tf.convert_to_tensor(
            discount_np, dtype=self.spec.discount.dtype)

        return discount

    def get_info(self, id):
        return {}, {}, {}

    def get_valid_ids(self, episode_length_padded):
        valid_ids_int = tf.ones(episode_length_padded)
        valid_ids = tf.cast(valid_ids_int, dtype=tf.bool)

        return valid_ids

# ---------------------------------------------------------------- #

class TFOffpolicyDatasetGeneratorMedicalRL(TFOffpolicyDatasetGenerator):
    def __init__(self, data, bounds, n_pads=1):
        data_formatter = AmsterdamFormatter()
        super().__init__(data_formatter, data, bounds, n_pads)

    def get_observation_action_spec(self, bounds):
        return get_observation_action_spec_medical_rl[self.space_type](bounds)

# ---------------------------------------------------------------- #

def load_or_create_dataset_medical_rl(dataset_dir, data, bounds, n_pads):

    try:
        print("Try loading dataset", end=" ")
        dataset = TFOffpolicyDataset.load(dataset_dir)

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        assert False

    except:
        print(); print(f"No dataset found in {dataset_dir}")

        generator = TFOffpolicyDatasetGeneratorMedicalRL(
            data=data, bounds=bounds,
            n_pads=n_pads, )

        generator.add_episodes_to_dataset(verbosity=1)

        dataset = generator.dataset

        if not tf.io.gfile.isdir(dataset_dir):
            tf.io.gfile.makedirs(dataset_dir)

        dataset.save(dataset_dir)

    return dataset

# ---------------------------------------------------------------- #

def load_or_create_dataset_medical_rl_simulator(
        dataset_dir,
        env, get_act,
        num_trajectory, max_trajectory_length,
        seed,
        by, n_pads):

    get_generator = lambda: TFOffpolicyDatasetGenerator_StepsEpisodes(
        env=env, get_act=get_act,
        num_trajectory=num_trajectory, max_trajectory_length=max_trajectory_length,
        seed=seed,
        by=by, n_pads=n_pads)

    dataset = load_or_create_dataset(dataset_dir, get_generator)

    return dataset

# ---------------------------------------------------------------- #
