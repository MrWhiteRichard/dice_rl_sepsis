# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from tf_agents.trajectories.time_step import time_step_spec as get_time_step_spec

from tqdm import tqdm

from dice_rl.data.dataset import EnvStep
from dice_rl.data.tf_offpolicy_dataset import TFOffpolicyDataset

from dice_rl_TU_Vienna.specs import get_step_num_spec
from dice_rl_TU_Vienna.applications.dice_rl.create_dataset import add_episodes_to_dataset
from dice_rl_TU_Vienna.applications.boyan_chain.specs import get_observation_action_spec_boyan_chain

from boyan_chain.environment import BoyanChain

from utils.general import SuppressPrint

# ---------------------------------------------------------------- #

class TFOffpolicyDatasetGeneratorBoyanChain:
    def __init__(self, N, kind, n_experience, seed=0, one_hot=False):
        self.n_experience = n_experience

        self.one_hot = one_hot

        self.env = BoyanChain(N, kind, seed) # type: ignore
        self.dataset = TFOffpolicyDataset(
            self.get_spec(), self.get_capacity(), )

    def get_spec(self):
        observation_spec, action_spec = get_observation_action_spec_boyan_chain(
            self.env.N, self.one_hot)
        time_step_spec = get_time_step_spec(observation_spec) # type: ignore
        step_num_spec = get_step_num_spec(step_num_max=2)

        return EnvStep(
            step_type  =time_step_spec.step_type,
            step_num   =step_num_spec,
            observation=time_step_spec.observation,
            action     =action_spec,
            reward     =time_step_spec.reward,
            discount   =time_step_spec.discount,
            policy_info={},
            env_info   ={},
            other_info ={},
        )

    def get_capacity(self):
        return self.n_experience * 3

    def add_experiences_to_dataset(self, verbosity=0):
        pbar = range(self.n_experience)
        if verbosity > 0:
            print("adding experience to dataset")
            pbar = tqdm(pbar)

        for _ in pbar:
            experience, valid_ids = self.get_experience()
            add_episodes_to_dataset(experience, valid_ids, self.dataset)

    def get_experience(self):
        step_type = self.get_step_type()
        step_num = self.get_step_num()

        observation, action, reward = self.get_observation_action_reward()

        discount = self.get_discount()

        policy_info = {}
        env_info = {}
        other_info = {}

        experience =  EnvStep(
            step_type, step_num,
            observation, action, reward,
            discount,
            policy_info, env_info, other_info)

        valid_ids = self.get_valid_ids()

        return experience, valid_ids

    def get_step_type(self):
        step_type_np = np.array([0, 1, 1])
        step_type = tf.convert_to_tensor(
            step_type_np, dtype=self.dataset.spec.step_type.dtype)

        return step_type

    def get_step_num(self):
        step_num_np = np.array([0, 1, 2])
        step_num = tf.convert_to_tensor(
            step_num_np, dtype=self.dataset.spec.step_num.dtype)

        return step_num

    def get_observation_action_reward(self):
        obs_0 = self.env.reset()
        act_0 = self.env.action_space.sample()
        _, rew_0, _, _ = self.env.step(act_0)

        obs = self.env.reset()
        act = self.env.action_space.sample()
        obs_prime, rew, _, _ = self.env.step(act)
        act_prime = self.env.action_space.sample()
        _, rew_prime, _, _ = self.env.step(act_prime)

        observation_np = np.array([obs_0, obs, obs_prime])
        action_np      = np.array([act_0, act, act_prime])
        reward_np      = np.array([rew_0, rew, rew_prime])

        if self.one_hot:
            I = np.identity(self.env.N + 1)
            observation_np = I[observation_np]

        observations_tf = tf.convert_to_tensor(
            observation_np, dtype=self.dataset.spec.observation.dtype)
        actions_tf = tf.convert_to_tensor(
            action_np, dtype=self.dataset.spec.action.dtype)
        rewards_tf = tf.convert_to_tensor(
            reward_np, dtype=self.dataset.spec.reward.dtype)

        observation = observations_tf
        action      = actions_tf
        reward      = rewards_tf

        return observation, action, reward

    def get_discount(self):
        discount_np = np.ones(3)
        discount = tf.convert_to_tensor(
            discount_np, dtype=self.dataset.spec.discount.dtype)

        return discount

    def get_valid_ids(self):
        valid_ids_int = tf.ones(3)
        valid_ids = tf.cast(valid_ids_int, dtype=tf.bool)

        return valid_ids

# ---------------------------------------------------------------- #

def load_or_create_dataset_boyan_chain(
    dataset_dir,
    N, kind, n_experience, seed, one_hot,
    verbosity=0):

    raise DeprecationWarning

    try:
        if verbosity == 1:
            print("Try loading dataset", end=" ")
            dataset = TFOffpolicyDataset.load(dataset_dir)
        else:
            with SuppressPrint():
                dataset = TFOffpolicyDataset.load(dataset_dir)

    except KeyboardInterrupt:
        if verbosity == 1: print("KeyboardInterrupt")
        assert False

    except:
        if verbosity == 1: print(); print(f"No dataset found in {dataset_dir}")
        generator = TFOffpolicyDatasetGeneratorBoyanChain(
            N=N, kind=kind, n_experience=n_experience, seed=seed, one_hot=one_hot)

        generator.add_experiences_to_dataset(verbosity=1)

        dataset = generator.dataset

        if not tf.io.gfile.isdir(dataset_dir):
            tf.io.gfile.makedirs(dataset_dir)

        dataset.save(dataset_dir)
    
    return dataset

# ---------------------------------------------------------------- #
