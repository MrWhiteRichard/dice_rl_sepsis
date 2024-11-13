# ---------------------------------------------------------------- #

import numpy as np

import torch

from dice_rl_TU_Vienna.estimators.tabular.analytical_solver import AnalyticalSolver

# ---------------------------------------------------------------- #

class AnalyticalSolverFrozenLake(AnalyticalSolver):
    def __init__(self, model, transitions):
        self.model = model
        self.transitions = transitions

        num_obs = 16
        n_act = 4

        super().__init__(num_obs, n_act)

    def get_act_probs(self, obs):
        distribution = self.model.policy.get_distribution( torch.tensor([[obs]]) )
        actions = torch.arange(4)
        logits = distribution.log_prob(actions).detach().numpy()
        probs = np.exp(logits)
        return probs
    
    def get_distributions(self):
        d0 = np.zeros([self.dim])
        dD = np.ones ([self.dim]) / ( self.dim )

        obs_init = 0
        i1, i2 = self.get_index(obs_init)
        d0[i1:i2] = self.get_act_probs(obs_init)

        P = np.zeros([self.dim] * 2)
        r = np.zeros([self.dim])

        for obs in range(16):
            for act in range(4):

                d = self.transitions["obs_next_dict"][(obs, act)]
                for obs_next, prob in zip(d["obs_next"], d["probs"] ):

                    i  = self.get_index(obs, act)
                    j1, j2 = self.get_index(obs_next)

                    P[i, j1:j2] = prob * self.get_act_probs(obs_next)

                r[i] = self.transitions["rew_dict"][(obs, act)]

        return d0, dD, P, r

# ---------------------------------------------------------------- #
