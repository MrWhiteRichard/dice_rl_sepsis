# ---------------------------------------------------------------- #

import numpy as np
import tensorflow as tf

from plugins.boyan_chain.environment import get_env
from plugins.boyan_chain.policy import TFPolicyBoyanChain

from dice_rl_TU_Vienna.estimators.tabular.analytical_solver import AnalyticalSolver

# ---------------------------------------------------------------- #

class AnalyticalSolverBoyanChain(AnalyticalSolver):
    def __init__(self,
        N: int,
        p: float,
        kind: str = "episodic",
        enumerate_by: str = "act"):

        self.N = N
        self.p = p
        self.kind = kind
        self.enumerate_by = enumerate_by

        self.policy = TFPolicyBoyanChain(N=N, p=p, tabular_continuous="tabular")
        self.env = get_env(seed=None, N=N, kind=kind)

        num_obs = self.N + 1
        n_act = 2

        super().__init__(num_obs, n_act)

    def assert_gamma(self, gamma):
        if self.kind == "episodic":   assert 0 < gamma < 1
        if self.kind == "continuing": assert gamma == 1

    def get_index(self, obs, act):
        if self.enumerate_by == "act": return obs * self.n_act + act
        if self.enumerate_by == "obs": return obs + self.num_obs * act

    def get_distributions(self):
        d0 = self.get_d0()
        dD = self.get_dD()
        P  = self.get_P()
        r  = self.get_r()

        return d0, dD, P, r

    def get_act_probs(self, obs):
        logits = self.policy._logits(obs)
        probs = tf.nn.softmax(logits)

        return probs

    def get_obs_next_probs(self, obs, act):

        obs_next = self.env.unwrapped.get_obs_next(obs, act)

        probs = np.identity(self.num_obs)[obs_next]

        if self.kind == "continuing" and obs_next == 0:
            probs = np.ones(self.num_obs) / self.num_obs

        return probs

    def get_d0(self):
        probs_obs_init = np.ones([self.num_obs, self.n_act]) / self.num_obs

        obs_init = np.arange(0, self.num_obs)
        probs_act_init = self.get_act_probs(obs_init)

        d0 = np.array(probs_obs_init * probs_act_init)

        if self.enumerate_by == "act": d0 = tf.reshape(d0,   [-1])
        if self.enumerate_by == "obs": d0 = tf.reshape(d0.T, [-1])

        self.d0 = d0
        return d0

    def get_dD(self):
        dD = np.ones(self.dim) / self.dim

        self.dD = dD
        return dD

    def get_P(self):
        P = np.zeros([self.dim, self.dim])

        for obs in range(self.num_obs):
            for act in range(self.n_act):
                i = self.get_index(obs, act)

                for obs_next, prob_obs_next in zip( range(self.num_obs), self.get_obs_next_probs(obs, act) ):
                    for act_next, prob_act_next in zip( range(self.n_act), self.get_act_probs(obs_next) ):
                        j = self.get_index(obs_next, act_next)

                        P[i, j] = prob_obs_next * prob_act_next

        if self.kind == "continuing":
            i0 = self.get_index(0, 0)
            i_1 = self.get_index(0, 1)
            P[i0] = self.d0
            P[i_1] = self.d0

        self.P = P
        return P

    def get_r(self):
        r = np.zeros(self.dim)

        i = self.get_index(0, 0)
        r[i] = 0
        i = self.get_index(0, 1)
        r[1] = 0

        if self.N >= 1:
            i = self.get_index(1, 0)
            r[i] = -2
            i = self.get_index(1, 1)
            r[i] = -2

        for obs in range(2, self.num_obs):
            for act in range(self.n_act):
                i = self.get_index(obs, act)
                r[i] = -3

        self.r = r
        return r

# ---------------------------------------------------------------- #

def test(N, p, gamma=0.99):
    assert 0 < gamma < 1

    evaluation_policy = TFPolicyBoyanChain(N, p) # type: ignore
    analytical_solver = BoyanChainAnalyticalSolver(evaluation_policy, "episodic", gamma) # type: ignore

    sdc, _ = analytical_solver.solve(gamma)
    sd_d = analytical_solver.dD
    sd_p = sdc * sd_d

    if N == 0:
        sd_states = np.array([1])

    elif N == 1:
        sd_states = (1 - gamma) / 2 * np.array([
            1 \
                + gamma * 2 / (1 - gamma),
            1,
        ])

    elif N == 2:
        sd_states = (1 - gamma) / 3 * np.array([
            1 \
                + gamma    * (3 - p) \
                + gamma**2 * 3 / (1 - gamma),
            1 + \
                p * gamma,
            1,
        ])

    elif N == 3:
        sd_states = (1 - gamma) / 4 * np.array([
            1 \
                + gamma    * (3 - p) \
                + gamma**2 * (4 - p**2) \
                + gamma**3 * 4 / (1 - gamma),
            1 + \
                gamma + p**2 * gamma**2,
            1 + \
                gamma * p,
            1,
        ])

    else:
        raise NotImplementedError

    A = 2
    sd_state_action = np.reshape(
        np.reshape( np.tile(sd_states, A), [A, len(sd_states)] ).T \
            * np.array([p, 1 - p]),
        -1
    )

    a = sd_p
    b = sd_state_action

    n = min( max( len(str(a)), len(str(b)) ), 64 )
    print("-" * n)
    print("solver:")
    print(a)
    print("-" * n)
    print("manually:")
    print(b)
    print("-" * n)

# ---------------------------------------------------------------- #
