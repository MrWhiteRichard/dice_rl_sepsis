# ---------------------------------------------------------------- #

import numpy as np
import pandas as pd

from tqdm import tqdm

from tf_agents.trajectories.time_step import TimeStep

# ---------------------------------------------------------------- #

def get_df(env, get_act, target_policy, path):

    try:
        print(f"trying to load df from {path}")
        df = pd.read_parquet(path)
        return df
    except:
        pass

    print("path not found, making new df")

    col_id  = []
    col_t   = []
    col_obs = []
    col_act = []
    col_rew = []
    col_probs_eval = []

    num_episodes = 100

    for id in tqdm(range(num_episodes)):

        t = 0
        obs, _ = env.reset()

        done = False
        while not done:
            act = get_act(obs)
            obs_next, rew, term, trunc, info = env.step(act)

            col_id .append(id)
            col_t  .append(t)
            col_obs.append(obs)
            col_act.append(act)
            col_rew.append(rew)

            time_step = TimeStep(
                step_type=0 if t == 0 else 1,
                reward=rew,
                discount=1,
                observation=obs,
            )
            probs_eval = target_policy._probs(time_step)
            col_probs_eval.append(probs_eval)

            t += 1
            obs = obs_next
            done = term or trunc

    df = pd.DataFrame({
        "id": col_id,
        "t": col_t,
        "obs": col_obs,
        "act": col_act,
        "rew": col_rew,
        "probs_eval": [np.array(probs_eval) for probs_eval in col_probs_eval],
    })

    df.to_parquet(path)

    return df

# ---------------------------------------------------------------- #
