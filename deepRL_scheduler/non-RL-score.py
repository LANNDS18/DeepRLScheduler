#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

import numpy as np

from hpc_rl_simulator.env import GymSchedulerEnv
from hpc_rl_simulator.scorer import Obs_Scorer
from hpc_rl_simulator.common import init_evaluation_env


def schedule_curr_sequence_reset(_env, score_fn, log=True):
    """schedule the sequence of jobs using heuristic algorithm."""

    job_queue_obs = _env.reset()[0]

    while True:

        action, min_value = min(enumerate(job_queue_obs), key=lambda pair: score_fn(pair[1]))
        obs, rwd, done, info = _env.step(action)
        job_queue_obs = obs[0]

        if done:
            record = info['performance_score']
            current_time = info['current_timestamp']
            break
    if log:
        print(f"Current Time Stamp: {current_time}")
        print(f'total performance matrix value: {record}')
    return record


def evaluate_score_fn(workload, score_fn, n_round=5):
    with open('ppo-conf.json', 'r') as f:
        config = json.load(f)

    env = init_evaluation_env(workload, GymSchedulerEnv, config, flatten_state_space=False)

    rewards = []
    for i in range(n_round):
        rewards.append(schedule_curr_sequence_reset(env, score_fn=score_fn, log=True))
    print(np.mean(rewards))


if __name__ == '__main__':
    evaluate_score_fn("./dataset/HPC2N-2002-2.2-cln.swf", Obs_Scorer.sjf_score)
    evaluate_score_fn("./dataset/HPC2N-2002-2.2-cln.swf", Obs_Scorer.fcfs_score)
    evaluate_score_fn("./dataset/HPC2N-2002-2.2-cln.swf", Obs_Scorer.smallest_score)
    evaluate_score_fn("./dataset/HPC2N-2002-2.2-cln.swf", Obs_Scorer.f1_score)
