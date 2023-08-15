#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import csv

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
            performance_log = info['performance matrix']
            break
    if log:
        print(f"Current Time Stamp: {current_time}")
        print(f'total performance matrix value: {record}')
    return record, performance_log.values()


def evaluate_score_fn(workload, score_fn, title, n_round=5, backfil=False):
    with open('ppo_configs/ppo-conf.json', 'r') as f:
        config = json.load(f)

    env = init_evaluation_env(workload, GymSchedulerEnv, config, flatten_state_space=False, backfill=backfil)

    rewards = []
    for i in range(n_round):
        record, score_dict = schedule_curr_sequence_reset(env, score_fn, log=True)
        rewards.extend(list(score_dict))

    if backfil:
        file_name = f'{title}_backfill.csv'
    else:
        file_name = f'{title}.csv'

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in rewards:
            writer.writerow([item])

    print(np.mean(rewards))


if __name__ == '__main__':
    backfil = False
    evaluate_score_fn("./dataset/SDSC-SP2-1998-4.2-cln.swf", Obs_Scorer.sjf_score, title='sjf', backfil=backfil)
    evaluate_score_fn("./dataset/SDSC-SP2-1998-4.2-cln.swf", Obs_Scorer.fcfs_score, title='fcfs', backfil=backfil)
    evaluate_score_fn("./dataset/SDSC-SP2-1998-4.2-cln.swf", Obs_Scorer.smallest_score, title='smallest', backfil=backfil)
    evaluate_score_fn("./dataset/SDSC-SP2-1998-4.2-cln.swf", Obs_Scorer.f1_score, title='f1', backfil=backfil)
