#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import json

import numpy as np
from stable_baselines3 import PPO

from hpc_rl_simulator.common import init_dir_from_args, init_evaluation_env
from hpc_rl_simulator.env import GymSchedulerEnv


def schedule_curr_sequence_reset(_env, model, log=True):
    """schedule the sequence of jobs using heuristic algorithm."""

    obs = _env.reset()

    while True:

        pi = model.predict(obs)
        action = pi[0]
        obs, rwd, done, info = _env.step(action)

        if done:
            record = info['performance_score']
            current_time = info['current_timestamp']
            performance_log = info['performance matrix']
            break

    if log:
        print(f"Current Time Stamp: {current_time}")
        print(f'total performance matrix value: {record}')
    return record, performance_log.values()


if __name__ == '__main__':

    evaluate_trace = 'ppo'
    backfil = True

    with open(f'SDSC-ppo-conf.json', 'r') as f:
        config = json.load(f)

    # init directories
    _, log_data_dir, workload_file = init_dir_from_args(config)
    # create environment
    env = init_evaluation_env(workload_file, GymSchedulerEnv, config, backfill=backfil)
    n_round = 5
    scores = []

    for i in range(n_round):
        model = PPO.load("data/trained_models/SDSC-SP2/bsld/SDSC-SP2-1998-4_ppo_bes_new.zip", env=env)
        record, score_dict = schedule_curr_sequence_reset(env, model, log=True)
        scores.extend(list(score_dict))

    if backfil:
        file_name = f'{evaluate_trace}_backfill.csv'
    else:
        file_name = f'{evaluate_trace}.csv'

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in scores:
            writer.writerow([item])

    print(np.mean(scores))
