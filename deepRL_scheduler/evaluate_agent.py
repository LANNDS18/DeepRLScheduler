#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from stable_baselines3 import PPO

from hpc_rl_simulator.env import GymSchedulerEnv
from hpc_rl_simulator.utils import init_dir_from_args, init_evaluation_env


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
            break

    if log:
        print(f"Current Time Stamp: {current_time}")
        print(f'total performance matrix value: {record}')
    return record


with open('ppo-conf.json', 'r') as f:
    config = json.load(f)

if __name__ == '__main__':

    # init directories
    model_dir, log_data_dir, workload_file = init_dir_from_args(config)
    # create environment
    env = init_evaluation_env(workload_file, GymSchedulerEnv, config, True)

    for i in range(1):
        model = PPO.load("trained_models/bsld/HPC2N-2002-2_ppo_HPC_optimal_1.zip", env=env)
        print(schedule_curr_sequence_reset(env, model, log=True))
