#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from stable_baselines3 import PPO

from hpc_rl_simulator.env import GymSchedulerEnv
from train_ppo import init_dir_from_args


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
    env = GymSchedulerEnv(
        workload_file="./dataset/HPC2N-2002-2.2-cln.swf",
        flatten_observation=True,
        trace_sample_range=[0.95, 1.0],
        back_fill=False,
        seed=0,
        use_fixed_job_sequence=True,
        customized_trace_len_range=(0, 1000)  # (0 + 10000) /2 = 5000
    )

    for i in range(1):
        model = PPO.load("trained_models/reward_space_experiment/HPC2N-2002-2ppo_HPC_small_reward.zip", env=env)
        print(schedule_curr_sequence_reset(env, model, False))
