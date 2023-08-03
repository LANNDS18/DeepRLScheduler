#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stable_baselines3.common.vec_env import DummyVecEnv


def init_evaluation_env(workload_path, ENV, config, flatten_state_space=True):
    customEnv = ENV(
        flatten_observation=flatten_state_space,
        trace_sample_range=[0.5, 1.0],
        workload_file=workload_path,
        job_score_type=config['score_type'],
        quiet=True,
        seed=config['seed'],
        use_fixed_job_sequence=True,
        customized_trace_len_range=(0, 4000)
    )
    return customEnv


def init_training_env(workload_path, ENV, config):
    customEnv = ENV(
        flatten_observation=True,
        workload_file=workload_path,
        job_score_type=config['score_type'],
        trace_sample_range=config['trace_sample_range'],
        quiet=False,
        seed=config['seed'],
    )
    return customEnv


def init_vec_training_env(workload_path, ENV, config, n_envs=3):
    env_list = []
    for i in range(n_envs):
        customEnv = ENV(
            flatten_observation=True,
            workload_file=workload_path,
            job_score_type=config['score_type'],
            trace_sample_range=config['trace_sample_range'],
            quiet=False if i == 0 else True,
            seed=config['seed'] + i
        )
        env_list.append(customEnv)
    envs = DummyVecEnv(
        [
            (lambda env: lambda: env)(outer_env) for outer_env in env_list
        ]
    )
    return envs
