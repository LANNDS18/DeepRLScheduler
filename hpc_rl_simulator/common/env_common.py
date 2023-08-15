#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stable_baselines3.common.vec_env import DummyVecEnv


def init_evaluation_env(workload_path, ENV, config, flatten_state_space=True, backfill=False):
    customEnv = ENV(
        flatten_observation=flatten_state_space,
        trace_sample_range=[0.6, 1.0],
        workload_file=workload_path,
        job_score_type=config['score_type'],
        quiet=True,
        back_fill=backfill,
        seed=config['seed'],
        use_fixed_job_sequence=True,
        customized_trace_len_range=(0, 20000),
        reward=config['reward'],
        k=config['reward_k'],
    )
    return customEnv


def init_validation_env(workload_path, ENV, config, flatten_state_space=True):
    customEnv = ENV(
        flatten_observation=flatten_state_space,
        trace_sample_range=[0.4, 0.6],
        workload_file=workload_path,
        job_score_type=config['score_type'],
        quiet=True,
        seed=config['seed'],
        back_fill=config['backfil'],
        use_fixed_job_sequence=True,
        customized_trace_len_range=(0, 10000),
        reward=config['reward'],
        k=config['reward_k'],
    )
    return customEnv


def init_training_env(workload_path, ENV, config):
    customEnv = ENV(
        flatten_observation=True,
        workload_file=workload_path,
        job_score_type=config['score_type'],
        trace_sample_range=config['trace_sample_range'],
        quiet=False,
        back_fill=config['backfil'],
        seed=config['seed'],
        reward=config['reward'],
        k=config['reward_k'],
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
            seed=config['seed'] + i,
            back_fill=config['backfil'],
            reward=config['reward'],
            k=config['reward_k'],
        )
        env_list.append(customEnv)
    envs = DummyVecEnv(
        [
            (lambda env: lambda: env)(outer_env) for outer_env in env_list
        ]
    )
    return envs
