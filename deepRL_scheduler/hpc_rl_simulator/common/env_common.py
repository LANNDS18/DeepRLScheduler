#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        quiet=False
    )
    return customEnv
