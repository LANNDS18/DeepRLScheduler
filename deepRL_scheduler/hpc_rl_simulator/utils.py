#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def display_message(message: str, quiet: bool) -> None:
    """
    Function to display a message

    Parameters:
    message (str): Message to be displayed
    quiet: whether display or not

    """
    if not quiet:
        print(message)


def init_training_env(workload_path, ENV, config):
    customEnv = ENV(
        flatten_observation=True,
        workload_file=workload_path,
        skip=config['skip'],
        job_score_type=config['score_type'],
        trace_sample_range=config['trace_sample_range'],
        quiet=False
    )
    return customEnv


def init_dir_from_args(config):
    score_type_dict = {0: 'bsld', 1: 'wait_time', 2: 'turnaround_time', 3: 'resource_utilization'}
    workload_name = config['workload'].split('/')[-1].split('.')[0]
    current_dir = os.getcwd()

    workload_file = os.path.join(current_dir, config['workload'])
    log_data_dir = os.path.join(current_dir, config['log_dir'])
    model_dir = config['model_dir'] + '/' + score_type_dict[config['score_type']] + '/' + workload_name
    print(model_dir)
    return model_dir, log_data_dir, workload_file


def extract_custom_kwargs(**kwargs):
    """Extract custom kwargs from kwargs"""
    custom_kwargs = {}
    filtered_kwargs = {}
    for key in kwargs:
        if key in ['actor_model', 'critic_model', 'attn', 'obs_shape']:
            custom_kwargs[key] = kwargs[key]
        else:
            filtered_kwargs[key] = kwargs[key]
    return custom_kwargs, filtered_kwargs
