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


def init_dir_from_args(config):
    score_type_dict = {0: 'bsld', 1: 'wait_time', 2: 'turnaround_time', 3: 'resource_utilization'}
    workload_name = config['workload'].split('/')[-1].split('.')[0]
    current_dir = os.getcwd()

    workload_file = os.path.join(current_dir, config['workload'])
    log_data_dir = os.path.join(current_dir, config['log_dir'])
    model_dir = config['model_dir'] + '/' + score_type_dict[config['score_type']] + '/' + workload_name
    return model_dir, log_data_dir, workload_file


def extract_custom_kwargs(**kwargs):
    """Extract custom kwargs from kwargs"""
    custom_kwargs = {}
    filtered_kwargs = {}
    for key in kwargs:
        if key in ['actor_model', 'critic_model', 'obs_shape']:
            custom_kwargs[key] = kwargs[key]
        else:
            filtered_kwargs[key] = kwargs[key]
    return custom_kwargs, filtered_kwargs


def lr_linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
