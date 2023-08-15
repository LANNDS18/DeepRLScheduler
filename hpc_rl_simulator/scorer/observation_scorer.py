#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from .scorer import ScheduleScorer
from ..job import Obs_Job


class Obs_Scorer(ScheduleScorer):
    def __init__(self, job_score_type):
        super(Obs_Scorer, self).__init__(job_score_type)

    @staticmethod
    def f1_score(obs_list):
        obs_job = Obs_Job.from_list(obs_list)
        submit_time = obs_job.normalized_submit_time
        request_processors = obs_job.normalized_request_procs
        request_time = obs_job.normalized_request_time
        real_flag = obs_job.real_flag
        can_schedule_now = obs_job.can_schedule_now

        return (np.log10(request_time if request_time > 0 else 0.1) * request_processors + 870 * np.log10(
            submit_time if submit_time > 0 else 0.1)) if real_flag and can_schedule_now else -1

    @staticmethod
    def f2_score(obs_list):
        obs_job = Obs_Job.from_list(obs_list)
        submit_time = obs_job.normalized_submit_time
        request_processors = obs_job.normalized_request_procs
        request_time = obs_job.normalized_request_time
        real_flag = obs_job.real_flag
        can_schedule_now = obs_job.can_schedule_now

        # f2: r^(1/2)*n + 25600 * log10(s)
        return np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time) \
            if real_flag and can_schedule_now else -1

    @staticmethod
    def sjf_score(obs_list):
        obs_job = Obs_Job.from_list(obs_list)
        request_time = obs_job.normalized_request_time
        submit_time = obs_job.normalized_submit_time
        real_flag = obs_job.real_flag
        can_schedule_now = obs_job.can_schedule_now

        # if request_time is the same, pick whichever submitted earlier
        return request_time if real_flag and can_schedule_now else -1  # , submit_time

    @staticmethod
    def smallest_score(obs_list):
        obs_job = Obs_Job.from_list(obs_list)
        request_processors = obs_job.normalized_request_procs
        submit_time = obs_job.normalized_submit_time
        real_flag = obs_job.real_flag
        can_schedule_now = obs_job.can_schedule_now

        # if request_time is the same, pick whichever submitted earlier
        return request_processors if real_flag and can_schedule_now else -1  # , submit_time

    @staticmethod
    def fcfs_score(obs_list):
        obs_job = Obs_Job.from_list(obs_list)
        submit_time = obs_job.normalized_submit_time
        real_flag = obs_job.real_flag
        can_schedule_now = obs_job.can_schedule_now
        return submit_time if real_flag and can_schedule_now else -1
