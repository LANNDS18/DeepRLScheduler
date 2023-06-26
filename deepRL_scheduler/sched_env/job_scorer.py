#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class JobScorer:
    def __init__(self, job_score_type):
        self.job_score_type = job_score_type

    def job_score(self, job):
        score_calculations = {
            0: self.average_bounded_slowdown,
            1: self.average_waiting_time,
            2: self.average_turnaround_time,
            3: self.resource_utilization,
            4: self.average_slowdown,
        }

        try:
            calculation = score_calculations[self.job_score_type]
        except KeyError:
            raise ValueError("Invalid job_score_type")

        return calculation(job)

    def post_process_score(self, scheduled_logs, num_job_in_batch, current_timestamp, start_job, max_procs):
        total_cpu_hour = 0
        if self.job_score_type == 3:
            total_cpu_hour = (current_timestamp - start_job.submit_time) * max_procs

        for i in scheduled_logs:
            if self.job_score_type in [0, 1, 2, 4]:
                scheduled_logs[i] /= num_job_in_batch
            elif self.job_score_type == 3:
                scheduled_logs[i] /= total_cpu_hour
            else:
                raise NotImplementedError("Invalid job_score_type")

    @staticmethod
    def f1_score(job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        return (np.log10(request_time if request_time > 0 else 0.1) * request_processors + 870 * np.log10(
            submit_time if submit_time > 0 else 0.1))

    @staticmethod
    def f2_score(job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f2: r^(1/2)*n + 25600 * log10(s)
        return np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time)

    @staticmethod
    def sjf_score(job):
        # run_time = job.run_time
        request_time = job.request_time
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier
        return request_time, submit_time

    @staticmethod
    def smallest_score(job):
        request_processors = job.request_number_of_processors
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier
        return request_processors, submit_time

    @staticmethod
    def fcfs_score(job):
        submit_time = job.submit_time
        return submit_time

    @staticmethod
    def average_bounded_slowdown(job):
        runtime = max(job.run_time, 10)
        return max(1.0, (job.scheduled_time - job.submit_time + job.run_time) / runtime)

    @staticmethod
    def average_waiting_time(job):
        return job.scheduled_time - job.submit_time

    @staticmethod
    def average_turnaround_time(job):
        return job.scheduled_time - job.submit_time + job.run_time

    @staticmethod
    def resource_utilization(job):
        return -job.run_time * job.request_number_of_processors

    @staticmethod
    def average_slowdown(job):
        return (job.scheduled_time - job.submit_time + job.run_time) / job.run_time
