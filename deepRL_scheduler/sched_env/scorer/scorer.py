#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class ScheduleScorer:
    def __init__(self, job_score_type):
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization
        self.job_score_type = job_score_type

    def scheduling_matrices(self, job):
        score_calculations = {
            0: self._average_bounded_slowdown,
            1: self._average_waiting_time,
            2: self._average_turnaround_time,
            3: self._resource_utilization,
            4: self._average_slowdown,
        }

        try:
            calculation = score_calculations[self.job_score_type]
        except KeyError:
            raise ValueError("Invalid job_score_type")

        return calculation(job)

    def post_process_matrices(self, scheduled_logs, num_job_in_batch, current_timestamp, start_job, max_procs):

        for i in scheduled_logs:
            if self.job_score_type in [0, 1, 2, 4]:
                scheduled_logs[i] /= num_job_in_batch
            elif self.job_score_type == 3:
                scheduled_logs[i] /= (current_timestamp - start_job.submit_time) * max_procs
            else:
                raise NotImplementedError("Invalid job_score_type")

        return scheduled_logs

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
    def _average_bounded_slowdown(job):
        runtime = max(job.run_time, 10)
        return max(1.0, (job.scheduled_time - job.submit_time + job.run_time) / runtime)

    @staticmethod
    def _average_waiting_time(job):
        return job.scheduled_time - job.submit_time

    @staticmethod
    def _average_turnaround_time(job):
        return job.scheduled_time - job.submit_time + job.run_time

    @staticmethod
    def _resource_utilization(job):
        return -job.run_time * job.request_number_of_processors

    @staticmethod
    def _average_slowdown(job):
        return (job.scheduled_time - job.submit_time + job.run_time) / job.run_time
