#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Obs_Job:
    def __init__(self,
                 normalized_submit_time,
                 normalized_wait_time,
                 normalized_request_time,
                 normalized_run_time,
                 normalized_request_procs,
                 normalized_request_memory,
                 normalized_user_id,
                 normalized_group_id,
                 can_schedule_now,
                 real_flag):
        self.normalized_submit_time = normalized_submit_time
        self.normalized_wait_time = normalized_wait_time
        self.normalized_request_time = normalized_request_time

        self.normalized_run_time = normalized_run_time
        self.normalized_request_procs = normalized_request_procs
        self.normalized_request_memory = normalized_request_memory
        self.normalized_user_id = normalized_user_id
        self.normalized_group_id = normalized_group_id
        self.can_schedule_now = can_schedule_now
        self.real_flag = real_flag

    @classmethod
    def from_list(cls, lst):
        return cls(*lst)

    def to_list(self):
        return [self.normalized_submit_time,
                self.normalized_wait_time,
                self.normalized_request_time,

                self.normalized_run_time,
                self.normalized_request_procs,
                self.normalized_request_memory,
                self.normalized_user_id,
                self.normalized_group_id,
                self.can_schedule_now,
                self.real_flag]


class JobTransition(Obs_Job):
    def __init__(self, job, normalized_submit_time, normalized_wait_time, normalized_request_time, normalized_run_time,
                 normalized_request_procs, normalized_request_memory, normalized_user_id, normalized_group_id,
                 can_schedule_now, real_flag):
        super().__init__(normalized_submit_time, normalized_wait_time, normalized_request_time, normalized_run_time,
                         normalized_request_procs, normalized_request_memory, normalized_user_id, normalized_group_id,
                         can_schedule_now, real_flag)
        self.job = job

    def get_job(self):
        return self.job

