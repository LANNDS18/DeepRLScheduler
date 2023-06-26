#!/usr/bin/env python
# -*- coding: utf-8 -*-

class Pool:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.running_job_id = -1
        self.is_free = True
        self.job_history = []

    def taken_by_job(self, job_id):
        if self.is_free:
            self.running_job_id = job_id
            self.is_free = False
            self.job_history.append(job_id)
            return True
        else:
            return False

    def release(self):
        if self.is_free:
            return -1
        else:
            self.is_free = True
            self.running_job_id = -1
            return 1

    def reset(self):
        self.is_free = True
        self.running_job_id = -1
        self.job_history = []

    def __eq__(self, other):
        return self.machine_id == other.machine_id

    def __str__(self):
        return "M[" + str(self.machine_id) + "] "
