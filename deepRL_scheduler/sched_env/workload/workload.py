#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from ..job import Job


class Workloads:

    def __init__(self):
        self.all_jobs = []
        self.max_allocated_proc = 0
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_job_id = 0

        self.max_requested_memory = 0
        self.max_user_id = 0
        self.max_group_id = 0
        self.max_executable_number = 0
        self.max_job_id = 0
        self.max_nodes = 0
        self.max_procs = 0
        self.end_time = 0

    @property
    def size(self):
        return len(self.all_jobs)

    def parse_swf(self, path):

        end_time = 0

        with open(path) as fp:
            for line in fp:
                if line.startswith(";"):
                    if line.startswith("; MaxNodes:"):
                        self.max_nodes = int(line.split(":")[1].strip())
                    if line.startswith("; MaxProcs:"):
                        self.max_procs = int(line.split(":")[1].strip())
                    continue

                j = Job(line)
                if j.run_time > self.max_exec_time:
                    self.max_exec_time = j.run_time
                if j.submit_time > self.end_time:
                    self.end_time = j.submit_time
                if j.run_time < self.min_exec_time:
                    self.min_exec_time = j.run_time
                if j.request_memory > self.max_requested_memory:
                    self.max_requested_memory = j.request_memory
                if j.user_id > self.max_user_id:
                    self.max_user_id = j.user_id
                if j.group_id > self.max_group_id:
                    self.max_group_id = j.group_id
                if j.executable_number > self.max_executable_number:
                    self.max_executable_number = j.executable_number
                if j.job_id > self.max_job_id:
                    self.max_job_id = j.job_id

                # filter those illegal data whose runtime < 0
                if j.run_time < 0:
                    j.run_time = 10
                if j.run_time > 0:
                    self.all_jobs.append(j)

                    if j.request_number_of_processors > self.max_allocated_proc:
                        self.max_allocated_proc = j.request_number_of_processors

        # if max_procs = 0, it means node/proc are the same.
        if self.max_procs == 0:
            self.max_procs = self.max_nodes

        print(
            f":WORKLOAD:\tMax Allocated Processors: {self.max_allocated_proc}\n"
            f":WORKLOAD:\tmax node: {self.max_nodes}\n"
            f":WORKLOAD:\tmax procs: {self.max_procs}\n"
            f":WORKLOAD:\tmax execution time: {self.max_exec_time}\n"
            f":WORKLOAD:\tnumber of jobs: {self.size}\n"
        )

        self.all_jobs.sort(key=lambda job: job.job_id)

    def reset(self):
        for job in self.all_jobs:
            job.scheduled_time = -1

    def __getitem__(self, item):
        return self.all_jobs[item]


if __name__ == "__main__":
    print("Loading the workloads...")
    load = Workloads()
    load.parse_swf('../../dataset/NASA-iPSC-1993-3.1-cln.swf')
    print("Finish loading the workloads...", type(load[0]))
    print(load.max_nodes, load.max_procs)
    print(load[0].__feature__())
    print(load[1].__feature__())
