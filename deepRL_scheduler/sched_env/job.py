#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re


class Job:
    def __init__(self, line="0        0      0    0   0     0    0   0  0 0  0   0   0  0  0 0 0 0"):
        line = line.strip()
        s_array = re.split("\\s+", line)
        self.job_id = int(s_array[0])
        self.submit_time = int(s_array[1])
        self.wait_time = int(s_array[2])
        self.run_time = int(s_array[3])
        self.number_of_allocated_processors = int(s_array[4])
        self.average_cpu_time_used = float(s_array[5])
        self.used_memory = int(s_array[6])

        # "requested number of processors" and "number of allocated processors" are typically mixed.
        # I do not know their difference clearly. But it seems to me using a larger one will be sufficient.
        self.request_number_of_processors = int(s_array[7])
        self.number_of_allocated_processors = max(self.number_of_allocated_processors,
                                                  self.request_number_of_processors)
        self.request_number_of_processors = self.number_of_allocated_processors

        self.request_number_of_nodes = -1

        # if we use the job's request time field
        # for model, request_time might be empty. In this case, we set request_time to the run_time
        self.request_time = int(s_array[8])
        if self.request_time == -1:
            self.request_time = self.run_time

        # if we use the run time as the most accurate request time
        # self.request_time = self.run_time + 60
        # if we gradually increase the accuracy of job's request time
        # with a percentage wrong estimation and round to a fixed time: 1,2,3,... hours.
        # this.requestTime = (int) (this.runTime + this.runTime * 0.4);
        # int roundsTo = 60 * 60; //round up to hours
        # this.requestTime = (this.requestTime / roundsTo + 1) * roundsTo;

        self.request_memory = int(s_array[9])
        self.status = int(s_array[10])
        self.user_id = int(s_array[11])
        self.group_id = int(s_array[12])
        self.executable_number = int(s_array[13])
        self.queue_number = int(s_array[14])

        try:
            self.partition_number = int(s_array[15])
        except ValueError:
            self.partition_number = 0

        self.proceeding_job_number = int(s_array[16])
        self.think_time_from_proceeding_job = int(s_array[17])

        self.random_id = self.submit_time

        self.scheduled_time = -1

        self.allocated_machines = None

        self.slurm_in_queue_time = 0
        self.slurm_age = 0
        self.slurm_job_size = 0.0
        self.slurm_fair = 0.0
        self.slurm_partition = 0
        self.slurm_qos = 0
        self.slurm_tres_cpu = 0.0

    def __eq__(self, other):
        return self.job_id == other.job_id

    def __lt__(self, other):
        return self.job_id < other.job_id

    def __str__(self):
        return (
            f'Job<{self.job_id}, {self.status}, start={self.submit_time}, '
            f'processors={self.request_number_of_processors}, '
            f'memory={self.request_memory} '
            f'duration={self.run_time}>'
        )

    def __feature__(self):
        return [self.submit_time, self.request_number_of_processors, self.request_time,
                self.user_id, self.group_id, self.executable_number, self.queue_number]
