#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from abc import ABC
from typing import Optional

import numpy as np
from gym.utils import seeding

from ..cluster import Cluster
from ..job import Job
from ..scorer import ScheduleScorer
from ..workload import Workloads
from ..common import display_message

from .env_conf import *


class HPCSchedulingSimulator(ABC):
    """
     HPCSchedulingSimulator is an abstract base class that provides the core functionality
     to simulate a HPC scheduling environment. It maintains the state of the
     environment and provides methods to manipulate the environment.

     It relies on other helper classes such as Workloads, Cluster,
     and ScheduleScorer to model the complete HPC environment.

     Each instance of this class represents a unique HPC environment, and it is parameterized by several settings
     such as the workload file to use, whether to allow backfilling of jobs, the scoring method for jobs, and a
     seed for random number generation.
     """

    def __init__(self,
                 workload_file,
                 back_fill=False,
                 job_score_type=0,
                 trace_sample_range=None,
                 seed=0,
                 quiet=False):

        super(HPCSchedulingSimulator, self).__init__()

        self.back_fill = back_fill
        self.trace_sample_range = np.array(trace_sample_range) if trace_sample_range else None
        self.np_random = None
        self.seed(seed)
        self.quiet = quiet

        self.penalty_job_score = None

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.complete_jobs = []

        self.obs_transitions = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0

        self.start_idx_last_reset = self.start

        self.loads = Workloads()
        self.cluster = None
        self.scorer = ScheduleScorer(job_score_type)

        self.scheduled_rl = {}
        self.enable_pre_workloads = False
        self.pre_workloads = []

        self._load_job_trace(workload_file)

        self.scheduled_a_job = False
        self.n_reset_simulator = 0

    def seed(self, seed=None):
        """
        Sets the seed for this environment's random number generator.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _load_job_trace(self, workload_file: str = ''):
        """
        Loads the job trace from the given file. Initialises the cluster and sets the penalty job score.
        """
        display_message(f":ENV:\tloading workloads from dataset: {workload_file}", self.quiet)
        self.loads.parse_swf(workload_file)
        self.cluster = Cluster(self.loads.max_nodes, self.loads.max_procs / self.loads.max_nodes)
        display_message(
            f":WORKLOAD:\tMax Allocated Processors: {self.loads.max_allocated_proc}\n"
            f":WORKLOAD:\tmax node: {self.loads.max_nodes}\n"
            f":WORKLOAD:\tmax procs: {self.loads.max_procs}\n"
            f":WORKLOAD:\tmax execution time: {self.loads.max_exec_time}\n"
            f":WORKLOAD:\tnumber of jobs: {self.loads.size}\n",
            self.quiet
        )

    def fill_pre_workloads(self, size: int):
        """
        Generates a set of pre-existing running jobs to fill the cluster based on a provided size.

        This function is useful when you want to simulate a scenario where the cluster
        has pre-existing running jobs upon initialization.

        Parameters:
        size: int
            The number of pre-existing jobs you want to generate.
        """

        if self.enable_pre_workloads:
            for i in range(size):
                _job = self.loads[self.start - i - 1]
                req_num_of_processors = _job.request_number_of_processors
                runtime_of_job = _job.request_time
                job_tmp = Job()

                job_tmp.job_id = (-2 - i)
                job_tmp.request_number_of_processors = req_num_of_processors
                job_tmp.run_time = runtime_of_job
                if self.cluster.fits(job_tmp):
                    self.running_jobs.append(job_tmp)
                    job_tmp.scheduled_time = max(0, (self.current_timestamp -
                                                     self.np_random.randint(0, max(runtime_of_job, 1))))
                    job_tmp.allocated_machines = self.cluster.allocate(job_tmp)
                    self.pre_workloads.append(job_tmp)
                else:
                    break

    def refill_pre_workloads(self):
        if self.enable_pre_workloads:
            for _job in self.pre_workloads:
                self.running_jobs.append(_job)
                _job.allocated_machines = self.cluster.allocate(_job)

    def move_to_next_event(self, next_release_time: float, next_release_machines: list):
        """
            Advances the simulator to the next event in time. The function compares the next job submission time and the
            next job completion time (next_release_time), and chooses the event that happens earlier in time. If a job
            is submitted, it is appended to the job queue; if a job is completed, it is removed from the running_jobs
            list and the associated resources are released from the cluster.

            Parameters:
            next_release_time: int
                The timestamp of the next job completion.

            next_release_machines: list
                The list of machines that will be released upon the completion of the next job.

            Returns:
            None
        """
        if self.next_arriving_job_idx < self.last_job_in_batch and \
                self.loads[self.next_arriving_job_idx].submit_time <= next_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
            self.job_queue.append(self.loads[self.next_arriving_job_idx])
            self.next_arriving_job_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_release_time)
            self.cluster.release(next_release_machines)
            self.running_jobs.pop(0)

    def schedule_job(self, job: Job):
        """
        Used to schedule a job. Assigns a current timestamp as the scheduled time for the job, allocates machines to the
        job from the cluster, and adds the job to the running jobs queue.
        It also calculates a score using the 'scheduling_matrices' method from the scorer, and maps the job ID to the
        calculated score in the 'scheduled_rl' dictionary. Finally, it removes the job from the job queue.

        Parameters:
        job (object): An Job object.
        """
        assert job.scheduled_time == -1
        job.scheduled_time = self.current_timestamp
        job.allocated_machines = self.cluster.allocate(job)
        self.running_jobs.append(job)
        score = self.scorer.scheduling_matrices(job)

        self.scheduled_rl[job.job_id] = score
        self.job_queue.remove(job)

        self.scheduled_a_job = True

    def check_next_release(self) -> tuple[float, list]:
        """
        This function sorts the running jobs based on the total time of the job (scheduled_time + run_time). It then
        returns the total time for the earliest finishing job and the machines allocated to that job.

        Parameters:
        None

        Returns:
        Tuple (next_resource_release_time, next_resource_release_machines):
            next_resource_release_time (int): The time at which the first job in the sorted 'running_jobs' queue will finish.
            next_resource_release_machines (list): The list of machines that are allocated to the first job in the
                                                   sorted 'running_jobs' queue.
        """
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
        next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
        next_resource_release_machines = self.running_jobs[0].allocated_machines
        return next_resource_release_time, next_resource_release_machines

    def skip_to_resource(self, job: Job):
        """
        This function attempts to allocate resources for the specified job in a "greedy" fashion.

        Parameters:
        job (Job): A Job object representing the job for which resources are to be allocated.
        """
        assert not self.cluster.fits(job)
        while not self.cluster.fits(job):
            next_release_time, next_release_machines = self.check_next_release()
            self.move_to_next_event(next_release_time, next_release_machines)

    def skip_to_resource_backfilling(self, large_job: Job):
        """
        This function aims to backfill smaller jobs until there are enough resources to schedule the large job.
        It does this by sorting the backfilling jobs according to the FCFS score.

        Parameters:
        large_job: Job
            The large job that needs to be scheduled.
        """

        assert not self.cluster.fits(large_job) and self.back_fill

        earliest_start_time = self.current_timestamp

        self.running_jobs.sort(key=lambda running: (running.scheduled_time + running.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= large_job.request_number_of_processors:
                break

        while not self.cluster.fits(large_job):
            self.job_queue.sort(key=lambda _j: self.scorer.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)
            for _j in job_queue_iter_copy:
                if self.cluster.fits(_j) and (self.current_timestamp + _j.request_time) < earliest_start_time:
                    self.schedule_job(_j)

            next_release_time, next_release_machines = self.check_next_release()
            self.move_to_next_event(next_release_time, next_release_machines)

    def process_job_queue(self):
        """
        Processes the job queue and handles job scheduling.

        Returns:
            bool:
            True if queue is empty and no jobs can be added.
            False if jobs exist in the queue or if a job was added.
        """
        if self.job_queue:
            return False
        elif self.next_arriving_job_idx >= self.last_job_in_batch:
            return True
        else:
            while not self.job_queue:
                if not self.running_jobs:
                    next_release_time = sys.maxsize
                    next_release_machines = []
                else:
                    next_release_time, next_release_machines = self.check_next_release()

                if self.loads[self.next_arriving_job_idx].submit_time <= next_release_time:
                    self.current_timestamp = max(self.current_timestamp,
                                                 self.loads[self.next_arriving_job_idx].submit_time)
                    self.job_queue.append(self.loads[self.next_arriving_job_idx])
                    self.next_arriving_job_idx += 1
                    return False
                else:
                    self.current_timestamp = max(self.current_timestamp, next_release_time)
                    self.cluster.release(next_release_machines)
                    self.running_jobs.pop(0)

        return True

    def noop_schedule(self):
        """
        The method provides a no-operation schedule.

        Instead of scheduling jobs, it simply moves forward to the next timestamp.

        The movement of the time can happen due to three reasons: 1) A new job is added, 2) A running job is finished,
        3) A skip time is reached.

        If the time after the skip is earlier than the submission of the next arriving job or the release of resources
        by a running job, then the current timestamp is updated to this skip time. If there are still jobs to arrive and
        the next arriving job's submission time is earlier than the next release of resources, the method updates the
        current timestamp to the job's submission time and adds it to the job queue.

        If the next release of resources happens earlier, the method updates the current timestamp to the release time,
        releases the resources and removes the running job.

        Returns:
        bool:
            The function returns False indicating that no job has been scheduled and not done.
        """
        next_time_after_skip = self.current_timestamp + SKIP_TIME

        next_release_time = sys.maxsize
        next_release_machines = []
        if self.running_jobs:
            next_release_time, next_release_machines = self.check_next_release()

        if self.next_arriving_job_idx >= self.last_job_in_batch:
            if not self.running_jobs:
                return True
            else:
                self.current_timestamp = next_time_after_skip
                self.current_timestamp = max(self.current_timestamp, next_release_time)
                self.cluster.release(next_release_machines)
                self.running_jobs.pop(0)
                return False

        if next_time_after_skip < min(self.loads[self.next_arriving_job_idx].submit_time, next_release_time):
            self.current_timestamp = next_time_after_skip
            return False

        if self.next_arriving_job_idx < self.last_job_in_batch and \
                self.loads[self.next_arriving_job_idx].submit_time <= next_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
            self.job_queue.append(self.loads[self.next_arriving_job_idx])
            self.next_arriving_job_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_release_time)
            self.cluster.release(next_release_machines)
            self.running_jobs.pop(0)
        return False

    def check_then_schedule(self, job: Optional[Job]) -> bool:
        """
        Check the legality and internal transition of provided action then call corresponding functions.

        Parameters:
        Job: job or None
            The job to be scheduled.

        Returns:
            bool:
            True if the simulation of current data complete.
            False if the simulation has not completed.
        """

        if not job:
            done = self.noop_schedule()
            return done

        else:
            if not self.cluster.fits(job):
                if self.back_fill:
                    self.skip_to_resource_backfilling(job)
                else:
                    self.skip_to_resource(job)

            self.schedule_job(job)
            done = self.process_job_queue()
            return done

    def reset_simulator(self, use_fixed_job_sequence=False, customized_trace_len_range=None, reset_num=50):
        """
        Resets the simulation environment by resetting the cluster and the loads, and initializing various
        instance variables to their starting values. It then optionally fills some pre-workloads and returns the initial
        observation.
        """

        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []

        self.obs_transitions = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.pre_workloads = []

        min_index, max_index = customized_trace_len_range if customized_trace_len_range is not None \
            else (MIN_JOB_SEQUENCE_SIZE, MAX_JOB_SEQUENCE_SIZE)

        if not use_fixed_job_sequence:
            job_sequence_size = self.np_random.randint(min_index, max_index)
            assert self.trace_sample_range is None or \
                   np.all(np.logical_and(self.trace_sample_range >= 0, self.trace_sample_range <= 1))

            if self.trace_sample_range is None:
                self.start = self.np_random.randint(0, (self.loads.size - job_sequence_size - 1))
            else:
                start, end = (self.loads.size * self.trace_sample_range).astype(int)
                assert end - start >= job_sequence_size + 1
                self.start = self.np_random.randint(start, (end - job_sequence_size - 1))
        else:
            job_sequence_size = int((min_index + max_index) / 2)
            if self.trace_sample_range is None:
                self.start = min([job_sequence_size * self.n_reset_simulator, self.loads.size - job_sequence_size])
            else:
                start, end = (self.loads.size * self.trace_sample_range).astype(int)

                start_ratio = (end - job_sequence_size - 1 - start) / reset_num

                move_start = int(min([start + start_ratio * self.n_reset_simulator, end - job_sequence_size - 1]))

                move_end = move_start + job_sequence_size + 1

                if move_end <= end and move_end - move_start >= job_sequence_size + 1:
                    start = move_start
                else:
                    start = end - job_sequence_size - 1
                assert end - start >= job_sequence_size + 1

                self.start = start

        self.start_idx_last_reset = self.start
        self.next_arriving_job_idx = self.start + 1
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.n_reset_simulator += 1

        self.fill_pre_workloads(job_sequence_size + self.np_random.randint(job_sequence_size))
