#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from abc import ABC
from gym.utils import seeding

from ..cluster import Cluster
from ..job import Job
from ..scorer import ScheduleScorer
from ..workload import Workloads

from .env_conf import *


class HPCSchedulingSimulator(ABC):
    def __init__(self,
                 workload_file,
                 back_fill=False,
                 skip=False,
                 job_score_type=0,
                 batch_job_slice=0,
                 seed=0):

        super(HPCSchedulingSimulator, self).__init__()

        self.penalty_job_score = None

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.complete_jobs = []

        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0

        self.start_idx_last_reset = self.start

        # sub-components
        self.loads = Workloads()
        self.cluster = None

        self.scorer = ScheduleScorer(job_score_type)

        self.scheduled_rl = {}
        self.pivot_job = False
        self.enable_pre_workloads = False
        self.pre_workloads = []

        self.back_fill = back_fill
        self.skip = skip

        self.batch_job_slice = batch_job_slice

        self.np_random = None
        self.seed(seed)

        self._load_job_trace(workload_file)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _load_job_trace(self, workload_file=''):
        print(f":ENV:\tloading workloads from dataset: {workload_file}")
        self.loads.parse_swf(workload_file)
        self.cluster = Cluster(self.loads.max_nodes, self.loads.max_procs / self.loads.max_nodes)
        self.penalty_job_score = JOB_SEQUENCE_SIZE * self.loads.max_exec_time / 10

    def fill_pre_workloads(self, size):
        # Generate some running jobs to randomly fill the cluster.
        # size = self.np_random.randint(2 * job_sequence_size)
        if self.enable_pre_workloads:
            for i in range(size):
                _job = self.loads[self.start - i - 1]
                req_num_of_processors = _job.request_number_of_processors
                runtime_of_job = _job.request_time
                job_tmp = Job()

                # to be different from the normal jobs; normal jobs have a job_id >= 0
                # The id cannot be -1 cause the invalid job id will be -1

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

    def skip_for_resources_greedy(self, job):
        """

        This function will be called when cluster cannot allocate resource to skip the time for waiting

        """
        assert not self.cluster.fits(job)

        while not self.cluster.fits(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running
            # job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running: (running.scheduled_time + running.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.next_arriving_job_idx < self.last_job_in_batch \
                    and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def forward_single_step_resources_back_fill(self, job, scheduled_logs):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.fits(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running: (running.scheduled_time + running.request_time))
        free_processors = self.cluster.free_node * self.cluster.num_procs_per_node
        for running_job in self.running_jobs:
            free_processors += len(running_job.allocated_machines) * self.cluster.num_procs_per_node
            earliest_start_time = (running_job.scheduled_time + running_job.request_time)
            if free_processors >= job.request_number_of_processors:
                break

        while not self.cluster.fits(job):

            # try to backfill as many jobs as possible. Use FCFS
            self.job_queue.sort(key=lambda _j: self.scorer.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)
            for _j in job_queue_iter_copy:
                if (self.current_timestamp + _j.request_time) < earliest_start_time:
                    if self.cluster.fits(_j):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = self.current_timestamp
                        _j.allocated_machines = self.cluster.allocate(_j)
                        self.running_jobs.append(_j)
                        score = self.scorer.scheduling_matrices(_j)  # calculated reward
                        scheduled_logs[_j.job_id] = score
                        self.job_queue.remove(_j)  # remove the job from job queue

            release_time, release_machines = self.check_next_release()

            self.move_to_next_event(release_time, release_machines)

        return scheduled_logs

    def reset_env_component(self):
        # print("start", self.start)
        self.cluster.reset()
        self.loads.reset()

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []

        self.pairs = []

    def move_to_next_event(self, next_release_time, next_release_machines):
        if self.next_arriving_job_idx < self.last_job_in_batch and \
                self.loads[self.next_arriving_job_idx].submit_time <= next_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
            self.job_queue.append(self.loads[self.next_arriving_job_idx])
            self.next_arriving_job_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_release_time)
            self.cluster.release(next_release_machines)
            self.running_jobs.pop(0)  # remove the first running job

    def schedule_job(self, job):
        assert job.scheduled_time == -1  # this job should never be scheduled before.
        job.scheduled_time = self.current_timestamp
        job.allocated_machines = self.cluster.allocate(job)
        self.running_jobs.append(job)
        score = self.scorer.scheduling_matrices(job)  # calculated reward

        self.scheduled_rl[job.job_id] = score
        self.job_queue.remove(job)  # remove the job from job queue

    def check_next_release(self):
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
        next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
        next_resource_release_machines = self.running_jobs[0].allocated_machines
        return next_resource_release_time, next_resource_release_machines

    def skip_to_resource(self, job):
        assert not self.cluster.fits(job)
        while not self.cluster.fits(job):
            next_release_time, next_release_machines = self.check_next_release()
            self.move_to_next_event(next_release_time, next_release_machines)

    def skip_to_resource_backfilling(self, large_job):
        # note that this function is only called when current large job can not be scheduled.
        assert not self.cluster.fits(large_job) and self.back_fill

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
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
        """Processes the job queue and handles job scheduling.

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
            # If job queue is empty, attempt to add jobs
            while not self.job_queue:
                if not self.running_jobs:
                    next_release_time = sys.maxsize
                    next_release_machines = []
                else:
                    next_release_time, next_release_machines = self.check_next_release()

                # If the next job's submit time is less than or equal to the next resource release time
                if self.loads[self.next_arriving_job_idx].submit_time <= next_release_time:
                    # Update current timestamp and move to next job
                    self.current_timestamp = max(self.current_timestamp,
                                                 self.loads[self.next_arriving_job_idx].submit_time)
                    self.job_queue.append(self.loads[self.next_arriving_job_idx])
                    self.next_arriving_job_idx += 1
                    return False
                else:
                    # Update current timestamp and remove this job from running jobs
                    self.current_timestamp = max(self.current_timestamp, next_release_time)
                    self.cluster.release(next_release_machines)
                    self.running_jobs.pop(0)

        return True  # Return True if no jobs were added

    def noop_schedule(self):
        # schedule nothing, just move forward to next timestamp.
        # It should 1) add a new job; 2) finish a running job; 3) reach skip time;
        next_time_after_skip = self.current_timestamp + SKIP_TIME

        next_release_time = sys.maxsize  # always add jobs if no resource can be released.
        next_release_machines = []
        if self.running_jobs:  # there are running
            next_release_time, next_release_machines = self.check_next_release()

        if self.next_arriving_job_idx >= self.last_job_in_batch and not self.running_jobs:
            if not self.pivot_job:
                self.pivot_job = True
                return False
            else:
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
            self.running_jobs.pop(0)  # remove the first running job.
        return False

    def check_then_schedule(self, action):

        job = self.pairs[action].get_job()
        if not job:
            done = self.noop_schedule()
            return done

        else:
            # make sure we move forward and release needed resources
            if not self.cluster.fits(job):
                if self.back_fill:
                    self.skip_to_resource_backfilling(job)
                else:
                    self.skip_to_resource(job)

            # we should be OK to schedule the job now
            self.schedule_job(job)
            done = self.process_job_queue()
            return done

    def build_critic_observation(self):
        raise NotImplementedError

    def build_observation(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError
