#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from abc import ABC

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from .cluster import Cluster
from .job import Job
from .job_scorer import JobScorer
from .workload import Workloads

MAX_QUEUE_SIZE = 128
MLP_SIZE = 256

MAX_WAIT_TIME = 12 * 60 * 60  # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60  # assume maximal runtime is 12 hours

# each job has three features: wait_time, requested_node, runtime, machine states,
JOB_FEATURES = 8
DEBUG = False

JOB_SEQUENCE_SIZE = 256
SKIP_TIME = 360  # skip 60 seconds


class HPCEnv(gym.Env, ABC):
    def __init__(self,
                 shuffle=False,
                 back_fill=False,
                 skip=False,
                 job_score_type=0,
                 batch_job_slice=0,
                 build_sjf=False,
                 seed=0):

        super(HPCEnv, self).__init__()

        print("Initialize Simple HPC Env")

        self.penalty_job_score = None

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)

        ob = spaces.Box(low=0.0, high=1.0, shape=(JOB_FEATURES * MAX_QUEUE_SIZE,), dtype=np.float32)

        self.observation_space = ob

        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.start_idx_last_reset = 0

        self.loads = None
        self.cluster = None

        self.bsld_algo_dict = {}
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []
        self.enable_pre_workloads = False
        self.pre_workloads = []

        self.shuffle = shuffle
        self.back_fill = back_fill
        self.skip = skip

        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization
        self.job_score_type = job_score_type

        self.scorer = JobScorer(job_score_type)

        self.batch_job_slice = batch_job_slice

        self.build_sjf = build_sjf
        self.sjf_scores = []

        self.np_random = None
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return self.np_random, _

    def load_job_trace(self, workload_file=''):
        print(f":ENV:\tloading workloads from dataset: {workload_file}")
        self.loads = Workloads(workload_file)
        self.cluster = Cluster(self.loads.max_nodes, self.loads.max_procs / self.loads.max_nodes)
        self.penalty_job_score = JOB_SEQUENCE_SIZE * self.loads.max_exec_time / 10

        if self.build_sjf:
            # this is for trajectory filtering.
            # calculate SJF scores for all sample sequence and save them here
            index = 0
            if self.batch_job_slice == 0:
                max_index = self.loads.size() - JOB_SEQUENCE_SIZE - 1

            else:
                max_index = min(self.batch_job_slice, self.loads.size()) - JOB_SEQUENCE_SIZE - 1

            print(f":ENV:\tmax index = {max_index} ... initializing SJF Score Array")

            while index <= max_index:
                index += 1
                if index % 100 == 0:
                    print(f":ENV:\tindex {index}")

                self.reset_env_component()

                self.current_timestamp = 0
                self.start = 0
                self.next_arriving_job_idx = 0
                self.last_job_in_batch = 0
                self.num_job_in_batch = 0
                self.scheduled_rl = {}
                self.penalty = 0
                self.pivot_job = False
                self.scheduled_scores = []

                job_sequence_size = JOB_SEQUENCE_SIZE
                self.pre_workloads = []

                self.start = index
                self.start_idx_last_reset = self.start
                self.num_job_in_batch = job_sequence_size
                self.last_job_in_batch = self.start + self.num_job_in_batch
                self.current_timestamp = self.loads[self.start].submit_time
                self.job_queue.append(self.loads[self.start])
                self.next_arriving_job_idx = self.start + 1

                self.fill_pre_workloads(job_sequence_size + self.np_random.randint(job_sequence_size))

                self.sjf_scores.append(sum(self.schedule_curr_sequence_reset(self.scorer.sjf_score).values()))

    @staticmethod
    def build_observation_space():
        # todo: represent the cluster into correct machine state

        # time_stamp = 1

        # job_state = spaces.Box(low=0.0, high=1.0, shape=(JOB_FEATURES * MAX_QUEUE_SIZE,), dtype=np.float32)

        # machine_state = spaces.Box(low=-1, high=100000, shape=(JOB_FEATURES * MAX_QUEUE_SIZE,), dtype=np.float32)
        pass

    def fill_pre_workloads(self, size):
        # Generate some running jobs to randomly fill the cluster.
        # size = self.np_random.randint(2 * job_sequence_size)
        if self.enable_pre_workloads:
            running_job_size = size
            for i in range(running_job_size):
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
                    job_tmp.scheduled_time = max(0,
                                                 (self.current_timestamp - self.np_random.randint(0, max(runtime_of_job,
                                                                                                         1))))
                    # job_tmp.scheduled_time = max(0, (self.current_timestamp - runtime_of_job/2))
                    job_tmp.allocated_machines = self.cluster.allocate(job_tmp)
                    self.pre_workloads.append(job_tmp)
                else:
                    break

    def refill_pre_workloads(self):
        if self.enable_pre_workloads:
            for _job in self.pre_workloads:
                self.running_jobs.append(_job)
                _job.allocated_machines = self.cluster.allocate(_job)

    def reset(self):
        self.reset_env_component()

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        job_sequence_size = JOB_SEQUENCE_SIZE

        self.pre_workloads = []

        assert self.batch_job_slice == 0 or self.batch_job_slice >= job_sequence_size

        if self.build_sjf:
            while True:
                if self.batch_job_slice == 0:
                    self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
                else:
                    self.start = self.np_random.randint(job_sequence_size,
                                                        (self.batch_job_slice - job_sequence_size - 1))

                if 10 < self.sjf_scores[self.start] < 150:
                    break

        else:
            if self.batch_job_slice == 0:
                self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
            else:
                self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))

        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

        self.fill_pre_workloads(job_sequence_size + self.np_random.randint(job_sequence_size))

        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.scorer.sjf_score).values()))
        self.scheduled_scores.append(sum(self.schedule_curr_sequence_reset(self.scorer.f1_score).values()))

        print(":ENV:\tScheduled Scores:", self.scheduled_scores)

        return self.build_observation(), self.build_critic_observation()

    def reset_for_test(self, num):

        self.reset_env_component()

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False
        self.scheduled_scores = []

        job_sequence_size = num
        assert self.batch_job_slice == 0 or self.batch_job_slice >= job_sequence_size
        if self.batch_job_slice == 0:
            self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
        else:
            self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))
        # self.start = start
        self.start_idx_last_reset = self.start
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.next_arriving_job_idx = self.start + 1

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
                        score = self.scorer.job_score(_j)  # calculated reward
                        scheduled_logs[_j.job_id] = score
                        self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
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
                self.running_jobs.pop(0)  # remove the first running job

        return scheduled_logs

    def reset_env_component(self):
        # print("start", self.start)
        self.cluster.reset()
        self.loads.reset()
        self.job_queue = []
        self.running_jobs = []
        self.visible_jobs = []
        self.pairs = []

    def heuristic_step(self, score_fn, scheduled_logs):

        self.job_queue.sort(key=lambda j: score_fn(j))
        job_for_scheduling = self.job_queue[0]

        if not self.cluster.fits(job_for_scheduling):
            if self.back_fill:
                scheduled_logs = self.forward_single_step_resources_back_fill(job_for_scheduling, scheduled_logs)
            else:
                self.skip_for_resources_greedy(job_for_scheduling)

        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        # print(self.current_timestamp)
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling)
        self.running_jobs.append(job_for_scheduling)
        score = self.scorer.job_score(job_for_scheduling)  # calculated reward
        scheduled_logs[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)

        not_empty = self.moveforward_for_job()

        return not_empty, scheduled_logs

    def schedule_curr_sequence_reset(self, score_fn):
        # schedule the sequence of jobs using heuristic algorithm. 
        scheduled_logs = {}
        # f = False
        # if score_fn.__name__ == "sjf_score":
        #     f = True
        #     num_total = 0
        # start_time = time.time()
        while True:
            not_empty, scheduled_logs = self.heuristic_step(score_fn, scheduled_logs)
            if not not_empty:
                break

        self.scorer.post_process_score(scheduled_logs, self.num_job_in_batch,
                                       self.current_timestamp, self.loads[self.start], self.loads.max_procs)
        # if f:
        #     print((time.time()-start_time)/num_total, num_total)
        # reset again

        # reset
        self.reset_env_component()

        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        self.refill_pre_workloads()

        return scheduled_logs

    def build_critic_observation(self):
        vector = np.zeros(JOB_SEQUENCE_SIZE * 3, dtype=float)
        earlist_job = self.loads[self.start_idx_last_reset]
        earlist_submit_time = earlist_job.submit_time
        pairs = []
        for i in range(self.start_idx_last_reset, self.last_job_in_batch + 1):
            job = self.loads[i]
            submit_time = job.submit_time - earlist_submit_time
            request_processors = job.request_number_of_processors
            request_time = job.request_time

            normalized_submit_time = min(float(submit_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
            normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
            normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1.0 - 1e-5)

            pairs.append([normalized_submit_time, normalized_run_time, normalized_request_nodes])

        for i in range(JOB_SEQUENCE_SIZE):
            vector[i * 3:(i + 1) * 3] = pairs[i]

        return vector

    def select_jobs_by_score(self, score_func):
        self.job_queue.sort(key=lambda j: score_func(j))
        return self.job_queue[:MAX_QUEUE_SIZE]

    def build_job_features(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        wait_time = self.current_timestamp - submit_time

        # make sure that larger value is better.
        normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
        normalized_run_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - 1e-5)
        normalized_request_nodes = min(float(request_processors) / float(self.loads.max_procs), 1.0 - 1e-5)

        '''
        @ddai: part 2 of OPTIMIZE_OBSV
        earliest_start_time = 1
        for fp, ts in free_processors_pair:
            if request_processors < fp:
                earliest_start_time = ts
                break
        normalized_earliest_start_time = min(float(earliest_start_time) / float(MAX_WAIT_TIME), 1.0 - 1e-5)
        '''

        # add extra parameters, include "Requested Memory", "User Id", "Groupd Id", "Exectuable Id",
        # if its value does not exist in the trace (-1), we set it to 1 by default.
        if job.request_memory == -1:
            normalized_request_memory = 1
        else:
            normalized_request_memory = min(float(job.request_memory) / float(self.loads.max_requested_memory),
                                            1.0 - 1e-5)

        if job.user_id == -1:
            normalized_user_id = 1
        else:
            normalized_user_id = min(float(job.user_id) / float(self.loads.max_user_id), 1.0 - 1e-5)

        if job.group_id == -1:
            normalized_group_id = 1
        else:
            normalized_group_id = min(float(job.group_id) / float(self.loads.max_group_id), 1.0 - 1e-5)

        if job.executable_number == -1:
            normalized_executable_id = 1
        else:
            normalized_executable_id = min(
                float(job.executable_number) / float(self.loads.max_executable_number), 1.0 - 1e-5)

        if self.cluster.fits(job):
            can_schedule_now = 1.0 - 1e-5
        else:
            can_schedule_now = 1e-5
        return [job, normalized_wait_time, normalized_run_time, normalized_request_nodes,
                normalized_request_memory, normalized_user_id, normalized_group_id,
                normalized_executable_id, can_schedule_now]

    @staticmethod
    def build_skip_features(pivot):
        if pivot:
            return [None, 1, 1, 1, 1, 1, 1, 1, 1]
        else:
            return [None, 1, 1, 1, 1, 1, 1, 1, 0]

    def build_observation(self):
        self.job_queue.sort(key=lambda j: self.scorer.fcfs_score(j))
        self.visible_jobs = self.job_queue[:MAX_QUEUE_SIZE]

        if self.shuffle:
            self.np_random.shuffle(self.visible_jobs)

        # @ddai: optimize the observable jobs
        self.visible_jobs = []
        if len(self.job_queue) <= MAX_QUEUE_SIZE:
            self.visible_jobs = self.job_queue[:len(self.job_queue)]
        else:
            visible_jobs_sets = [
                self.select_jobs_by_score(self.scorer.f1_score),
                self.select_jobs_by_score(self.scorer.f2_score),
                self.select_jobs_by_score(self.scorer.sjf_score),
                self.select_jobs_by_score(self.scorer.smallest_score),
            ]

            shuffled = list(self.job_queue)
            self.np_random.shuffle(shuffled)
            visible_jobs_sets.append(shuffled[:MAX_QUEUE_SIZE])

            print(visible_jobs_sets)

            index = 0
            while index < MAX_QUEUE_SIZE:
                for visible_jobs in visible_jobs_sets:
                    if index >= MAX_QUEUE_SIZE:
                        break
                    job = visible_jobs[index]
                    if job not in self.visible_jobs:
                        self.visible_jobs.append(job)
                        index += 1

        # Code for OPTIMIZE_OBSV omitted for brevity

        vector = np.zeros(MAX_QUEUE_SIZE * JOB_FEATURES, dtype=float)
        self.pairs = []
        for i in range(MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs):
                job = self.visible_jobs[i]
                self.pairs.append(self.build_job_features(job))
            elif self.skip and not self.pairs:
                self.pairs.append(self.build_skip_features(self.pivot_job))
            else:
                self.pairs.append(self.build_skip_features(None))

        for i in range(MAX_QUEUE_SIZE):
            vector[i * JOB_FEATURES:(i + 1) * JOB_FEATURES] = self.pairs[i][1:]

        return vector

    def move_forward_in_time(self, next_resource_release_time, next_resource_release_machines):
        if self.next_arriving_job_idx < self.last_job_in_batch and \
                self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
            self.job_queue.append(self.loads[self.next_arriving_job_idx])
            self.next_arriving_job_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
            self.cluster.release(next_resource_release_machines)
            self.running_jobs.pop(0)  # remove the first running job

    def schedule_job(self, job):
        assert job.scheduled_time == -1  # this job should never be scheduled before.
        job.scheduled_time = self.current_timestamp
        job.allocated_machines = self.cluster.allocate(job)
        self.running_jobs.append(job)
        score = self.scorer.job_score(job)  # calculated reward
        self.scheduled_rl[job.job_id] = score
        self.job_queue.remove(job)  # remove the job from job queue

    def find_next_resource_release(self):
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
        next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
        next_resource_release_machines = self.running_jobs[0].allocated_machines
        return next_resource_release_time, next_resource_release_machines

    def skip_for_resources(self, job):
        assert not self.cluster.fits(job)

        while not self.cluster.fits(job):
            next_resource_release_time, next_resource_release_machines = self.find_next_resource_release()
            self.move_forward_in_time(next_resource_release_time, next_resource_release_machines)

    def moveforward_for_resources_backfill(self, job):
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
            self.job_queue.sort(key=lambda _j: self.scorer.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)
            for _j in job_queue_iter_copy:
                if self.cluster.fits(_j) and (self.current_timestamp + _j.request_time) < earliest_start_time:
                    self.schedule_job(_j)

            next_resource_release_time, next_resource_release_machines = self.find_next_resource_release()
            self.move_forward_in_time(next_resource_release_time, next_resource_release_machines)

    def moveforward_for_job(self):
        if self.job_queue:
            return True

        # if we need to add job, but can not add anymore, return False indicating the job_queue is for sure empty now.
        if self.next_arriving_job_idx >= self.last_job_in_batch:
            assert not self.job_queue
            return False

        # move forward to add jobs into job queue.
        while not self.job_queue:
            if not self.running_jobs:  # there are no running jobs
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machines

            if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True  # job added
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def skip_schedule(self):
        # schedule nothing, just move forward to next timestamp. It should 1) add a new job; 2) finish a running job;
        # 3) reach skip time
        next_time_after_skip = self.current_timestamp + SKIP_TIME

        next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
        next_resource_release_machines = []
        if self.running_jobs:  # there are running jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machines

        if self.next_arriving_job_idx >= self.last_job_in_batch and not self.running_jobs:
            if not self.pivot_job:
                self.pivot_job = True
                return False, 0
            else:
                return False, 0

        if next_time_after_skip < min(self.loads[self.next_arriving_job_idx].submit_time, next_resource_release_time):
            self.current_timestamp = next_time_after_skip
            return False, 0

        if self.next_arriving_job_idx < self.last_job_in_batch and \
                self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
            self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
            self.job_queue.append(self.loads[self.next_arriving_job_idx])
            self.next_arriving_job_idx += 1
        else:
            self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
            self.cluster.release(next_resource_release_machines)
            self.running_jobs.pop(0)  # remove the first running job.
        return False, 0

    def schedule(self, job_for_scheduling):
        # make sure we move forward and release needed resources
        if not self.cluster.fits(job_for_scheduling):
            if self.back_fill:
                self.moveforward_for_resources_backfill(job_for_scheduling)
            else:
                self.skip_for_resources(job_for_scheduling)

        # we should be OK to schedule the job now
        assert job_for_scheduling.scheduled_time == -1  # this job should never be scheduled before.
        job_for_scheduling.scheduled_time = self.current_timestamp
        job_for_scheduling.allocated_machines = self.cluster.allocate(job_for_scheduling)
        self.running_jobs.append(job_for_scheduling)
        score = self.scorer.job_score(job_for_scheduling)  # calculated reward
        self.scheduled_rl[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)  # remove the job from job queue

        # after scheduling, check if job queue is empty, try to add jobs.
        not_empty = self.moveforward_for_job()

        if not_empty:
            # job_queue is not empty
            return False
        else:
            # job_queue is empty and can not add new jobs as we reach the end of the sequence
            return True

    def step(self, a):
        job_for_scheduling = self.pairs[a][0]
        if not job_for_scheduling:
            done, _ = self.skip_schedule()
        else:
            job_for_scheduling = self.pairs[a][0]
            done = self.schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, 0, 0, 0]
        else:
            self.scorer.post_process_score(self.scheduled_rl, self.num_job_in_batch,
                                           self.current_timestamp, self.loads[self.start], self.loads.max_procs)
            rl_total = sum(self.scheduled_rl.values())
            best_total = min(self.scheduled_scores)
            sjf = self.scheduled_scores[0]
            f1 = self.scheduled_scores[1]
            rwd2 = (best_total - rl_total)
            rwd = -rl_total
            return [None, rwd, True, rwd2, sjf, f1]

    def step_for_test(self, a):
        job_for_scheduling = self.pairs[a][0]

        if not job_for_scheduling:
            # print("SKIP", end=" ")
            done, _ = self.skip_schedule()
        else:
            job_for_scheduling = self.pairs[a][0]
            done = self.schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, None]
        else:
            self.scorer.post_process_score(self.scheduled_rl, self.num_job_in_batch,
                                           self.current_timestamp, self.loads[self.start], self.loads.max_procs)
            rl_total = sum(self.scheduled_rl.values())
            return [None, rl_total, True, None]
