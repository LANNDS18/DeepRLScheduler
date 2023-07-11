#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gym

import numpy as np

from gym import spaces
from abc import ABC

from .scheduler_simulator import HPCSchedulingSimulator

MAX_QUEUE_SIZE = 128
MLP_SIZE = 256

MAX_WAIT_TIME = 12 * 60 * 60  # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60  # assume maximal runtime is 12 hours

# each job has three features: wait_time, requested_node, runtime, machine states,
JOB_FEATURES = 9
DEBUG = False

JOB_SEQUENCE_SIZE = 256
SKIP_TIME = 360  # skip 60 seconds


class GymSchedulerEnv(HPCSchedulingSimulator, gym.Env, ABC):
    def __init__(self,
                 shuffle=False,
                 back_fill=False,
                 skip=False,
                 job_score_type=0,
                 batch_job_slice=0,
                 seed=0):

        HPCSchedulingSimulator.__init__(self,
                                        shuffle=shuffle,
                                        back_fill=back_fill,
                                        skip=skip,
                                        job_score_type=job_score_type,
                                        batch_job_slice=batch_job_slice,
                                        seed=seed
                                        )

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)

        ob = spaces.Box(low=0.0, high=1.0, shape=(JOB_FEATURES * MAX_QUEUE_SIZE,), dtype=np.float32)

        self.observation_space = ob

    @staticmethod
    def build_observation_space():
        # todo: represent the cluster into correct machine state

        # time_stamp = 1

        # job_state = spaces.Box(low=0.0, high=1.0, shape=(JOB_FEATURES * MAX_QUEUE_SIZE,), dtype=np.float32)

        # machine_state = spaces.Box(low=-1, high=100000, shape=(JOB_FEATURES * MAX_QUEUE_SIZE,), dtype=np.float32)
        pass

    def seed(self, seed=None):
        return super().seed(seed)

    def reset(self, **kwargs):
        self.reset_env_component()

        self.current_timestamp = 0
        self.start = 0
        self.next_arriving_job_idx = 0
        self.last_job_in_batch = 0
        self.num_job_in_batch = 0
        self.scheduled_rl = {}
        self.penalty = 0
        self.pivot_job = False

        job_sequence_size = JOB_SEQUENCE_SIZE

        self.pre_workloads = []

        assert self.batch_job_slice == 0 or self.batch_job_slice >= job_sequence_size

        if self.batch_job_slice == 0:
            self.start = self.np_random.randint(job_sequence_size, (self.loads.size() - job_sequence_size - 1))
        else:
            self.start = self.np_random.randint(job_sequence_size, (self.batch_job_slice - job_sequence_size - 1))

        self.start_idx_last_reset = self.start
        self.next_arriving_job_idx = self.start + 1
        self.num_job_in_batch = job_sequence_size
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])

        self.fill_pre_workloads(job_sequence_size + self.np_random.randint(job_sequence_size))

        return self.build_observation()

    def build_critic_observation(self):
        vector = np.zeros(JOB_SEQUENCE_SIZE * 3, dtype=float)
        earliest_job = self.loads[self.start_idx_last_reset]
        earliest_submit_time = earliest_job.submit_time
        pairs = []
        for i in range(self.start_idx_last_reset, self.last_job_in_batch + 1):
            job = self.loads[i]
            submit_time = job.submit_time - earliest_submit_time
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
                normalized_executable_id, can_schedule_now, 1]

    def build_skip_features(self):
        if self.pivot_job:
            return [None, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        else:
            return [None, 1, 1, 1, 1, 1, 1, 1, 0, 0]

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

            index = 0
            global_index = 0
            while index < MAX_QUEUE_SIZE:
                for visible_jobs in visible_jobs_sets:
                    job = visible_jobs[global_index]
                    if job not in self.visible_jobs:
                        self.visible_jobs.append(job)
                        index += 1
                global_index += 1

        # Code for OPTIMIZE_OBSV omitted for brevity

        vector = np.zeros(MAX_QUEUE_SIZE * JOB_FEATURES, dtype=float)
        self.pairs = []
        add_skip = False
        for i in range(MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs):
                job = self.visible_jobs[i]
                self.pairs.append(self.build_job_features(job))
            elif self.skip and not add_skip:
                add_skip = True
                self.pairs.append(self.build_skip_features())
            else:
                self.pairs.append([None, 0, 1, 1, 1, 1, 1, 1, 0, 0])

        for i in range(MAX_QUEUE_SIZE):
            vector[i * JOB_FEATURES:(i + 1) * JOB_FEATURES] = self.pairs[i][1:]

        return np.array(vector, dtype=np.float32)

    def check_then_schedule(self, job):
        # make sure we move forward and release needed resources
        if not self.cluster.fits(job):
            if self.back_fill:
                self.skip_to_resource_backfilling(job)
            else:
                self.skip_to_resource(job)

        # we should be OK to schedule the job now
        self.schedule_job(job)
        not_done = self.process_job_queue()

        if not_done:
            return False
        else:
            return True

    def step(self, a):
        job_for_scheduling = self.pairs[a][0]
        if not job_for_scheduling:
            done = self.noop_schedule()
        else:
            done = self.check_then_schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, {'current_timestamp': self.current_timestamp}]
        else:
            self.scheduled_rl = self.scorer.post_process_matrices(self.scheduled_rl, self.num_job_in_batch,
                                                                  self.current_timestamp, self.loads[self.start],
                                                                  self.loads.max_procs)

            rl_total = sum(self.scheduled_rl.values())
            rwd = -rl_total
            return [None, rwd, True, {'current_timestamp': self.current_timestamp}]
