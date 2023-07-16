#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym

import numpy as np

from gym import spaces
from abc import ABC

from .scheduler_simulator import HPCSchedulingSimulator

from .env_conf import *


class GymSchedulerEnv(HPCSchedulingSimulator, gym.Env, ABC):
    def __init__(self,
                 workload_file,
                 back_fill=False,
                 skip=False,
                 job_score_type=0,
                 batch_job_slice=0,
                 seed=0):

        HPCSchedulingSimulator.__init__(self,
                                        workload_file=workload_file,
                                        back_fill=back_fill,
                                        skip=skip,
                                        job_score_type=job_score_type,
                                        batch_job_slice=batch_job_slice,
                                        seed=seed
                                        )

        job_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(MAX_QUEUE_SIZE, JOB_FEATURES),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = job_space
        # self.build_observation_space()

    def build_observation_space(self):
        job_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(MAX_QUEUE_SIZE, JOB_FEATURES),
            dtype=np.float32
        )

        cluster_space = spaces.box.Box(
            low=-1.0,
            high=1.0,
            shape=self.cluster.state.shape,
            dtype=np.float32
        )

        self.observation_space = gym.spaces.tuple.Tuple(
            (
                job_space,
                cluster_space
            )
        )

        self.observation_space.n = np.sum(  # type: ignore
            [
                np.prod(e.shape) if isinstance(e, gym.spaces.box.Box) else e.n
                for e in self.observation_space
            ]
        )

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

    def build_job_features(self, job, epsilon=1e-6):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        run_time = job.run_time
        request_time = job.request_time
        wait_time = self.current_timestamp - submit_time

        # make sure that larger value is better.
        normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - epsilon)
        normalized_request_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - epsilon)
        normalized_run_time = min(float(run_time) / float(self.loads.max_exec_time), 1.0 - epsilon)
        normalized_request_procs = min(float(request_processors) / float(self.loads.max_procs), 1.0 - epsilon)
        normalized_submit_time = min(float(submit_time) / float(self.loads.end_time), 1.0 - epsilon)

        if job.request_memory == -1:
            normalized_request_memory = 1
        else:
            normalized_request_memory = min(float(job.request_memory) / float(self.loads.max_requested_memory),
                                            1.0 - epsilon)

        if job.user_id == -1:
            normalized_user_id = 1
        else:
            normalized_user_id = min(float(job.user_id) / float(self.loads.max_user_id), 1.0 - epsilon)

        if job.group_id == -1:
            normalized_group_id = 1
        else:
            normalized_group_id = min(float(job.group_id) / float(self.loads.max_group_id), 1.0 - epsilon)

        if job.executable_number == -1:
            normalized_executable_id = 1
        else:
            normalized_executable_id = min(
                float(job.executable_number) / float(self.loads.max_executable_number), 1.0 - epsilon)

        if self.cluster.fits(job):
            can_schedule_now = 1.0 - epsilon
        else:
            can_schedule_now = epsilon

        return [job, normalized_submit_time, normalized_wait_time, normalized_request_time, normalized_run_time,
                normalized_request_procs, normalized_request_memory, normalized_user_id, normalized_group_id,
                normalized_executable_id, can_schedule_now, 1]

    def build_skip_features(self):
        if self.pivot_job:
            return [None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        else:
            return [None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]

    def build_cluster_state(self):
        return np.array(self.cluster.state)

    def build_observation(self):
        self.job_queue.sort(key=lambda j: self.scorer.fcfs_score(j))

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
                self.pairs.append([None, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0])

        vector = np.vstack(np.array([pair[1:] for pair in self.pairs]), dtype='float32')

        return vector  # , self.build_cluster_state()

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
            scheduled_rl = self.scorer.post_process_matrices(self.scheduled_rl, self.num_job_in_batch,
                                                             self.current_timestamp, self.loads[self.start],
                                                             self.loads.max_procs)

            rl_total = sum(scheduled_rl.values())
            rwd = -rl_total
            return [None, rwd, True, {'current_timestamp': self.current_timestamp}]
