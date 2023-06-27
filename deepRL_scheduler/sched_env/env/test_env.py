#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC

from .scheduler_env import HPCEnv


class TestEnv(HPCEnv, ABC):
    def __init__(self,
                 shuffle=False,
                 back_fill=False,
                 skip=False,
                 job_score_type=0,
                 batch_job_slice=0,
                 build_sjf=False,
                 seed=0):

        super(HPCEnv, self).__init__(shuffle, back_fill, skip, job_score_type, batch_job_slice, build_sjf, seed)

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
