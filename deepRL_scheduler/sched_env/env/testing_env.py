#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC

from .gym_env import GymSchedulerEnv


class TestEnv(GymSchedulerEnv, ABC):
    def __init__(self,
                 workload_file,
                 back_fill=False,
                 skip=False,
                 job_score_type=0,
                 batch_job_slice=0,
                 seed=0):

        GymSchedulerEnv.__init__(self,
                                 workload_file=workload_file,
                                 back_fill=back_fill,
                                 skip=skip,
                                 job_score_type=job_score_type,
                                 batch_job_slice=batch_job_slice,
                                 seed=seed)

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
            done = self.noop_schedule()
        else:
            job_for_scheduling = self.pairs[a][0]
            done = self.check_then_schedule(job_for_scheduling)

        if not done:
            obs = self.build_observation()
            return [obs, 0, False, None]
        else:
            self.scheduled_rl = self.scorer.post_process_matrices(self.scheduled_rl, self.num_job_in_batch,
                                                                  self.current_timestamp, self.loads[self.start],
                                                                  self.loads.max_procs)
            rl_total = sum(self.scheduled_rl.values())
            return [None, rl_total, True, None]

    def heuristic_reset(self):
        self.reset_env_component()

        self.current_timestamp = self.loads[self.start].submit_time
        self.job_queue.append(self.loads[self.start])
        self.last_job_in_batch = self.start + self.num_job_in_batch
        self.next_arriving_job_idx = self.start + 1

        self.refill_pre_workloads()

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
        score = self.scorer.scheduling_matrices(job_for_scheduling)  # calculated reward
        scheduled_logs[job_for_scheduling.job_id] = score
        self.job_queue.remove(job_for_scheduling)

        not_empty = self.process_job_queue()

        return not_empty, scheduled_logs
