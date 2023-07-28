#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Dict

import gym
import numpy as np
import os

from .simulator import DeepRmSimulator
from ..workload.distributional_generator import build as build_workload
from ..scheduler.null_scheduler import NullScheduler

BACKLOG_SIZE = 200
MAXIMUM_NUMBER_OF_ACTIVE_JOBS = 40  # Number of colors in image
MAX_TIME_TRACKING_SINCE_LAST_JOB = 10

TIME_HORIZON = 100
JOB_SLOTS = 5

NUMBER_OF_PROCESSORS = 128

TRACE_FILE = os.path.join("./dataset/HPC2N-2002-2.2-cln.swf")

SWF_WORKLOAD = {
    'tracefile': TRACE_FILE,
    'processors': NUMBER_OF_PROCESSORS,
}


MAXIMUM_JOB_LENGTH = 1500
MAXIMUM_JOB_SIZE = 128
NEW_JOB_RATE = 0.7
SMALL_JOB_CHANCE = 0.8
DEFAULT_WORKLOAD = {
    'type': 'Distribution',
    'new_job_rate': NEW_JOB_RATE,
    'max_job_size': MAXIMUM_JOB_SIZE,
    'max_job_len': MAXIMUM_JOB_LENGTH,
    'small_job_chance': SMALL_JOB_CHANCE,
}

DEFAULT_WORKLOAD = {
    'type': 'k',
    'trace_file': TRACE_FILE,
    'processors': NUMBER_OF_PROCESSORS,
}


class RewardJobs(IntEnum):
    ALL = (0,)
    JOB_SLOTS = (1,)
    WAITING = (2,)
    RUNNING_JOB_SLOTS = (3,)

    @staticmethod
    def from_str(reward_range: str):
        reward_range = reward_range.upper().replace('-', '_')
        if reward_range in RewardJobs.__members__:
            return RewardJobs[reward_range]
        else:
            raise ValueError(
                f'{reward_range} is not a valid RewardJobs range. '
                f'Valid options are: {list(RewardJobs.__members__.keys())}.'
            )


class BaseRmEnv(ABC, gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    job_slots: int
    time_limit: int
    job_num_cap: int
    time_horizon: int
    color_index: List[int]
    color_cache: Dict[int, int]
    simulator: DeepRmSimulator

    @abstractmethod
    def __init__(self, **kwargs):
        self.color_cache = {}
        self.renderer = kwargs.get('renderer', None)
        self.shuffle_colors = kwargs.get('shuffle_colors', False)
        self.job_num_cap = kwargs.get(
            'job_num_cap', MAXIMUM_NUMBER_OF_ACTIVE_JOBS
        )

        self.reward_jobs = RewardJobs.from_str(
            kwargs.get('reward_jobs', 'all')
        )

        self.gamma = kwargs.get('gamma', 1.0)

        self.time_horizon = kwargs.get(
            'time_horizon', TIME_HORIZON
        )  # number of time steps in the graph

        time_limit = kwargs.get('time_limit', 109583573)
        if time_limit is None:
            self.time_limit = 1
            self.update_time_limit = True
        else:
            self.time_limit = time_limit
            self.update_time_limit = False

        step = 1.0 / self.job_num_cap
        # zero is already present and set to "no job there"
        self.colormap = np.arange(start=step, stop=1, step=step)
        if self.shuffle_colors:
            np.random.shuffle(self.colormap)
        self.color_index = list(range(len(self.colormap)))

        # Number of jobs to show
        self.job_slots = kwargs.get('job_slots', JOB_SLOTS)

        self.reward_mapper = {
            RewardJobs.ALL: lambda: self.scheduler.jobs_in_system,
            RewardJobs.WAITING: lambda: self.scheduler.queue_admission,
            RewardJobs.JOB_SLOTS: lambda: self.scheduler.queue_admission[
                : self.job_slots
            ],
            RewardJobs.RUNNING_JOB_SLOTS: lambda: self.scheduler.queue_running
            + self.scheduler.queue_admission[: self.job_slots],
        }

        self.backlog_size = kwargs.get('backlog_size', BACKLOG_SIZE)
        self.processors = kwargs.get('processors', NUMBER_OF_PROCESSORS)

        self.workload_config = kwargs.get('workload', DEFAULT_WORKLOAD)
        wl = build_workload(self.workload_config)

        scheduler = NullScheduler(
            self.processors
        )
        self.simulator = DeepRmSimulator(
            wl,
            scheduler,
            job_slots=self.job_slots,
        )

    def reset(self) -> np.ndarray:
        scheduler = NullScheduler(
            self.processors
        )
        wl = build_workload(self.workload_config)
        if self.update_time_limit and hasattr(wl, 'trace'):
            self.time_limit = (
                wl.trace[-1].submission_time +  # type: ignore
                wl.trace[-1].execution_time  # type: ignore
            )
        self.simulator.reset(wl, scheduler)
        return self.state

    def _render_state(self):
        state, jobs, backlog = self.scheduler.state(
            self.time_horizon, self.job_slots
        )
        s = self._convert_state(
            state,
            jobs,
            backlog,
            (
                (self.simulator.current_time - self.simulator.last_job_time)
                / MAX_TIME_TRACKING_SINCE_LAST_JOB
            ),
        )
        return s

    def build_current_state(self, current):

        ret = np.zeros((self.time_horizon, sum(current[0][:-1])))

        for t in range(self.time_horizon):
            for k, v in current[t][-1].items():
                ret[t][slice(*k)] = v
        return [ret]

    def build_job_slots(self, wait):
        processors = np.zeros(
            (
                self.job_slots,
                self.time_horizon,
                self.scheduler.number_of_processors,
            )
        )
        for i, j in enumerate(wait):
            if j.requested_processors == -1:
                break
            time_slice = slice(
                0,
                self.time_horizon
                if j.requested_time > self.time_horizon
                else j.requested_time,
            )
            processors[i, time_slice, : j.requested_processors] = 1.0
        return processors

    def _convert_state(self, current, wait, backlog, time):
        current = self.build_current_state(current)
        wait = self.build_job_slots(wait)
        backlog_width = self.backlog_size // self.time_horizon
        backlog = np.ones(self.time_horizon * backlog_width) * backlog
        unique = set(np.unique(current[0])) - {0.0}
        if len(unique) > self.job_num_cap:
            raise AssertionError('Number of jobs > number of colors')
        available_colors = list(
            set(self.color_index)
            - set(
                [self.color_cache[int(j)] for j in unique if j in self.color_cache]
            )
        )
        need_color = unique - set(self.color_cache.keys())
        for i, j in enumerate(need_color):
            self.color_cache[int(j)] = available_colors[i]
        for j in unique:  # noqa
            for resource in current:
                resource[resource == int(j)] = self.colormap[self.color_cache[int(j)]]

        return (
            np.array(current),
            np.array(wait),
            backlog.reshape((self.time_horizon, -1)),
            np.ones((self.time_horizon, 1)) * min(1.0, time),
        )

    def render(self, mode='human'):
        if self.renderer is None:
            from .render import DeepRmRenderer

            self.renderer = DeepRmRenderer(mode)
        rgb = self.renderer.render(self._render_state())
        return rgb

    def seed(self, seed=None):
        if seed is None:
            seed = random.randint(0, 99999999)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]


    def f2_score(self, job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f2: r^(1/2)*n + 25600 * log10(s)
        return np.sqrt(request_time) * request_processors + 25600 * np.log10(submit_time)

    @staticmethod
    def f3_score(job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f3: r * n + 6860000 * log10(s)
        return request_time * request_processors + 6860000 * np.log10(submit_time)


    @staticmethod
    def f4_score(job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        # run_time = job.run_time
        # f4: r * sqrt(n) + 530000 * log10(s)
        return request_time * np.sqrt(request_processors) + 530000 * np.log10(submit_time)

    @staticmethod
    def sjf_score(job):
        # run_time = job.run_time
        request_time = job.request_time
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier
        return request_time, submit_time

    @staticmethod
    def smallest_score(job):
        request_processors = job.request_number_of_processors
        submit_time = job.submit_time
        # if request_time is the same, pick whichever submitted earlier
        return request_processors, submit_time

    @staticmethod
    def wfp_score(job):
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time - job.submit_time
        return -np.power(float(waiting_time) / request_time, 3) * request_processors

    @staticmethod
    def uni_score(job):
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        request_time = job.request_time
        waiting_time = job.scheduled_time - job.submit_time

        return -(waiting_time + 1e-15) / (np.log2(request_processors + 1e-15) * request_time)

    @staticmethod
    def fcfs_score(job):
        submit_time = job.submit_time
        return submit_time

    def compute_reward(self, joblist):
        return -np.sum([1 / j.execution_time for j in joblist])

    @property
    def reward(self):
        return self.compute_reward(self.reward_mapper[self.reward_jobs]())

    @property
    def stats(self):
        return self.scheduler.stats

    @property
    @abstractmethod
    def state(self):
        raise NotImplementedError

    @property
    def scheduler(self) -> NullScheduler:
        return self.simulator.scheduler
