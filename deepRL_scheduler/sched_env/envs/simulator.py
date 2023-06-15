#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, List, Optional, Union, cast

from sched_env.job import Job
from sched_env.scheduler import NullScheduler
from sched_env.workload import SwfGenerator

WorkloadGeneratorType = Union[SwfGenerator]


class DeepRmSimulator:
    scheduler: NullScheduler
    workload: SwfGenerator

    def __init__(
            self,
            workload_generator: SwfGenerator,
            scheduler: NullScheduler,
            job_slots: Optional[int] = None,
    ):
        self.scheduler = scheduler
        self.workload = workload_generator
        self.job_slots = slice(0, job_slots)
        self.simulator = TimeBasedDeepRmSimulator(
            self.workload,
            self.scheduler,
            self.job_slots,
        )
        self.reset(self.workload, scheduler)

    def rl_step(
            self,
            action: Optional[int],
            listjobs: Optional[Callable[[], List[Job]]],
    ) -> List[List[Job]]:
        return self.simulator.rl_step(
            action if action is not None else -1,
            listjobs if listjobs else lambda: self.scheduler.jobs_in_system,
        )

    @property
    def current_time(self):
        return self.simulator.current_time

    @property
    def last_job_time(self):
        return self.simulator.last_job_time

    def reset(self, workload, scheduler):
        self.scheduler = scheduler
        self.workload = workload
        self.simulator = TimeBasedDeepRmSimulator(
            self.workload,
            self.scheduler,
            self.job_slots,
        )


class TimeBasedDeepRmSimulator:
    last_job_time: int
    scheduler: NullScheduler
    job_slots: slice

    def __init__(
            self,
            workload_generator: WorkloadGeneratorType,
            scheduler: NullScheduler,
            job_slots: slice,
    ):

        self.scheduler = scheduler
        self.simulation_start_time = 0
        self.workload = workload_generator
        self.current_time = self.last_job_time = 0
        self.job_slots = job_slots

        if isinstance(workload_generator, SwfGenerator):
            first_job_time = cast(
                Job, workload_generator.pick()
            ).submission_time - 1
            workload_generator.current_time = first_job_time
            scheduler.job_events.time = first_job_time
            scheduler.current_time = first_job_time

    def step(self, _=True):
        """Not implemented in DeepRmSimulator"""
        raise NotImplementedError('This simulator cannot follow the base API')

    def rl_step(
            self, action: int, listjobs: Callable[[], List[Job]]
    ) -> List[List[Job]]:
        """Returns a list of jobs for each successful intermediate time step."""

        if self.scheduler.step(action):
            return [[]]
        else:
            self.current_time += 1
            j = self.workload.step()
            if j:
                self.scheduler.submit(j)
                self.last_job_time = self.current_time
            self.scheduler.forward_time()
            return [listjobs()]
