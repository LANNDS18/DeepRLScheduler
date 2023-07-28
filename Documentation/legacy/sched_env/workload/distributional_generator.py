#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sched_env import workload as wl


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa E501

import itertools
import random
from collections import namedtuple
from typing import Optional, List

from sched_env import workload as wl, job

JobParameters = namedtuple('JobParameters', ['small', 'large'])


class DeepRmWorkloadGenerator(wl.DistributionalWorkloadGenerator):
    def __init__(self, *args: wl.BinomialWorkloadGenerator):
        super().__init__(max([w.length for w in args]))

        self.generators = args
        self.counter = itertools.count(1)

        for generator in self.generators:
            generator.counter = self.counter

    def step(self, offset=1) -> List[Optional[job.Job]]:
        return self.generators[
            random.randint(0, len(self.generators) - 1)
        ].step()

    def __len__(self):
        return self.generators[0].length

    def pick(self):
        return self.step()

    @staticmethod
    def build(
        new_job_rate,
        small_job_chance,
        max_job_len,
        max_job_size,
        ignore_memory=False,
        min_large_job_len=None,
        max_small_job_len=None,
        min_small_job_len=None,
        min_dominant_job_size=None,
        min_other_job_size=None,
        max_other_job_size=None,
        runtime_estimates=None,
        estimate_parameters=None,
    ) -> 'DeepRmWorkloadGenerator':
        # Time-related job parameters {{{
        small_job_time_lower = (
            1 if min_small_job_len is None else min_small_job_len
        )
        small_job_time_upper = (
            max(max_job_len // 5, 1)
            if max_small_job_len is None
            else max_small_job_len
        )
        large_job_time_lower = (
            int(max_job_len * (2 / 3))
            if min_large_job_len is None
            else min_large_job_len
        )
        large_job_time_upper = max_job_len

        dominant_resource_lower = (
            max_job_size // 2
            if min_dominant_job_size is None
            else min_dominant_job_size
        )
        dominant_resource_upper = max_job_size
        other_resource_lower = (
            1 if min_other_job_size is None else min_other_job_size
        )
        other_resource_upper = (
            max_job_size // 5
            if max_other_job_size is None
            else max_other_job_size
        )
        # }}}

        cpu_dominant_parameters = JobParameters(  # {{{
            job.JobParameters(
                small_job_time_lower,
                small_job_time_upper,
                dominant_resource_lower,
                dominant_resource_upper,
                other_resource_lower,
                other_resource_upper,
            ),
            job.JobParameters(
                large_job_time_lower,
                large_job_time_upper,
                dominant_resource_lower,
                dominant_resource_upper,
                other_resource_lower,
                other_resource_upper,
            ),
        )  # }}}

        mem_dominant_parameters = JobParameters(  # {{{
            job.JobParameters(
                small_job_time_lower,
                small_job_time_upper,
                other_resource_lower,
                other_resource_upper,
                dominant_resource_lower,
                dominant_resource_upper,
            ),
            job.JobParameters(
                large_job_time_lower,
                large_job_time_upper,
                other_resource_lower,
                other_resource_upper,
                dominant_resource_lower,
                dominant_resource_upper,
            ),
        )  # }}}

        generators = (
            wl.BinomialWorkloadGenerator(
                new_job_rate,
                small_job_chance,
                cpu_dominant_parameters.small,
                cpu_dominant_parameters.large,
                runtime_estimates=runtime_estimates,
                estimate_parameters=estimate_parameters,
            ),
            wl.BinomialWorkloadGenerator(
                new_job_rate,
                small_job_chance,
                mem_dominant_parameters.small,
                mem_dominant_parameters.large,
                runtime_estimates=runtime_estimates,
                estimate_parameters=estimate_parameters,
            ),
        )

        return DeepRmWorkloadGenerator(
            *generators[: (1 if ignore_memory else None)]
        )


def build(workload_config: dict):
    type = workload_config['type']
    kwargs = {k: v for k, v in workload_config.items() if k != 'type'}
    if type == 'Distribution':
        return DeepRmWorkloadGenerator.build(**kwargs)
    return wl.SwfGenerator(**kwargs)
