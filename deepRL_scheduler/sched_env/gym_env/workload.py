import math
import random
import warnings
from collections import namedtuple
from math import log2

import parallelworkloads as pwa

from deepRL_scheduler.sched_env import workload as wl, job

JobParameters = namedtuple('JobParameters', ['small', 'large'])


class SyntheticWorkloadGenerator(wl.TraceGenerator):
    """A synthetic workload generator based on realistic models."""

    def __init__(
            self,
            length,
            nodes,
            start_time=8,
            random_seed=0,
            restart=False,
            uniform_proportion=0.95,
            cdf_break=0.5,
            runtime_estimates=None,
            estimate_parameters=None,
    ):
        """Synthetic workload generator based on Lublin's work.

        Parameters
        ----------
            length : int
                number of jobs to generate
            nodes : int
                number of compute nodes in the system
            start_time : int
                hour of day in which to start simulation
            random_seed : int
                random seed to use to generate jobs
            restart : bool
                whether to restart after a sample finishes
            uniform_proportion : float
                tunes the proportion between the first and second uniform
                distributions in the two-stage uniform process
            cdf_break : float
                whether to move the break closer to the inferior or superior
                limit. A value closer to 0 will (tend to) produce bigger jobs,
                while a value closer to 1 will (tend to) produce smaller jobs
            runtime_estimates : {'gaussian', 'tsafrir', None}
                whether to include runtime estimates and the method used
                to compute them:
                * None generates perfect estimate (estimates equal run time)
                * 'gaussian' generates estimates with zero-mean Gaussian noise
                  added to them
                * 'tsafrir' uses Dan Tsafrir's model of user runtime estimates
                  to generate estimates
            estimate_parameters : Union[float, List[Tuple[float, float]]
                the parameters used for generating user estimates.
                Depends on :param:`runtime_estimates`.
                When `runtime_estimates` is 'gaussian', this is a single
                floating-point number that sets the standard deviation of the
                noise.
                When `runtime_estimates` is 'tsafrir', this is a list of
                floating-point pairs specifying a histogram (time, number of
                jobs) of job runtime popularity.
        """
        random.seed(random_seed)

        self.lublin = pwa.lublin99.Lublin99(False, random_seed, length)
        self.lublin.start = start_time
        self.random_seed = random_seed
        self.nodes = nodes

        uniform_low_prob = 0.8
        log2_size = log2(nodes)
        min_umed = log2_size - 3.5
        max_umed = log2_size - 1.5
        breaking_point = cdf_break * min_umed + (1 - cdf_break) * max_umed

        self.lublin.setParallelJobProbabilities(
            False,
            uniform_low_prob,
            breaking_point,
            log2_size,
            uniform_proportion,
        )

        self.runtime_estimates = runtime_estimates
        self.estimate_parameters = estimate_parameters

        trace = self.refresh_jobs()
        super().__init__(restart, trace)

    def refresh_jobs(self):
        """Refreshes the underlying job list."""
        jobs = self.lublin.generate()
        if self.runtime_estimates:
            if self.runtime_estimates == 'tsafrir':
                if self.estimate_parameters is not None:
                    warnings.warn(
                        'Setting tsafrir parameters is currently unsupported'
                    )
                tsafrir = pwa.tsafrir05.Tsafrir05(jobs)
                jobs = tsafrir.generate(jobs)
            elif self.runtime_estimates == 'gaussian':
                for j in jobs:
                    j.reqTime = math.ceil(
                        random.gauss(
                            j.runTime, self.estimate_parameters * j.runTime
                        )
                    )
                    if j.reqTime < 1:
                        j.reqTime = 1
            else:
                raise ValueError(
                    f'Unsupported estimate type {self.runtime_estimates}'
                )

        self.trace = [job.Job.from_swf_job(j) for j in jobs]
        return self.trace


def build(workload_config: dict):
    kwargs = {k: v for k, v in workload_config.items() if k != 'dict'}
    return SyntheticWorkloadGenerator(**kwargs)

