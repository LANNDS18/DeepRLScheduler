#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""trace - A trace-based workload generator

Inherits from the base WorkloadGenerator and uses the swf_parser to parse SWF
files.
"""

from itertools import takewhile
from typing import Iterator, Optional

from .base import WorkloadGenerator
from .swf_parser import parse as parse_swf
from ..job import Job


class SwfGenerator(WorkloadGenerator):
    """A trace-based (workload log) generator.

    Supports starting the parsing after an offset, and also supports reading a
    pre-specified number of jobs.

    Parameters
    ----------
        trace_file : str
            The path to the filed to be parsed and used as input for workload
            generation.
        processors : int
            The number of processors in this trace
        restart : bool
            Whether to restart from the beginning of the file when we reach
            its end (or, in the case we're using an offset and a length, to
            restart from the offset up to the length)
    """

    def __init__(
            self,
            trace_file,
            processors,
            offset=0,
            length=None,
            restart=True,
    ):

        self.current_time = 0
        self.restart = restart
        self.current_element = 0

        self.trace = list(parse_swf(trace_file, processors))
        self.trace_file = trace_file

        if length is None:
            length = len(self.trace)
        else:
            length = length if length <= len(self.trace) else len(self.trace)

        self.trace = self.trace[offset:offset + length]

        self.current_element = 0

    def step(self, offset=1):
        """ "Samples" jobs from the trace file.

        Parameters
        ----------
            offset : int
                The amount to offset the current time step
        """
        if offset < 0:
            raise ValueError('Submission time must be positive')
        if self.current_element >= len(self.trace):
            if self.restart:
                self.current_element = 0
                for job in self.trace:
                    job.submission_time += self.current_time
            else:
                raise StopIteration('Workload finished')
        submission_time = self.current_time + offset
        jobs = takewhile(
            lambda j: j[1].submission_time <= submission_time,
            enumerate(
                self.trace[self.current_element:], self.current_element
            ),
        )
        self.current_time = submission_time
        jobs = list(jobs)
        if jobs:
            self.current_element = jobs[-1][0] + 1
            return [j for (i, j) in jobs]
        return []

    def step_to_next_job(self):
        """ "Samples" jobs from the trace file."""

        if self.current_element >= len(self.trace):
            if self.restart:
                self.current_element = 0
                for job in self.trace:
                    job.submission_time += self.current_time
            else:
                raise StopIteration('Workload finished')

        job = self.pick()

        if job:
            self.current_element = self.current_element + 1
            self.current_time = job.submission_time
            return job

        return None

    @property
    def last_event_time(self):
        """The submission time of the last generated job"""
        offset = (
            self.current_element
            if self.current_element < len(self.trace)
            else -1
        )
        return self.trace[offset].submission_time

    def __len__(self):
        return len(self.trace)

    def __next__(self) -> Job:
        if self.current_element >= len(self.trace):
            if self.restart:
                self.current_element = 0
            else:
                raise StopIteration()
        job = self.trace[self.current_element]
        self.current_element += 1
        return job

    def __iter__(self) -> Iterator[Optional[Job]]:
        return iter(self.trace)

    def __getitem__(self, item):
        return self.trace[item]

    def pick(self) -> Optional[Job]:
        job = next(self)
        if self.current_element > 0:
            self.current_element -= 1
        return job
