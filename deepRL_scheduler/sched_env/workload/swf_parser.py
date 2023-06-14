"""
parser - Parser for the Standard Workload Format (SWF)

A full description of the format, with meanings for each field is available on
the web at http://www.cs.huji.ac.il/labs/parallel/workload/swf.html.
"""

import logging
import sys
from enum import IntEnum
from itertools import takewhile
from typing import Iterator, Optional, Sequence, Callable

from deepRL_scheduler.sched_env.job import Job, SwfJobStatus

# Configure logger
logger = logging.getLogger(__name__)


class SwfFields(IntEnum):
    """Enumeration for the Fields of the Standard Workload Format."""
    JOB_ID = 0
    SUBMITTED = 1
    WAIT_TIME = 2
    EXEC_TIME = 3
    ALLOC_PROCS = 4
    AVG_CPU_USAGE = 5
    USED_MEM = 6
    REQ_PROCS = 7
    REQ_TIME = 8
    REQ_MEM = 9
    STATUS = 10
    USER_ID = 11
    GROUP_ID = 12
    EXECUTABLE = 13
    QUEUE_NUM = 14
    PART_NUM = 15
    PRECEDING_JOB = 16
    THINK_TIME = 17


CONVERTERS = {
    key: int if key != SwfFields.AVG_CPU_USAGE else float for key in SwfFields
}


def parse(filename, processors):
    """
    Parser for SWF job files.

    Args:
        filename: path to the file to be parsed
        processors: number of processors

    Yields:
        job: a Job object constructed from the fields in the file
    """
    with open(filename, 'r') as file:
        for line_number, line in enumerate(file):
            if ';' in line:
                continue

            fields = [
                CONVERTERS[SwfFields(i)](field)
                for i, field in enumerate(line.strip().split())
            ]

            job = Job(
                fields[SwfFields.JOB_ID],
                fields[SwfFields.SUBMITTED],
                fields[SwfFields.EXEC_TIME],
                fields[SwfFields.ALLOC_PROCS],
                fields[SwfFields.AVG_CPU_USAGE],
                fields[SwfFields.USED_MEM],
                fields[SwfFields.REQ_PROCS],
                fields[SwfFields.REQ_TIME],
                fields[SwfFields.REQ_MEM],
                SwfJobStatus(fields[SwfFields.STATUS]),
                fields[SwfFields.USER_ID],
                fields[SwfFields.GROUP_ID],
                fields[SwfFields.EXECUTABLE],
                fields[SwfFields.QUEUE_NUM],
                fields[SwfFields.PART_NUM],
                fields[SwfFields.PRECEDING_JOB],
                fields[SwfFields.THINK_TIME],
                fields[SwfFields.WAIT_TIME],
            )

            if job.requested_memory < 0 < job.memory_use:
                job.requested_memory = job.memory_use

            if job.requested_processors < 0 < job.processors_allocated:
                job.requested_processors = job.processors_allocated

            if job.requested_processors < 1:
                logger.warning(f'Ignoring wrong processor indication for job {job.id}')
                continue

            if job.execution_time < 0.1 or job.submission_time < 0:
                logger.warning(f'Ignoring unclear execution time and/or submission time for job {job.id}')
                continue

            if job.requested_time < job.execution_time:
                job.requested_time = job.execution_time

            if job.requested_processors > processors:
                job.requested_processors = processors

            yield job


class SwfWorkload:
    restart: bool
    trace: Sequence[Job]
    refresh_jobs: Optional[Callable] = None

    def __init__(self, path):
        self.all_jobs = []
        self.max = 0
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_job_id = 0

        self.max_requested_memory = 0
        self.max_user_id = 0
        self.max_group_id = 0
        self.max_executable_number = 0
        self.max_job_id = 0
        self.max_nodes = 0
        self.max_procs = 0
        self.ignore_memory = True

        with open(path) as fp:
            for line in fp:
                if line.startswith(";"):
                    if line.startswith("; MaxNodes:"):
                        self.max_nodes = int(line.split(":")[1].strip())
                    if line.startswith("; MaxProcs:"):
                        self.max_procs = int(line.split(":")[1].strip())
                    if self.max_procs == 0:
                        self.max_procs = self.max_nodes

        self.all_jobs = list(parse(path, self.max_procs))
        self.all_jobs.sort(key=lambda job: job.id)

        self.current_time = 0
        self.restart = False
        self.current_element = 0

        if path is not None:
            self.trace = self.all_jobs
        else:
            self.trace = []

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
                if self.refresh_jobs is not None:
                    self.refresh_jobs()
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
                if self.refresh_jobs is not None:
                    self.refresh_jobs()
            else:
                raise StopIteration()
        job = self.trace[self.current_element]
        self.current_element += 1
        return job

    def __iter__(self) -> Iterator[Optional[Job]]:
        return iter(self.trace)

    def pick(self) -> Optional[Job]:
        job = next(self)
        if self.current_element > 0:
            self.current_element -= 1
        return job

    def size(self):
        return len(self.all_jobs)

    def reset(self):
        for job in self.all_jobs:
            job.submission_time = -1


def build(workload_config: dict):
    kwargs = {k: v for k, v in workload_config.items() if k != 'dict'}
    return SwfWorkload(**kwargs)


if __name__ == "__main__":
    print("Loading the workloads...")
    load = list(parse("../../dataset/NASA-iPSC-1993-3.1-cln.swf", 256))
    load.sort(key=lambda job: job.id)
    print(load[0])
