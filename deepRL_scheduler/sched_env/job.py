"""
    job - Classes for jobs in the simulator.
"""

import enum
import warnings
from collections import namedtuple

from deepRL_scheduler.sched_env.resources import Resource

JobState = namedtuple(
    'JobState',
    [
        'submission_time',
        'requested_time',
        'requested_memory',
        'requested_processors',
        'queue_size',
        'queued_work',
        'free_processors',
    ],
)


class JobStatus(enum.IntEnum):
    """An enumeration for different states of a job within our simulator."""

    SUBMITTED = 0
    RUNNING = 1
    WAITING = 2
    COMPLETED = 3
    SCHEDULED = 4
    INVALID = -1


class SwfJobStatus(enum.IntEnum):
    """An enumeration for different states of a job in the SWF_.

    .. _SWF: https://www.cs.huji.ac.il/labs/parallel/workload/swf.html
    """

    FAILED = 0
    COMPLETED = 1
    PARTIAL_TO_BE_CONTINUED = 2
    PARTIAL_LAST_COMPLETED = 3
    PARTIAL_LAST_FAILED = 4
    CANCELLED = 5
    MEANINGLESS = -1


class Job:
    """A job in the system.

    This follows the fields of the `Standard Workload Format
    <https://www.cs.huji.ac.il/labs/parallel/workload/swf.html>`_ with a couple
    of helper methods to compute slowdown and bounded slowdown of a job. The
    initializer arguments follow the same ordering and have the same meaning
    as those in the SWF description.

    This makes use of the :class:`schedgym.resource.Resource` class to keep
    track of the assigned resources to the job. Resource assignment itself is
    performed by
    :func:`schedgym.scheduler.scheduler.Scheduler.assign_schedule`.

    The figure below shows the relationship between jobs, resources, and the
    basic data structure for resource management (`IntervalTree`).

    .. image:: /img/job-resource.svg
    """

    resources: Resource

    SWF_JOB_MAP = {
        'jobId': 'id',
        'submissionTime': 'submission_time',
        'waitTime': 'wait_time',
        'runTime': 'execution_time',
        'allocProcs': 'processors_allocated',
        'avgCpuUsage': 'average_cpu_use',
        'usedMem': 'memory_use',
        'reqProcs': 'requested_processors',
        'reqTime': 'requested_time',
        'reqMem': 'requested_memory',
        'status': 'status',
        'userId': 'user_id',
        'groupId': 'group_id',
        'executable': 'executable',
        'queueNum': 'queue_number',
        'partNum': 'partition_number',
        'precedingJob': 'preceding_job_id',
        'thinkTime': 'think_time',
    }

    def __init__(
            self,
            job_id=-1,
            submission_time=-1,
            execution_time=-1,
            processors_allocated=-1,
            average_cpu_use=-1,
            memory_use=-1,
            requested_processors=-1,
            requested_time=-1,
            requested_memory=-1,
            status=JobStatus.INVALID,
            user_id=-1,
            group_id=-1,
            executable=-1,
            queue_number=-1,
            partition_number=-1,
            preceding_job_id=-1,
            think_time=-1,
            wait_time=-1,
            ignore_memory=True,
    ):
        self.id: int = job_id
        self.submission_time: int = submission_time
        self.execution_time: int = execution_time
        self.requested_time: int = requested_time
        self.requested_processors: int = requested_processors
        self.processors_allocated: int = processors_allocated
        self.average_cpu_use: int = average_cpu_use
        self.memory_use: int = memory_use
        self.requested_memory: int = requested_memory
        self.status: JobStatus = status
        self.user_id: int = user_id
        self.group_id: int = group_id
        self.executable: int = executable
        self.queue_number: int = queue_number
        self.partition_number: int = partition_number
        self.preceding_job_id: int = preceding_job_id
        self.think_time = think_time
        self.wait_time = wait_time

        self.resources = Resource()
        self.first_scheduling_promise: int = -1
        self.start_time: int = -1
        self.finish_time: int = -1
        self.ignore_memory = ignore_memory
        self.slot_position: int = -1
        self.free_processors = -1
        self.queued_work = -1
        self.queue_size = -1

    def __str__(self):
        return (
            f'Job<{self.id}, {self.status.name}, start={self.start_time}, '
            f'processors={self.requested_processors}, '
            f'memory={self.requested_memory} '
            f'duration={self.execution_time}>'
        )

    __repr__ = __str__

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    @property
    def proper(self):
        """Checks whether this job is a proper job with assigned resources.

        Returns:
            bool: True if the job is proper, and False otherwise.
        """
        processors, memory = self.resources.measure()
        return processors == self.requested_processors and (
                self.ignore_memory or memory == self.requested_memory
        )

    @property
    def slowdown(self):
        """Computes the slowdown of the current job."""
        if self.finish_time < 0:
            warnings.warn(
                f'Failed to obtain slowdown for job {self}. '
                'It may not have finished yet.'
            )
            return -1
        return (
                self.finish_time - self.submission_time
        ) / self.execution_time

    @property
    def bounded_slowdown(self):
        """Gives the bounded slowdown of a job"""
        if self.finish_time < 0:
            warnings.warn(
                f'Failed to obtain avg bounded slowdown for job {self}.'
                'It may not have finished yet.'
            )
            return -1
        return max(
            1.0,
            (self.finish_time - self.submission_time)
            / max(10, self.execution_time),
        )

    @property
    def resource_utilization(self):
        """Returns the resource utilization of the current job."""
        if self.status == JobStatus.SUBMITTED or self.status == JobStatus.WAITING:
            return 0

        elif self.status == JobStatus.COMPLETED:
            return 1

        else:  # If the job is running
            return self.average_cpu_use / self.requested_processors

    def waiting_time(self, current_time_step):
        """Returns the waiting time of the current job."""
        if self.status == JobStatus.RUNNING or self.status == JobStatus.COMPLETED:
            return self.start_time - self.submission_time
        else:  # If the job is submitted or waiting
            return current_time_step - self.submission_time

    @property
    def swf(self):
        """Returns an SWF representation of this job"""
        return (
            f'{self.id} {self.submission_time} {self.wait_time} '
            f'{self.execution_time} {self.processors_allocated} '
            f'{self.average_cpu_use} '
            f'{self.memory_use} {self.requested_processors} '
            f'{self.requested_time} {self.requested_memory} '
            f'{self.swfstatus} {self.user_id} {self.group_id} '
            f'{self.executable} {self.queue_number} '
            f'{self.partition_number} {self.preceding_job_id} '
            f'{self.think_time}'
        )

    @property
    def swfstatus(self):
        """Returns the job status in the format expected by the SWF."""
        if self.status == JobStatus.COMPLETED:
            return SwfJobStatus.COMPLETED
        return SwfJobStatus.MEANINGLESS

    @staticmethod
    def from_swf_job(swf_job):
        """Converts an SWF job to our internal job format."""
        new_job = Job()
        for key, value in Job.SWF_JOB_MAP.items():
            tmp = getattr(swf_job, key)
            setattr(new_job, value, int(tmp) if 'time' in value else tmp)

        new_job.status = JobStatus.SUBMITTED
        new_job.requested_processors = new_job.processors_allocated
        if new_job.requested_time == -1:
            new_job.requested_time = new_job.execution_time

        return new_job

    @property
    def state(self):
        return JobState(
            self.submission_time,
            self.requested_time,
            self.requested_memory,
            self.requested_processors,
            self.queue_size,
            self.queued_work,
            self.free_processors,
        )
