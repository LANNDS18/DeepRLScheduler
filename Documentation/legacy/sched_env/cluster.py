#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
cluster - Classes for cluster management
"""

import copy
from typing import Tuple, Iterable, Optional

from . import pool

from .job import Job, Resource
from .event import JobEvent, EventType


class Cluster:
    """A cluster as a set of resources.

    Parameters
    ----------
        processors : int
            The number of processors in this cluster
        used_processors : Optional[Resource]
            Processors already in use in this cluster
    """

    processors: pool.ResourcePool

    def __init__(
            self,
            processors,
            used_processors: Optional = None,

    ):
        self.processors = pool.ResourcePool(
            pool.ResourceType.CPU, processors, used_processors
        )

    @property
    def free_resources(self) -> int:
        """The set of resources *not* in use in this cluster."""
        return self.processors.free_resources

    def fits(self, job: Job) -> bool:
        """Checks whether a job fits in this cluster.

        Parameters
        ----------
            job : Job
                The job to check against in this cluster
        """
        return self.processors.fits(job.requested_processors)

    def allocate(self, job: Job) -> None:
        """Checks whether a job fits the system and allocates resources for it.

        Parameters
        ----------
            job : Job
                The job to allocate resources to.
        """
        if not self.fits(job):
            raise AssertionError(
                f'Unable to allocate resources for {job} in {self}'
            )
        self.processors.allocate(job.resources.processors)

    def clone(self):
        """Clones this Cluster"""
        return copy.deepcopy(self)

    def find(self, job: Job) -> Resource:
        """Finds resources for a job.

        If the job fits in the system, this will return a set of resources that
        can be used by a job. If it doesn't, will return an empty set of
        resources (which evaluate to False in boolean expressions).

        Parameters
        ----------
            job : Job
                The job to find resources to.
        """
        p = self.processors.find(job.requested_processors, job.id)
        if not p:
            return Resource()
        return Resource(p)

    def free(self, job: Job) -> None:
        """Frees the resources used by a job.

        Parameters
        ----------
            job : Job
                The job to free resources from.
        """
        self.processors.free(job.resources.processors)

    def find_resources_at_time(
            self, time: int, job: Job, events: Iterable[JobEvent]
    ) -> Resource:
        """Finds resources for a job at a given time step.

        To find an allocation for a job, we have to iterate through the
        queue of events and evaluating the state of the system given that set
        of events to check whether a given job would fit the system.

        Since this method can be called with time stamps in the far future, we
        are required to play events to find the exact configuration in the
        future.

        Parameters
        ----------
            time : int
                The time at which to check whether the job fits the system
            job : Job
                The job to check
            events : Iterable[JobEvent]
                A set of events that will play out in the future
        """

        def valid(e, t):
            return t + 1 <= e.time < job.requested_time + t

        used = Resource(self.processors.used_pool)
        for event in (e for e in events if (valid(e, time) and e.type == EventType.JOB_START)):
            for i in event.processors:
                used.processors.add(i)
        used.processors.merge_overlaps()

        return Cluster(self.processors.size, used.processors).find(job)

    @property
    def state(self) -> Tuple[int, int, dict]:
        """Gets the current state of the cluster as numpy arrays.

        Returns:
            Tuple: a pair containing the number of processors used
        """
        processors = (
            self.processors.free_resources,
            self.processors.used_resources,
            {(i.begin, i.end): i.data for i in self.processors.used_pool},
        )

        return processors

    def __bool__(self):
        return (
                self.processors.free_resources != 0
        )

    def __repr__(self):
        return (
            f'Cluster({self.processors})'
        )

    def __str__(self):
        return (
            f'Cluster({self.processors})'
        )
