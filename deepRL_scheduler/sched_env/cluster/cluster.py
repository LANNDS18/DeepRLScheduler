import math

from typing import Tuple
from .pool import Pool


class Cluster:
    def __init__(self, node_num, num_procs_per_node):
        self.total_node = node_num
        self.free_node = node_num
        self.used_node = 0
        self.num_procs_per_node = num_procs_per_node
        self.all_nodes = []

        for i in range(self.total_node):
            self.all_nodes.append(Pool(i))

        self.state_list = {}

    @property
    def free_resources(self) -> int:
        """The set of resources *not* in use in this cluster."""
        return self.free_node

    def fits(self, job):
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes > self.free_node:
            return False
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes <= self.free_node:
            return True

        request_node = int(math.ceil(float(job.request_number_of_processors) / float(self.num_procs_per_node)))
        job.request_number_of_nodes = request_node
        if request_node > self.free_node:
            return False
        else:
            return True

    def allocate(self, job):
        allocated_nodes = []
        request_node = int(math.ceil(float(job.request_number_of_processors) / float(self.num_procs_per_node)))

        if request_node > self.free_node:
            return []

        allocated = 0

        for m in self.all_nodes:
            if allocated == request_node:
                return allocated_nodes
            if m.taken_by_job(job.job_id):
                allocated += 1
                self.used_node += 1
                self.free_node -= 1
                allocated_nodes.append(m)

        if allocated == request_node:
            return allocated_nodes

        print("Error in allocation, there are enough free resources but can not allocated!")
        return []

    def release(self, releases):
        self.used_node -= len(releases)
        self.free_node += len(releases)

        for m in releases:
            m.release()

    def is_idle(self):
        if self.used_node == 0:
            return True
        return False

    def reset(self):
        self.used_node = 0
        self.free_node = self.total_node
        for m in self.all_nodes:
            m.reset()

    @property
    def state(self) -> Tuple[dict, int, int]:
        """Gets the current state of the cluster as numpy arrays.

        Returns:
            Tuple: a pair containing the number of processors used
        """
        self.state_list = {a.machine_id: a.running_job_id if a.running_job_id else -1 for a in self.all_nodes}
        processors = (
            self.state_list,
            self.free_resources,
            self.used_node,
        )

        return processors
