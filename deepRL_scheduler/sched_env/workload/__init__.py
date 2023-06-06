"""workload - Package for generators of load for a cluster.

Supports generative workloads, based on probability distributions, and
trace-based workloads in the Standard Workload Format.
"""

from .base import WorkloadGenerator
from .traces import TraceGenerator, SwfGenerator

__all__ = [
    'WorkloadGenerator',
    'TraceGenerator',
    'SwfGenerator',
]