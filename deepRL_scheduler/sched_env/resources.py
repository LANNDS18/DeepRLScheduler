#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""resource - basic resource unit

This module has two classes:
  1. `PrimaryResource`, an enumeration for the different supported types (CPU)
  2. The basic resource group, which is comprised of CPU only now
"""

import copy
import enum
from typing import Tuple

from intervaltree import IntervalTree


class PrimaryResource(enum.IntEnum):
    """Enumeration for identifying the various supported resource types."""

    CPU = 0


class Resource(object):
    """The basic resource group.

    This groups IntervalTrees into as many resources that can are supported in
    the system.

    This is referenced by a :class:`schedgym.job.Job` to represent *which
    specific resources* are being used by that job.

    Parameters
    ----------
        processors : IntervalTree
            An interval tree that defines a set of processors
    """

    processors: IntervalTree
    """IntervalTree that stores processors used"""

    def __init__(
        self,
        processors: IntervalTree = IntervalTree(),
    ):
        self.processors = copy.copy(processors)

    def measure(self) -> int:
        """Returns the total amount of resources in use.

        Returns:
            Tuple: A tuple containing the amount of resources used for each
            resource type supported.
        """
        processors = sum([i.end - i.begin for i in self.processors])
        return processors

    def __bool__(self) -> bool:
        return bool(self.processors)

    def __repr__(self):
        return f'Resource({self.processors})'

    def __str__(self):
        return f'Resource({self.processors})'
