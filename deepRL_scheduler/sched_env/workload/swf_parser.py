#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
swf_parser - Parser for the Standard Workload Format (SWF)

A full description of the format, with meanings for each field is available on
the web at http://www.cs.huji.ac.il/labs/parallel/workload/swf.html.
"""

from enum import IntEnum

import logging

from ..job import Job, SwfJobStatus

logger = logging.getLogger(__name__)  # pylint: disable=C


class SwfFields(IntEnum):
    """Fields of the Standard Workload Format."""

    JOB_ID = 0
    SUBMITTED = 1
    WAIT_TIME = 2
    RUN_TIME = 3
    NUM_PROCS = 4
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
    """Parser for SWF job files.

    The SWF is a simple format with commented lines starting with the ';'
    character and other lines separated by spaces.

    Parsing, therefore, involves splitting the lines and associating each
    column of the file with a field.
    """

    with open(filename, 'r') as fp:  # pylint: disable=C
        for line in fp:
            if ';' in line:
                continue
            fields = line.strip().split()
            fields = [  # Converts all fields according to our rules
                CONVERTERS[SwfFields(i)](f) for i, f in enumerate(fields)
            ]

            job = Job(
                fields[SwfFields.JOB_ID],
                fields[SwfFields.SUBMITTED],
                fields[SwfFields.RUN_TIME],
                fields[SwfFields.NUM_PROCS],
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
