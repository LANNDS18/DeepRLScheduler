#!/usr/bin/env python
# -*- coding: utf-8 -*-

MAX_QUEUE_SIZE = 128

MAX_WAIT_TIME = 12 * 60 * 60  # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60  # assume maximal runtime is 12 hours

# each job has 11 features
JOB_FEATURES = 11
# skip 60 seconds
SKIP_TIME = 60

MIN_JOB_SEQUENCE_SIZE = 256
MAX_JOB_SEQUENCE_SIZE = 1024
