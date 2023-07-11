MAX_QUEUE_SIZE = 128
MLP_SIZE = 256

MAX_WAIT_TIME = 12 * 60 * 60  # assume maximal wait time is 12 hours.
MAX_RUN_TIME = 12 * 60 * 60  # assume maximal runtime is 12 hours

# each job has three features: wait_time, requested_node, runtime, machine states,
JOB_FEATURES = 9
DEBUG = False

JOB_SEQUENCE_SIZE = 256
SKIP_TIME = 360  # skip 60 seconds
