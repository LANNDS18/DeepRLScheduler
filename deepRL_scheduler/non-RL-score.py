#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from sched_env.env import TestEnv
from sched_env.scorer import ScheduleScorer


def schedule_curr_sequence_reset(_env, score_fn):
    """schedule the sequence of jobs using heuristic algorithm."""

    _env.reset()
    _env.heuristic_reset()
    print(f"Current Time Stamp: {_env.current_timestamp}")
    scheduled_logs = {}

    while True:
        not_done, scheduled_logs = _env.heuristic_step(score_fn, scheduled_logs)
        if not not_done:
            break

    scheduled_logs = _env.scorer.post_process_matrices(scheduled_logs, _env.num_job_in_batch,
                                                       _env.current_timestamp, _env.loads[_env.start],
                                                       _env.loads.max_procs)

    print(f"Current Time Stamp: {_env.current_timestamp}")
    _env.heuristic_reset()
    return scheduled_logs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default="./dataset/HPC2N-2002-2.2-cln.swf")  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = TestEnv(batch_job_slice=700, back_fill=False, seed=0)
    env.load_job_trace(workload_file=workload_file)

    print(sum(schedule_curr_sequence_reset(env, ScheduleScorer.sjf_score).values()))
    print(sum(schedule_curr_sequence_reset(env, ScheduleScorer.fcfs_score).values()))
    print(sum(schedule_curr_sequence_reset(env, ScheduleScorer.smallest_score).values()))
    print(sum(schedule_curr_sequence_reset(env, ScheduleScorer.f1_score).values()))

