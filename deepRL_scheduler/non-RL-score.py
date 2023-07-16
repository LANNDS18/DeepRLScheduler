#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from sched_env.env import TestEnv
from sched_env.scorer import Obs_Scorer


def schedule_curr_sequence_reset(_env, score_fn):
    """schedule the sequence of jobs using heuristic algorithm."""

    obs = _env.reset()
    print(f"Current Time Stamp: {_env.current_timestamp}")

    while True:

        min_index, min_value = min(enumerate(obs), key=lambda pair: score_fn(pair[1]))
        obs, rwd, done, _ = _env.step(min_index)

        if done:
            break

    print(f"Current Time Stamp: {_env.current_timestamp}")
    _env.reset()
    return rwd


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default="./dataset/HPC2N-2002-2.2-cln.swf")  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = TestEnv(workload_file=workload_file, batch_job_slice=700, back_fill=False, seed=0)

    print(schedule_curr_sequence_reset(env, Obs_Scorer.sjf_score))
    print(schedule_curr_sequence_reset(env, Obs_Scorer.fcfs_score))
    print(schedule_curr_sequence_reset(env, Obs_Scorer.smallest_score))
    print(schedule_curr_sequence_reset(env, Obs_Scorer.f1_score))
