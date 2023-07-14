#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from sched_env.env import TestEnv
from sched_env.scorer import Obs_Scorer


def schedule_curr_sequence_reset(_env, score_fn):
    """schedule the sequence of jobs using heuristic algorithm."""

    obs = _env.reset()

    print(f"Current Time Stamp: {_env.current_timestamp}")

    action = 0

    scheduled_logs = {}

    while True:

        state, rwd, done, info = _env.step(action)
        action = 0
        score = 0

        for i, ob in enumerate(obs):
            if score_fn(ob) < score:
                score = score_fn(ob)
                action = i

        if done:
            print("reward", rwd)
            print("action", action)
            break

    scheduled_logs = _env.scorer.post_process_matrices(scheduled_logs, _env.num_job_in_batch,
                                                       _env.current_timestamp, _env.loads[_env.start],
                                                       _env.loads.max_procs)

    # print(f"Current Time Stamp: {_env.current_timestamp}")
    _env.reset()

    return {1: 1}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default="./dataset/HPC2N-2002-2.2-cln.swf")  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = TestEnv(workload_file=workload_file, batch_job_slice=0, back_fill=False, seed=0)

    print(sum(schedule_curr_sequence_reset(env, Obs_Scorer.sjf_score).values()))
    print(sum(schedule_curr_sequence_reset(env, Obs_Scorer.fcfs_score).values()))
    print(sum(schedule_curr_sequence_reset(env, Obs_Scorer.smallest_score).values()))
    print(sum(schedule_curr_sequence_reset(env, Obs_Scorer.f1_score).values()))
