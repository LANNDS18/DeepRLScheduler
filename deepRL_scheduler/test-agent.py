#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from sched_env.env import GymSchedulerEnv

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default="./dataset/HPC2N-2002-2.2-cln.swf")  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = GymSchedulerEnv(workload_file=workload_file, batch_job_slice=700, back_fill=False, seed=0)
    env.reset()

    for i in range(5000):
        action = env.action_space.sample()
        state, rwd, done, info = env.step(action)
        print(info)
        print(done)
