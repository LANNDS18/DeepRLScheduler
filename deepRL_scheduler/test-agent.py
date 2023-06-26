import os
from sched_env.scheduler_env import HPCEnv

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default="./dataset/HPC2N-2002-2.2-cln.swf")  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = HPCEnv(batch_job_slice=700, build_sjf=True, back_fill=False, seed=0)
    env.load_job_trace(workload_file=workload_file)
    env.reset()

    for i in range(500):
        action = env.action_space.sample()
        state, rwd, done, rwd2, sjf, f1 = env.step(action)
        if done:
            print(i)
