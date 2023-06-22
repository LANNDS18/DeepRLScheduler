import os
from sched_env_v2.HPCSimPickJobs import HPCEnv

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default="./dataset/HPC2N-2002-2.2-cln.swf")  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = HPCEnv(batch_job_slice=700, build_sjf=True)
    env.seed(0)
    env.my_init(workload_file=workload_file)

    env.reset()

    for _ in range(500):
        action = env.action_space.sample()
        ob = env.step(action)
