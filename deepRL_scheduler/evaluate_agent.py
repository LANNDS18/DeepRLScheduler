from train_ppo import init_env, init_dir_from_args
import json
from stable_baselines3 import PPO


def schedule_curr_sequence_reset(_env, model, log=True):
    """schedule the sequence of jobs using heuristic algorithm."""

    obs = _env.reset()

    while True:

        pi = model.predict(obs)
        action = pi[0]
        obs, rwd, done, info = _env.step(action)

        if done:
            record = info['performance matrix']
            current_time = info['current_timestamp']
            break

    if log:
        print(f"Current Time Stamp: {current_time}")
        print(f'total performance matrix value: {sum(record.values())}')
    _env.reset()
    return rwd


with open('ppo-conf.json', 'r') as f:
    config = json.load(f)

# init directories
model_dir, log_data_dir, workload_file = init_dir_from_args(config)
# create environment

from sched_env.env import GymSchedulerEnv

for i in range(10):
    slice = 200000 - i * 10000
    print(slice)
    env = GymSchedulerEnv(
        workload_file="./dataset/HPC2N-2002-2.2-cln.swf",
        flatten_observation=True,
        batch_job_slice=slice,
        back_fill=False,
        seed=0
    )
    model = PPO.load("./trained_models/bsld/HPC2N-2002-2ppo_HPC.zip", env=env)
    print(schedule_curr_sequence_reset(env, model, False))
