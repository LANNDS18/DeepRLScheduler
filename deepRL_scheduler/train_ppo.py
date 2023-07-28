import os
import json

from stable_baselines3 import PPO

from sched_env.agent.ppo import *
from sched_env.env import GymSchedulerEnv


def init_env(workload_path, config):
    customEnv = GymSchedulerEnv(
        flatten_observation=True,
        workload_file=workload_path,
        skip=config['skip'],
        job_score_type=config['score_type'],
        batch_job_slice=config['batch_job_slice'],
    )
    return customEnv


def init_dir_from_args(config):
    score_type_dict = {0: 'bsld', 1: 'wait_time', 2: 'turnaround_time', 3: 'resource_utilization'}
    workload_name = config['workload'].split('/')[-1].split('.')[0]
    current_dir = os.getcwd()

    workload_file = os.path.join(current_dir, config['workload'])
    log_data_dir = os.path.join(current_dir, config['log_dir'])
    model_dir = config['model_dir'] + '/' + score_type_dict[config['score_type']] + '/' + workload_name
    print(model_dir)
    return model_dir, log_data_dir, workload_file


if __name__ == '__main__':

    with open('ppo-conf.json', 'r') as f:
        config = json.load(f)

    # init directories
    model_dir, log_data_dir, workload_file = init_dir_from_args(config)
    # create environment
    env = init_env(workload_file, config)
    if config['trained_model']:
        model = PPO.load(config['trained_model'], env=env)
    else:
        available_models = CustomTorchModel.list_models()
        if config['actor_model'] in available_models and config['critic_model'] in available_models:
            policy_args = {'attn': False, 'actor_model': config['actor_model'], 'critic_model': config['critic_model']}
        else:
            print(
                f"Invalid model name: {config['actor_model']} or {config['critic_model']} "
                f"not in available models: {', '.join(available_models)}")
            print(f"Using default model: kernel and critic_lg")
            policy_args = {}

        model = PPO(
            CustomActorCriticPolicy,
            env,
            learning_rate=config['lr'],
            n_steps=config['rollout_steps'],
            n_epochs=config['epochs'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            normalize_advantage=False,
            target_kl=config['target_kl'],
            stats_window_size=config['stats_window_size'],
            policy_kwargs=policy_args,
            tensorboard_log=log_data_dir,
            verbose=config['verbose'],
            seed=config['seed'],
            device=config['device'],
        )
        env_steps = config['rollout_steps'] * config['num_rollouts']
        print(":AGENT-PPO: Learning")
        model.learn(400000)
        model.save(f"{model_dir}ppo_HPC")
        print(f"Trained model saved at: {model_dir}ppo_HPC")
