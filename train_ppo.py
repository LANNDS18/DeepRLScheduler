#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

from stable_baselines3 import PPO

from hpc_rl_simulator.agent import CustomActorCriticPolicy, available_models
from hpc_rl_simulator.common import init_dir_from_args, EvalCallback, init_validation_env, lr_linear_schedule, \
    init_vec_training_env
from hpc_rl_simulator.env import GymSchedulerEnv

if __name__ == '__main__':

    with open('ppo_configs/4-different-dataset/SDSC-SP2.json', 'r') as f:
        config = json.load(f)

    # init directories
    model_dir, log_data_dir, workload_file = init_dir_from_args(config)
    # create environment
    env = init_vec_training_env(workload_file, GymSchedulerEnv, config)

    if config['trained_model']:
        print(f"load model from {config['trained_model']}")
        model = PPO.load(config['trained_model'], env=env)
    else:
        if config['actor_model'] in available_models and config['critic_model'] in available_models:
            policy_args = {
                'actor_model': config['actor_model'],
                'critic_model': config['critic_model']
            }
        else:
            print(
                f"Invalid model name: {config['actor_model']} or {config['critic_model']} "
                f"not in available models: {', '.join(available_models)}")
            print(f"Using default model: kernel and critic_lg")
            policy_args = {}

        model = PPO(
            CustomActorCriticPolicy,
            env,
            learning_rate=lr_linear_schedule(config['lr']),
            n_steps=config['rollout_steps'],
            batch_size=config['batch_size'],
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
        env_steps = config['rollout_steps'] * config['batch_size']
        print(":AGENT-PPO: Learning")

        eval_env = init_validation_env(workload_file, GymSchedulerEnv, config)
        eval_callback = EvalCallback(eval_env, save_path=f"{model_dir}_ppo_bes_new")

        model.learn(total_timesteps=400000, callback=[eval_callback])
        model.save(f"{model_dir}_ppo")
        print(f":AGENT-PPO: Trained model saved at: {model_dir}_ppo_new")
