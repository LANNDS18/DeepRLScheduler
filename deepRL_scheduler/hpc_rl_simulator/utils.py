#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Optional, Union

import gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv


def evaluate_policy(
        model,
        env: Union[gym.Env, VecEnv],
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
):

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_performance_score = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(observations, state=states, episode_start=episode_starts,
                                        deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                done = dones[i]
                episode_starts[i] = done

                if dones[i]:
                    score = infos[i]['performance_score']
                    episode_rewards.append(current_rewards[i])
                    episode_lengths.append(current_lengths[i])
                    episode_performance_score.append(score)
                    episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_performance_score = np.mean(episode_performance_score)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward, mean_performance_score


class EvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_episodes=1, verbose=0):
        super(EvalCallback, self).__init__(verbose)
        self._eval_env = eval_env
        self._eval_env.reset()
        self._eval_episodes = eval_episodes

    def _on_rollout_start(self):
        mean_reward, reward_std, performance_score = evaluate_policy(
            self.model,
            self._eval_env,
            n_eval_episodes=self._eval_episodes
        )
        print(f"Evaluation: mean reward = {mean_reward}, std reward = {reward_std}")
        self.logger.record("eval/evaluation_env_mean_reward", mean_reward)
        self.logger.record("eval/evaluation_env_std_reward", reward_std)
        self.logger.record("eval/performance matrix", performance_score)
        return True

    def _on_step(self) -> bool:
        return True


def display_message(message: str, quiet: bool) -> None:
    """
    Function to display a message

    Parameters:
    message (str): Message to be displayed
    quiet: whether display or not

    """
    if not quiet:
        print(message)


def init_evaluation_env(workload_path, ENV, config):
    customEnv = ENV(
        flatten_observation=True,
        trace_sample_range=[0.5, 1.0],
        workload_file=workload_path,
        skip=config['skip'],
        job_score_type=config['score_type'],
        quiet=True,
        seed=config['seed'],
        use_fixed_job_sequence=True,
        customized_trace_len_range=(0, 20000)
    )
    customEnv = DummyVecEnv([lambda: customEnv])
    return customEnv


def init_training_env(workload_path, ENV, config):
    customEnv = ENV(
        flatten_observation=True,
        workload_file=workload_path,
        skip=config['skip'],
        job_score_type=config['score_type'],
        trace_sample_range=config['trace_sample_range'],
        quiet=False
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


def extract_custom_kwargs(**kwargs):
    """Extract custom kwargs from kwargs"""
    custom_kwargs = {}
    filtered_kwargs = {}
    for key in kwargs:
        if key in ['actor_model', 'critic_model', 'obs_shape']:
            custom_kwargs[key] = kwargs[key]
        else:
            filtered_kwargs[key] = kwargs[key]
    return custom_kwargs, filtered_kwargs
