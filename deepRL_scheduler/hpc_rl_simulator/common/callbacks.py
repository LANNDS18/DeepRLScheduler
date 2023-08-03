#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv


def evaluate_policy(
        model,
        env,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        reward_threshold: Optional[float] = None,
        return_episode_rewards: bool = False,
):
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
    def __init__(self, eval_env, eval_episodes=5, verbose=0, early_stop_tolerance=10, save_path='./data/logs'):
        super(EvalCallback, self).__init__(verbose)
        self._eval_env = eval_env
        self._eval_env.reset()
        self._eval_episodes = eval_episodes
        self.ignore_first_call = True

        self.early_stop_tolerance = early_stop_tolerance
        self.early_stop_count = 0
        self.best_record = -np.inf

        self.save_path = save_path

    def _on_rollout_start(self):
        # avoid evaluate first call
        if self.ignore_first_call:
            self.ignore_first_call = False
            return True

        mean_reward, reward_std, performance_score = evaluate_policy(
            self.model,
            self._eval_env,
            n_eval_episodes=self._eval_episodes
        )

        self._eval_env.n_reset_simulator = 0

        self.early_stop_count += 1
        if performance_score >= self.best_record:
            self.early_stop_count = 0
            self.best_record = performance_score
            print(f"New record: {performance_score}, Saving new best model to {self.save_path}")
            self.model.save(self.save_path)

        print(f"Evaluation: mean reward = {mean_reward}, std reward = {reward_std}")
        self.logger.record("eval/evaluation_env_mean_reward", mean_reward)
        self.logger.record("eval/performance matrix", performance_score)

        if self.early_stop_count >= self.early_stop_tolerance:
            print(f"No improvement made for {self.early_stop_tolerance} episodes, early stop")
            return False

        return True

    def _on_step(self) -> bool:
        if self.early_stop_count >= self.early_stop_tolerance:
            return False
        return True
