#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import gym

import numpy as np

from gym import spaces
from abc import ABC
from typing import Tuple, List

from .scheduler_simulator import HPCSchedulingSimulator
from .env_conf import *

from ..job import JobTransition, Job


class GymSchedulerEnv(HPCSchedulingSimulator, gym.Env, ABC):
    """
    Reinforcement learning environment that simulates a High Performance Computing (HPC) scheduler.
    It extends the 'HPCSchedulingSimulator' class and the 'gym.Env' class from OpenAI Gym.

    Attributes:
    self.flatten_observation (bool): A flag indicate whether flatten the observation into one-dim vector or not.
    self.done (bool): A flag indicating whether the episode is over.
    self.passed_step(int): Number of steps passed in the current episode
    self.action_space (gym.spaces.Discrete): The action space of the environment, which is a discrete space
        containing MAX_QUEUE_SIZE possible actions.

    self.observation_space (gym.spaces): The observation space of the environment, which is a tuple
        space containing fixed-size arrays of values that represent the state of the environment.
    """

    def __init__(self,
                 workload_file: str,
                 flatten_observation: bool = False,
                 back_fill: bool = False,
                 skip: bool = False,
                 job_score_type: int = 0,
                 trace_sample_range: List = None,
                 seed: int = 0,
                 quiet: bool = True,
                 customized_trace_len_range: Tuple = None,
                 use_fixed_job_sequence: bool = False,
                 ):

        HPCSchedulingSimulator.__init__(self,
                                        workload_file=workload_file,
                                        back_fill=back_fill,
                                        skip=skip,
                                        job_score_type=job_score_type,
                                        trace_sample_range=trace_sample_range,
                                        seed=seed,
                                        quiet=quiet
                                        )

        self.flatten_observation = flatten_observation
        self.done = False
        self.passed_step = 0
        self.use_fixed_job_sequence = use_fixed_job_sequence
        self.customized_trace_len_range = customized_trace_len_range

        self.action_space = spaces.Discrete(MAX_QUEUE_SIZE)
        self.observation_space = None
        self.build_observation_space()

    def build_observation_space(self):
        """
        This function builds the observation space for the environment. This space consists of a tuple of two spaces:
        the job space which represents the jobs in the queue, and the cluster space which represents the state of the
        cluster.
        """
        job_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(MAX_QUEUE_SIZE, JOB_FEATURES),
            dtype=np.float32
        )

        cluster_space = spaces.box.Box(
            low=-1.0,
            high=1.0,
            shape=self.cluster.state.shape,
            dtype=np.float32
        )

        if not self.flatten_observation:
            self.observation_space = gym.spaces.tuple.Tuple(
                (
                    job_space,
                    cluster_space
                )
            )

            self.observation_space.n = np.sum(  # type: ignore
                [
                    np.prod(e.shape) if isinstance(e, gym.spaces.box.Box) else e.n
                    for e in self.observation_space
                ]
            )
        else:
            # Calculate the total size of the flattened observation space
            total_size = np.prod(job_space.shape) + np.prod(cluster_space.shape)

            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(total_size,),
                dtype=np.float32
            )

            self.observation_space.n = total_size

    def seed(self, seed=None):
        """
        This function sets the seed for this environment's random number generator.

        Parameters:
            seed (int, optional): The seed for the random number generator. Defaults to None.

        Returns:
            list of int: The seed for the random number generator.
        """
        return super().seed(seed)

    def _jobs_by_score(self, score_func) -> list[Job]:
        """
        This helper function sorts the job queue based on a given score function, and then returns the top jobs up to
        the maximum queue size.

        Parameters:
            score_func (function): The function used to score the jobs in the queue.

        Returns:
            list of Job: The top jobs from the job queue based on the score function.
        """
        self.job_queue.sort(key=lambda j: score_func(j))
        return self.job_queue[:MAX_QUEUE_SIZE]

    def _job_features(self, job: Job, epsilon: float = 1e-6) -> JobTransition:
        """
        Extracts and normalizes the features of a given job. If a feature value is not available
        (denoted by -1), it is set to 1. If the job fits in the cluster, the 'can_schedule_now' feature is set to 1,
        otherwise it is set to 0.

        Parameters:
            job (Job): The job whose features are to be extracted.
            epsilon (float, optional): A small value used to avoid division by zero errors. Defaults to 1e-6.

        Returns:
            JobTransition: A JobTransition object representing the features of the job.
        """
        submit_time = job.submit_time
        request_processors = job.request_number_of_processors
        run_time = job.run_time
        request_time = job.request_time
        wait_time = self.current_timestamp - submit_time

        # make sure that larger value is better.
        normalized_wait_time = min(float(wait_time) / float(MAX_WAIT_TIME), 1.0 - epsilon)
        normalized_request_time = min(float(request_time) / float(self.loads.max_exec_time), 1.0 - epsilon)
        normalized_run_time = min(float(run_time) / float(self.loads.max_exec_time), 1.0 - epsilon)
        normalized_request_procs = min(float(request_processors) / float(self.loads.max_procs), 1.0 - epsilon)
        normalized_submit_time = min(float(submit_time) / float(self.loads.end_time), 1.0 - epsilon)

        if job.request_memory == -1:
            normalized_request_memory = 1
        else:
            normalized_request_memory = min(float(job.request_memory) / float(self.loads.max_requested_memory),
                                            1.0 - epsilon)

        if job.user_id == -1:
            normalized_user_id = 1
        else:
            normalized_user_id = min(float(job.user_id) / float(self.loads.max_user_id), 1.0 - epsilon)

        if job.group_id == -1:
            normalized_group_id = 1
        else:
            normalized_group_id = min(float(job.group_id) / float(self.loads.max_group_id), 1.0 - epsilon)

        if job.executable_number == -1:
            normalized_executable_id = 1
        else:
            normalized_executable_id = min(
                float(job.executable_number) / float(self.loads.max_executable_number), 1.0 - epsilon)

        if self.cluster.fits(job):
            can_schedule_now = 1.0 - epsilon
        else:
            can_schedule_now = epsilon

        return JobTransition.from_list(
            [job, normalized_submit_time, normalized_wait_time, normalized_request_time, normalized_run_time,
             normalized_request_procs, normalized_request_memory, normalized_user_id, normalized_group_id,
             normalized_executable_id, can_schedule_now, 1])

    def _skip_features(self) -> JobTransition:
        """
        Returns a JobTransition object representing the features of a skipped job. If 'pivot_job'
        is True, the last feature is set to 0, otherwise it is set to 1.

        Returns:
            JobTransition: A JobTransition object representing the features of a skipped job.
        """
        if self.pivot_job:
            return JobTransition.from_list([None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
        else:
            return JobTransition.from_list([None, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0])

    def build_cluster_state(self) -> np.ndarray:
        """
        Returns the normalised state of the cluster as a numpy array.
        """
        state = []
        for i, node in enumerate(self.cluster.state):
            state.append(min(max(float(node) / float(self.loads.max_job_id), -1), 1.0))
        return np.array(state)

    def build_job_queue_state(self) -> np.ndarray:
        """
        Building the state of current visible job queue

        Returns:
            vector (np.ndarray): A numpy array representing the list of job transitions.
        """

        self.job_queue.sort(key=lambda j: self.scorer.fcfs_score(j))

        self.visible_jobs = []
        if len(self.job_queue) <= MAX_QUEUE_SIZE:
            self.visible_jobs = self.job_queue[:len(self.job_queue)]
        else:
            visible_jobs_sets = [
                self._jobs_by_score(self.scorer.f1_score),
                self._jobs_by_score(self.scorer.f2_score),
                self._jobs_by_score(self.scorer.sjf_score),
                self._jobs_by_score(self.scorer.smallest_score),
            ]

            shuffled = list(self.job_queue)
            self.np_random.shuffle(shuffled)
            visible_jobs_sets.append(shuffled[:MAX_QUEUE_SIZE])

            index = 0
            global_index = 0
            while index < MAX_QUEUE_SIZE:
                for visible_jobs in visible_jobs_sets:
                    job = visible_jobs[global_index]
                    if job not in self.visible_jobs:
                        self.visible_jobs.append(job)
                        index += 1
                global_index += 1

        self.obs_transitions = []
        add_skip = False
        for i in range(MAX_QUEUE_SIZE):
            if i < len(self.visible_jobs):
                job = self.visible_jobs[i]
                self.obs_transitions.append(self._job_features(job))
            elif self.skip and not add_skip:
                add_skip = True
                self.obs_transitions.append(self._skip_features())
            else:
                self.obs_transitions.append(JobTransition.from_list([None, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]))

        vector = np.vstack(np.array([pair.to_list() for pair in self.obs_transitions]), dtype='float32')

        return vector

    def get_reward(self) -> float:

        """
        Uses the 'post_process_matrices' method from the scorer to process the 'scheduled_rl' and obtain a reward for
        each job. It then returns the negative sum of these rewards as the total reward.

        Returns:
            rl_total (float): The total reward, calculated as the negative sum of individual job rewards.
        """

        complete_job_reward = self.scorer.post_process_matrices(copy.copy(self.scheduled_rl),
                                                                self.current_timestamp,
                                                                self.loads[self.start],
                                                                self.loads.max_procs)

        rl_total = -sum(complete_job_reward.values())
        return rl_total

    def get_observation(self) -> np.ndarray:
        """
        Get observation from the current state

        Returns:
             observation (gym.spaces.tuple or gym.spaces.Box):
             A tuple containing the initial state of the job queue and the cluster.
             Or a flatten vector containing or flatten information.
        """
        obs_job, obs_cluster = (self.build_job_queue_state(), self.build_cluster_state())

        if self.flatten_observation:
            observation = np.concatenate((np.reshape(obs_job, -1), np.reshape(obs_cluster, -1)), dtype='float32')

        else:
            observation = (obs_job, obs_cluster)

        return observation

    def reset(self, **kwargs) -> gym.spaces.tuple:
        """
        Resets the simulation environment by resetting the cluster and the loads, and initializing various instance
        variables to their starting values. It then optionally fills some pre-workloads and returns the initial
        observation.

        Parameters:
            kwargs (dict): A dictionary of keyword arguments. These arguments are not used in this function.

        Returns:
            observation: A vector containing the initial state of the job queue and the cluster.
        """

        self.reset_simulator(self.use_fixed_job_sequence, self.customized_trace_len_range)
        self.build_observation_space()

        self.done = False
        self.passed_step = 0
        observation = self.get_observation()

        return observation

    def step(self, a: int) -> list[gym.spaces.tuple, float, bool, tuple]:
        """
        This function is the core of the environment's interaction loop. It takes an action, checks and schedules
        the corresponding job, and then calculates the reward. It also builds the new observation and returns a list
        containing the new observation, the reward, a boolean indicating whether the episode is done, and a
        dictionary containing the current timestamp and the performance matrix.

        Parameters:
            a (int): The action taken by the agent, which is an integer representing the index of the job in the job
            queue.

        Returns:
            List [observation, reward, self.done, additional_info]:
                observation (gym.spaces.tuple): A tuple containing the new state of the job queue and the cluster.
                reward (float): The reward for taking action 'a'.
                self.done (bool): A boolean indicating whether the episode is done.
                additional_info (dict): A dictionary containing the current timestamp and the performance matrix.
        """

        self.done = self.check_then_schedule(a)
        reward = self.get_reward()
        observation = self.get_observation()

        return [
            observation,
            reward,
            self.done,
            {'current_timestamp': self.current_timestamp,
             'performance matrix': self.scheduled_rl,
             'start': self.start}
        ]
