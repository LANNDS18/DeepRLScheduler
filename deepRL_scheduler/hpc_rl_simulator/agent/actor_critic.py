#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Categorical

from .models import PPOTorchModels


class CustomActorCriticPolicy(ActorCriticPolicy):
    """Custom Actor Critic Policy for HPC RL"""

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 activation_fn=nn.Tanh,
                 *args,
                 **kwargs):
        """Initialize the policy based on Stable Baselines' ActorCriticPolicy"""

        self.custom_kwargs = {
            'actor_model': kwargs.pop('actor_model', 'kernel'),
            'critic_model': kwargs.pop('critic_model', 'critic_linear_large'),
            'attn': kwargs.pop('attn', False),
            'obs_shape': observation_space.shape
        }

        # call parent constructor
        super(CustomActorCriticPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PPOTorchModels(**self.custom_kwargs)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get the action and the value for a given observation"""
        # Preprocess the observation if needed
        features = self.extract_features(observation)
        latent_pi, mask = self.mlp_extractor.forward_actor(features)
        actions = self.get_actions(latent_pi, mask, deterministic=deterministic)
        return actions

    def get_actions(self, latent_pi, mask, deterministic=False, ret_dist=False, sample=True):
        """Sample actions from the policy"""
        actions = torch.squeeze(self.action_net(latent_pi)) + (mask - 1) * 1000000
        distribution = Categorical(logits=actions)
        if sample:
            actions = torch.argmax(distribution.probs) if deterministic else distribution.sample()

        return (actions, distribution) if ret_dist else actions

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Forward pass in all the networks (actor and critic)
        Parameter:
            obs: Observation
            deterministic: Whether to use deterministic actions action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, mask, latent_value = self.mlp_extractor(features)
        actions, distribution = self.get_actions(latent_pi,
                                                 mask=mask,
                                                 deterministic=False,
                                                 ret_dist=True)
        # Evaluate the values for the given observations
        values = torch.flatten(self.value_net(latent_value))
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        """
        Evaluate actions according to the current policy, given the observations.
        Parameters:
            obs: Observation
            actions: Actions
        Returns
            estimated value, log likelihood of taking those actions, entropy of the action distribution.
        """

        features = self.extract_features(obs)
        latent_pi, mask, latent_value = self.mlp_extractor(features)

        _, distribution = self.get_actions(latent_pi,
                                           mask,
                                           sample=False,
                                           ret_dist=True)

        values = torch.flatten(self.value_net(latent_value))
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def build_actor_output(self):
        """
        Forward pass for actor network in the last layer
        """
        net = nn.Linear(in_features=self.mlp_extractor.latent_dim_pi, out_features=1)
        return net

    def build_value_output(self):
        """
        Forward pass for critic network in the last layer
        """
        net = nn.Linear(in_features=self.mlp_extractor.latent_dim_pi, out_features=1)
        return net

    def _build(self, lr_schedule) -> None:
        """
        Create the networks and the optimizer.
        Parameters:
            lr_schedule: Learning rate schedule lr_schedule(1) is the initial learning rate
        """

        self._build_mlp_extractor()
        self.action_net = self.build_actor_output()
        self.value_net = self.build_value_output()

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

            self.optimizer_kwargs['lr'] = lr_schedule(1)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(),
            **self.optimizer_kwargs
        )
