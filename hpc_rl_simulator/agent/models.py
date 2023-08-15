#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.functional import relu, softmax

from ..env.scheduler_simulator import JOB_FEATURES, MAX_QUEUE_SIZE

available_models = ['actor_kernel',
                    'actor_conv',
                    'actor_kernel_attention',
                    'actor_mlp',
                    'critic_linear_large',
                    'critic_linear_deep',
                    'critic_linear_small']


class Kernel_Attention_Policy(nn.Module):
    def __init__(self, latent_dim_pi):
        super(Kernel_Attention_Policy, self).__init__()
        self.query_layer = nn.Linear(in_features=JOB_FEATURES + latent_dim_pi, out_features=32)
        self.key_layer = nn.Linear(in_features=JOB_FEATURES + latent_dim_pi, out_features=32)
        self.value_layer = nn.Linear(in_features=JOB_FEATURES + latent_dim_pi, out_features=32)
        self.attn_layer = nn.MultiheadAttention(embed_dim=32, num_heads=4)
        self.fc1 = nn.Linear(in_features=32, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=latent_dim_pi)

    def forward(self, x_concat):
        queries = relu(self.query_layer(x_concat))
        keys = relu(self.key_layer(x_concat))
        values = relu(self.value_layer(x_concat))
        output, _ = self.attn_layer(queries, keys, values)
        x_emb = softmax(output, dim=1)
        x_emb = relu(self.fc1(x_emb))
        x = relu(self.fc2(x_emb))
        return x


class PPOTorchModels(nn.Module):
    """Custom model for PPO"""

    def __init__(self, actor_model, critic_model, obs_shape):
        """ Initialize the custom model"""

        super(PPOTorchModels, self).__init__()
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.latent_dim_pi = JOB_FEATURES
        self.latent_dim_vf = JOB_FEATURES

        self.obs_shape = obs_shape
        self.cluster_size = self.obs_shape[0] - MAX_QUEUE_SIZE * JOB_FEATURES

        self.actor_kernel = nn.Sequential(
            nn.Linear(in_features=JOB_FEATURES + self.latent_dim_pi, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=self.latent_dim_pi),
            nn.ReLU()
        )

        self.actor_kernel_attention = Kernel_Attention_Policy(self.latent_dim_pi)

        self.critic_linear_encoder_large = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE * JOB_FEATURES + self.cluster_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.latent_dim_vf)
        )

        self.cluster_encoder = nn.Sequential(
            nn.Linear(in_features=self.cluster_size, out_features=self.latent_dim_pi)
        )

        self.critic_linear_encoder_deep = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE * JOB_FEATURES + self.cluster_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.latent_dim_vf)
        )

        self.critic_linear_encoder_small = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE * JOB_FEATURES + self.cluster_size, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.latent_dim_vf)
        )

        self.models = {
            'actor_kernel': self.actor_kernel,
            'actor_kernel_attention': self.actor_kernel_attention,
            'critic_linear_large': self.critic_linear_encoder_large,
            'critic_linear_deep': self.critic_linear_encoder_deep,
            'critic_linear_small': self.critic_linear_encoder_small,
        }

        self.display_model_name()

    @staticmethod
    def list_models():
        return available_models

    def display_model_name(self):
        print(f'Actor Model: {self.actor_model}, Critic Model: {self.critic_model}')

    def forward(self, observation):
        """Forward pass for both actor and critic"""
        pi = self.forward_actor(observation)
        return pi, self.forward_critic(observation)

    def forward_actor(self, x):
        """Forward pass for actor
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor"""
        x_jobs = x[:, :-self.cluster_size]
        x_cluster = x[:, -self.cluster_size:]

        if self.actor_model in ['actor_kernel', 'actor_kernel_attention']:
            x_jobs = torch.reshape(x_jobs, shape=(-1, MAX_QUEUE_SIZE, JOB_FEATURES))
            x_cluster = self.cluster_encoder(x_cluster)
            x_cluster = x_cluster.unsqueeze(1).repeat(1, x_jobs.shape[1], 1)
            x_concat = torch.concat((x_jobs, x_cluster), dim=2)
            x = self.models[self.actor_model](x_concat)
            return x
        else:
            raise NotImplementedError(f'Actor model: {self.actor_model} not implemented')

    def forward_critic(self, x):
        """Forward pass for critic
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor"""
        x_jobs = x[:, :-self.cluster_size]
        x_cluster = x[:, -self.cluster_size:]

        x_cluster = torch.reshape(x_cluster, shape=(-1, self.cluster_size))
        x_concat = torch.cat((x_jobs, x_cluster), dim=-1)
        x = self.models[self.critic_model](x_concat)

        return x
