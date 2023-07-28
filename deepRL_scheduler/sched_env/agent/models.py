import numpy as np
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from torch.nn.functional import relu, softmax

from ..env.scheduler_simulator import JOB_FEATURES, MAX_QUEUE_SIZE

available_models = ['kernel',
                    'conv',
                    'critic_lg',
                    'critic_un_lg',
                    'critic_un',
                    'critic_sm']


class PPOTorchModels(nn.Module):
    """Custom model for PPO"""

    def __init__(self, actor_model, critic_model, attn):
        """ Initialize the custom model
        :param actor_model: (str) the name of the actor model
        :param critic_model: (str) the name of the critic model"""

        super(PPOTorchModels, self).__init__()
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.m = int(np.sqrt(MAX_QUEUE_SIZE))
        self.latent_dim_pi = JOB_FEATURES
        self.latent_dim_vf = JOB_FEATURES
        self.attn = attn

        self.attn_layer = MultiheadAttention(JOB_FEATURES - 1, 1, batch_first=True)

        self.kernel_net = nn.Sequential(
            nn.Linear(in_features=JOB_FEATURES - 1, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=self.latent_dim_pi),
            nn.ReLU()
        )

        self.lenet = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=16, out_features=self.latent_dim_pi)
        )

        self.critic_lg1 = nn.Sequential(
            nn.Linear(in_features=JOB_FEATURES - 1, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1)
        )

        self.critic_lg2 = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.latent_dim_vf),
            nn.ReLU()
        )

        self.critic_un_lg = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE * (JOB_FEATURES - 1), out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=self.latent_dim_vf)
        )

        self.critic_un = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE * (JOB_FEATURES - 1), out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=self.latent_dim_vf)
        )

        self.critic_sm = nn.Sequential(
            nn.Linear(in_features=MAX_QUEUE_SIZE * (JOB_FEATURES - 1), out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=self.latent_dim_vf)
        )

        self.models = {
            'kernel': self.kernel_net,
            'conv': self.lenet,
            'critic_lg': [self.critic_lg1, self.critic_lg2],
            'critic_un_lg': self.critic_un_lg,
            'critic_un': self.critic_un,
            'critic_sm': self.critic_sm,
        }

    @staticmethod
    def list_models():
        return available_models

    def forward(self, observation):
        """Forward pass for both actor and critic"""
        pi, mask = self.forward_actor(observation)
        return pi, mask, self.forward_critic(observation)

    def forward_actor(self, x):
        """Forward pass for actor
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor"""
        x_job = x[:, :JOB_FEATURES * MAX_QUEUE_SIZE]

        x_cluster = x[:, JOB_FEATURES * MAX_QUEUE_SIZE:]

        if self.actor_model == 'conv':
            x = torch.reshape(x_job, (-1, self.m, self.m, JOB_FEATURES))
            x = self.models[self.actor_model](x)
        elif self.actor_model in ['kernel', 'attn']:
            x = x_job.reshape(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
            mask = x[:, :, -1]
            mask = mask.squeeze()
            x = x[:, :, :-1]
            if self.attn:
                queries = relu(nn.Linear(in_features=JOB_FEATURES - 1, out_features=32)(x))
                keys = relu(nn.Linear(in_features=JOB_FEATURES - 1, out_features=32)(x))
                values = relu(nn.Linear(in_features=JOB_FEATURES - 1, out_features=32)(x))
                output = self.attn_layer(queries, keys, values)[0]
                x = softmax(output, dim=1)
                x = relu(nn.Linear(in_features=x.shape[-1], out_features=16)(x))
                x = relu(nn.Linear(in_features=16, out_features=self.latent_dim_pi)(x))
            else:
                x = self.models[self.actor_model](x)
            return x, mask
        else:
            raise NotImplementedError(f'Actor model: {self.actor_model} not implemented')

        return x

    def forward_critic(self, x):
        """Forward pass for critic
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor"""
        x_reshape = x[:, :JOB_FEATURES * MAX_QUEUE_SIZE]

        if not self.critic_model == 'critic_lg':
            x = torch.reshape(x_reshape, shape=(-1, MAX_QUEUE_SIZE, JOB_FEATURES))
            x = x[:, :, :-1]
            x = torch.reshape(x, shape=(-1, MAX_QUEUE_SIZE * (JOB_FEATURES - 1)))
            x = self.models[self.critic_model](x)
        else:
            x = x_reshape.reshape(-1, MAX_QUEUE_SIZE, JOB_FEATURES)
            x = x[:, :, :-1]
            x = self.models[self.critic_model][0](x)
            x = torch.squeeze(x, dim=2)
            x = self.models[self.critic_model][1](x)
        return x
