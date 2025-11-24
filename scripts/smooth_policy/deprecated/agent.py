import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Simple MLP Gaussian Policy + Critic for PPO
class Agent(nn.Module):
    def __init__(self, envs, init_reward_scaling=1.0):
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        # reward scaling is tied to the agent
        reward_scaling = torch.tensor(init_reward_scaling)
        self.register_buffer("reward_scaling", reward_scaling)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def forward(self, x):
        # check if x is a tensor, and if need unsqueeze
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return self.get_action_and_value(x)[0] # just return the action, as a tensor

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_mean(self, x):
        return self.actor_mean(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)