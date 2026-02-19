import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

# default constants
LOG_STD_MIN = -1
LOG_STD_MAX = 1
EPS = 1e-6

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def build_activation(activation_type: str, leaky_relu_negative_slope: float = 0.01) -> nn.Module:
    activation_type = activation_type.lower()
    if activation_type == "tanh":
        return nn.Tanh()
    if activation_type == "leaky_relu":
        return nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
    raise ValueError(f"Unsupported activation_type: {activation_type}. Use 'tanh' or 'leaky_relu'.")

# Simple MLP Gaussian Policy + Critic for PPO (actions clipped into some pre-determined range)
class Agent(nn.Module):
    def __init__(
        self,
        envs,
        action_scale=1.0,
        action_bias=0.0,
        hidden_size=64,
        activation_type="tanh",
        leaky_relu_negative_slope=0.01,
    ): # preliminary calculation
        super().__init__()
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        act_dim = int(np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            build_activation(activation_type, leaky_relu_negative_slope),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            build_activation(activation_type, leaky_relu_negative_slope),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            build_activation(activation_type, leaky_relu_negative_slope),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            build_activation(activation_type, leaky_relu_negative_slope),
        )
        self.actor_mean_head = layer_init(nn.Linear(hidden_size, act_dim), std=0.01) # action initially close to 0, exploration guided by logstd
        # self.actor_logstd_head = layer_init(nn.Linear(64, act_dim), std=3)

        # forget about per-state logstd, just use a fixed one for now
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.register_buffer("action_scale", torch.tensor(action_scale))
        self.register_buffer("action_bias", torch.tensor(action_bias))

        self.register_buffer("LOG_STD_MIN", torch.tensor(LOG_STD_MIN))
        self.register_buffer("LOG_STD_MAX", torch.tensor(LOG_STD_MAX))
        self.register_buffer("EPS", torch.tensor(EPS))

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_mean_and_logstd(self, x):
        x = self.actor(x)
        mean = self.actor_mean_head(x)

        # logstd = torch.tanh(self.actor_logstd_head(x))
        # logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (logstd + 1)

        logstd = torch.tanh(self.actor_logstd.expand_as(mean))
        logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (logstd + 1)

        return mean, logstd

    def forward(self, x):
        with torch.no_grad(): # only for evaluation
            # check if x is a tensor, and if need unsqueeze
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            return self.get_action_and_value(x)[0] # just return the action, as a tensor

    def get_action_and_value(self, x, action=None):
        action_mean, action_logstd = self.get_action_mean_and_logstd(x)
        action_std = torch.exp(action_logstd)
        normal = Normal(action_mean, action_std)

        # # get actions
        if action is None:
            x_t = normal.rsample()
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
        else:
            y_t = (action - self.action_bias) / self.action_scale
            x_t = torch.atanh(torch.clamp(y_t, -1 + self.EPS, 1 - self.EPS)) # reverse tanh 
        
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + self.EPS)
        log_prob = log_prob.sum(1)
        mean = torch.tanh(action_mean) * self.action_scale + self.action_bias

        return action, log_prob, mean, self.critic(x)
