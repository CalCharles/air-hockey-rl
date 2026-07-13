# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation (RMA)
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Ported from hora/hora/algo/models/models.py for air-hockey Box2D.
# ProprioAdaptTConv is parameterized for air-hockey proprio history
# (default 6-dim: paddle/puck pos+valid context; no action dims).
# Policy input is obs ⊕ z only — no TD3-style last-action append.
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProprioAdaptTConv(nn.Module):
    """Temporal CNN adaptation module φ(·).

    Expects proprio history of temporal length T=30 so the Conv1d stack
    (k9/s2, k5/s1, k5/s1) yields temporal length 3 before the low-dim proj.
    """

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_channels: int = 32):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, hidden_channels, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(hidden_channels * 3, latent_dim)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, T, hidden)
        x = x.permute((0, 2, 1))  # (N, hidden, T)
        x = self.temporal_aggregation(x)  # (N, hidden, 3) for T=30
        x = self.low_dim_proj(x.flatten(1))
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['proprio_adapt']
        # Default 6 = paddle/puck pos+valid context entry (no action dims).
        proprio_hist_input_dim = kwargs.get('proprio_hist_input_dim', 6)
        latent_dim = self.priv_mlp[-1] if self.priv_info else 8
        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'])

            if self.priv_info_stage2:
                self.adapt_tconv = ProprioAdaptTConv(
                    input_dim=proprio_hist_input_dim,
                    latent_dim=latent_dim,
                )

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self.adapt_tconv(obs_dict['proprio_hist'])
                # during supervised training, extrin has gt label
                extrin_gt = self.env_mlp(obs_dict['priv_info']) if 'priv_info' in obs_dict else extrin
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.env_mlp(obs_dict['priv_info'])
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)

        x = self.actor_mlp(obs)
        value = self.value(x)
        mu = self.mu(x)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result
