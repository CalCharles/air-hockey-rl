"""Scratch actor with a learned absolute aim point in the canonical workspace.

The environment still receives its original normalized displacement command.
Only policy output coordinates change; no target is sent around the canonical
action mapping, PID or two-step target smoother. No base/residual controller.
"""
from __future__ import annotations

import numpy as np
import torch

from scripts.td3.deterministic_agent import DeterministicAgent


class AbsoluteTargetAgent(DeterministicAgent):
    def __init__(self, envs, **kwargs):
        super().__init__(envs, **kwargs)
        if int(np.prod(envs.single_observation_space.shape)) not in (30, 32):
            raise ValueError("Absolute target actor requires canonical history observations")
        if int(np.prod(envs.single_action_space.shape)) != 2:
            raise ValueError("Absolute target actor requires two displacement actions")
        # Public canonical geometry/command scales, not sampled episode physics.
        # x bounds: [-.83, -.45] + 1.3359; y: [-.33, .39] - .0376.
        self.register_buffer("target_center", torch.tensor([.6959, -.0076]))
        self.register_buffer("target_half_span", torch.tensor([.19, .36]))
        self.register_buffer("displacement_scales", torch.tensor([.26, .12]))

    def get_action(self, x):
        aim = self.target_center + self.target_half_span * torch.tanh(self.get_action_mean(x))
        # Read only the delayed observed paddle xy, never current simulator state.
        displacement = (aim - x[..., 12:14]) / self.displacement_scales
        return torch.tanh(displacement) * self.action_scale + self.action_bias
