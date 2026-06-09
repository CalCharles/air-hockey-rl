import numpy as np
import torch
import torch.nn as nn

'''
This is an abstract class that serves as the base for transformers in RL.
'''

class TrajectoryModel(nn.Module):
    """
    Abstract base class for transformer-based context encoders in RL.

    Subclasses take a sequence of observations and produce a context vector
    that can be appended to the actor's policy observation.

    Input shape:  (batch, seq_len, obs_dim)
    Output shape: (batch, context_dim)
    """

    def __init__(self, obs_dim: int, context_dim: int, max_length: int | None = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_dim = context_dim
        self.max_length = max_length

    def forward(self, obs_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_sequence: (batch, seq_len, obs_dim)
        Returns:
            context_vector: (batch, context_dim)
        """
        raise NotImplementedError
