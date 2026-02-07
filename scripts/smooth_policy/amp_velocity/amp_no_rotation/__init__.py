"""
AMP (Adversarial Motion Priors) package for air hockey training.

This package implements AMP to learn smooth, natural motion policies by
combining reinforcement learning with imitation learning from expert demonstrations.
"""

from .discriminator import Discriminator
from .normalizer import Normalizer
from .replay_buffer import ReplayBuffer
from .demo_loader import DemoLoader

__all__ = [
    'Discriminator',
    'Normalizer',
    'ReplayBuffer',
    'DemoLoader',
]

__version__ = '1.0.0'
