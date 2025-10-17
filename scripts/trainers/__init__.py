"""
Modular training framework for air hockey RL agents.

This package provides a clean, extensible architecture for training
different RL algorithms with shared common functionality.
"""

from .base_trainer import BaseTrainer
from .sac_trainer import SACTrainer  
from .ppo_trainer import PPOTrainer
from .trainer_factory import TrainerFactory

__all__ = [
    'BaseTrainer',
    'SACTrainer', 
    'PPOTrainer',
    'TrainerFactory'
]
