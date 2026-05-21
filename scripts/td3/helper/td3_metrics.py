"""TensorBoard metric helpers for TD3 simulation training.

See notes/docs/training/td3-algorithm.md for the training loop overview.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch.utils.tensorboard import SummaryWriter


def tensor_mean_items(values: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: tensor.mean().item() for key, tensor in values.items()}


def log_scalar_metrics(writer: SummaryWriter, metrics: Dict[str, float], global_step: int) -> None:
    for name, value in metrics.items():
        writer.add_scalar(name, value, global_step)


def initialize_train_metrics() -> Dict[str, float]:
    return {
        "losses/q_loss": 0.0,
        "losses/q_total_loss": 0.0,
        "losses/actor_loss": 0.0,
        "losses/q1_mean": 0.0,
        "losses/actor_norm_q_mean": 0.0,
        "debug/bellman_target_original_mean": 0.0,
        "debug/next_q_h_mean": 0.0,
        "rewards/sampled_reward_mean": 0.0,
        "rewards/sampled_reward_min": 0.0,
        "rewards/sampled_reward_std": 0.0,
        "rewards/sampled_reward_positive_count": 0.0,
        "rewards/sampled_reward_positive_fraction": 0.0,
        "rewards/sampled_reward_positive_mean": 0.0,
        "rewards/sampled_reward_positive_std": 0.0,
        "replay/per_beta": 0.0,
        "replay/per_is_weight_mean": 1.0,
        "replay/per_sampled_priority_mean": 0.0,
        "replay/per_priority_td_error_mean": 0.0,
        "replay/critic_per_sample_count": 0.0,
        "replay/critic_uniform_sample_count": 0.0,
        "replay/critic_per_sample_fraction": 0.0,
        "replay/success_buffer_size": 0.0,
        "replay/failure_buffer_size": 0.0,
        "replay/critic_success_sample_count": 0.0,
        "replay/critic_failure_sample_count": 0.0,
        "replay/critic_success_sample_fraction": 0.0,
        "replay/critic_failure_sample_fraction": 0.0,
        "replay/episode_return_success_threshold": 0.0,
        "replay/recent_episode_window_count": 0.0,
    }
