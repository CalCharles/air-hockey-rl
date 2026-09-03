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
    """Reduced metric set (2026-09 throughput cleanup).

    Only the scalars that are actually consulted when diagnosing a run are
    kept. Everything here is a cheap read of a value the graphed update step
    already produces; the dropped reward-distribution / sample-count metrics
    were either constants of the config or never looked at.
    """
    return {
        "losses/q_loss": 0.0,
        "losses/q_total_loss": 0.0,
        "losses/actor_loss": 0.0,
        "losses/q1_mean": 0.0,
        "losses/actor_norm_q_mean": 0.0,
        "debug/bellman_target_original_mean": 0.0,
        "debug/next_q_h_mean": 0.0,
        "rewards/sampled_reward_mean": 0.0,
        "replay/per_beta": 0.0,
        "replay/per_priority_td_error_mean": 0.0,
        "replay/success_buffer_size": 0.0,
        "replay/failure_buffer_size": 0.0,
        "replay/episode_return_success_threshold": 0.0,
    }
