"""Metric-bundle and periodic-log helpers for TD3 training loops.

Three pure dict-returning functions for the per-step `train_metrics` bag
(merged once per cycle via the existing `log_scalar_metrics`):
- `build_target_q_debug_metrics` — Bellman-target / next-Q means
- `build_critic_metrics` — losses, sampled rewards, PER stats, replay sizes,
  multi-critic Q stats when num_critics > 2
- `build_actor_metrics` — actor loss, norm_q

One direct-writer for the every-500-step rollup of rolling-window episode
stats, puck-hit / e-stop rates, and exploration-primitive fractions:
- `write_periodic_episode_stats`

Reusable from `td3_training.py` and `td3_training_dr.py`.
"""

from collections import deque
from typing import Dict, List

import numpy as np
import torch


def build_target_q_debug_metrics(
    bellman_target_original: torch.Tensor,
    next_q_value_h: torch.Tensor,
) -> Dict[str, float]:
    return {
        "debug/bellman_target_original_mean": bellman_target_original.mean().item(),
        "debug/next_q_h_mean": next_q_value_h.mean().item(),
    }


def build_critic_metrics(
    *,
    qi_h_list: List[torch.Tensor],
    qi_loss_list: List[torch.Tensor],
    q_total_loss: torch.Tensor,
    q1_h: torch.Tensor,
    num_critics: int,
    sampled_rewards: torch.Tensor,
    sampled_weights: torch.Tensor,
    sampled_priorities: torch.Tensor,
    priority_td_error: torch.Tensor,
    per_beta: float,
    per_enabled: bool,
    per_sample_count: int,
    uniform_sample_count: int,
    batch_size: int,
    success_batch_count: int,
    failure_batch_count: int,
    len_success_rb: int,
    len_failure_rb: int,
    episode_return_success_threshold: float,
    recent_episode_window_count: int,
) -> Dict[str, float]:
    positive_reward_mask = sampled_rewards > 0.0
    positive_reward_count = float(positive_reward_mask.sum().item())
    minibatch_size = max(int(sampled_rewards.numel()), 1)
    positive_rewards = sampled_rewards[positive_reward_mask]
    priority_td_error_mean = (
        priority_td_error.mean().item()
        if per_enabled and per_sample_count > 0
        else 0.0
    )
    metrics: Dict[str, float] = {
        "losses/q_loss": sum(l.item() for l in qi_loss_list) / num_critics,
        "losses/q_total_loss": q_total_loss.item(),
        "losses/q1_mean": q1_h.mean().item(),
        "rewards/sampled_reward_mean": sampled_rewards.mean().item(),
        "rewards/sampled_reward_min": sampled_rewards.min().item(),
        "rewards/sampled_reward_std": sampled_rewards.std(unbiased=False).item(),
        "rewards/sampled_reward_positive_count": positive_reward_count,
        "rewards/sampled_reward_positive_fraction": (
            positive_reward_count / float(minibatch_size)
        ),
        "rewards/sampled_reward_positive_mean": (
            positive_rewards.mean().item() if positive_rewards.numel() > 0 else 0.0
        ),
        "rewards/sampled_reward_positive_std": (
            positive_rewards.std(unbiased=False).item()
            if positive_rewards.numel() > 0
            else 0.0
        ),
        "replay/per_beta": per_beta,
        "replay/per_is_weight_mean": sampled_weights.mean().item(),
        "replay/per_sampled_priority_mean": sampled_priorities.mean().item(),
        "replay/per_priority_td_error_mean": priority_td_error_mean,
        "replay/critic_per_sample_count": float(per_sample_count),
        "replay/critic_uniform_sample_count": float(uniform_sample_count),
        "replay/critic_per_sample_fraction": (
            float(per_sample_count) / float(max(batch_size, 1))
        ),
        "replay/success_buffer_size": float(len_success_rb),
        "replay/failure_buffer_size": float(len_failure_rb),
        "replay/critic_success_sample_count": float(success_batch_count),
        "replay/critic_failure_sample_count": float(failure_batch_count),
        "replay/critic_success_sample_fraction": (
            float(success_batch_count) / float(max(batch_size, 1))
        ),
        "replay/critic_failure_sample_fraction": (
            float(failure_batch_count) / float(max(batch_size, 1))
        ),
        "replay/episode_return_success_threshold": float(episode_return_success_threshold),
        "replay/recent_episode_window_count": float(recent_episode_window_count),
    }
    if num_critics > 2:
        for ci, qh in enumerate(qi_h_list, start=1):
            if ci == 1:
                continue  # already logged as q1_mean
            metrics[f"losses/q{ci}_mean"] = qh.mean().item()
        all_h = torch.stack(qi_h_list, dim=0)
        metrics["losses/q_min_mean"] = all_h.min(dim=0).values.mean().item()
        metrics["losses/q_mean_mean"] = all_h.mean().item()
    return metrics


def build_actor_metrics(
    actor_loss: torch.Tensor,
    norm_q: torch.Tensor,
) -> Dict[str, float]:
    return {
        "losses/actor_loss": actor_loss.item(),
        "losses/actor_norm_q_mean": norm_q.mean().item(),
    }


def write_periodic_episode_stats(
    writer,
    global_step: int,
    *,
    rolling_episode_stats_window: deque,
    rolling_step_stats_window: deque,
    interval_paddle_puck_collisions: float,
    interval_env_steps: int,
    interval_primitive_env_steps: int,
    interval_primitive_horizontal_env_steps: int,
) -> None:
    """Emit the every-500-step rollup: rolling-window episode summaries,
    per-env-step puck-hit rates, and exploration-primitive fractions.

    Caller is responsible for resetting the interval_* counters after this call.
    """
    if rolling_episode_stats_window:
        rolling_returns = [item[1] for item in rolling_episode_stats_window]
        rolling_lengths = [item[2] for item in rolling_episode_stats_window]
        rolling_success = [item[3] for item in rolling_episode_stats_window]
        avg_return = float(np.mean(rolling_returns))
        min_return = float(np.min(rolling_returns))
        max_return = float(np.max(rolling_returns))
        avg_success = float(np.mean(rolling_success))
        avg_episode_length = float(np.mean(rolling_lengths))
        print(
            f"Step {global_step}: Rolling(2k) Avg Return: {avg_return:.2f}, "
            f"Min: {min_return:.2f}, Max: {max_return:.2f}, "
            f"Success Rate: {avg_success:.2f}, Avg Episode Length: {avg_episode_length:.2f}, "
            f"Episodes: {len(rolling_episode_stats_window)}"
        )
        writer.add_scalar("charts/avg_episodic_return", avg_return, global_step)
        writer.add_scalar("charts/min_episodic_return", min_return, global_step)
        writer.add_scalar("charts/max_episodic_return", max_return, global_step)
        writer.add_scalar("charts/avg_success_rate", avg_success, global_step)
        writer.add_scalar("charts/rolling2k_avg_episode_return", avg_return, global_step)
        writer.add_scalar("charts/rolling2k_avg_episode_length", avg_episode_length, global_step)
        writer.add_scalar("charts/rolling2k_episode_count", len(rolling_episode_stats_window), global_step)
    else:
        print(f"Step {global_step}: No episodes in rolling 2k-step window")

    rolling_window_env_steps = int(sum(item[1] for item in rolling_step_stats_window))
    rolling_window_puck_hits = float(sum(item[2] for item in rolling_step_stats_window))
    rolling_puck_hits_per_env_step = (
        rolling_window_puck_hits / float(rolling_window_env_steps)
        if rolling_window_env_steps > 0
        else 0.0
    )
    print(
        f"Step {global_step}: Rolling(2k) Puck Hits: {int(rolling_window_puck_hits)}, "
        f"Puck Hits/env-step: {rolling_puck_hits_per_env_step:.4f}"
    )
    writer.add_scalar("charts/rolling2k_puck_hits_total", rolling_window_puck_hits, global_step)
    writer.add_scalar("charts/rolling2k_puck_hits_per_env_step", rolling_puck_hits_per_env_step, global_step)

    collisions_per_env_step = (
        interval_paddle_puck_collisions / max(interval_env_steps, 1)
        if interval_env_steps > 0
        else 0.0
    )
    print(
        f"Step {global_step}: Paddle-Puck Collisions (last interval): "
        f"{int(interval_paddle_puck_collisions)} total, {collisions_per_env_step:.4f} per env-step"
    )
    writer.add_scalar(
        "contacts/interval_paddle_puck_collisions_total",
        interval_paddle_puck_collisions, global_step,
    )
    writer.add_scalar(
        "contacts/interval_paddle_puck_collisions_per_env_step",
        collisions_per_env_step, global_step,
    )
    primitive_fraction = (
        interval_primitive_env_steps / max(interval_env_steps, 1)
        if interval_env_steps > 0
        else 0.0
    )
    primitive_horizontal_fraction = (
        interval_primitive_horizontal_env_steps / max(interval_primitive_env_steps, 1)
        if interval_primitive_env_steps > 0
        else 0.0
    )
    print(
        f"Step {global_step}: Primitive Actions (last interval): "
        f"{interval_primitive_env_steps}/{interval_env_steps} env-steps ({primitive_fraction:.4f}), "
        f"horizontal-dominant: {interval_primitive_horizontal_env_steps}/{interval_primitive_env_steps} "
        f"({primitive_horizontal_fraction:.4f})"
    )
    writer.add_scalar(
        "exploration/interval_primitive_env_steps",
        interval_primitive_env_steps, global_step,
    )
    writer.add_scalar(
        "exploration/interval_primitive_env_step_fraction",
        primitive_fraction, global_step,
    )
    writer.add_scalar(
        "exploration/interval_primitive_horizontal_env_steps",
        interval_primitive_horizontal_env_steps, global_step,
    )
    writer.add_scalar(
        "exploration/interval_primitive_horizontal_fraction",
        primitive_horizontal_fraction, global_step,
    )
