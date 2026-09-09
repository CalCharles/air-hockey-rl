"""Critic and actor replay sampling with PER/uniform mixing and success/failure splits.

See notes/docs/training/replay-and-episodes.md for high-level design.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def critic_success_failure_counts(
    batch_size: int,
    success_fraction: float,
    success_available: bool,
    failure_available: bool,
) -> Tuple[int, int]:
    if not success_available and not failure_available:
        return 0, 0
    if not success_available:
        return 0, int(batch_size)
    if not failure_available:
        return int(batch_size), 0
    success_count = int(round(batch_size * float(success_fraction)))
    success_count = min(max(success_count, 0), int(batch_size))
    failure_count = int(batch_size) - success_count
    return success_count, failure_count


def concat_replay_samples(sample_chunks: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not sample_chunks:
        raise ValueError("Cannot concatenate empty replay sample chunks.")
    tensor_keys = (
        "observations",
        "next_observations",
        "actions",
        "prev_actions",
        "rewards",
        "dones",
        "weights",
        "sampled_priorities",
        "history",                  # Note: Actual values present only when context_len > 0
    )
    combined = {
        key: torch.cat([chunk[key] for chunk in sample_chunks], dim=0)
        for key in tensor_keys
        if key in sample_chunks[0]
    }
    return combined


def sample_critic_source_chunk(
    replay_buffer,
    sample_count: int,
    per_enabled: bool,
    per_beta: float,
    critic_per_fraction: float,
) -> Dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive for source chunk sampling.")
    if per_enabled:
        per_count = int(round(sample_count * float(critic_per_fraction)))
        per_count = min(max(per_count, 0), sample_count)
        uniform_count = sample_count - per_count
        sample_chunks: List[Dict[str, torch.Tensor]] = []
        per_indices = None
        if per_count > 0:
            per_data = replay_buffer.sample(per_count, beta=per_beta)
            sample_chunks.append(per_data)
            per_indices = per_data["indices"]
        if uniform_count > 0:
            sample_chunks.append(replay_buffer.sample_uniform(uniform_count))
        data = concat_replay_samples(sample_chunks)
        return {
            "data": data,
            "per_count": per_count,
            "uniform_count": uniform_count,
            "per_indices": per_indices,
        }

    data = replay_buffer.sample(sample_count)
    data["weights"] = torch.ones((sample_count,), dtype=torch.float32, device=data["rewards"].device)
    data["sampled_priorities"] = torch.zeros((sample_count,), dtype=torch.float32, device=data["rewards"].device)
    return {
        "data": data,
        "per_count": 0,
        "uniform_count": sample_count,
        "per_indices": None,
    }


def sample_actor_source_chunk(replay_buffer, sample_count: int, per_enabled: bool) -> Dict[str, torch.Tensor]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive for actor source chunk sampling.")
    if per_enabled:
        data = replay_buffer.sample_uniform(sample_count)
    else:
        data = replay_buffer.sample(sample_count)
    return data
