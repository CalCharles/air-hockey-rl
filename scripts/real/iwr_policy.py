"""IWR (interaction-weighted sampling) policy wrapper for real rollouts/eval.

IWR checkpoints are produced by the same external trainer as SGCRL and share
the identical actor architecture / ``agent`` dict layout. They differ only in
how the trainer saves them:

  * ``algorithm_name == 'interaction_weighted_sampling'`` (vs ``'sgcrl'``)
  * filename convention ``*_interaction_weighted_sampling_*.pkl``

Inference is the same tanh-squashed Gaussian actor on
``concat(state_30, desired_goal_2)``.
"""
from __future__ import annotations

from pathlib import Path

import torch

from scripts.real.sgcrl_policy import (
    SGCRLDeterministicPolicy,
    load_gcrl_style_deterministic_policy,
)

IWR_ALGORITHM_NAME = "interaction_weighted_sampling"


def load_iwr_deterministic_policy(
    *,
    model_path: str | Path,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
) -> SGCRLDeterministicPolicy:
    """Build a deterministic policy from an IWR ``.pkl`` checkpoint."""
    return load_gcrl_style_deterministic_policy(
        model_path=model_path,
        env_obs_dim=env_obs_dim,
        env_act_dim=env_act_dim,
        device=device,
        agent_label="iwr",
        expected_algorithm_name=IWR_ALGORITHM_NAME,
    )
