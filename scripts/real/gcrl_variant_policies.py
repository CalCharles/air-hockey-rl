"""Goal-conditioned policy loaders for additional GCRL trainer algorithms.

Each variant shares the SGCRL/IWR actor layout (tanh-squashed Gaussian on
``concat(state, desired_goal)``) and differs only in the saved
``algorithm_name`` (and, for SAC-weighted-HER, the filename convention).
"""
from __future__ import annotations

from pathlib import Path

import torch

from scripts.real.sgcrl_policy import (
    SGCRLDeterministicPolicy,
    load_gcrl_style_deterministic_policy,
)

CRTR_ALGORITHM_NAME = "crtr"
SAC_GCRL_ALGORITHM_NAME = "sac"
SAC_HER_ALGORITHM_NAME = "sac_her"
PPO_GCRL_ALGORITHM_NAME = "ppo"


def _load_variant_policy(
    *,
    model_path: str | Path,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
    agent_label: str,
    expected_algorithm_name: str,
    filename_must_contain: str | None = None,
    filename_must_not_contain: str | None = None,
) -> SGCRLDeterministicPolicy:
    model_path = Path(model_path)
    basename = model_path.name.lower()
    if filename_must_contain is not None:
        if filename_must_contain.lower() not in basename:
            raise SystemExit(
                f"--agent {agent_label} expects a checkpoint filename containing "
                f"{filename_must_contain!r}, got {model_path.name!r}."
            )
    if filename_must_not_contain is not None:
        if filename_must_not_contain.lower() in basename:
            raise SystemExit(
                f"--agent {agent_label} expects a checkpoint filename without "
                f"{filename_must_not_contain!r}, got {model_path.name!r}."
            )
    return load_gcrl_style_deterministic_policy(
        model_path=model_path,
        env_obs_dim=env_obs_dim,
        env_act_dim=env_act_dim,
        device=device,
        agent_label=agent_label,
        expected_algorithm_name=expected_algorithm_name,
    )


def load_crtr_deterministic_policy(
    *,
    model_path: str | Path,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
) -> SGCRLDeterministicPolicy:
    """Build a deterministic policy from a CRTR ``.pkl`` checkpoint."""
    return _load_variant_policy(
        model_path=model_path,
        env_obs_dim=env_obs_dim,
        env_act_dim=env_act_dim,
        device=device,
        agent_label="crtr",
        expected_algorithm_name=CRTR_ALGORITHM_NAME,
    )


def load_sac_gcrl_deterministic_policy(
    *,
    model_path: str | Path,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
) -> SGCRLDeterministicPolicy:
    """Build a deterministic policy from a SAC-GCRL ``.pkl`` checkpoint."""
    return _load_variant_policy(
        model_path=model_path,
        env_obs_dim=env_obs_dim,
        env_act_dim=env_act_dim,
        device=device,
        agent_label="sac-gcrl",
        expected_algorithm_name=SAC_GCRL_ALGORITHM_NAME,
        filename_must_contain="sac_gcrl",
    )


def load_sac_her_deterministic_policy(
    *,
    model_path: str | Path,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
) -> SGCRLDeterministicPolicy:
    """Build a deterministic policy from a SAC-HER ``.pkl`` checkpoint."""
    return _load_variant_policy(
        model_path=model_path,
        env_obs_dim=env_obs_dim,
        env_act_dim=env_act_dim,
        device=device,
        agent_label="sac-her",
        expected_algorithm_name=SAC_HER_ALGORITHM_NAME,
        filename_must_contain="sac_her",
        filename_must_not_contain="weighted",
    )


def load_sac_weighted_her_deterministic_policy(
    *,
    model_path: str | Path,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
) -> SGCRLDeterministicPolicy:
    """Build a deterministic policy from a SAC-weighted-HER ``.pkl`` checkpoint."""
    return _load_variant_policy(
        model_path=model_path,
        env_obs_dim=env_obs_dim,
        env_act_dim=env_act_dim,
        device=device,
        agent_label="sac-weighted-her",
        expected_algorithm_name=SAC_HER_ALGORITHM_NAME,
        filename_must_contain="sac_weighted_her",
    )


def load_ppo_gcrl_deterministic_policy(
    *,
    model_path: str | Path,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
) -> SGCRLDeterministicPolicy:
    """Build a deterministic policy from a PPO-GCRL ``.pkl`` checkpoint."""
    return _load_variant_policy(
        model_path=model_path,
        env_obs_dim=env_obs_dim,
        env_act_dim=env_act_dim,
        device=device,
        agent_label="ppo-gcrl",
        expected_algorithm_name=PPO_GCRL_ALGORITHM_NAME,
        filename_must_contain="ppo_gcrl",
    )
