"""TD3-style checkpoint directory helpers for RMA.

Each periodic / best save writes a directory:
  <run_dir>/phase{1,2}/checkpoint_<agent_steps>/
    args.yaml
    config.yaml
    model.pth   (phase 1) or model.ckpt (phase 2)
    multi_env_eval.json   (written by eval callback)
    eval_0.gif            (written by eval callback)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml


def sanitize_for_yaml(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_yaml(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_yaml(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def write_args_and_config(
    out_dir: str,
    args_dict: Optional[Dict[str, Any]],
    air_hockey_cfg: Optional[Dict[str, Any]],
) -> None:
    """Write TD3-style args.yaml + config.yaml into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    if args_dict is not None:
        with open(os.path.join(out_dir, "args.yaml"), "w") as f:
            yaml.dump(sanitize_for_yaml(args_dict), f, default_flow_style=False)
    if air_hockey_cfg is not None:
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.dump(sanitize_for_yaml(air_hockey_cfg), f, default_flow_style=False)


def save_weight_bundle(
    path: str,
    model: torch.nn.Module,
    running_mean_std: Optional[torch.nn.Module] = None,
    value_mean_std: Optional[torch.nn.Module] = None,
    sa_mean_std: Optional[torch.nn.Module] = None,
) -> str:
    """Save RMA ActorCritic + normalizer state dicts to ``path``."""
    weights: Dict[str, Any] = {"model": model.state_dict()}
    if running_mean_std is not None:
        weights["running_mean_std"] = running_mean_std.state_dict()
    if value_mean_std is not None:
        weights["value_mean_std"] = value_mean_std.state_dict()
    if sa_mean_std is not None:
        weights["sa_mean_std"] = sa_mean_std.state_dict()
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(weights, path)
    return path


def save_checkpoint_dir(
    ckpt_dir: str,
    *,
    model: torch.nn.Module,
    args_dict: Optional[Dict[str, Any]],
    air_hockey_cfg: Optional[Dict[str, Any]],
    model_filename: str = "model.pth",
    running_mean_std: Optional[torch.nn.Module] = None,
    value_mean_std: Optional[torch.nn.Module] = None,
    sa_mean_std: Optional[torch.nn.Module] = None,
) -> str:
    """Create a TD3-style checkpoint directory and return the model path."""
    write_args_and_config(ckpt_dir, args_dict, air_hockey_cfg)
    model_path = os.path.join(ckpt_dir, model_filename)
    return save_weight_bundle(
        model_path,
        model=model,
        running_mean_std=running_mean_std,
        value_mean_std=value_mean_std,
        sa_mean_std=sa_mean_std,
    )
