"""Wandb helpers shared by RMA phase-1 / phase-2 trainers."""

from __future__ import annotations

from typing import Any, Dict, Optional


def wandb_log(metrics: Dict[str, Any], step: int, prefix: str = "") -> None:
    """Best-effort wandb.log; no-op if wandb is unavailable or not initialized."""
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    payload = {f"{prefix}{k}": v for k, v in metrics.items()} if prefix else dict(metrics)
    wandb.log(payload, step=int(step))


def wandb_finish() -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()
