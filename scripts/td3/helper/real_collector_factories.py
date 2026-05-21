"""Factories for real-collector TD3 helpers (decoupled from async_td3_real CLI types)."""

from __future__ import annotations

from typing import Protocol

import torch

from scripts.td3.helper.exploration_selector import (
    PrimitiveExplorationSelector,
)


class RealCollectorPrimitiveExplorationConfig(Protocol):
    exploration_primitive_chance_start: float
    exploration_primitive_chance: float
    exploration_primitive_chance_anneal_steps: int
    exploration_primitive_steps: int
    exploration_direction_y_component_weight: float
    exploration_action_delta_x: float
    exploration_action_delta_y: float
    exploration_same_direction_min_angle_deg: float
    exploration_same_direction_max_angle_deg: float
    exploration_same_direction_min_magnitude: float
    exploration_same_direction_max_magnitude: float


def _linear_anneal(start: float, end: float, step: int, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return end
    progress = min(max(step, 0) / float(anneal_steps), 1.0)
    return start + progress * (end - start)


def _primitive_exploration_chance_for_step(
    args: RealCollectorPrimitiveExplorationConfig, step: int
) -> float:
    return _linear_anneal(
        args.exploration_primitive_chance_start,
        args.exploration_primitive_chance,
        step,
        args.exploration_primitive_chance_anneal_steps,
    )


def build_primitive_exploration_selector_for_real_collector(
    args: RealCollectorPrimitiveExplorationConfig,
    device: torch.device,
    *,
    initial_total_steps: int = 0,
) -> PrimitiveExplorationSelector:
    return PrimitiveExplorationSelector(
        num_envs=1,
        chance=float(_primitive_exploration_chance_for_step(args, step=initial_total_steps)),
        takeover_steps=int(args.exploration_primitive_steps),
        device=device,
        dtype=torch.float32,
        direction_y_component_weight=float(args.exploration_direction_y_component_weight),
        action_delta_x=float(args.exploration_action_delta_x),
        action_delta_y=float(args.exploration_action_delta_y),
        same_direction_min_angle_deg=float(args.exploration_same_direction_min_angle_deg),
        same_direction_max_angle_deg=float(args.exploration_same_direction_max_angle_deg),
        same_direction_min_magnitude=float(args.exploration_same_direction_min_magnitude),
        same_direction_max_magnitude=float(args.exploration_same_direction_max_magnitude),
    )
