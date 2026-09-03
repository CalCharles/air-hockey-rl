"""Selection/state manager for primitive TD3 exploration takeover.

Two primitives only:
  0) stand_still      — outputs near-zero action.
  1) same_direction   — samples one direction + magnitude, repeats that
                        directional move for the takeover window.

`same_direction` supports two sampling modes:
  - legacy (range fields unset): uniform unit direction (with a y-component
    weight bias) × uniform magnitude in [0.1, 1.0].
  - simulator-space range (all four range fields set): direction sampled from
    an angle range, magnitude from a displacement range, projected into the
    action box.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scripts.td3.helper.exploration_primitives import (
    actions_from_direction_and_magnitude,
    project_displacement_to_action_box,
    sample_simulator_displacements_from_ranges,
    sample_uniform_magnitude,
    sample_unit_directions,
    stand_still_actions,
)


@dataclass
class PrimitiveIds:
    STAND_STILL: int = 0
    SAME_DIRECTION_SMALL_MAG: int = 1


class PrimitiveExplorationSelector:
    def __init__(
        self,
        num_envs: int,
        chance: float,
        takeover_steps: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        direction_y_component_weight: float = 1.0,
        action_delta_x: float = 0.26,
        action_delta_y: float = 0.12,
        same_direction_min_angle_deg: float | None = None,
        same_direction_max_angle_deg: float | None = None,
        same_direction_min_magnitude: float | None = None,
        same_direction_max_magnitude: float | None = None,
    ):
        if takeover_steps <= 0:
            raise ValueError("takeover_steps must be > 0")
        self.num_envs = int(num_envs)
        self.chance = float(chance)
        self.takeover_steps = int(takeover_steps)
        self.device = device
        self.dtype = dtype
        self.direction_y_component_weight = float(max(direction_y_component_weight, 1e-6))
        self.action_delta_x = float(action_delta_x)
        self.action_delta_y = float(action_delta_y)
        self.same_direction_min_angle_deg = self._maybe_float(same_direction_min_angle_deg)
        self.same_direction_max_angle_deg = self._maybe_float(same_direction_max_angle_deg)
        self.same_direction_min_magnitude = self._maybe_float(same_direction_min_magnitude)
        self.same_direction_max_magnitude = self._maybe_float(same_direction_max_magnitude)
        self._validate_optional_range(
            "same_direction",
            self.same_direction_min_angle_deg,
            self.same_direction_max_angle_deg,
            self.same_direction_min_magnitude,
            self.same_direction_max_magnitude,
        )
        self.primitive_ids = PrimitiveIds()
        self.primitive_weights = torch.tensor(
            [0.5, 0.5], dtype=self.dtype, device=self.device
        )

        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.primitive_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.direction = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.magnitude = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)

    @staticmethod
    def _maybe_float(value: float | None) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _validate_optional_range(
        primitive_name: str,
        min_angle_deg: float | None,
        max_angle_deg: float | None,
        min_magnitude: float | None,
        max_magnitude: float | None,
    ) -> None:
        values = (min_angle_deg, max_angle_deg, min_magnitude, max_magnitude)
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ValueError(
                f"{primitive_name} simulator-space range requires angle min/max and magnitude min/max"
            )
        if min_magnitude is not None and min_magnitude < 0.0:
            raise ValueError(f"{primitive_name} min_magnitude must be >= 0")
        if (
            min_magnitude is not None
            and max_magnitude is not None
            and max_magnitude < min_magnitude
        ):
            raise ValueError(f"{primitive_name} max_magnitude must be >= min_magnitude")

    def _uses_simulator_space_range(self, primitive_name: str) -> bool:
        return getattr(self, f"{primitive_name}_min_angle_deg") is not None

    def _sample_simulator_range(
        self,
        primitive_name: str,
        count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return sample_simulator_displacements_from_ranges(
            count=count,
            min_angle_deg=getattr(self, f"{primitive_name}_min_angle_deg"),
            max_angle_deg=getattr(self, f"{primitive_name}_max_angle_deg"),
            min_magnitude=getattr(self, f"{primitive_name}_min_magnitude"),
            max_magnitude=getattr(self, f"{primitive_name}_max_magnitude"),
            max_delta_x=self.action_delta_x,
            max_delta_y=self.action_delta_y,
            device=self.device,
            dtype=self.dtype,
        )

    def _clear_mask(self, mask: torch.Tensor) -> None:
        if not torch.any(mask):
            return
        self.active[mask] = False
        self.steps_remaining[mask] = 0
        self.primitive_id[mask] = -1
        self.direction[mask] = 0
        self.magnitude[mask] = 0

    def reset(self, done_mask: torch.Tensor) -> None:
        mask = done_mask.to(device=self.device, dtype=torch.bool)
        self._clear_mask(mask)

    def set_primitive_weights(
        self,
        stand_still: float,
        same_direction: float,
    ) -> None:
        weights = torch.tensor(
            [stand_still, same_direction],
            dtype=self.dtype,
            device=self.device,
        )
        if torch.any(weights < 0):
            raise ValueError("Primitive weights must be non-negative.")
        weight_sum = weights.sum()
        if float(weight_sum.item()) <= 0.0:
            raise ValueError("At least one primitive weight must be > 0.")
        self.primitive_weights = weights / weight_sum

    def _activate_new_primitives(self) -> torch.Tensor:
        inactive = ~self.active
        if not torch.any(inactive):
            return inactive
        trigger = torch.rand(self.num_envs, device=self.device) < self.chance
        activate_mask = inactive & trigger
        if not torch.any(activate_mask):
            return activate_mask

        indices = torch.nonzero(activate_mask, as_tuple=False).squeeze(-1)
        count = int(indices.numel())
        sample_weights = self.primitive_weights.clone()
        weight_sum = sample_weights.sum()
        if float(weight_sum.item()) <= 0.0:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        sample_weights = sample_weights / weight_sum
        sampled_primitive = torch.multinomial(sample_weights, count, replacement=True).to(torch.long)

        self.active[indices] = True
        self.steps_remaining[indices] = self.takeover_steps
        self.primitive_id[indices] = sampled_primitive

        same_dir_mask = sampled_primitive == self.primitive_ids.SAME_DIRECTION_SMALL_MAG
        if torch.any(same_dir_mask):
            same_indices = indices[same_dir_mask]
            same_count = int(same_indices.numel())
            if self._uses_simulator_space_range("same_direction"):
                directions, magnitudes, _ = self._sample_simulator_range("same_direction", same_count)
                self.direction[same_indices] = directions
                self.magnitude[same_indices] = magnitudes
            else:
                self.direction[same_indices] = sample_unit_directions(
                    same_count,
                    self.device,
                    self.dtype,
                    y_component_weight=self.direction_y_component_weight,
                )
                self.magnitude[same_indices] = sample_uniform_magnitude(
                    same_count, max_magnitude=1.0, min_magnitude=0.1, device=self.device, dtype=self.dtype
                )

        return activate_mask

    def apply(
        self,
        proposed_actions: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        return_stats: bool = False,
    ):
        if proposed_actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected proposed_actions to have batch {self.num_envs}, got {proposed_actions.shape[0]}"
            )
        _ = self._activate_new_primitives()

        actions = proposed_actions.clone()
        active_indices = torch.nonzero(self.active, as_tuple=False).squeeze(-1)
        primitive_applied_count = int(active_indices.numel())
        primitive_horizontal_dominant_count = 0
        if active_indices.numel() == 0:
            if return_stats:
                return actions, {
                    "primitive_applied_count": 0,
                    "primitive_horizontal_dominant_count": 0,
                }
            return actions

        active_primitive = self.primitive_id[active_indices]
        stand_mask = active_primitive == self.primitive_ids.STAND_STILL
        if torch.any(stand_mask):
            stand_indices = active_indices[stand_mask]
            actions[stand_indices] = stand_still_actions(int(stand_indices.numel()), self.device, self.dtype)

        same_mask = active_primitive == self.primitive_ids.SAME_DIRECTION_SMALL_MAG
        if torch.any(same_mask):
            same_indices = active_indices[same_mask]
            if self._uses_simulator_space_range("same_direction"):
                same_displacements = actions_from_direction_and_magnitude(
                    self.direction[same_indices], self.magnitude[same_indices]
                )
                same_actions, _, _ = project_displacement_to_action_box(
                    same_displacements,
                    max_delta_x=self.action_delta_x,
                    max_delta_y=self.action_delta_y,
                )
                actions[same_indices] = same_actions
            else:
                actions[same_indices] = actions_from_direction_and_magnitude(
                    self.direction[same_indices], self.magnitude[same_indices]
                )

        actions = torch.clamp(actions, action_low, action_high)
        primitive_horizontal_dominant_count = int(
            (
                torch.abs(actions[active_indices, 0])
                > torch.abs(actions[active_indices, 1])
            ).sum().item()
        )

        self.steps_remaining[active_indices] = self.steps_remaining[active_indices] - 1
        finished_mask = self.active & (self.steps_remaining <= 0)
        self._clear_mask(finished_mask)
        if return_stats:
            return actions, {
                "primitive_applied_count": primitive_applied_count,
                "primitive_horizontal_dominant_count": primitive_horizontal_dominant_count,
            }
        return actions

    def state_dict(self) -> dict:
        return {
            "num_envs": self.num_envs,
            "chance": self.chance,
            "takeover_steps": self.takeover_steps,
            "same_direction_min_angle_deg": self.same_direction_min_angle_deg,
            "same_direction_max_angle_deg": self.same_direction_max_angle_deg,
            "same_direction_min_magnitude": self.same_direction_min_magnitude,
            "same_direction_max_magnitude": self.same_direction_max_magnitude,
            "primitive_weights": self.primitive_weights.detach().clone().cpu(),
            "active": self.active.detach().clone().cpu(),
            "steps_remaining": self.steps_remaining.detach().clone().cpu(),
            "primitive_id": self.primitive_id.detach().clone().cpu(),
            "direction": self.direction.detach().clone().cpu(),
            "magnitude": self.magnitude.detach().clone().cpu(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "primitive_weights" in state_dict:
            loaded_weights = state_dict["primitive_weights"].to(self.device)
            if loaded_weights.numel() == self.primitive_weights.numel():
                weight_sum = loaded_weights.sum().clamp_min(1e-8)
                self.primitive_weights.copy_(loaded_weights / weight_sum)
        self.same_direction_min_angle_deg = self._maybe_float(
            state_dict.get("same_direction_min_angle_deg", self.same_direction_min_angle_deg)
        )
        self.same_direction_max_angle_deg = self._maybe_float(
            state_dict.get("same_direction_max_angle_deg", self.same_direction_max_angle_deg)
        )
        self.same_direction_min_magnitude = self._maybe_float(
            state_dict.get("same_direction_min_magnitude", self.same_direction_min_magnitude)
        )
        self.same_direction_max_magnitude = self._maybe_float(
            state_dict.get("same_direction_max_magnitude", self.same_direction_max_magnitude)
        )
        # Mid-episode primitive state is not carried across resume; clear it.
        self.active.zero_()
        self.steps_remaining.zero_()
        self.primitive_id.fill_(-1)
        self.direction.zero_()
        self.magnitude.zero_()


class NumpyPrimitiveExplorationSelector:
    """numpy backend of `PrimitiveExplorationSelector` for the CPU rollout path.

    Same semantics, same public surface (`chance`, `apply`, `reset`,
    `set_primitive_weights`, `state_dict` / `load_state_dict` with the same
    keys and CPU-tensor values) so checkpoints interoperate with the torch
    class. Only the legacy sampling mode is implemented (uniform unit
    direction with y-weight × uniform magnitude in [0.1, 1]); the trainer
    falls back to the torch class when the simulator-space range fields are
    set. The torch class stays the reference implementation (it also serves
    the real-robot runtime); ~30 tiny torch CPU ops per step cost ~0.14 ms,
    the numpy version ~0.03 ms.
    """

    def __init__(
        self,
        num_envs: int,
        chance: float,
        takeover_steps: int,
        direction_y_component_weight: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if takeover_steps <= 0:
            raise ValueError("takeover_steps must be > 0")
        import numpy as np

        self._np = np
        self.num_envs = int(num_envs)
        self.chance = float(chance)
        self.takeover_steps = int(takeover_steps)
        self.direction_y_component_weight = float(max(direction_y_component_weight, 1e-6))
        self.primitive_ids = PrimitiveIds()
        self.rng = np.random.default_rng(seed)
        self.primitive_weights = np.array([0.5, 0.5], dtype=np.float64)
        self.active = np.zeros(self.num_envs, dtype=bool)
        self.steps_remaining = np.zeros(self.num_envs, dtype=np.int64)
        self.primitive_id = np.full((self.num_envs,), -1, dtype=np.int64)
        self.direction = np.zeros((self.num_envs, 2), dtype=np.float32)
        self.magnitude = np.zeros(self.num_envs, dtype=np.float32)

    def _clear(self, mask) -> None:
        if not mask.any():
            return
        self.active[mask] = False
        self.steps_remaining[mask] = 0
        self.primitive_id[mask] = -1
        self.direction[mask] = 0.0
        self.magnitude[mask] = 0.0

    def reset(self, done_mask) -> None:
        mask = self._np.asarray(done_mask, dtype=bool).reshape(-1)
        self._clear(mask)

    def set_primitive_weights(self, stand_still: float, same_direction: float) -> None:
        weights = self._np.array([stand_still, same_direction], dtype=self._np.float64)
        if (weights < 0).any():
            raise ValueError("Primitive weights must be non-negative.")
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("At least one primitive weight must be > 0.")
        self.primitive_weights = weights / total

    def _activate(self) -> None:
        np = self._np
        inactive = ~self.active
        if not inactive.any():
            return
        activate = inactive & (self.rng.random(self.num_envs) < self.chance)
        if not activate.any():
            return
        idx = np.flatnonzero(activate)
        sampled = self.rng.choice(2, size=idx.size, p=self.primitive_weights)
        self.active[idx] = True
        self.steps_remaining[idx] = self.takeover_steps
        self.primitive_id[idx] = sampled
        same = idx[sampled == self.primitive_ids.SAME_DIRECTION_SMALL_MAG]
        if same.size:
            angles = 2.0 * np.pi * self.rng.random(same.size)
            d = np.stack((np.cos(angles), np.sin(angles) * self.direction_y_component_weight), axis=-1)
            d /= np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-8)
            self.direction[same] = d.astype(np.float32)
            self.magnitude[same] = (0.1 + 0.9 * self.rng.random(same.size)).astype(np.float32)

    def apply(self, proposed_actions, action_low, action_high, return_stats: bool = False):
        """`proposed_actions`: (num_envs, 2) ndarray. Returns ndarray (+ stats)."""
        np = self._np
        actions = np.array(proposed_actions, dtype=np.float32, copy=True)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected batch {self.num_envs}, got {actions.shape[0]}")
        self._activate()
        active_idx = np.flatnonzero(self.active)
        applied = int(active_idx.size)
        horizontal = 0
        if applied:
            prim = self.primitive_id[active_idx]
            stand = active_idx[prim == self.primitive_ids.STAND_STILL]
            if stand.size:
                actions[stand] = 0.0
            same = active_idx[prim == self.primitive_ids.SAME_DIRECTION_SMALL_MAG]
            if same.size:
                actions[same] = self.direction[same] * self.magnitude[same][:, None]
            np.clip(actions, action_low, action_high, out=actions)
            horizontal = int((np.abs(actions[active_idx, 0]) > np.abs(actions[active_idx, 1])).sum())
            self.steps_remaining[active_idx] -= 1
            self._clear(self.active & (self.steps_remaining <= 0))
        else:
            np.clip(actions, action_low, action_high, out=actions)
        if return_stats:
            return actions, {
                "primitive_applied_count": applied,
                "primitive_horizontal_dominant_count": horizontal,
            }
        return actions

    def state_dict(self) -> dict:
        return {
            "num_envs": self.num_envs,
            "chance": self.chance,
            "takeover_steps": self.takeover_steps,
            "same_direction_min_angle_deg": None,
            "same_direction_max_angle_deg": None,
            "same_direction_min_magnitude": None,
            "same_direction_max_magnitude": None,
            "primitive_weights": torch.as_tensor(self.primitive_weights, dtype=torch.float32),
            "active": torch.as_tensor(self.active.copy()),
            "steps_remaining": torch.as_tensor(self.steps_remaining.copy()),
            "primitive_id": torch.as_tensor(self.primitive_id.copy()),
            "direction": torch.as_tensor(self.direction.copy()),
            "magnitude": torch.as_tensor(self.magnitude.copy()),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "primitive_weights" in state_dict:
            w = torch.as_tensor(state_dict["primitive_weights"]).detach().cpu().double().numpy().reshape(-1)
            if w.size == 2 and w.sum() > 0:
                self.primitive_weights = w / w.sum()
        # Mid-episode primitive state is not carried across resume; clear it.
        self.active[:] = False
        self.steps_remaining[:] = 0
        self.primitive_id[:] = -1
        self.direction[:] = 0.0
        self.magnitude[:] = 0.0
