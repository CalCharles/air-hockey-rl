"""Selection/state manager for primitive TD3 exploration takeover."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scripts.td3.helper.exploration_primitives import (
    actions_from_direction_and_magnitude,
    max_magnitude_for_directions_in_action_box,
    project_displacement_to_action_box,
    sample_directions_from_angle_range,
    sample_simulator_displacements_from_ranges,
    sample_target_distances,
    sample_uniform_magnitude,
    sample_unit_directions,
    stand_still_actions,
)


@dataclass
class PrimitiveIds:
    STAND_STILL: int = 0
    SAME_DIRECTION_SMALL_MAG: int = 1
    Y_ALIGNED_DIRECTION: int = 2
    TARGET_POSITION_DIRECTIONAL: int = 3


class PrimitiveExplorationSelector:
    def __init__(
        self,
        num_envs: int,
        chance: float,
        takeover_steps: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        direction_y_component_weight: float = 1.0,
        target_min_distance: float = 0.2,
        target_max_distance: float = 0.5,
        target_action_delta_x: float = 0.26,
        target_action_delta_y: float = 0.12,
        same_direction_min_angle_deg: float | None = None,
        same_direction_max_angle_deg: float | None = None,
        same_direction_min_magnitude: float | None = None,
        same_direction_max_magnitude: float | None = None,
        y_aligned_min_angle_deg: float | None = None,
        y_aligned_max_angle_deg: float | None = None,
        y_aligned_min_magnitude: float | None = None,
        y_aligned_max_magnitude: float | None = None,
        target_position_directional_min_angle_deg: float | None = None,
        target_position_directional_max_angle_deg: float | None = None,
        target_position_directional_min_magnitude: float | None = None,
        target_position_directional_max_magnitude: float | None = None,
        target_takeover_steps: int | None = None,
    ):
        if takeover_steps <= 0:
            raise ValueError("takeover_steps must be > 0")
        self.num_envs = int(num_envs)
        self.chance = float(chance)
        self.takeover_steps = int(takeover_steps)
        self.device = device
        self.dtype = dtype
        self.direction_y_component_weight = float(max(direction_y_component_weight, 1e-6))
        self.target_min_distance = float(target_min_distance)
        self.target_max_distance = float(target_max_distance)
        self.target_action_delta_x = float(target_action_delta_x)
        self.target_action_delta_y = float(target_action_delta_y)
        self.same_direction_min_angle_deg = self._maybe_float(same_direction_min_angle_deg)
        self.same_direction_max_angle_deg = self._maybe_float(same_direction_max_angle_deg)
        self.same_direction_min_magnitude = self._maybe_float(same_direction_min_magnitude)
        self.same_direction_max_magnitude = self._maybe_float(same_direction_max_magnitude)
        self.y_aligned_min_angle_deg = self._maybe_float(y_aligned_min_angle_deg)
        self.y_aligned_max_angle_deg = self._maybe_float(y_aligned_max_angle_deg)
        self.y_aligned_min_magnitude = self._maybe_float(y_aligned_min_magnitude)
        self.y_aligned_max_magnitude = self._maybe_float(y_aligned_max_magnitude)
        self.target_position_directional_min_angle_deg = self._maybe_float(
            target_position_directional_min_angle_deg
        )
        self.target_position_directional_max_angle_deg = self._maybe_float(
            target_position_directional_max_angle_deg
        )
        self.target_position_directional_min_magnitude = self._maybe_float(
            target_position_directional_min_magnitude
        )
        self.target_position_directional_max_magnitude = self._maybe_float(
            target_position_directional_max_magnitude
        )
        self.target_takeover_steps = int(target_takeover_steps or takeover_steps)
        if self.target_min_distance < 0.0:
            raise ValueError("target_min_distance must be >= 0")
        if self.target_max_distance < self.target_min_distance:
            raise ValueError("target_max_distance must be >= target_min_distance")
        if self.target_takeover_steps <= 0:
            raise ValueError("target_takeover_steps must be > 0")
        self._validate_optional_range(
            "same_direction",
            self.same_direction_min_angle_deg,
            self.same_direction_max_angle_deg,
            self.same_direction_min_magnitude,
            self.same_direction_max_magnitude,
        )
        self._validate_optional_range(
            "y_aligned",
            self.y_aligned_min_angle_deg,
            self.y_aligned_max_angle_deg,
            self.y_aligned_min_magnitude,
            self.y_aligned_max_magnitude,
        )
        self._validate_optional_range(
            "target_position_directional",
            self.target_position_directional_min_angle_deg,
            self.target_position_directional_max_angle_deg,
            self.target_position_directional_min_magnitude,
            self.target_position_directional_max_magnitude,
        )
        self.primitive_ids = PrimitiveIds()
        self.primitive_weights = torch.tensor(
            [0.25, 0.25, 0.25, 0.25], dtype=self.dtype, device=self.device
        )

        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.primitive_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.direction = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.magnitude = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        self.target_position = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.target_displacement = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.target_action = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)

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
            max_delta_x=self.target_action_delta_x,
            max_delta_y=self.target_action_delta_y,
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
        self.target_position[mask] = 0
        self.target_displacement[mask] = 0
        self.target_action[mask] = 0

    def reset(self, done_mask: torch.Tensor) -> None:
        mask = done_mask.to(device=self.device, dtype=torch.bool)
        self._clear_mask(mask)

    def set_primitive_weights(
        self,
        stand_still: float,
        same_direction: float,
        y_aligned: float,
        target_position_directional: float = 0.0,
    ) -> None:
        weights = torch.tensor(
            [
                stand_still,
                same_direction,
                y_aligned,
                target_position_directional,
            ],
            dtype=self.dtype,
            device=self.device,
        )
        if torch.any(weights < 0):
            raise ValueError("Primitive weights must be non-negative.")
        weight_sum = weights.sum()
        if float(weight_sum.item()) <= 0.0:
            raise ValueError("At least one primitive weight must be > 0.")
        self.primitive_weights = weights / weight_sum

    def _sample_y_aligned_directions(self, y_alignment_sign: torch.Tensor) -> torch.Tensor:
        count = int(y_alignment_sign.numel())
        if self._uses_simulator_space_range("y_aligned"):
            directions = sample_directions_from_angle_range(
                count=count,
                min_angle_deg=self.y_aligned_min_angle_deg,
                max_angle_deg=self.y_aligned_max_angle_deg,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            directions = sample_unit_directions(
                count,
                self.device,
                self.dtype,
                y_component_weight=self.direction_y_component_weight,
            )
        nonzero_mask = y_alignment_sign != 0
        if torch.any(nonzero_mask):
            desired_sign = y_alignment_sign[nonzero_mask].to(self.dtype)
            aligned_directions = directions[nonzero_mask]
            aligned_directions[:, 1] = torch.abs(aligned_directions[:, 1]) * desired_sign
            direction_norm = torch.norm(aligned_directions, dim=-1, keepdim=True).clamp_min(1e-8)
            directions[nonzero_mask] = aligned_directions / direction_norm
        return directions

    def _activate_new_primitives(
        self,
        y_alignment_sign: torch.Tensor | None = None,
        current_paddle_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        if y_alignment_sign is None:
            sampled_y_alignment_sign = torch.zeros(count, dtype=torch.long, device=self.device)
        else:
            sampled_y_alignment_sign = y_alignment_sign[indices].to(device=self.device, dtype=torch.long)

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

        y_aligned_mask = sampled_primitive == self.primitive_ids.Y_ALIGNED_DIRECTION
        if torch.any(y_aligned_mask):
            aligned_indices = indices[y_aligned_mask]
            aligned_count = int(aligned_indices.numel())
            aligned_signs = sampled_y_alignment_sign[y_aligned_mask]
            self.direction[aligned_indices] = self._sample_y_aligned_directions(aligned_signs)
            if self._uses_simulator_space_range("y_aligned"):
                feasible_max = max_magnitude_for_directions_in_action_box(
                    self.direction[aligned_indices],
                    max_delta_x=self.target_action_delta_x,
                    max_delta_y=self.target_action_delta_y,
                )
                capped_max = torch.clamp(feasible_max, min=0.0, max=float(self.y_aligned_max_magnitude))
                lower = torch.minimum(
                    capped_max,
                    torch.full_like(capped_max, float(self.y_aligned_min_magnitude)),
                )
                self.magnitude[aligned_indices] = (
                    lower
                    + torch.rand(aligned_count, device=self.device, dtype=self.dtype) * (capped_max - lower)
                )
            else:
                self.magnitude[aligned_indices] = sample_uniform_magnitude(
                    aligned_count, max_magnitude=1.0, min_magnitude=0.1, device=self.device, dtype=self.dtype
                )

        target_position_mask = sampled_primitive == self.primitive_ids.TARGET_POSITION_DIRECTIONAL
        if torch.any(target_position_mask):
            target_indices = indices[target_position_mask]
            target_count = int(target_indices.numel())
            if self._uses_simulator_space_range("target_position_directional"):
                _, _, target_displacements = self._sample_simulator_range(
                    "target_position_directional",
                    target_count,
                )
            else:
                target_directions = sample_unit_directions(
                    target_count,
                    self.device,
                    self.dtype,
                    y_component_weight=self.direction_y_component_weight,
                )
                target_distances = sample_target_distances(
                    target_count,
                    min_distance=self.target_min_distance,
                    max_distance=self.target_max_distance,
                    device=self.device,
                    dtype=self.dtype,
                )
                target_displacements = actions_from_direction_and_magnitude(target_directions, target_distances)
            target_actions, achievable_displacements, _ = project_displacement_to_action_box(
                target_displacements,
                max_delta_x=self.target_action_delta_x,
                max_delta_y=self.target_action_delta_y,
            )
            self.target_displacement[target_indices] = target_displacements
            self.target_action[target_indices] = target_actions
            if current_paddle_position is not None:
                sampled_current_pos = current_paddle_position[target_indices].to(
                    device=self.device, dtype=self.dtype
                )
                self.target_position[target_indices] = sampled_current_pos + target_displacements
            else:
                self.target_position[target_indices] = achievable_displacements
            self.steps_remaining[target_indices] = self.target_takeover_steps

        return activate_mask

    def apply(
        self,
        proposed_actions: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        y_alignment_sign: torch.Tensor | None = None,
        current_paddle_position: torch.Tensor | None = None,
        current_puck_position: torch.Tensor | None = None,
        current_puck_velocity: torch.Tensor | None = None,
        return_stats: bool = False,
    ):
        if proposed_actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected proposed_actions to have batch {self.num_envs}, got {proposed_actions.shape[0]}"
            )
        del current_puck_position, current_puck_velocity
        _ = self._activate_new_primitives(
            y_alignment_sign=y_alignment_sign,
            current_paddle_position=current_paddle_position,
        )

        actions = proposed_actions.clone()
        active_indices = torch.nonzero(self.active, as_tuple=False).squeeze(-1)
        primitive_applied_count = int(active_indices.numel())
        primitive_horizontal_dominant_count = 0
        target_position_directional_applied_count = 0
        if active_indices.numel() == 0:
            if return_stats:
                return actions, {
                    "primitive_applied_count": 0,
                    "primitive_horizontal_dominant_count": 0,
                    "target_position_directional_applied_count": 0,
                }
            return actions

        active_primitive = self.primitive_id[active_indices]
        target_position_mask = active_primitive == self.primitive_ids.TARGET_POSITION_DIRECTIONAL
        target_position_directional_applied_count = int(target_position_mask.sum().item())
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
                    max_delta_x=self.target_action_delta_x,
                    max_delta_y=self.target_action_delta_y,
                )
                actions[same_indices] = same_actions
            else:
                actions[same_indices] = actions_from_direction_and_magnitude(
                    self.direction[same_indices], self.magnitude[same_indices]
                )

        y_aligned_mask = active_primitive == self.primitive_ids.Y_ALIGNED_DIRECTION
        if torch.any(y_aligned_mask):
            aligned_indices = active_indices[y_aligned_mask]
            if self._uses_simulator_space_range("y_aligned"):
                aligned_displacements = actions_from_direction_and_magnitude(
                    self.direction[aligned_indices], self.magnitude[aligned_indices]
                )
                aligned_actions, _, _ = project_displacement_to_action_box(
                    aligned_displacements,
                    max_delta_x=self.target_action_delta_x,
                    max_delta_y=self.target_action_delta_y,
                )
                actions[aligned_indices] = aligned_actions
            else:
                actions[aligned_indices] = actions_from_direction_and_magnitude(
                    self.direction[aligned_indices], self.magnitude[aligned_indices]
                )
        if torch.any(target_position_mask):
            target_indices = active_indices[target_position_mask]
            actions[target_indices] = self.target_action[target_indices]

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
                "target_position_directional_applied_count": target_position_directional_applied_count,
            }
        return actions

    def state_dict(self) -> dict:
        return {
            "num_envs": self.num_envs,
            "chance": self.chance,
            "takeover_steps": self.takeover_steps,
            "target_takeover_steps": self.target_takeover_steps,
            "same_direction_min_angle_deg": self.same_direction_min_angle_deg,
            "same_direction_max_angle_deg": self.same_direction_max_angle_deg,
            "same_direction_min_magnitude": self.same_direction_min_magnitude,
            "same_direction_max_magnitude": self.same_direction_max_magnitude,
            "y_aligned_min_angle_deg": self.y_aligned_min_angle_deg,
            "y_aligned_max_angle_deg": self.y_aligned_max_angle_deg,
            "y_aligned_min_magnitude": self.y_aligned_min_magnitude,
            "y_aligned_max_magnitude": self.y_aligned_max_magnitude,
            "target_position_directional_min_angle_deg": self.target_position_directional_min_angle_deg,
            "target_position_directional_max_angle_deg": self.target_position_directional_max_angle_deg,
            "target_position_directional_min_magnitude": self.target_position_directional_min_magnitude,
            "target_position_directional_max_magnitude": self.target_position_directional_max_magnitude,
            "primitive_weights": self.primitive_weights.detach().clone().cpu(),
            "active": self.active.detach().clone().cpu(),
            "steps_remaining": self.steps_remaining.detach().clone().cpu(),
            "primitive_id": self.primitive_id.detach().clone().cpu(),
            "direction": self.direction.detach().clone().cpu(),
            "magnitude": self.magnitude.detach().clone().cpu(),
            "target_position": self.target_position.detach().clone().cpu(),
            "target_displacement": self.target_displacement.detach().clone().cpu(),
            "target_action": self.target_action.detach().clone().cpu(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "primitive_weights" in state_dict:
            loaded_weights = state_dict["primitive_weights"].to(self.device)
            # Tolerate old 5- or 6-element weight vectors (dropped POLICY_TAKEOVER
            # at slot 3, dropped PRE_CONTACT_HIT_VARIANT at slot 5). Re-pack the
            # surviving entries in the new 4-slot layout.
            if loaded_weights.numel() == 6:
                loaded_weights = torch.stack(
                    [loaded_weights[0], loaded_weights[1], loaded_weights[2], loaded_weights[4]]
                )
            elif loaded_weights.numel() == 5:
                loaded_weights = torch.stack(
                    [loaded_weights[0], loaded_weights[1], loaded_weights[2], loaded_weights[4]]
                )
            elif loaded_weights.numel() == 3:
                loaded_weights = torch.cat(
                    [loaded_weights, torch.zeros(1, dtype=loaded_weights.dtype, device=self.device)],
                    dim=0,
                )
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
        self.y_aligned_min_angle_deg = self._maybe_float(
            state_dict.get("y_aligned_min_angle_deg", self.y_aligned_min_angle_deg)
        )
        self.y_aligned_max_angle_deg = self._maybe_float(
            state_dict.get("y_aligned_max_angle_deg", self.y_aligned_max_angle_deg)
        )
        self.y_aligned_min_magnitude = self._maybe_float(
            state_dict.get("y_aligned_min_magnitude", self.y_aligned_min_magnitude)
        )
        self.y_aligned_max_magnitude = self._maybe_float(
            state_dict.get("y_aligned_max_magnitude", self.y_aligned_max_magnitude)
        )
        self.target_position_directional_min_angle_deg = self._maybe_float(
            state_dict.get(
                "target_position_directional_min_angle_deg",
                self.target_position_directional_min_angle_deg,
            )
        )
        self.target_position_directional_max_angle_deg = self._maybe_float(
            state_dict.get(
                "target_position_directional_max_angle_deg",
                self.target_position_directional_max_angle_deg,
            )
        )
        self.target_position_directional_min_magnitude = self._maybe_float(
            state_dict.get(
                "target_position_directional_min_magnitude",
                self.target_position_directional_min_magnitude,
            )
        )
        self.target_position_directional_max_magnitude = self._maybe_float(
            state_dict.get(
                "target_position_directional_max_magnitude",
                self.target_position_directional_max_magnitude,
            )
        )
        # Mid-episode primitive state from old checkpoints used a different ID
        # space; force-clear exploration state on resume rather than translate.
        self.active.zero_()
        self.steps_remaining.zero_()
        self.primitive_id.fill_(-1)
        self.direction.zero_()
        self.magnitude.zero_()
        self.target_position.zero_()
        self.target_displacement.zero_()
        self.target_action.zero_()
