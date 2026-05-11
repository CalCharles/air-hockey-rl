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
    POLICY_TAKEOVER: int = 3
    TARGET_POSITION_DIRECTIONAL: int = 4
    PRE_CONTACT_HIT_VARIANT: int = 5


class PrimitiveExplorationSelector:
    _HIT_VARIANT_MODE_FIXED_TARGET = 0
    _HIT_VARIANT_MODE_REPEAT_DELTA = 1

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
        pre_contact_hit_variant_chance: float = 0.15,
        pre_contact_hit_variant_steps: int = 5,
        pre_contact_hit_variant_distance_threshold: float = 0.2,
        pre_contact_hit_variant_scale_min: float = 0.5,
        pre_contact_hit_variant_scale_max: float = 1.5,
        pre_contact_hit_variant_min_upward_displacement_x: float = 0.12,
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
        self.pre_contact_hit_variant_chance = float(pre_contact_hit_variant_chance)
        self.pre_contact_hit_variant_steps = int(pre_contact_hit_variant_steps)
        self.pre_contact_hit_variant_distance_threshold = float(pre_contact_hit_variant_distance_threshold)
        self.pre_contact_hit_variant_scale_min = float(pre_contact_hit_variant_scale_min)
        self.pre_contact_hit_variant_scale_max = float(pre_contact_hit_variant_scale_max)
        self.pre_contact_hit_variant_min_upward_displacement_x = float(
            pre_contact_hit_variant_min_upward_displacement_x
        )
        if self.target_min_distance < 0.0:
            raise ValueError("target_min_distance must be >= 0")
        if self.target_max_distance < self.target_min_distance:
            raise ValueError("target_max_distance must be >= target_min_distance")
        if self.target_takeover_steps <= 0:
            raise ValueError("target_takeover_steps must be > 0")
        if not (0.0 <= self.pre_contact_hit_variant_chance <= 1.0):
            raise ValueError("pre_contact_hit_variant_chance must be between 0 and 1")
        if self.pre_contact_hit_variant_steps <= 0:
            raise ValueError("pre_contact_hit_variant_steps must be > 0")
        if self.pre_contact_hit_variant_distance_threshold < 0.0:
            raise ValueError("pre_contact_hit_variant_distance_threshold must be >= 0")
        if self.pre_contact_hit_variant_scale_max < self.pre_contact_hit_variant_scale_min:
            raise ValueError("pre_contact_hit_variant_scale_max must be >= pre_contact_hit_variant_scale_min")
        if self.pre_contact_hit_variant_min_upward_displacement_x < 0.0:
            raise ValueError("pre_contact_hit_variant_min_upward_displacement_x must be >= 0")
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
            [0.25, 0.25, 0.25, 0.25, 0.0, 0.0], dtype=self.dtype, device=self.device
        )

        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.primitive_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.direction = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.magnitude = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        self.target_position = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.target_displacement = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.target_action = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.hit_variant_mode = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.hit_variant_scale = torch.ones(self.num_envs, dtype=self.dtype, device=self.device)
        self.hit_variant_base_action = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.hit_variant_running_action = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)

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
        self.hit_variant_mode[mask] = -1
        self.hit_variant_scale[mask] = 1
        self.hit_variant_base_action[mask] = 0
        self.hit_variant_running_action[mask] = 0

    def reset(self, done_mask: torch.Tensor) -> None:
        mask = done_mask.to(device=self.device, dtype=torch.bool)
        self._clear_mask(mask)

    def set_primitive_weights(
        self,
        stand_still: float,
        same_direction: float,
        y_aligned: float,
        policy_takeover: float = 0.0,
        target_position_directional: float = 0.0,
        pre_contact_hit_variant: float = 0.0,
    ) -> None:
        weights = torch.tensor(
            [
                stand_still,
                same_direction,
                y_aligned,
                policy_takeover,
                target_position_directional,
                pre_contact_hit_variant,
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

    def _activate_pre_contact_hit_variants(
        self,
        proposed_actions: torch.Tensor,
        current_paddle_position: torch.Tensor | None,
        current_puck_position: torch.Tensor | None,
        current_puck_velocity: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            self.pre_contact_hit_variant_chance <= 0.0
            or current_paddle_position is None
            or current_puck_position is None
            or current_puck_velocity is None
        ):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        inactive = ~self.active
        if not torch.any(inactive):
            return inactive
        distances = torch.norm(
            current_puck_position.to(device=self.device, dtype=self.dtype)
            - current_paddle_position.to(device=self.device, dtype=self.dtype),
            dim=-1,
        )
        puck_falling = current_puck_velocity.to(device=self.device, dtype=self.dtype)[:, 0] > 0.0
        upward_displacement_x = torch.clamp(
            -proposed_actions[:, 0] * float(self.target_action_delta_x),
            min=0.0,
        )
        paddle_upward_action = (
            upward_displacement_x >= self.pre_contact_hit_variant_min_upward_displacement_x
        )
        close_enough = distances <= self.pre_contact_hit_variant_distance_threshold
        trigger = (
            inactive
            & puck_falling
            & paddle_upward_action
            & close_enough
            & (torch.rand(self.num_envs, device=self.device) < self.pre_contact_hit_variant_chance)
        )
        if not torch.any(trigger):
            return trigger
        trigger_indices = torch.nonzero(trigger, as_tuple=False).squeeze(-1)
        count = int(trigger_indices.numel())
        scaled_actions = proposed_actions[trigger_indices].clone()
        scales = sample_uniform_magnitude(
            count=count,
            min_magnitude=self.pre_contact_hit_variant_scale_min,
            max_magnitude=self.pre_contact_hit_variant_scale_max,
            device=self.device,
            dtype=self.dtype,
        )
        scaled_actions = scaled_actions * scales.unsqueeze(-1)
        variant_modes = torch.randint(
            low=0,
            high=2,
            size=(count,),
            device=self.device,
            dtype=torch.long,
        )
        self.active[trigger_indices] = True
        self.steps_remaining[trigger_indices] = self.pre_contact_hit_variant_steps
        self.primitive_id[trigger_indices] = self.primitive_ids.PRE_CONTACT_HIT_VARIANT
        self.hit_variant_mode[trigger_indices] = variant_modes
        self.hit_variant_scale[trigger_indices] = scales
        self.hit_variant_base_action[trigger_indices] = scaled_actions
        self.hit_variant_running_action[trigger_indices] = proposed_actions[trigger_indices]
        return trigger

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
        # Pre-contact hit variant must only activate via trigger-gated path.
        sample_weights[self.primitive_ids.PRE_CONTACT_HIT_VARIANT] = 0.0
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
        _ = self._activate_pre_contact_hit_variants(
            proposed_actions=proposed_actions,
            current_paddle_position=current_paddle_position,
            current_puck_position=current_puck_position,
            current_puck_velocity=current_puck_velocity,
        )
        _ = self._activate_new_primitives(
            y_alignment_sign=y_alignment_sign,
            current_paddle_position=current_paddle_position,
        )

        actions = proposed_actions.clone()
        active_indices = torch.nonzero(self.active, as_tuple=False).squeeze(-1)
        primitive_applied_count = int(active_indices.numel())
        primitive_horizontal_dominant_count = 0
        policy_takeover_applied_count = 0
        target_position_directional_applied_count = 0
        pre_contact_hit_variant_applied_count = 0
        if active_indices.numel() == 0:
            if return_stats:
                return actions, {
                    "primitive_applied_count": 0,
                    "primitive_horizontal_dominant_count": 0,
                    "policy_takeover_applied_count": 0,
                    "target_position_directional_applied_count": 0,
                    "pre_contact_hit_variant_applied_count": 0,
                }
            return actions

        active_primitive = self.primitive_id[active_indices]
        policy_takeover_mask = active_primitive == self.primitive_ids.POLICY_TAKEOVER
        policy_takeover_applied_count = int(policy_takeover_mask.sum().item())
        target_position_mask = active_primitive == self.primitive_ids.TARGET_POSITION_DIRECTIONAL
        target_position_directional_applied_count = int(target_position_mask.sum().item())
        pre_contact_hit_variant_mask = active_primitive == self.primitive_ids.PRE_CONTACT_HIT_VARIANT
        pre_contact_hit_variant_applied_count = int(pre_contact_hit_variant_mask.sum().item())
        if torch.any(pre_contact_hit_variant_mask):
            hit_indices = active_indices[pre_contact_hit_variant_mask]
            invalid_mode_mask = self.hit_variant_mode[hit_indices] < 0
            if torch.any(invalid_mode_mask):
                self._clear_mask(hit_indices[invalid_mode_mask])
            active_indices = torch.nonzero(self.active, as_tuple=False).squeeze(-1)
            primitive_applied_count = int(active_indices.numel())
            active_primitive = self.primitive_id[active_indices]
            policy_takeover_mask = active_primitive == self.primitive_ids.POLICY_TAKEOVER
            policy_takeover_applied_count = int(policy_takeover_mask.sum().item())
            target_position_mask = active_primitive == self.primitive_ids.TARGET_POSITION_DIRECTIONAL
            target_position_directional_applied_count = int(target_position_mask.sum().item())
            pre_contact_hit_variant_mask = active_primitive == self.primitive_ids.PRE_CONTACT_HIT_VARIANT
            pre_contact_hit_variant_applied_count = int(pre_contact_hit_variant_mask.sum().item())
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
        if torch.any(pre_contact_hit_variant_mask):
            hit_indices = active_indices[pre_contact_hit_variant_mask]
            hit_modes = self.hit_variant_mode[hit_indices]
            fixed_target_mask = hit_modes == self._HIT_VARIANT_MODE_FIXED_TARGET
            if torch.any(fixed_target_mask):
                fixed_indices = hit_indices[fixed_target_mask]
                actions[fixed_indices] = self.hit_variant_base_action[fixed_indices]
            repeat_delta_mask = hit_modes == self._HIT_VARIANT_MODE_REPEAT_DELTA
            if torch.any(repeat_delta_mask):
                repeat_indices = hit_indices[repeat_delta_mask]
                next_running_action = self.hit_variant_running_action[repeat_indices] + self.hit_variant_base_action[
                    repeat_indices
                ]
                actions[repeat_indices] = next_running_action
                self.hit_variant_running_action[repeat_indices] = next_running_action

        actions = torch.clamp(actions, action_low, action_high)
        if torch.any(pre_contact_hit_variant_mask):
            hit_indices = active_indices[pre_contact_hit_variant_mask]
            self.hit_variant_running_action[hit_indices] = torch.clamp(
                self.hit_variant_running_action[hit_indices], action_low, action_high
            )
        directional_active_indices = active_indices[~policy_takeover_mask]
        primitive_horizontal_dominant_count = int(
            (
                torch.abs(actions[directional_active_indices, 0])
                > torch.abs(actions[directional_active_indices, 1])
            ).sum().item()
        )

        self.steps_remaining[active_indices] = self.steps_remaining[active_indices] - 1
        finished_mask = self.active & (self.steps_remaining <= 0)
        self._clear_mask(finished_mask)
        if return_stats:
            return actions, {
                "primitive_applied_count": primitive_applied_count,
                "primitive_horizontal_dominant_count": primitive_horizontal_dominant_count,
                "policy_takeover_applied_count": policy_takeover_applied_count,
                "target_position_directional_applied_count": target_position_directional_applied_count,
                "pre_contact_hit_variant_applied_count": pre_contact_hit_variant_applied_count,
            }
        return actions

    def state_dict(self) -> dict:
        return {
            "num_envs": self.num_envs,
            "chance": self.chance,
            "takeover_steps": self.takeover_steps,
            "target_takeover_steps": self.target_takeover_steps,
            "pre_contact_hit_variant_chance": self.pre_contact_hit_variant_chance,
            "pre_contact_hit_variant_steps": self.pre_contact_hit_variant_steps,
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
            "pre_contact_hit_variant_distance_threshold": self.pre_contact_hit_variant_distance_threshold,
            "pre_contact_hit_variant_scale_min": self.pre_contact_hit_variant_scale_min,
            "pre_contact_hit_variant_scale_max": self.pre_contact_hit_variant_scale_max,
            "pre_contact_hit_variant_min_upward_displacement_x": (
                self.pre_contact_hit_variant_min_upward_displacement_x
            ),
            "primitive_weights": self.primitive_weights.detach().clone().cpu(),
            "active": self.active.detach().clone().cpu(),
            "steps_remaining": self.steps_remaining.detach().clone().cpu(),
            "primitive_id": self.primitive_id.detach().clone().cpu(),
            "direction": self.direction.detach().clone().cpu(),
            "magnitude": self.magnitude.detach().clone().cpu(),
            "target_position": self.target_position.detach().clone().cpu(),
            "target_displacement": self.target_displacement.detach().clone().cpu(),
            "target_action": self.target_action.detach().clone().cpu(),
            "hit_variant_mode": self.hit_variant_mode.detach().clone().cpu(),
            "hit_variant_scale": self.hit_variant_scale.detach().clone().cpu(),
            "hit_variant_base_action": self.hit_variant_base_action.detach().clone().cpu(),
            "hit_variant_running_action": self.hit_variant_running_action.detach().clone().cpu(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if "primitive_weights" in state_dict:
            loaded_weights = state_dict["primitive_weights"].to(self.device)
            if loaded_weights.numel() == 3:
                loaded_weights = torch.cat(
                    [loaded_weights, torch.zeros(3, dtype=loaded_weights.dtype, device=self.device)],
                    dim=0,
                )
            elif loaded_weights.numel() == 4:
                loaded_weights = torch.cat(
                    [loaded_weights, torch.zeros(2, dtype=loaded_weights.dtype, device=self.device)],
                    dim=0,
                )
            elif loaded_weights.numel() == 5:
                loaded_weights = torch.cat(
                    [loaded_weights, torch.zeros(1, dtype=loaded_weights.dtype, device=self.device)],
                    dim=0,
                )
            if loaded_weights.numel() == self.primitive_weights.numel():
                weight_sum = loaded_weights.sum().clamp_min(1e-8)
                self.primitive_weights.copy_(loaded_weights / weight_sum)
        self.pre_contact_hit_variant_chance = float(
            state_dict.get("pre_contact_hit_variant_chance", self.pre_contact_hit_variant_chance)
        )
        self.pre_contact_hit_variant_steps = int(
            state_dict.get("pre_contact_hit_variant_steps", self.pre_contact_hit_variant_steps)
        )
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
        self.pre_contact_hit_variant_distance_threshold = float(
            state_dict.get(
                "pre_contact_hit_variant_distance_threshold",
                self.pre_contact_hit_variant_distance_threshold,
            )
        )
        self.pre_contact_hit_variant_scale_min = float(
            state_dict.get("pre_contact_hit_variant_scale_min", self.pre_contact_hit_variant_scale_min)
        )
        self.pre_contact_hit_variant_scale_max = float(
            state_dict.get("pre_contact_hit_variant_scale_max", self.pre_contact_hit_variant_scale_max)
        )
        self.pre_contact_hit_variant_min_upward_displacement_x = float(
            state_dict.get(
                "pre_contact_hit_variant_min_upward_displacement_x",
                self.pre_contact_hit_variant_min_upward_displacement_x,
            )
        )
        self.active.copy_(state_dict["active"].to(self.device))
        self.steps_remaining.copy_(state_dict["steps_remaining"].to(self.device))
        self.primitive_id.copy_(state_dict["primitive_id"].to(self.device))
        self.direction.copy_(state_dict["direction"].to(self.device))
        self.magnitude.copy_(state_dict["magnitude"].to(self.device))
        if "target_position" in state_dict:
            self.target_position.copy_(state_dict["target_position"].to(self.device))
        if "target_displacement" in state_dict:
            self.target_displacement.copy_(state_dict["target_displacement"].to(self.device))
        if "target_action" in state_dict:
            self.target_action.copy_(state_dict["target_action"].to(self.device))
        if "hit_variant_mode" in state_dict:
            self.hit_variant_mode.copy_(state_dict["hit_variant_mode"].to(self.device))
        if "hit_variant_scale" in state_dict:
            self.hit_variant_scale.copy_(state_dict["hit_variant_scale"].to(self.device))
        if "hit_variant_base_action" in state_dict:
            self.hit_variant_base_action.copy_(state_dict["hit_variant_base_action"].to(self.device))
        if "hit_variant_running_action" in state_dict:
            self.hit_variant_running_action.copy_(state_dict["hit_variant_running_action"].to(self.device))