"""Selection/state manager for primitive TD3 exploration takeover."""

from dataclasses import dataclass

import torch

from scripts.smooth_policy.amp_history.amp_training.td3.exploration_primitives import (
    actions_from_direction_and_magnitude,
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


class PrimitiveExplorationSelector:
    def __init__(
        self,
        num_envs: int,
        chance: float,
        takeover_steps: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        direction_y_component_weight: float = 1.0,
    ):
        if takeover_steps <= 0:
            raise ValueError("takeover_steps must be > 0")
        self.num_envs = int(num_envs)
        self.chance = float(chance)
        self.takeover_steps = int(takeover_steps)
        self.device = device
        self.dtype = dtype
        self.direction_y_component_weight = float(max(direction_y_component_weight, 1e-6))
        self.primitive_ids = PrimitiveIds()
        self.primitive_weights = torch.tensor(
            [0.25, 0.25, 0.25, 0.25], dtype=self.dtype, device=self.device
        )

        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.primitive_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.direction = torch.zeros((self.num_envs, 2), dtype=self.dtype, device=self.device)
        self.magnitude = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)

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
        y_aligned: float,
        policy_takeover: float = 0.0,
    ) -> None:
        weights = torch.tensor(
            [stand_still, same_direction, y_aligned, policy_takeover],
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

    def _activate_new_primitives(self, y_alignment_sign: torch.Tensor | None = None) -> torch.Tensor:
        inactive = ~self.active
        if not torch.any(inactive):
            return inactive
        trigger = torch.rand(self.num_envs, device=self.device) < self.chance
        activate_mask = inactive & trigger
        if not torch.any(activate_mask):
            return activate_mask

        indices = torch.nonzero(activate_mask, as_tuple=False).squeeze(-1)
        count = int(indices.numel())
        sampled_primitive = torch.multinomial(self.primitive_weights, count, replacement=True).to(torch.long)
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
            self.magnitude[aligned_indices] = sample_uniform_magnitude(
                aligned_count, max_magnitude=1.0, min_magnitude=0.1, device=self.device, dtype=self.dtype
            )

        return activate_mask

    def apply(
        self,
        proposed_actions: torch.Tensor,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        y_alignment_sign: torch.Tensor | None = None,
        return_stats: bool = False,
    ):
        if proposed_actions.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected proposed_actions to have batch {self.num_envs}, got {proposed_actions.shape[0]}"
            )
        _ = self._activate_new_primitives(y_alignment_sign=y_alignment_sign)

        actions = proposed_actions.clone()
        active_indices = torch.nonzero(self.active, as_tuple=False).squeeze(-1)
        primitive_applied_count = int(active_indices.numel())
        primitive_horizontal_dominant_count = 0
        policy_takeover_applied_count = 0
        if active_indices.numel() == 0:
            if return_stats:
                return actions, {
                    "primitive_applied_count": 0,
                    "primitive_horizontal_dominant_count": 0,
                    "policy_takeover_applied_count": 0,
                }
            return actions

        active_primitive = self.primitive_id[active_indices]
        policy_takeover_mask = active_primitive == self.primitive_ids.POLICY_TAKEOVER
        policy_takeover_applied_count = int(policy_takeover_mask.sum().item())
        stand_mask = active_primitive == self.primitive_ids.STAND_STILL
        if torch.any(stand_mask):
            stand_indices = active_indices[stand_mask]
            actions[stand_indices] = stand_still_actions(int(stand_indices.numel()), self.device, self.dtype)

        same_mask = active_primitive == self.primitive_ids.SAME_DIRECTION_SMALL_MAG
        if torch.any(same_mask):
            same_indices = active_indices[same_mask]
            actions[same_indices] = actions_from_direction_and_magnitude(
                self.direction[same_indices], self.magnitude[same_indices]
            )

        y_aligned_mask = active_primitive == self.primitive_ids.Y_ALIGNED_DIRECTION
        if torch.any(y_aligned_mask):
            aligned_indices = active_indices[y_aligned_mask]
            actions[aligned_indices] = actions_from_direction_and_magnitude(
                self.direction[aligned_indices], self.magnitude[aligned_indices]
            )

        actions = torch.clamp(actions, action_low, action_high)
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
            }
        return actions

    def state_dict(self) -> dict:
        return {
            "num_envs": self.num_envs,
            "chance": self.chance,
            "takeover_steps": self.takeover_steps,
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
            if loaded_weights.numel() == 3:
                loaded_weights = torch.cat(
                    [loaded_weights, torch.zeros(1, dtype=loaded_weights.dtype, device=self.device)],
                    dim=0,
                )
            if loaded_weights.numel() == self.primitive_weights.numel():
                weight_sum = loaded_weights.sum().clamp_min(1e-8)
                self.primitive_weights.copy_(loaded_weights / weight_sum)
        self.active.copy_(state_dict["active"].to(self.device))
        self.steps_remaining.copy_(state_dict["steps_remaining"].to(self.device))
        self.primitive_id.copy_(state_dict["primitive_id"].to(self.device))
        self.direction.copy_(state_dict["direction"].to(self.device))
        self.magnitude.copy_(state_dict["magnitude"].to(self.device))
