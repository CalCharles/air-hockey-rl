"""
Demo data loader for AMP training with support for:
- legacy 5-step position-history discriminator observations
- optional action-conditioned features
- optional puck-conditioned features
- long-window bucketed sampling for temporal discriminator heads
"""

import torch
from pathlib import Path

from scripts.smooth_policy.amp_history.amp_training.feature_processing import (
    PUCK_FEATURE_DIM,
    build_puck_discriminator_features_torch,
    normalize_action_history_batch,
    normalize_position_history_batch,
    normalize_position_sequence_batch,
    sample_bucketed_indices_torch,
)


class DemoLoaderPositionHistory:
    """Loader for expert demonstration data used by AMP discriminators."""

    def __init__(
        self,
        demo_path,
        device='cuda',
        use_actions=None,
        use_puck=None,
        puck_vertical_axis=0,
        puck_downward_positive_direction=1.0,
        puck_downward_speed_max=0.75,
        puck_speed_dt=0.05,
        puck_noise_std=0.03,
        puck_vertical_pos_min=-1.0,
        puck_vertical_pos_max=1.0,
        position_key='position_sequences',
        puck_key='puck_sequences',
        sample_bucketed_points=False,
        bucket_window_len=30,
        bucket_num_bins=3,
        bucket_samples_per_bin=2,
        puck_current_index=2,
    ):
        self.demo_path = Path(demo_path)
        self.device = device

        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo data not found at: {demo_path}")

        self.puck_vertical_axis = int(puck_vertical_axis)
        self.puck_downward_positive_direction = float(puck_downward_positive_direction)
        self.puck_downward_speed_max = float(puck_downward_speed_max)
        self.puck_speed_dt = float(puck_speed_dt)
        self.puck_noise_std = float(puck_noise_std)
        self.puck_vertical_pos_min = float(puck_vertical_pos_min)
        self.puck_vertical_pos_max = float(puck_vertical_pos_max)

        self.sample_bucketed_points = bool(sample_bucketed_points)
        self.bucket_window_len = int(bucket_window_len)
        self.bucket_num_bins = int(bucket_num_bins)
        self.bucket_samples_per_bin = int(bucket_samples_per_bin)
        self.puck_current_index = int(puck_current_index)

        print(f"Loading demonstration position history data from: {demo_path}")
        data = torch.load(demo_path, map_location=device)

        self.has_actions = 'action_sequences' in data
        self.has_puck = puck_key in data
        self.use_actions = self.has_actions if use_actions is None else bool(use_actions)
        self.use_puck = self.has_puck if use_puck is None else bool(use_puck)

        if self.use_actions and not self.has_actions:
            raise ValueError("use_actions=True but dataset does not contain 'action_sequences'.")
        if self.use_puck and not self.has_puck:
            raise ValueError(f"use_puck=True but dataset does not contain '{puck_key}'.")

        self.action_sequences = None
        self.puck_sequences = None
        self.position_history = None

        if self.sample_bucketed_points:
            if self.use_actions:
                raise ValueError("sample_bucketed_points=True currently does not support action-conditioned observations.")
            if position_key not in data:
                raise ValueError(f"Dataset does not contain required key '{position_key}' for long-window sampling.")
            self.position_history = data[position_key].to(device=device, dtype=torch.float32)
            if self.position_history.dim() != 3 or self.position_history.shape[1] != self.bucket_window_len or self.position_history.shape[2] != 2:
                raise ValueError(
                    f"Unexpected long position history shape: {self.position_history.shape}. "
                    f"Expected [N, {self.bucket_window_len}, 2]"
                )
            sampled_points = 2 + self.bucket_num_bins * self.bucket_samples_per_bin
            self.demo_obs_base = torch.empty((self.position_history.shape[0], (sampled_points - 1) * 2), device=device)
            self.is_normalized = False
        elif 'normalized_position_history' in data:
            normalized = data['normalized_position_history'].to(device=device, dtype=torch.float32)
            if normalized.dim() != 2 or normalized.shape[1] < 8:
                raise ValueError(
                    f"Unexpected normalized data shape: {normalized.shape}. Expected [N, >=8]"
                )
            self.demo_obs_base = normalized[:, :8]
            self.is_normalized = True
        else:
            if position_key not in data:
                # Legacy fallback
                position_key = 'position_sequences' if 'position_sequences' in data else 'position_history'
            if position_key not in data:
                raise ValueError(
                    "Dataset must contain 'normalized_position_history', 'position_history', "
                    f"or '{position_key}' key"
                )
            self.position_history = data[position_key].to(device=device, dtype=torch.float32)
            self.is_normalized = False
            if self.position_history.dim() != 3 or self.position_history.shape[-1] != 2:
                raise ValueError(
                    f"Unexpected position history shape: {self.position_history.shape}. Expected [N, T, 2]"
                )
            if self.position_history.shape[1] == 5:
                self.demo_obs_base = normalize_position_history_batch(self.position_history)
            else:
                self.demo_obs_base = normalize_position_sequence_batch(self.position_history)

        if self.use_actions:
            self.action_sequences = data['action_sequences'].to(device=device, dtype=torch.float32)
            if self.action_sequences.dim() != 3 or self.action_sequences.shape[-1] != 2:
                raise ValueError(
                    f"Unexpected action sequence shape: {self.action_sequences.shape}. Expected [N, T, 2]"
                )
            if self.action_sequences.shape[0] != self.demo_obs_base.shape[0]:
                raise ValueError("Position and action sequence counts must match.")
            transition_actions = (
                self.action_sequences[:, :-1, :]
                if self.action_sequences.shape[1] == 5
                else self.action_sequences[:, :4, :]
            )
            flattened_actions = normalize_action_history_batch(transition_actions)
            self.demo_obs_base = torch.cat([self.demo_obs_base, flattened_actions], dim=-1)

        if self.use_puck:
            self.puck_sequences = data[puck_key].to(device=device, dtype=torch.float32)
            if self.puck_sequences.dim() != 3 or self.puck_sequences.shape[-1] != 2:
                raise ValueError(
                    f"Unexpected puck sequence shape: {self.puck_sequences.shape}. Expected [N, T, 2]"
                )
            if self.puck_sequences.shape[0] != self.demo_obs_base.shape[0]:
                raise ValueError("Position and puck sequence counts must match.")

        self.num_demos = self.demo_obs_base.shape[0]
        self.stats = data.get('stats', None)
        self.obs_dim = self.demo_obs_base.shape[1] + (PUCK_FEATURE_DIM if self.use_puck else 0)

        print(f"✓ Loaded {self.num_demos:,} demonstration observations")
        print(f"  Base observation shape: {self.demo_obs_base.shape}")
        print(f"  Observation dim used for discriminator: {self.obs_dim}")
        print(f"  Dataset has actions: {self.has_actions}, using actions: {self.use_actions}")
        print(f"  Dataset has puck: {self.has_puck}, using puck: {self.use_puck}")
        self._print_statistics()

    def _sample_base_obs(self, indices):
        if not self.sample_bucketed_points:
            return self.demo_obs_base[indices]

        position_window = self.position_history[indices]
        batch_size = position_window.shape[0]
        sampled_indices = sample_bucketed_indices_torch(
            batch_size,
            window_len=self.bucket_window_len,
            num_bins=self.bucket_num_bins,
            samples_per_bin=self.bucket_samples_per_bin,
            device=position_window.device,
        )
        gather_idx = sampled_indices.unsqueeze(-1).expand(-1, -1, 2)
        sampled_positions = torch.gather(position_window, dim=1, index=gather_idx)
        return normalize_position_sequence_batch(sampled_positions)

    def sample(self, batch_size):
        """Sample random batch of demonstration observations."""
        if batch_size > self.num_demos:
            indices = torch.randint(0, self.num_demos, (batch_size,), device=self.device)
        else:
            indices = torch.randperm(self.num_demos, device=self.device)[:batch_size]

        sampled_obs = self._sample_base_obs(indices)
        if self.use_puck:
            sampled_puck = self.puck_sequences[indices]
            puck_features = build_puck_discriminator_features_torch(
                sampled_puck,
                current_index=self.puck_current_index,
                vertical_axis=self.puck_vertical_axis,
                downward_positive_direction=self.puck_downward_positive_direction,
                downward_speed_max=self.puck_downward_speed_max,
                speed_dt=self.puck_speed_dt,
                noise_std=self.puck_noise_std,
                vertical_pos_min=self.puck_vertical_pos_min,
                vertical_pos_max=self.puck_vertical_pos_max,
            )
            sampled_obs = torch.cat([sampled_obs, puck_features], dim=-1)
        return sampled_obs

    def get_all(self):
        """Return all demonstration observations."""
        if self.sample_bucketed_points:
            base = self._sample_base_obs(torch.arange(self.num_demos, device=self.device))
        else:
            base = self.demo_obs_base
        if not self.use_puck:
            return base
        puck_features = build_puck_discriminator_features_torch(
            self.puck_sequences,
            current_index=self.puck_current_index,
            vertical_axis=self.puck_vertical_axis,
            downward_positive_direction=self.puck_downward_positive_direction,
            downward_speed_max=self.puck_downward_speed_max,
            speed_dt=self.puck_speed_dt,
            noise_std=0.0,
            vertical_pos_min=self.puck_vertical_pos_min,
            vertical_pos_max=self.puck_vertical_pos_max,
        )
        return torch.cat([base, puck_features], dim=-1)

    def get_stats(self):
        return self.stats

    def get_obs_dim(self):
        return self.obs_dim

    def __len__(self):
        return self.num_demos

    def _print_statistics(self):
        print("\n  Demonstration data statistics:")
        print(f"    sample_bucketed_points: {self.sample_bucketed_points}")
        print(f"    use_puck: {self.use_puck}")
        print(f"    obs_dim: {self.obs_dim}")
        if self.sample_bucketed_points:
            sampled_points = 2 + self.bucket_num_bins * self.bucket_samples_per_bin
            print(
                "    long-window sampling: "
                f"window={self.bucket_window_len}, points={sampled_points}, "
                f"bins={self.bucket_num_bins}, per_bin={self.bucket_samples_per_bin}"
            )
