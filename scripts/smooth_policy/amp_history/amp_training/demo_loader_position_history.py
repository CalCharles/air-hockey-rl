"""
Demo data loader for AMP training with 5 consecutive position history.

Loads expert demonstration data and provides random sampling for discriminator training.
Uses only paddle (x, y) positions, not velocities.

Supports position history format: [N, 5, 2] where each entry contains 5 consecutive (x, y) positions.
Compatible with datasets using keys: 'position_sequences', 'position_history', 'normalized_position_history', or 'state_pairs'.

Optionally supports action-conditioned discrimination:
- If dataset contains 'action_sequences' key, loads actions and appends to observations
- Actions are delta target positions: action = desired_pose - pose
- Uses 4 transition actions for a 5-state window (s1->s2, s2->s3, s3->s4, s4->s5)
- Output format becomes [N, 16] = 8 relative positions + 8 action dims

Optionally supports puck-conditioned discrimination:
- If dataset contains 'puck_sequences' key, loads aligned 5-step puck windows
- Appends 4 puck features per sample:
  [noised_curr_x, noised_curr_y, direction_sign, downward_speed_bin(-1/0/1)]
"""

import torch
from pathlib import Path

from scripts.smooth_policy.amp_history.amp_training.feature_processing import (
    PUCK_FEATURE_DIM,
    build_puck_discriminator_features_torch,
    normalize_action_history_batch,
    normalize_position_history_batch,
)


class DemoLoaderPositionHistory:
    """
    Loader for expert demonstration data with 5 consecutive positions (position history).
    
    Loads position sequences from .pt files and provides random sampling for 
    training the discriminator.
    
    Optionally supports action-conditioned discrimination when dataset contains
    'action_sequences' key. In this mode, outputs are [N, 16] = 8 positions + 8 action dims.
    """
    
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
    ):
        """Initialize demo loader for position/action/puck discriminator features."""
        self.demo_path = Path(demo_path)
        self.device = device

        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo data not found at: {demo_path}")

        self.puck_vertical_axis = int(puck_vertical_axis)
        self.puck_downward_positive_direction = float(puck_downward_positive_direction)
        self.puck_downward_speed_max = float(puck_downward_speed_max)
        self.puck_speed_dt = float(puck_speed_dt)
        self.puck_noise_std = float(puck_noise_std)

        print(f"Loading demonstration position history data from: {demo_path}")
        data = torch.load(demo_path, map_location=device)

        self.has_actions = 'action_sequences' in data
        self.has_puck = 'puck_sequences' in data
        self.use_actions = self.has_actions if use_actions is None else bool(use_actions)
        self.use_puck = self.has_puck if use_puck is None else bool(use_puck)
        if self.use_actions and not self.has_actions:
            raise ValueError("use_actions=True but dataset does not contain 'action_sequences'.")
        if self.use_puck and not self.has_puck:
            raise ValueError("use_puck=True but dataset does not contain 'puck_sequences'.")

        self.action_sequences = None
        self.puck_sequences = None

        # Auto-detect dataset format and build base [N, 8] position features.
        if 'normalized_position_history' in data:
            print("  Detected pre-normalized position history dataset")
            normalized = data['normalized_position_history'].to(device=device, dtype=torch.float32)
            if normalized.dim() != 2 or normalized.shape[1] < 8:
                raise ValueError(
                    f"Unexpected normalized data shape: {normalized.shape}. Expected [N, >=8]"
                )
            self.demo_obs_base = normalized[:, :8]
            self.is_normalized = True

        elif 'position_history' in data or 'position_sequences' in data:
            key = 'position_sequences' if 'position_sequences' in data else 'position_history'
            print(f"  Detected non-normalized position dataset (key: '{key}') - applying normalization")
            self.position_history = data[key].to(device=device, dtype=torch.float32)
            self.is_normalized = False

            if self.position_history.dim() != 3 or self.position_history.shape[1] != 5 or self.position_history.shape[2] != 2:
                raise ValueError(
                    f"Unexpected position history shape: {self.position_history.shape}. Expected [N, 5, 2]"
                )

            print("  Normalizing position history...")
            self.demo_obs_base = normalize_position_history_batch(self.position_history)
            print("  ✓ Normalization complete")

        elif 'state_pairs' in data:
            print("  Detected state_pairs format - converting to position history")
            state_pairs = data['state_pairs'].to(device=device, dtype=torch.float32)
            positions = state_pairs[:, :, :2]
            print("  ⚠ Warning: Building 5-position history from 2-position pairs.")
            print("    For best results, prepare a dedicated position_history dataset.")

            num_pairs = positions.shape[0]
            num_sequences = num_pairs - 4
            if num_sequences <= 0:
                raise ValueError(
                    f"Not enough pairs ({num_pairs}) to create position history. Need at least 5."
                )

            position_history_list = []
            for i in range(num_sequences):
                position_history_list.append(positions[i:i+5, 0, :])
            self.position_history = torch.stack(position_history_list, dim=0)
            self.is_normalized = False

            print("  Normalizing position history...")
            self.demo_obs_base = normalize_position_history_batch(self.position_history)
            print("  ✓ Normalization complete")

        else:
            raise ValueError(
                "Dataset must contain 'normalized_position_history', 'position_history', "
                "'position_sequences', or 'state_pairs' key"
            )

        # Append normalized transition-action features when requested.
        if self.use_actions:
            self.action_sequences = data['action_sequences'].to(device=device, dtype=torch.float32)
            print(f"  ✓ Using action sequences: {self.action_sequences.shape}")
            if self.action_sequences.dim() != 3 or self.action_sequences.shape[-1] != 2:
                raise ValueError(
                    f"Unexpected action sequence shape: {self.action_sequences.shape}. Expected [N, T, 2]"
                )
            if self.action_sequences.shape[1] < 4:
                raise ValueError(
                    f"Action sequence length must be at least 4 for transition features, got {self.action_sequences.shape[1]}"
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
            print("  ✓ Appended 4 transition actions to observations")

        # Keep raw puck windows for per-sample noise injection at sampling time.
        if self.use_puck:
            self.puck_sequences = data['puck_sequences'].to(device=device, dtype=torch.float32)
            print(f"  ✓ Using puck sequences: {self.puck_sequences.shape}")
            if self.puck_sequences.dim() != 3 or self.puck_sequences.shape[1:] != (5, 2):
                raise ValueError(
                    f"Unexpected puck sequence shape: {self.puck_sequences.shape}. Expected [N, 5, 2]"
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
    
    def sample(self, batch_size):
        """
        Sample random batch of demonstration observations.
        
        Args:
            batch_size: Number of observations to sample
            
        Returns:
            Tensor of shape [batch_size, obs_dim].
        """
        if batch_size > self.num_demos:
            # Sample with replacement if requesting more than available
            indices = torch.randint(0, self.num_demos, (batch_size,), device=self.device)
        else:
            # Sample without replacement
            indices = torch.randperm(self.num_demos, device=self.device)[:batch_size]
        
        sampled_obs = self.demo_obs_base[indices]
        if self.use_puck:
            sampled_puck = self.puck_sequences[indices]
            puck_features = build_puck_discriminator_features_torch(
                sampled_puck,
                current_index=2,
                vertical_axis=self.puck_vertical_axis,
                downward_positive_direction=self.puck_downward_positive_direction,
                downward_speed_max=self.puck_downward_speed_max,
                speed_dt=self.puck_speed_dt,
                noise_std=self.puck_noise_std,
            )
            sampled_obs = torch.cat([sampled_obs, puck_features], dim=-1)
        return sampled_obs
    
    def get_all(self):
        """Return all demonstration observations."""
        if not self.use_puck:
            return self.demo_obs_base
        puck_features = build_puck_discriminator_features_torch(
            self.puck_sequences,
            current_index=2,
            vertical_axis=self.puck_vertical_axis,
            downward_positive_direction=self.puck_downward_positive_direction,
            downward_speed_max=self.puck_downward_speed_max,
            speed_dt=self.puck_speed_dt,
            noise_std=0.0,
        )
        return torch.cat([self.demo_obs_base, puck_features], dim=-1)
    
    def get_stats(self):
        """Return dataset statistics if available."""
        return self.stats
    
    def get_obs_dim(self):
        """
        Return the observation dimension.
        
        Returns:
            int: observation dimension after selected feature concatenation.
        """
        return self.obs_dim
    
    def __len__(self):
        """Return number of demonstration observations."""
        return self.num_demos
    
    def _print_statistics(self):
        """Print statistics about the demonstration data."""
        print(f"\n  Demonstration position history data statistics:")
        all_obs = self.get_all()
        
        # Statistics for normalized position format
        # [pos2_x, pos2_y, pos3_x, pos3_y, pos4_x, pos4_y, pos5_x, pos5_y]
        dim_names = [
            'Position 2 Relative X',
            'Position 2 Relative Y', 
            'Position 3 Relative X',
            'Position 3 Relative Y',
            'Position 4 Relative X',
            'Position 4 Relative Y',
            'Position 5 Relative X',
            'Position 5 Relative Y'
        ]
        
        # Add action dimension names if we have actions
        if self.use_actions:
            dim_names.extend([
                'Action 1->2 Delta X', 'Action 1->2 Delta Y',
                'Action 2->3 Delta X', 'Action 2->3 Delta Y',
                'Action 3->4 Delta X', 'Action 3->4 Delta Y',
                'Action 4->5 Delta X', 'Action 4->5 Delta Y',
            ])
        if self.use_puck:
            dim_names.extend([
                'Puck Current X (no-noise stats)',
                'Puck Current Y (no-noise stats)',
                'Puck Direction Sign',
                'Puck Downward Speed Bin',
            ])
        
        obs_type = f"{len(dim_names)}D - relative positions"
        if self.use_actions:
            obs_type += " + action"
        if self.use_puck:
            obs_type += " + puck"
        print(f"\n  Normalized observations ({obs_type}):")
        for dim_idx, dim_name in enumerate(dim_names):
            values = all_obs[:, dim_idx]
            print(f"    {dim_name}: "
                  f"mean={values.mean():.4f}, "
                  f"std={values.std():.4f}, "
                  f"min={values.min():.4f}, "
                  f"max={values.max():.4f}")
        
        # Compute displacement statistics
        # Displacement from pos1 to pos2
        disp_12 = torch.sqrt(all_obs[:, 0]**2 + all_obs[:, 1]**2).mean()
        # Displacement from pos1 to pos5
        disp_15 = torch.sqrt(all_obs[:, 6]**2 + all_obs[:, 7]**2).mean()
        print(f"\n  Displacement check:")
        print(f"    Position 1→2 displacement (average): {disp_12:.6f}")
        print(f"    Position 1→5 displacement (average): {disp_15:.6f}")
        
        # Action magnitude if available
        if self.use_actions:
            action_vectors = all_obs[:, 8:16].reshape(-1, 2)
            action_mag = torch.sqrt(action_vectors[:, 0]**2 + action_vectors[:, 1]**2).mean()
            print(f"    Transition action magnitude (average): {action_mag:.6f}")

        if self.use_puck:
            puck_slice_start = 16 if self.use_actions else 8
            puck_bins = all_obs[:, puck_slice_start + 3]
            unique_bins, counts = torch.unique(puck_bins, return_counts=True)
            print(
                "    Puck downward speed bin counts: "
                f"{dict(zip(unique_bins.tolist(), counts.tolist()))}"
            )
        
        # Check for any NaN or Inf
        if not torch.isfinite(all_obs).all():
            print("\n  ⚠ WARNING: Demo data contains NaN or Inf values!")
