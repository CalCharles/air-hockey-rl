"""
Demo data loader for AMP training.

Loads expert demonstration data prepared by prepare_amp_dataset.py and provides
random sampling for discriminator training.

Supports both normalized and non-normalized datasets:
- Normalized: [N, 6] format (pre-processed)
- Non-normalized: [N, 2, 4] format (normalized at load time)
"""

import torch
from pathlib import Path


def compute_rotation_matrix_torch(velocity):
    """
    Compute rotation matrix to align velocity with (1, 0) direction.
    Batched PyTorch version.
    
    Args:
        velocity: 2D velocity vectors [batch, 2] with [vx, vy]
        
    Returns:
        torch.Tensor: Rotation matrices [batch, 2, 2]
    """
    vx = velocity[:, 0]
    vy = velocity[:, 1]
    magnitude = torch.sqrt(vx**2 + vy**2)
    
    # Handle near-zero velocities by using identity matrix
    eps = 1e-8
    is_near_zero = magnitude < eps
    
    # Normalize velocity to get current direction
    vx_norm = torch.where(is_near_zero, torch.ones_like(vx), vx / magnitude)
    vy_norm = torch.where(is_near_zero, torch.zeros_like(vy), vy / magnitude)
    
    # Rotation matrix to align (vx_norm, vy_norm) with (1, 0)
    # R = [[vx_norm, vy_norm], [-vy_norm, vx_norm]]
    batch_size = velocity.shape[0]
    rotation_matrix = torch.zeros(batch_size, 2, 2, device=velocity.device, dtype=velocity.dtype)
    rotation_matrix[:, 0, 0] = vx_norm
    rotation_matrix[:, 0, 1] = vy_norm
    rotation_matrix[:, 1, 0] = -vy_norm
    rotation_matrix[:, 1, 1] = vx_norm
    
    return rotation_matrix


def normalize_state_pair_batch(state_pairs):
    """
    Normalize batched state pairs to relative coordinates.
    
    Process:
    1. Translate both states so first position is at (0, 0)
    2. Rotate both states so first velocity points in (1, 0) direction
    3. Return first velocity and second state (first position is always [0,0])
    
    Args:
        state_pairs: Tensor of shape [batch, 2, 4] containing [state1, state2]
                    where each state is [x_pos, y_pos, x_vel, y_vel]
    
    Returns:
        torch.Tensor: Normalized states of shape [batch, 6]
                     [vel1_x, vel1_y, pos2_x, pos2_y, vel2_x, vel2_y]
    """
    # Extract states
    state1 = state_pairs[:, 0, :]  # [batch, 4]
    state2 = state_pairs[:, 1, :]  # [batch, 4]
    
    # Extract positions and velocities
    pos1 = state1[:, :2]  # [batch, 2]
    vel1 = state1[:, 2:]  # [batch, 2]
    pos2 = state2[:, :2]  # [batch, 2]
    vel2 = state2[:, 2:]  # [batch, 2]
    
    # Step 1: Translate so first position is at origin
    pos2_translated = pos2 - pos1  # [batch, 2]
    
    # Step 2: Compute rotation matrices to align first velocity with (1, 0)
    rotation_matrices = compute_rotation_matrix_torch(vel1)  # [batch, 2, 2]
    
    # Apply rotation to positions and velocities
    # pos1_rotated is always [0, 0] so we don't compute it
    pos2_rotated = torch.bmm(rotation_matrices, pos2_translated.unsqueeze(-1)).squeeze(-1)  # [batch, 2]
    vel1_rotated = torch.bmm(rotation_matrices, vel1.unsqueeze(-1)).squeeze(-1)  # [batch, 2]
    vel2_rotated = torch.bmm(rotation_matrices, vel2.unsqueeze(-1)).squeeze(-1)  # [batch, 2]
    
    # Step 3: Concatenate first velocity and second state
    # First position [0, 0] contains no information so not included
    normalized_state = torch.cat([vel1_rotated, pos2_rotated, vel2_rotated], dim=-1)  # [batch, 6]
    
    return normalized_state


class DemoLoader:
    """
    Loader for expert demonstration data.
    
    Loads state pairs from .pt files created by prepare_amp_dataset.py and
    provides random sampling for training the discriminator.
    """
    
    def __init__(self, demo_path, device='cuda'):
        """
        Initialize demo loader.
        
        Args:
            demo_path: Path to .pt file containing demonstration data
            device: Device to store tensors on
        """
        self.demo_path = Path(demo_path)
        self.device = device
        
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo data not found at: {demo_path}")
        
        # Load data
        print(f"Loading demonstration data from: {demo_path}")
        data = torch.load(demo_path, map_location=device)
        
        # Auto-detect dataset format
        if 'normalized_states' in data:
            # Pre-normalized dataset [N, 6]
            print("  Detected pre-normalized dataset")
            self.demo_obs = data['normalized_states'].to(device=device, dtype=torch.float32)
            self.is_normalized = True
            
            if self.demo_obs.dim() != 2 or self.demo_obs.shape[1] != 6:
                raise ValueError(f"Unexpected normalized data shape: {self.demo_obs.shape}. "
                               f"Expected [N, 6]")
        
        elif 'state_pairs' in data:
            # Non-normalized dataset [N, 2, 4] - apply normalization
            print("  Detected non-normalized dataset - applying normalization")
            self.state_pairs = data['state_pairs'].to(device=device, dtype=torch.float32)
            self.is_normalized = False
            
            if self.state_pairs.dim() != 3 or self.state_pairs.shape[1] != 2 or self.state_pairs.shape[2] != 4:
                raise ValueError(f"Unexpected state pairs shape: {self.state_pairs.shape}. "
                               f"Expected [N, 2, 4]")
            
            # Apply normalization to convert [N, 2, 4] -> [N, 6]
            print("  Normalizing state pairs...")
            self.demo_obs = normalize_state_pair_batch(self.state_pairs)
            print(f"  ✓ Normalization complete")
        
        else:
            raise ValueError("Dataset must contain either 'normalized_states' or 'state_pairs' key")
        
        self.num_demos = self.demo_obs.shape[0]
        
        # Store statistics if available
        self.stats = data.get('stats', None)
        
        print(f"✓ Loaded {self.num_demos:,} demonstration observations")
        print(f"  Observation shape: {self.demo_obs.shape}")
        
        # Print data statistics
        self._print_statistics()
    
    def sample(self, batch_size):
        """
        Sample random batch of demonstration observations.
        
        Args:
            batch_size: Number of observations to sample
            
        Returns:
            Tensor of shape [batch_size, obs_dim]
        """
        if batch_size > self.num_demos:
            # Sample with replacement if requesting more than available
            indices = torch.randint(0, self.num_demos, (batch_size,), device=self.device)
        else:
            # Sample without replacement
            indices = torch.randperm(self.num_demos, device=self.device)[:batch_size]
        
        return self.demo_obs[indices]
    
    def get_all(self):
        """Return all demonstration observations."""
        return self.demo_obs
    
    def get_stats(self):
        """Return dataset statistics if available."""
        return self.stats
    
    def __len__(self):
        """Return number of demonstration observations."""
        return self.num_demos
    
    def _print_statistics(self):
        """Print statistics about the demonstration data."""
        print(f"\n  Demonstration data statistics:")
        
        # Statistics for normalized 6D format
        # [vel1_x, vel1_y, pos2_x, pos2_y, vel2_x, vel2_y]
        dim_names = [
            'First X Velocity',
            'First Y Velocity', 
            'Relative X Position',
            'Relative Y Position',
            'Relative X Velocity',
            'Relative Y Velocity'
        ]
        
        print(f"\n  Normalized observations (6D):")
        for dim_idx, dim_name in enumerate(dim_names):
            values = self.demo_obs[:, dim_idx]
            print(f"    {dim_name}: "
                  f"mean={values.mean():.4f}, "
                  f"std={values.std():.4f}, "
                  f"min={values.min():.4f}, "
                  f"max={values.max():.4f}")
        
        # Verify first velocity y-component is near zero (should be after normalization)
        vel1_y_mean = self.demo_obs[:, 1].abs().mean()
        print(f"\n  Normalization check:")
        print(f"    First velocity Y-component (should be ~0): mean_abs={vel1_y_mean:.6f}")
        
        # Check for any NaN or Inf
        if not torch.isfinite(self.demo_obs).all():
            print("\n  ⚠ WARNING: Demo data contains NaN or Inf values!")
