"""
Demo data loader for AMP triplet training.

Loads expert demonstration data prepared by prepare_amp_triplet_dataset.py and provides
random sampling for discriminator training.

Supports both normalized and non-normalized datasets:
- Normalized: [N, 10] format (pre-processed with translation only)
- Non-normalized: [N, 3, 4] format (normalized at load time with translation only)
"""

import torch
from pathlib import Path


def normalize_state_triplet_batch(state_triplets):
    """
    Normalize batched state triplets to relative coordinates (translation only, no rotation).
    
    Process:
    1. Translate all states so first position is at (0, 0)
    2. Keep first velocity and all relative positions/velocities
    
    Args:
        state_triplets: Tensor of shape [batch, 3, 4] containing [state1, state2, state3]
                       where each state is [x_pos, y_pos, x_vel, y_vel]
    
    Returns:
        torch.Tensor: Normalized states of shape [batch, 10]
                     [vel1_x, vel1_y, rel_pos2_x, rel_pos2_y, rel_vel2_x, rel_vel2_y,
                      rel_pos3_x, rel_pos3_y, rel_vel3_x, rel_vel3_y]
    """
    # Extract states
    state1 = state_triplets[:, 0, :]  # [batch, 4]
    state2 = state_triplets[:, 1, :]  # [batch, 4]
    state3 = state_triplets[:, 2, :]  # [batch, 4]
    
    # Extract positions and velocities
    pos1 = state1[:, :2]  # [batch, 2]
    vel1 = state1[:, 2:]  # [batch, 2]
    pos2 = state2[:, :2]  # [batch, 2]
    vel2 = state2[:, 2:]  # [batch, 2]
    pos3 = state3[:, :2]  # [batch, 2]
    vel3 = state3[:, 2:]  # [batch, 2]
    
    # Step 1: Translate so first position is at origin
    pos2_translated = pos2 - pos1  # [batch, 2]
    pos3_translated = pos3 - pos1  # [batch, 2]
    
    # Step 2: Concatenate first velocity and all relative states
    # No rotation is applied - keep velocities as-is
    normalized_state = torch.cat([
        vel1,              # First velocity (2D)
        pos2_translated,   # Relative position 2 (2D)
        vel2,              # Second velocity (2D)
        pos3_translated,   # Relative position 3 (2D)
        vel3               # Third velocity (2D)
    ], dim=-1)  # [batch, 10]
    
    return normalized_state


class DemoLoader:
    """
    Loader for expert demonstration data using state triplets.
    
    Loads state triplets from .pt files created by prepare_amp_triplet_dataset.py and
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
            # Pre-normalized dataset [N, 10]
            print("  Detected pre-normalized dataset")
            self.demo_obs = data['normalized_states'].to(device=device, dtype=torch.float32)
            self.is_normalized = True
            
            if self.demo_obs.dim() != 2 or self.demo_obs.shape[1] != 10:
                raise ValueError(f"Unexpected normalized data shape: {self.demo_obs.shape}. "
                               f"Expected [N, 10]")
        
        elif 'state_triplets' in data:
            # Non-normalized dataset [N, 3, 4] - apply normalization
            print("  Detected non-normalized dataset - applying normalization")
            self.state_triplets = data['state_triplets'].to(device=device, dtype=torch.float32)
            self.is_normalized = False
            
            if self.state_triplets.dim() != 3 or self.state_triplets.shape[1] != 3 or self.state_triplets.shape[2] != 4:
                raise ValueError(f"Unexpected state triplets shape: {self.state_triplets.shape}. "
                               f"Expected [N, 3, 4]")
            
            # Apply normalization to convert [N, 3, 4] -> [N, 10]
            print("  Normalizing state triplets (translation only)...")
            self.demo_obs = normalize_state_triplet_batch(self.state_triplets)
            print(f"  ✓ Normalization complete")
        
        else:
            raise ValueError("Dataset must contain either 'normalized_states' or 'state_triplets' key")
        
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
        
        # Statistics for normalized 10D format
        # [vel1_x, vel1_y, rel_pos2_x, rel_pos2_y, rel_vel2_x, rel_vel2_y,
        #  rel_pos3_x, rel_pos3_y, rel_vel3_x, rel_vel3_y]
        dim_names = [
            'First X Velocity',
            'First Y Velocity',
            'Relative X Position 2',
            'Relative Y Position 2',
            'Second X Velocity',
            'Second Y Velocity',
            'Relative X Position 3',
            'Relative Y Position 3',
            'Third X Velocity',
            'Third Y Velocity'
        ]
        
        print(f"\n  Normalized observations (10D):")
        for dim_idx, dim_name in enumerate(dim_names):
            values = self.demo_obs[:, dim_idx]
            print(f"    {dim_name}: "
                  f"mean={values.mean():.4f}, "
                  f"std={values.std():.4f}, "
                  f"min={values.min():.4f}, "
                  f"max={values.max():.4f}")
        
        # Check for any NaN or Inf
        if not torch.isfinite(self.demo_obs).all():
            print("\n  ⚠ WARNING: Demo data contains NaN or Inf values!")
