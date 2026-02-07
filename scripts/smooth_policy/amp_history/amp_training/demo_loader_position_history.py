"""
Demo data loader for AMP training with 5 consecutive position history.

Loads expert demonstration data and provides random sampling for discriminator training.
Uses only paddle (x, y) positions, not velocities.

Supports position history format: [N, 5, 2] where each entry contains 5 consecutive (x, y) positions.
Compatible with datasets using keys: 'position_sequences', 'position_history', 'normalized_position_history', or 'state_pairs'.
"""

import torch
from pathlib import Path


def normalize_position_history_batch(position_history):
    """
    Normalize batched position history to relative coordinates (translation only).
    
    Process:
    1. Translate all positions so the first position is at (0, 0)
    2. Remove the first position (now [0, 0], contains no information)
    3. Return the remaining 4 relative positions (8 dimensions)
    
    Args:
        position_history: Tensor of shape [batch, 5, 2] containing 5 consecutive (x, y) positions
    
    Returns:
        torch.Tensor: Normalized positions of shape [batch, 8]
                     [pos2_x, pos2_y, pos3_x, pos3_y, pos4_x, pos4_y, pos5_x, pos5_y]
                     (all relative to pos1 which is at origin)
    """
    # Extract the first position
    pos1 = position_history[:, 0, :]  # [batch, 2]
    
    # Translate all positions so first is at origin
    translated = position_history - pos1.unsqueeze(1)  # [batch, 5, 2]
    
    # Remove first position (now [0, 0]) and flatten the remaining 4 positions
    normalized_state = translated[:, 1:, :].reshape(-1, 8)  # [batch, 8]
    
    return normalized_state


class DemoLoaderPositionHistory:
    """
    Loader for expert demonstration data with 5 consecutive positions (position history).
    
    Loads position sequences from .pt files and provides random sampling for 
    training the discriminator.
    """
    
    def __init__(self, demo_path, device='cuda'):
        """
        Initialize demo loader for position history.
        
        Args:
            demo_path: Path to .pt file containing demonstration data
            device: Device to store tensors on
        """
        self.demo_path = Path(demo_path)
        self.device = device
        
        if not self.demo_path.exists():
            raise FileNotFoundError(f"Demo data not found at: {demo_path}")
        
        # Load data
        print(f"Loading demonstration position history data from: {demo_path}")
        data = torch.load(demo_path, map_location=device)
        
        # Auto-detect dataset format
        if 'normalized_position_history' in data:
            # Pre-normalized dataset [N, 8]
            print("  Detected pre-normalized position history dataset")
            self.demo_obs = data['normalized_position_history'].to(device=device, dtype=torch.float32)
            self.is_normalized = True
            
            if self.demo_obs.dim() != 2 or self.demo_obs.shape[1] != 8:
                raise ValueError(f"Unexpected normalized data shape: {self.demo_obs.shape}. "
                               f"Expected [N, 8]")
        
        elif 'position_history' in data or 'position_sequences' in data:
            # Non-normalized dataset [N, 5, 2] - apply normalization
            # Support both 'position_history' and 'position_sequences' keys
            key = 'position_sequences' if 'position_sequences' in data else 'position_history'
            print(f"  Detected non-normalized position dataset (key: '{key}') - applying normalization")
            self.position_history = data[key].to(device=device, dtype=torch.float32)
            self.is_normalized = False
            
            if self.position_history.dim() != 3 or self.position_history.shape[1] != 5 or self.position_history.shape[2] != 2:
                raise ValueError(f"Unexpected position history shape: {self.position_history.shape}. "
                               f"Expected [N, 5, 2]")
            
            # Apply normalization to convert [N, 5, 2] -> [N, 8]
            print("  Normalizing position history...")
            self.demo_obs = normalize_position_history_batch(self.position_history)
            print(f"  ✓ Normalization complete")
        
        elif 'state_pairs' in data:
            # Fallback: Convert from state_pairs format [N, 2, 4] to position sequences
            # This creates position history by taking consecutive pairs
            print("  Detected state_pairs format - converting to position history")
            state_pairs = data['state_pairs'].to(device=device, dtype=torch.float32)
            
            # Extract only positions (x, y) from state pairs
            # state_pairs: [N, 2, 4] where each state is [x, y, vx, vy]
            positions = state_pairs[:, :, :2]  # [N, 2, 2]
            
            # Build position history by sliding window over consecutive pairs
            # We need 5 positions, so we need to chain pairs together
            # For simplicity, construct from overlapping windows of raw position data
            print("  ⚠ Warning: Building 5-position history from 2-position pairs.")
            print("    For best results, prepare a dedicated position_history dataset.")
            
            # Simple approach: use 5 consecutive pairs to get 5 positions
            # Assume pairs are ordered sequentially
            num_pairs = positions.shape[0]
            num_sequences = num_pairs - 4  # Need 5 consecutive starting positions
            
            if num_sequences <= 0:
                raise ValueError(f"Not enough pairs ({num_pairs}) to create position history. Need at least 5.")
            
            # Build position history sequences
            position_history_list = []
            for i in range(num_sequences):
                # Take first position from each of 5 consecutive pairs
                seq = positions[i:i+5, 0, :]  # [5, 2]
                position_history_list.append(seq)
            
            self.position_history = torch.stack(position_history_list, dim=0)  # [N-4, 5, 2]
            self.is_normalized = False
            
            # Apply normalization
            print("  Normalizing position history...")
            self.demo_obs = normalize_position_history_batch(self.position_history)
            print(f"  ✓ Normalization complete")
        
        else:
            raise ValueError("Dataset must contain 'normalized_position_history', 'position_history', 'position_sequences', or 'state_pairs' key")
        
        self.num_demos = self.demo_obs.shape[0]
        
        # Store statistics if available
        self.stats = data.get('stats', None)
        
        print(f"✓ Loaded {self.num_demos:,} demonstration position history observations")
        print(f"  Observation shape: {self.demo_obs.shape}")
        
        # Print data statistics
        self._print_statistics()
    
    def sample(self, batch_size):
        """
        Sample random batch of demonstration observations.
        
        Args:
            batch_size: Number of observations to sample
            
        Returns:
            Tensor of shape [batch_size, 8] (4 relative positions × 2 coords)
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
        print(f"\n  Demonstration position history data statistics:")
        
        # Statistics for normalized 8D format
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
        
        print(f"\n  Normalized observations (8D - relative positions):")
        for dim_idx, dim_name in enumerate(dim_names):
            values = self.demo_obs[:, dim_idx]
            print(f"    {dim_name}: "
                  f"mean={values.mean():.4f}, "
                  f"std={values.std():.4f}, "
                  f"min={values.min():.4f}, "
                  f"max={values.max():.4f}")
        
        # Compute displacement statistics
        # Displacement from pos1 to pos2
        disp_12 = torch.sqrt(self.demo_obs[:, 0]**2 + self.demo_obs[:, 1]**2).mean()
        # Displacement from pos1 to pos5
        disp_15 = torch.sqrt(self.demo_obs[:, 6]**2 + self.demo_obs[:, 7]**2).mean()
        print(f"\n  Displacement check:")
        print(f"    Position 1→2 displacement (average): {disp_12:.6f}")
        print(f"    Position 1→5 displacement (average): {disp_15:.6f}")
        
        # Check for any NaN or Inf
        if not torch.isfinite(self.demo_obs).all():
            print("\n  ⚠ WARNING: Demo data contains NaN or Inf values!")
