#!/usr/bin/env python3
"""
Prepare AMP dataset from a single trajectory for overfitting verification.

This script extracts a specific number of frames from a single trajectory HDF5 file
and creates a PyTorch tensor dataset consisting of consecutive pairs of states.

This is useful for:
- Verifying that AMP training can overfit on a small dataset
- Testing the training pipeline with minimal data
- Debugging discriminator and policy learning

Each state is a 4D vector: [x_position, y_position, x_velocity, y_velocity]

Output format: PyTorch tensor of shape [N-1, 2, 4] where:
    - N-1 = number of consecutive state pairs (max_frames - 1)
    - 2 = current state and next state
    - 4 = [x_pos, y_pos, x_vel, y_vel]
"""

import h5py
import numpy as np
import torch
from pathlib import Path
import argparse


def load_trajectory_data(filepath, max_frames=None):
    """
    Load trajectory data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 trajectory file
        max_frames: Maximum number of frames to load (None = all frames)
        
    Returns:
        numpy.ndarray: Train values array (T, 32) where T <= max_frames
    """
    print(f"Loading trajectory data from: {filepath}")
    with h5py.File(filepath, 'r') as f:
        train_vals = f['train_vals'][:]
        total_frames = train_vals.shape[0]
        print(f"  Total frames in file: {total_frames}")
        
        if max_frames is not None and max_frames < total_frames:
            train_vals = train_vals[:max_frames]
            print(f"  Using first {max_frames} frames")
        else:
            print(f"  Using all {total_frames} frames")
    
    return train_vals


def extract_state_vector(train_vals, idx):
    """
    Extract 4D state vector at given timestep index.
    
    According to FIELD_DOCUMENTATION.md:
    - Field 5: pose_x (X position in meters)
    - Field 6: pose_y (Y position in meters)
    - Field 11: speed_vx (X velocity in m/s)
    - Field 12: speed_vy (Y velocity in m/s)
    
    Args:
        train_vals: Full trajectory array (T x 32)
        idx: Timestep index
    
    Returns:
        np.ndarray: [x_pos, y_pos, x_vel, y_vel] shape (4,)
    """
    state = np.array([
        train_vals[idx, 5],   # x position
        train_vals[idx, 6],   # y position
        train_vals[idx, 11],  # x velocity
        train_vals[idx, 12]   # y velocity
    ], dtype=np.float32)
    
    return state


def extract_state_pairs(train_vals):
    """
    Extract all consecutive state pairs from a single trajectory.
    
    Args:
        train_vals: Full trajectory array (T x 32)
    
    Returns:
        np.ndarray: Array of state pairs, shape (T-1, 2, 4)
    """
    n_timesteps = train_vals.shape[0]
    pairs = []
    
    for t in range(n_timesteps - 1):
        current_state = extract_state_vector(train_vals, t)
        next_state = extract_state_vector(train_vals, t + 1)
        pairs.append(np.stack([current_state, next_state]))
    
    return np.array(pairs, dtype=np.float32)


def validate_dataset(dataset):
    """
    Perform validation checks on the dataset.
    
    Args:
        dataset: np.ndarray to validate
        
    Returns:
        bool: True if all checks pass
    """
    print("\nValidating dataset...")
    
    # Check for NaN/Inf
    if np.any(~np.isfinite(dataset)):
        print("  ⚠ WARNING: Dataset contains NaN or infinite values!")
        return False
    
    # Check shape
    assert len(dataset.shape) == 3, f"Expected 3D array, got {len(dataset.shape)}D"
    assert dataset.shape[1] == 2, f"Expected 2 states per pair, got {dataset.shape[1]}"
    assert dataset.shape[2] == 4, f"Expected 4D state vector, got {dataset.shape[2]}D"
    
    print("  ✓ All validation checks passed")
    return True


def print_dataset_statistics(dataset, trajectory_file):
    """Print comprehensive statistics about the dataset."""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\nSource trajectory:")
    print(f"  File: {trajectory_file}")
    print(f"  Frames used: {dataset.shape[0] + 1}")
    
    print(f"\nDataset size:")
    print(f"  Total state pairs: {dataset.shape[0]:,}")
    print(f"  Shape: {dataset.shape}")
    print(f"  Memory size: {dataset.nbytes / (1024**2):.2f} MB")
    
    # Compute statistics for each state dimension
    all_states = dataset.reshape(-1, 4)
    dim_names = ['X Position (m)', 'Y Position (m)', 
                 'X Velocity (m/s)', 'Y Velocity (m/s)']
    
    print(f"\nState vector statistics:")
    for i, name in enumerate(dim_names):
        values = all_states[:, i]
        print(f"  {name}:")
        print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"    Mean: {values.mean():.4f}")
        print(f"    Std: {values.std():.4f}")


def save_dataset(dataset, output_path, trajectory_file, max_frames):
    """
    Save dataset as PyTorch tensor file.
    
    Args:
        dataset: np.ndarray of shape (N, 2, 4)
        output_path: Where to save the .pt file
        trajectory_file: Source trajectory file path
        max_frames: Number of frames used
    """
    # Convert to torch tensor
    tensor_dataset = torch.from_numpy(dataset).float()
    
    # Create save dictionary
    save_dict = {
        'state_pairs': tensor_dataset,
        'dataset_shape': tensor_dataset.shape,
        'source_trajectory': str(trajectory_file),
        'max_frames': max_frames,
        'num_pairs': tensor_dataset.shape[0],
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved dataset to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare AMP dataset from a single trajectory for overfitting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--trajectory-file',
        type=str,
        default='/nfs/data/airhockey/trajectory_data442.hdf5',
        help='Path to specific trajectory HDF5 file'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=100,
        help='Maximum number of frames to use from the trajectory'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='scripts/smooth_policy/amp_no_rotation_data/trajectory_data442_first100.pt',
        help='Output path for the PyTorch dataset file'
    )
    
    return parser.parse_args()


def main():
    """Main function to prepare single trajectory AMP dataset."""
    args = parse_args()
    
    print("="*80)
    print("SINGLE TRAJECTORY AMP DATASET PREPARATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Trajectory file: {args.trajectory_file}")
    print(f"  Max frames: {args.max_frames}")
    print(f"  Output path: {args.output_path}")
    
    # Check if trajectory file exists
    trajectory_path = Path(args.trajectory_file)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {args.trajectory_file}")
    
    # Load trajectory data
    train_vals = load_trajectory_data(args.trajectory_file, args.max_frames)
    
    # Extract state pairs
    print(f"\nExtracting state pairs...")
    dataset = extract_state_pairs(train_vals)
    print(f"  ✓ Extracted {dataset.shape[0]} state pairs")
    
    # Print statistics
    print_dataset_statistics(dataset, args.trajectory_file)
    
    # Validate
    if not validate_dataset(dataset):
        print("\n⚠ WARNING: Validation failed! Proceeding with save anyway...")
    
    # Save dataset
    save_dataset(dataset, args.output_path, args.trajectory_file, args.max_frames)
    
    print("\n" + "="*80)
    print("✓ Dataset preparation complete!")
    print("="*80)
    
    # Print usage example
    print("\nTo load the dataset in Python:")
    print("  import torch")
    print(f"  data = torch.load('{args.output_path}')")
    print("  state_pairs = data['state_pairs']  # Shape: (N, 2, 4)")
    print("  current_states = state_pairs[:, 0, :]  # (N, 4)")
    print("  next_states = state_pairs[:, 1, :]     # (N, 4)")
    print(f"\nTo use with AMP training:")
    print(f"  python scripts/smooth_policy/amp_no_rotation/amp_training_lsgan.py \\")
    print(f"    --args-file scripts/smooth_policy/amp_no_rotation/overfit_single_trajectory_args.yaml \\")
    print(f"    --demo-data-path {args.output_path}")


if __name__ == '__main__':
    main()
