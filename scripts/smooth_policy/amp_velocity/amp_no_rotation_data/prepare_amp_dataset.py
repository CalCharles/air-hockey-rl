#!/usr/bin/env python3
"""
Prepare AMP (Adversarial Motion Priors) dataset from real robot trajectories.

This script processes all trajectory HDF5 files from /nfs/data/airhockey and creates
a PyTorch tensor dataset consisting of consecutive pairs of states.

Each state is a 4D vector: [x_position, y_position, x_velocity, y_velocity]

Output format: PyTorch tensor of shape [N, 2, 4] where:
    - N = total number of consecutive state pairs across all trajectories
    - 2 = current state and next state
    - 4 = [x_pos, y_pos, x_vel, y_vel]
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse


def find_trajectory_files(data_dir):
    """
    Find all trajectory HDF5 files in the data directory.
    
    Args:
        data_dir: Path to directory containing trajectory files
        
    Returns:
        list: Sorted list of Path objects for trajectory files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all trajectory_data*.hdf5 files
    trajectory_files = sorted(data_path.glob('trajectory_data*.hdf5'))
    
    if not trajectory_files:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")
    
    return trajectory_files


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


def filter_trajectory(train_vals, min_length=50, safety_check=True):
    """
    Filter out bad trajectories based on quality criteria.
    
    Args:
        train_vals: Full trajectory array
        min_length: Minimum trajectory length
        safety_check: Whether to check safety flags (field 4)
    
    Returns:
        bool: True if trajectory should be kept
    """
    # Check minimum length
    if train_vals.shape[0] < min_length:
        return False
    
    # Check for safety violations
    if safety_check:
        safety_flags = train_vals[:, 4]  # Field 4: safety
        if np.any(safety_flags == 0):  # 0 = unsafe
            return False
    
    # Check for NaN or infinite values in key fields
    key_fields = train_vals[:, [5, 6, 11, 12]]
    if np.any(~np.isfinite(key_fields)):
        return False
    
    return True


def process_all_trajectories(data_dir, min_length=50, max_trajectories=None, 
                            safety_check=True):
    """
    Process all trajectory files and collect state pairs.
    
    Args:
        data_dir: Path to /nfs/data/airhockey
        min_length: Minimum trajectory length to include
        max_trajectories: Optional limit on number of trajectories
        safety_check: Whether to filter based on safety flags
    
    Returns:
        tuple: (dataset, stats)
            - dataset: np.ndarray of shape (N, 2, 4)
            - stats: dict with processing statistics
    """
    trajectory_files = find_trajectory_files(data_dir)
    
    if max_trajectories is not None:
        trajectory_files = trajectory_files[:max_trajectories]
    
    print(f"Found {len(trajectory_files)} trajectory files")
    
    all_pairs = []
    stats = {
        'total_files': len(trajectory_files),
        'valid_trajectories': 0,
        'skipped_trajectories': 0,
        'total_pairs': 0,
        'total_timesteps': 0,
        'error_files': []
    }
    
    for file_path in tqdm(trajectory_files, desc="Processing trajectories"):
        try:
            # Load trajectory data
            with h5py.File(file_path, 'r') as f:
                train_vals = f['train_vals'][:]
            
            # Filter trajectory
            if not filter_trajectory(train_vals, min_length, safety_check):
                stats['skipped_trajectories'] += 1
                continue
            
            # Extract state pairs
            pairs = extract_state_pairs(train_vals)
            all_pairs.append(pairs)
            
            stats['valid_trajectories'] += 1
            stats['total_pairs'] += pairs.shape[0]
            stats['total_timesteps'] += train_vals.shape[0]
            
        except Exception as e:
            print(f"\nError processing {file_path.name}: {e}")
            stats['skipped_trajectories'] += 1
            stats['error_files'].append((file_path.name, str(e)))
            continue
    
    # Concatenate all pairs
    if not all_pairs:
        raise ValueError("No valid trajectories found!")
    
    dataset = np.concatenate(all_pairs, axis=0)
    
    return dataset, stats


def save_dataset(dataset, output_path, stats=None):
    """
    Save dataset as PyTorch tensor file.
    
    Args:
        dataset: np.ndarray of shape (N, 2, 4)
        output_path: Where to save the .pt file
        stats: Optional statistics dictionary
    """
    # Convert to torch tensor
    tensor_dataset = torch.from_numpy(dataset).float()
    
    # Create save dictionary
    save_dict = {
        'state_pairs': tensor_dataset,
        'dataset_shape': tensor_dataset.shape,
    }
    
    if stats is not None:
        # Convert stats to be JSON-serializable (remove Path objects)
        stats_copy = stats.copy()
        if 'error_files' in stats_copy:
            stats_copy['error_files'] = [(str(f), e) for f, e in stats_copy['error_files']]
        save_dict['stats'] = stats_copy
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved dataset to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")


def print_dataset_statistics(dataset, stats):
    """Print comprehensive statistics about the dataset."""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\nFiles processed:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Valid trajectories: {stats['valid_trajectories']}")
    print(f"  Skipped trajectories: {stats['skipped_trajectories']}")
    
    if stats['error_files']:
        print(f"  Files with errors: {len(stats['error_files'])}")
    
    print(f"\nDataset size:")
    print(f"  Total state pairs: {dataset.shape[0]:,}")
    print(f"  Total timesteps: {stats['total_timesteps']:,}")
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare AMP dataset from real robot trajectories',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='/nfs/data/airhockey',
        help='Directory containing trajectory HDF5 files'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='amp_dataset.pt',
        help='Output path for the PyTorch dataset file'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Minimum trajectory length to include'
    )
    parser.add_argument(
        '--max-trajectories',
        type=int,
        default=None,
        help='Maximum number of trajectories to process (for testing)'
    )
    parser.add_argument(
        '--no-safety-check',
        action='store_true',
        help='Disable safety flag filtering'
    )
    
    return parser.parse_args()


def main():
    """Main function to prepare AMP dataset."""
    args = parse_args()
    
    print("="*80)
    print("AMP DATASET PREPARATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output path: {args.output_path}")
    print(f"  Min trajectory length: {args.min_length}")
    print(f"  Max trajectories: {args.max_trajectories or 'unlimited'}")
    print(f"  Safety check: {not args.no_safety_check}")
    
    # Process all trajectories
    dataset, stats = process_all_trajectories(
        args.data_dir,
        min_length=args.min_length,
        max_trajectories=args.max_trajectories,
        safety_check=not args.no_safety_check
    )
    
    # Print statistics
    print_dataset_statistics(dataset, stats)
    
    # Validate
    if not validate_dataset(dataset):
        print("\n⚠ WARNING: Validation failed! Proceeding with save anyway...")
    
    # Save dataset
    output_path = Path(args.output_path)
    save_dataset(dataset, output_path, stats)
    
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


if __name__ == '__main__':
    main()
