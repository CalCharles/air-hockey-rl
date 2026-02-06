#!/usr/bin/env python3
"""
Prepare AMP dataset with 5 consecutive position states from real robot trajectories.

This script processes trajectory HDF5 files and creates a PyTorch tensor dataset 
consisting of 5 consecutive position states.

Each state is a 2D vector: [x_position, y_position]

Output format: PyTorch tensor of shape [N, 5, 2] where:
    - N = total number of consecutive state sequences across all trajectories
    - 5 = five consecutive states (t, t+1, t+2, t+3, t+4)
    - 2 = [x_pos, y_pos]
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse


def load_demo_list(demo_list_path):
    """
    Load list of trajectory names from a text file.
    
    Args:
        demo_list_path: Path to text file with one trajectory name per line
        
    Returns:
        list: List of trajectory IDs (e.g., ['442', '441', '440'])
    """
    with open(demo_list_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Extract numeric IDs from names (handles both "442" and "trajectory_data442")
    trajectory_ids = []
    for line in lines:
        if line.startswith('trajectory_data'):
            trajectory_ids.append(line.replace('trajectory_data', ''))
        else:
            trajectory_ids.append(line)
    
    return trajectory_ids


def find_trajectory_files(data_dir, demo_list=None):
    """
    Find trajectory HDF5 files in the data directory.
    
    Args:
        data_dir: Path to directory containing trajectory files
        demo_list: Optional list of trajectory IDs to filter
        
    Returns:
        list: Sorted list of Path objects for trajectory files
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all trajectory_data*.hdf5 files
    all_trajectory_files = sorted(data_path.glob('trajectory_data*.hdf5'))
    
    if not all_trajectory_files:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")
    
    # Filter by demo list if provided
    if demo_list is not None:
        filtered_files = []
        for traj_file in all_trajectory_files:
            # Extract ID from filename (e.g., "trajectory_data442.hdf5" -> "442")
            traj_id = traj_file.stem.replace('trajectory_data', '')
            if traj_id in demo_list:
                filtered_files.append(traj_file)
        
        if not filtered_files:
            raise ValueError(f"None of the specified trajectories found in {data_dir}")
        
        return filtered_files
    
    return all_trajectory_files


def extract_position_vector(train_vals, idx):
    """
    Extract 2D position vector at given timestep index.
    
    According to FIELD_DOCUMENTATION.md:
    - Field 5: pose_x (X position in meters)
    - Field 6: pose_y (Y position in meters)
    
    Args:
        train_vals: Full trajectory array (T x 32)
        idx: Timestep index
    
    Returns:
        np.ndarray: [x_pos, y_pos] shape (2,)
    """
    position = np.array([
        train_vals[idx, 5],   # x position
        train_vals[idx, 6],   # y position
    ], dtype=np.float32)
    
    return position


def extract_position_sequences(train_vals, sequence_length=5):
    """
    Extract all consecutive position sequences from a single trajectory.
    
    Args:
        train_vals: Full trajectory array (T x 32)
        sequence_length: Number of consecutive states to include (default: 5)
    
    Returns:
        np.ndarray: Array of position sequences, shape (T-sequence_length+1, sequence_length, 2)
    """
    n_timesteps = train_vals.shape[0]
    sequences = []
    
    # Need at least sequence_length timesteps for a sequence
    for t in range(n_timesteps - sequence_length + 1):
        sequence = []
        for i in range(sequence_length):
            position = extract_position_vector(train_vals, t + i)
            sequence.append(position)
        sequences.append(np.stack(sequence))
    
    return np.array(sequences, dtype=np.float32)


def filter_trajectory(train_vals, min_length=50, safety_check=True):
    """
    Filter out bad trajectories based on quality criteria.
    
    Args:
        train_vals: Full trajectory array
        min_length: Minimum trajectory length (need at least 5 for sequences)
        safety_check: Whether to check safety flags (field 4)
    
    Returns:
        bool: True if trajectory should be kept
    """
    # Check minimum length (need at least 5 timesteps for sequences)
    if train_vals.shape[0] < max(min_length, 5):
        return False
    
    # Check for safety violations
    if safety_check:
        safety_flags = train_vals[:, 4]  # Field 4: safety
        if np.any(safety_flags == 0):  # 0 = unsafe
            return False
    
    # Check for NaN or infinite values in position fields
    position_fields = train_vals[:, [5, 6]]
    if np.any(~np.isfinite(position_fields)):
        return False
    
    return True


def process_all_trajectories(data_dir, min_length=50, max_trajectories=None, 
                            safety_check=True, demo_list=None, sequence_length=5):
    """
    Process all trajectory files and collect position sequences.
    
    Args:
        data_dir: Path to directory containing trajectory files
        min_length: Minimum trajectory length to include
        max_trajectories: Optional limit on number of trajectories
        safety_check: Whether to filter based on safety flags
        demo_list: Optional list of trajectory IDs to process
        sequence_length: Number of consecutive states per sequence (default: 5)
    
    Returns:
        tuple: (dataset, stats)
            - dataset: np.ndarray of shape (N, sequence_length, 2)
            - stats: dict with processing statistics
    """
    trajectory_files = find_trajectory_files(data_dir, demo_list)
    
    if max_trajectories is not None:
        trajectory_files = trajectory_files[:max_trajectories]
    
    print(f"Found {len(trajectory_files)} trajectory files to process")
    if demo_list is not None:
        print(f"  (filtered by demo list: {len(demo_list)} trajectories specified)")
    
    all_sequences = []
    stats = {
        'total_files': len(trajectory_files),
        'valid_trajectories': 0,
        'skipped_trajectories': 0,
        'total_sequences': 0,
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
            
            # Extract position sequences
            sequences = extract_position_sequences(train_vals, sequence_length)
            all_sequences.append(sequences)
            
            stats['valid_trajectories'] += 1
            stats['total_sequences'] += sequences.shape[0]
            stats['total_timesteps'] += train_vals.shape[0]
            
        except Exception as e:
            print(f"\nError processing {file_path.name}: {e}")
            stats['skipped_trajectories'] += 1
            stats['error_files'].append((file_path.name, str(e)))
            continue
    
    # Concatenate all sequences
    if not all_sequences:
        raise ValueError("No valid trajectories found!")
    
    dataset = np.concatenate(all_sequences, axis=0)
    
    return dataset, stats


def save_dataset(dataset, output_path, stats=None):
    """
    Save dataset as PyTorch tensor file.
    
    Args:
        dataset: np.ndarray of shape (N, 5, 2)
        output_path: Where to save the .pt file
        stats: Optional statistics dictionary
    """
    # Convert to torch tensor
    tensor_dataset = torch.from_numpy(dataset).float()
    
    # Create save dictionary
    save_dict = {
        'position_sequences': tensor_dataset,
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
    print(f"  Total position sequences: {dataset.shape[0]:,}")
    print(f"  Total timesteps: {stats['total_timesteps']:,}")
    print(f"  Shape: {dataset.shape}")
    print(f"  Memory size: {dataset.nbytes / (1024**2):.2f} MB")
    
    # Compute statistics for each position dimension
    all_positions = dataset.reshape(-1, 2)
    dim_names = ['X Position (m)', 'Y Position (m)']
    
    print(f"\nPosition vector statistics:")
    for i, name in enumerate(dim_names):
        values = all_positions[:, i]
        print(f"  {name}:")
        print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"    Mean: {values.mean():.4f}")
        print(f"    Std: {values.std():.4f}")


def validate_dataset(dataset, sequence_length=5):
    """
    Perform validation checks on the dataset.
    
    Args:
        dataset: np.ndarray to validate
        sequence_length: Expected sequence length
        
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
    assert dataset.shape[1] == sequence_length, f"Expected {sequence_length} states per sequence, got {dataset.shape[1]}"
    assert dataset.shape[2] == 2, f"Expected 2D position vector, got {dataset.shape[2]}D"
    
    print("  ✓ All validation checks passed")
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare AMP dataset with 5 consecutive position states from real robot trajectories',
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
        default='scripts/smooth_policy/amp_history/amp_training/amp_data/amp_position_dataset.pt',
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
    parser.add_argument(
        '--demo-list',
        type=str,
        default=None,
        help='Path to text file containing list of trajectory names to process (one per line)'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=5,
        help='Number of consecutive states per sequence'
    )
    
    return parser.parse_args()


def main():
    """Main function to prepare AMP position dataset."""
    args = parse_args()
    
    print("="*80)
    print("AMP POSITION DATASET PREPARATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output path: {args.output_path}")
    print(f"  Min trajectory length: {args.min_length}")
    print(f"  Max trajectories: {args.max_trajectories or 'unlimited'}")
    print(f"  Safety check: {not args.no_safety_check}")
    print(f"  Sequence length: {args.sequence_length}")
    
    # Load demo list if provided
    demo_list = None
    if args.demo_list:
        print(f"  Demo list file: {args.demo_list}")
        demo_list = load_demo_list(args.demo_list)
        print(f"  Loaded {len(demo_list)} trajectory IDs from demo list")
    
    # Process all trajectories
    dataset, stats = process_all_trajectories(
        args.data_dir,
        min_length=args.min_length,
        max_trajectories=args.max_trajectories,
        safety_check=not args.no_safety_check,
        demo_list=demo_list,
        sequence_length=args.sequence_length
    )
    
    # Print statistics
    print_dataset_statistics(dataset, stats)
    
    # Validate
    if not validate_dataset(dataset, args.sequence_length):
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
    print("  position_sequences = data['position_sequences']  # Shape: (N, 5, 2)")
    print("  state1 = position_sequences[:, 0, :]  # (N, 2)")
    print("  state2 = position_sequences[:, 1, :]  # (N, 2)")
    print("  state3 = position_sequences[:, 2, :]  # (N, 2)")
    print("  state4 = position_sequences[:, 3, :]  # (N, 2)")
    print("  state5 = position_sequences[:, 4, :]  # (N, 2)")


if __name__ == '__main__':
    main()
