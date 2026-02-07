#!/usr/bin/env python3
"""
Prepare AMP dataset from multiple trajectories for overfitting verification.

This script extracts a specific number of frames from multiple trajectory HDF5 files
and creates a PyTorch tensor dataset consisting of consecutive pairs of states.

This is useful for:
- Overfitting on a small set of diverse demonstrations
- Testing the training pipeline with controlled data size
- Debugging discriminator and policy learning with multiple motion patterns

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
import argparse
import json


def load_trajectory_data(filepath, max_frames=None):
    """
    Load trajectory data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 trajectory file
        max_frames: Maximum number of frames to load (None = all frames)
        
    Returns:
        numpy.ndarray: Train values array (T, 32) where T <= max_frames
    """
    print(f"  Loading: {filepath}")
    with h5py.File(filepath, 'r') as f:
        train_vals = f['train_vals'][:]
        total_frames = train_vals.shape[0]
        
        if max_frames is not None and max_frames < total_frames:
            train_vals = train_vals[:max_frames]
            print(f"    Using first {max_frames} of {total_frames} frames")
        else:
            print(f"    Using all {total_frames} frames")
    
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


def process_trajectories(trajectory_configs):
    """
    Process multiple trajectories and collect state pairs.
    
    Args:
        trajectory_configs: List of dicts with 'path' and optional 'max_frames'
        
    Returns:
        tuple: (dataset, metadata)
            - dataset: np.ndarray of shape (N, 2, 4)
            - metadata: dict with processing information
    """
    all_pairs = []
    trajectory_info = []
    
    print("\nProcessing trajectories:")
    print("="*80)
    
    for i, config in enumerate(trajectory_configs):
        filepath = config['path']
        max_frames = config.get('max_frames', None)
        
        # Check if file exists
        if not Path(filepath).exists():
            print(f"  ⚠ WARNING: File not found, skipping: {filepath}")
            continue
        
        try:
            # Load trajectory data
            train_vals = load_trajectory_data(filepath, max_frames)
            
            # Extract state pairs
            pairs = extract_state_pairs(train_vals)
            all_pairs.append(pairs)
            
            # Store trajectory info
            trajectory_info.append({
                'index': i,
                'path': str(filepath),
                'frames_used': train_vals.shape[0],
                'pairs_extracted': pairs.shape[0]
            })
            
            print(f"    ✓ Extracted {pairs.shape[0]} state pairs")
            
        except Exception as e:
            print(f"  ⚠ ERROR processing {filepath}: {e}")
            continue
    
    print("="*80)
    
    if not all_pairs:
        raise ValueError("No valid trajectories were processed!")
    
    # Concatenate all pairs
    dataset = np.concatenate(all_pairs, axis=0)
    
    metadata = {
        'num_trajectories': len(trajectory_info),
        'total_pairs': dataset.shape[0],
        'trajectories': trajectory_info
    }
    
    return dataset, metadata


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


def print_dataset_statistics(dataset, metadata):
    """Print comprehensive statistics about the dataset."""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    print(f"\nSource trajectories: {metadata['num_trajectories']}")
    for traj_info in metadata['trajectories']:
        print(f"  [{traj_info['index']}] {Path(traj_info['path']).name}")
        print(f"      Frames: {traj_info['frames_used']}, Pairs: {traj_info['pairs_extracted']}")
    
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


def save_dataset(dataset, output_path, metadata):
    """
    Save dataset as PyTorch tensor file.
    
    Args:
        dataset: np.ndarray of shape (N, 2, 4)
        output_path: Where to save the .pt file
        metadata: Processing metadata dict
    """
    # Convert to torch tensor
    tensor_dataset = torch.from_numpy(dataset).float()
    
    # Create save dictionary
    save_dict = {
        'state_pairs': tensor_dataset,
        'dataset_shape': tensor_dataset.shape,
        'metadata': metadata
    }
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved dataset to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")


def parse_trajectory_list(trajectory_list_str):
    """
    Parse trajectory list from command line argument.
    
    Supports formats:
    - Comma-separated paths: "path1.hdf5,path2.hdf5,path3.hdf5"
    - JSON array: '[{"path": "path1.hdf5", "max_frames": 100}, ...]'
    
    Args:
        trajectory_list_str: String representation of trajectory list
        
    Returns:
        list: List of trajectory config dicts
    """
    # Try parsing as JSON first
    if trajectory_list_str.strip().startswith('['):
        try:
            configs = json.loads(trajectory_list_str)
            # Ensure each config has a 'path' key
            for config in configs:
                if 'path' not in config:
                    raise ValueError("Each trajectory config must have a 'path' key")
            return configs
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    # Otherwise, treat as comma-separated paths
    paths = [p.strip() for p in trajectory_list_str.split(',')]
    return [{'path': p} for p in paths if p]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Prepare AMP dataset from multiple trajectories',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  # 5 trajectories with default frames
  python %(prog)s --trajectories "/nfs/data/airhockey/trajectory_data442.hdf5,/nfs/data/airhockey/trajectory_data441.hdf5,/nfs/data/airhockey/trajectory_data440.hdf5,/nfs/data/airhockey/trajectory_data439.hdf5,/nfs/data/airhockey/trajectory_data438.hdf5"
  
  # With custom frame limits per trajectory (JSON format)
  python %(prog)s --trajectories '[{"path": "/nfs/data/airhockey/trajectory_data442.hdf5", "max_frames": 100}, {"path": "/nfs/data/airhockey/trajectory_data441.hdf5", "max_frames": 150}]'
  
  # Using config file
  python %(prog)s --config-file my_trajectories.json
        """
    )
    
    # Trajectory specification (mutually exclusive)
    traj_group = parser.add_mutually_exclusive_group(required=True)
    traj_group.add_argument(
        '--trajectories',
        type=str,
        help='Comma-separated list of trajectory paths, or JSON array of configs'
    )
    traj_group.add_argument(
        '--config-file',
        type=str,
        help='JSON file containing trajectory configurations'
    )
    
    # Global settings
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Default max frames for all trajectories (can be overridden per-trajectory in JSON)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='scripts/smooth_policy/amp_no_rotation_data/multi_trajectory_dataset.pt',
        help='Output path for the PyTorch dataset file'
    )
    
    return parser.parse_args()


def main():
    """Main function to prepare multi-trajectory AMP dataset."""
    args = parse_args()
    
    print("="*80)
    print("MULTI-TRAJECTORY AMP DATASET PREPARATION")
    print("="*80)
    
    # Load trajectory configurations
    if args.config_file:
        print(f"\nLoading configuration from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            trajectory_configs = json.load(f)
    else:
        trajectory_configs = parse_trajectory_list(args.trajectories)
    
    # Apply global max_frames if specified and not overridden
    if args.max_frames is not None:
        for config in trajectory_configs:
            if 'max_frames' not in config:
                config['max_frames'] = args.max_frames
    
    print(f"\nConfiguration:")
    print(f"  Number of trajectories: {len(trajectory_configs)}")
    print(f"  Output path: {args.output_path}")
    
    # Process all trajectories
    dataset, metadata = process_trajectories(trajectory_configs)
    
    # Print statistics
    print_dataset_statistics(dataset, metadata)
    
    # Validate
    if not validate_dataset(dataset):
        print("\n⚠ WARNING: Validation failed! Proceeding with save anyway...")
    
    # Save dataset
    save_dataset(dataset, args.output_path, metadata)
    
    print("\n" + "="*80)
    print("✓ Dataset preparation complete!")
    print("="*80)
    
    # Print usage example
    print("\nTo load the dataset in Python:")
    print("  import torch")
    print(f"  data = torch.load('{args.output_path}')")
    print("  state_pairs = data['state_pairs']  # Shape: (N, 2, 4)")
    print("  metadata = data['metadata']  # Processing information")
    print(f"\nTo use with AMP training:")
    print(f"  python scripts/smooth_policy/amp_no_rotation/amp_training_lsgan.py \\")
    print(f"    --args-file scripts/smooth_policy/amp_no_rotation/overfit_multi_trajectory_args.yaml \\")
    print(f"    --demo-data-path {args.output_path}")


if __name__ == '__main__':
    main()
