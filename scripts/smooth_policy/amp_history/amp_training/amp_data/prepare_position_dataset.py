#!/usr/bin/env python3
"""
Prepare AMP dataset with 5 consecutive position states from real robot trajectories.

This script processes trajectory HDF5 files and creates a PyTorch tensor dataset 
consisting of 5 consecutive position states, optionally with associated actions.

Each state is a 2D vector: [x_position, y_position]
Each action is a 2D vector: [delta_x, delta_y] (desired_pose - pose)

Output format (position only): PyTorch tensor of shape [N, 5, 2] where:
    - N = total number of consecutive state sequences across all trajectories
    - 5 = five consecutive states (t, t+1, t+2, t+3, t+4)
    - 2 = [x_pos, y_pos]

Output format (with actions): Additional tensor of shape [N, 4, 2] where:
    - N = total number of consecutive state sequences across all trajectories
    - 4 = actions for transitions between consecutive states in the 5-state window
          (s1→s2, s2→s3, s3→s4, s4→s5)
    - 2 = [delta_x, delta_y]

Output format (with puck windows): Additional tensor of shape [N, 5, 2] where:
    - N = total number of aligned sequences across all trajectories
    - 5 = puck window aligned with each 5-state paddle window: [t, t+1, t+2, t+3, t+4]
    - 2 = [puck_x, puck_y]

Action data source (per FIELD_DOCUMENTATION.md):
    - Fields 26-27: desired_x, desired_y (commanded target position)
    - Fields 5-6: pose_x, pose_y (current position)
    - Action = desired_pose[:2] - pose[:2]

Puck data source:
    - Fields 32-33: puck_x, puck_y
    - Required when --include-puck is enabled.
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import argparse

from scripts.smooth_policy.amp_history.amp_training.feature_processing import (
    build_puck_discriminator_features_numpy,
)


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


def extract_action_vector(train_vals, idx):
    """
    Extract 2D action vector (delta target position) at given timestep index.
    Action is normalized to unit norm for consistent scale.
    
    According to FIELD_DOCUMENTATION.md:
    - Field 26: desired_x (commanded X position in meters)
    - Field 27: desired_y (commanded Y position in meters)
    - Field 5: pose_x (current X position in meters)
    - Field 6: pose_y (current Y position in meters)
    
    Action = desired_pose - pose (delta target position), normalized to norm 1
    
    Args:
        train_vals: Full trajectory array (T x 32)
        idx: Timestep index
    
    Returns:
        np.ndarray: [delta_x, delta_y] shape (2,), normalized to unit norm
    """
    action = np.array([
        train_vals[idx, 26] - train_vals[idx, 5],   # delta x
        train_vals[idx, 27] - train_vals[idx, 6],   # delta y
    ], dtype=np.float32)
    
    # Normalize to unit norm (handle zero vector case)
    norm = np.linalg.norm(action)
    if norm > 1e-8:
        action = action / norm
    
    return action


def extract_puck_vector(train_vals, idx):
    """
    Extract 2D puck position vector at given timestep index.

    According to FIELD_DOCUMENTATION.md:
    - Field 32: puck_x
    - Field 33: puck_y
    """
    if train_vals.shape[1] < 34:
        raise ValueError(
            f"Puck fields require at least 34 columns (found {train_vals.shape[1]})."
        )
    puck_position = np.array([
        train_vals[idx, 32],  # puck_x
        train_vals[idx, 33],  # puck_y
    ], dtype=np.float32)
    return puck_position


def extract_position_sequences(
    train_vals,
    sequence_length=5,
    include_actions=False,
    include_puck=False,
    puck_window_size=5,
):
    """
    Extract all consecutive position sequences from a single trajectory.
    
    Args:
        train_vals: Full trajectory array (T x 32)
        sequence_length: Number of consecutive states to include (default: 5)
        include_actions: Whether to also extract action sequences (default: False)
        include_puck: Whether to also extract puck windows (default: False)
        puck_window_size: Puck window size aligned to paddle sequence (default: 5)
    
    Returns:
        If include_actions=False and include_puck=False:
            np.ndarray: Array of position sequences, shape (T-sequence_length+1, sequence_length, 2)
        If include_actions=True and include_puck=False:
            tuple: (position_sequences, action_sequences)
                - position_sequences: shape (T-sequence_length+1, sequence_length, 2)
                - action_sequences: shape (T-sequence_length+1, sequence_length-1, 2)
        If include_actions=False and include_puck=True:
            tuple: (position_sequences, puck_sequences)
                - puck_sequences: shape (N, puck_window_size, 2)
        If include_actions=True and include_puck=True:
            tuple: (position_sequences, action_sequences, puck_sequences)
    """
    if include_puck:
        if puck_window_size != 5:
            raise ValueError("puck_window_size must be 5 for AMP puck processing.")
        if puck_window_size % 2 == 0:
            raise ValueError("puck_window_size must be odd.")

    n_timesteps = train_vals.shape[0]
    position_sequences = []
    action_sequences = [] if include_actions else None
    puck_sequences = [] if include_puck else None

    max_start = n_timesteps - sequence_length + 1
    
    # Need at least sequence_length timesteps for a sequence
    for t in range(max_start):
        position_seq = []
        action_seq = [] if include_actions else None
        
        for i in range(sequence_length):
            position = extract_position_vector(train_vals, t + i)
            position_seq.append(position)
            
            if include_actions and i < (sequence_length - 1):
                # Use only transition actions between states in the sequence.
                # For states [s1, s2, s3, s4, s5], keep actions [s1→s2, s2→s3, s3→s4, s4→s5].
                # Excludes the action at s5, which corresponds to the next transition.
                action = extract_action_vector(train_vals, t + i)
                action_seq.append(action)
        
        position_sequences.append(np.stack(position_seq))
        if include_actions:
            action_sequences.append(np.stack(action_seq))
        if include_puck:
            # Aligned puck window [t, t+1, t+2, t+3, t+4].
            # In downstream processing with current_index=2, this corresponds to
            # using the puck position 2 timesteps before the final paddle state.
            puck_start = t
            puck_end_exclusive = t + puck_window_size
            puck_seq = [
                extract_puck_vector(train_vals, puck_idx)
                for puck_idx in range(puck_start, puck_end_exclusive)
            ]
            puck_sequences.append(np.stack(puck_seq))
    
    position_array = np.array(position_sequences, dtype=np.float32)
    
    if include_actions and include_puck:
        action_array = np.array(action_sequences, dtype=np.float32)
        puck_array = np.array(puck_sequences, dtype=np.float32)
        return position_array, action_array, puck_array
    if include_actions:
        action_array = np.array(action_sequences, dtype=np.float32)
        return position_array, action_array
    if include_puck:
        puck_array = np.array(puck_sequences, dtype=np.float32)
        return position_array, puck_array
    
    return position_array


def filter_trajectory(
    train_vals,
    min_length=50,
    safety_check=True,
    include_actions=False,
    include_puck=False,
):
    """
    Filter out bad trajectories based on quality criteria.
    
    Args:
        train_vals: Full trajectory array
        min_length: Minimum trajectory length (need at least 5 for sequences)
        safety_check: Whether to check safety flags (field 4)
        include_actions: Whether to also check action fields for validity
        include_puck: Whether to also check puck fields for validity
    
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
    
    # Check for NaN or infinite values in action fields (desired_pose)
    if include_actions:
        desired_pose_fields = train_vals[:, [26, 27]]
        if np.any(~np.isfinite(desired_pose_fields)):
            return False

    # Check puck field availability and validity
    if include_puck:
        if train_vals.shape[1] < 34:
            return False
        puck_fields = train_vals[:, [32, 33]]
        if np.any(~np.isfinite(puck_fields)):
            return False
    
    return True


def process_all_trajectories(data_dir, min_length=50, max_trajectories=None,
                            safety_check=True, demo_list=None, sequence_length=5,
                            include_actions=False, include_puck=False):
    """
    Process all trajectory files and collect position sequences.
    
    Args:
        data_dir: Path to directory containing trajectory files
        min_length: Minimum trajectory length to include
        max_trajectories: Optional limit on number of trajectories
        safety_check: Whether to filter based on safety flags
        demo_list: Optional list of trajectory IDs to process
        sequence_length: Number of consecutive states per sequence (default: 5)
        include_actions: Whether to also extract action sequences (default: False)
        include_puck: Whether to also extract aligned puck windows (default: False)
    
    Returns:
        If include_actions=False and include_puck=False:
            tuple: (position_dataset, stats)
                - position_dataset: np.ndarray of shape (N, sequence_length, 2)
                - stats: dict with processing statistics
        If include_actions=True and include_puck=False:
            tuple: (position_dataset, action_dataset, stats)
                - position_dataset: np.ndarray of shape (N, sequence_length, 2)
                - action_dataset: np.ndarray of shape (N, sequence_length-1, 2)
                - stats: dict with processing statistics
        If include_actions=False and include_puck=True:
            tuple: (position_dataset, puck_dataset, stats)
        If include_actions=True and include_puck=True:
            tuple: (position_dataset, action_dataset, puck_dataset, stats)
    """
    trajectory_files = find_trajectory_files(data_dir, demo_list)
    
    if max_trajectories is not None:
        trajectory_files = trajectory_files[:max_trajectories]
    
    print(f"Found {len(trajectory_files)} trajectory files to process")
    if demo_list is not None:
        print(f"  (filtered by demo list: {len(demo_list)} trajectories specified)")
    
    all_position_sequences = []
    all_action_sequences = [] if include_actions else None
    all_puck_sequences = [] if include_puck else None
    stats = {
        'total_files': len(trajectory_files),
        'valid_trajectories': 0,
        'skipped_trajectories': 0,
        'total_sequences': 0,
        'total_timesteps': 0,
        'error_files': [],
        'include_actions': include_actions,
        'include_puck': include_puck,
    }
    
    for file_path in tqdm(trajectory_files, desc="Processing trajectories"):
        try:
            # Load trajectory data
            with h5py.File(file_path, 'r') as f:
                train_vals = f['train_vals'][:]
            
            # Filter trajectory
            if not filter_trajectory(
                train_vals,
                min_length,
                safety_check,
                include_actions=include_actions,
                include_puck=include_puck,
            ):
                stats['skipped_trajectories'] += 1
                continue
            
            # Extract position sequences (and optionally action sequences)
            result = extract_position_sequences(
                train_vals,
                sequence_length=sequence_length,
                include_actions=include_actions,
                include_puck=include_puck,
            )

            if include_actions and include_puck:
                position_sequences, action_sequences, puck_sequences = result
                all_position_sequences.append(position_sequences)
                all_action_sequences.append(action_sequences)
                all_puck_sequences.append(puck_sequences)
            elif include_actions:
                position_sequences, action_sequences = result
                all_position_sequences.append(position_sequences)
                all_action_sequences.append(action_sequences)
            elif include_puck:
                position_sequences, puck_sequences = result
                all_position_sequences.append(position_sequences)
                all_puck_sequences.append(puck_sequences)
            else:
                all_position_sequences.append(result)
            
            stats['valid_trajectories'] += 1
            stats['total_sequences'] += position_sequences.shape[0] if (include_actions or include_puck) else result.shape[0]
            stats['total_timesteps'] += train_vals.shape[0]
            
        except Exception as e:
            print(f"\nError processing {file_path.name}: {e}")
            stats['skipped_trajectories'] += 1
            stats['error_files'].append((file_path.name, str(e)))
            continue
    
    # Concatenate all sequences
    if not all_position_sequences:
        raise ValueError("No valid trajectories found!")
    
    position_dataset = np.concatenate(all_position_sequences, axis=0)
    
    if include_actions and include_puck:
        action_dataset = np.concatenate(all_action_sequences, axis=0)
        puck_dataset = np.concatenate(all_puck_sequences, axis=0)
        return position_dataset, action_dataset, puck_dataset, stats
    if include_actions:
        action_dataset = np.concatenate(all_action_sequences, axis=0)
        return position_dataset, action_dataset, stats
    if include_puck:
        puck_dataset = np.concatenate(all_puck_sequences, axis=0)
        return position_dataset, puck_dataset, stats
    
    return position_dataset, stats


def save_dataset(position_dataset, output_path, stats=None, action_dataset=None, puck_dataset=None):
    """
    Save dataset as PyTorch tensor file.
    
    Args:
        position_dataset: np.ndarray of shape (N, 5, 2) for positions
        output_path: Where to save the .pt file
        stats: Optional statistics dictionary
        action_dataset: Optional np.ndarray of shape (N, 4, 2) for transition actions
        puck_dataset: Optional np.ndarray of shape (N, 5, 2) for aligned puck windows
    """
    # Convert to torch tensor
    position_tensor = torch.from_numpy(position_dataset).float()
    
    # Create save dictionary
    save_dict = {
        'position_sequences': position_tensor,
        'dataset_shape': position_tensor.shape,
    }
    
    # Add action sequences if provided
    if action_dataset is not None:
        action_tensor = torch.from_numpy(action_dataset).float()
        save_dict['action_sequences'] = action_tensor
        save_dict['has_actions'] = True
    else:
        save_dict['has_actions'] = False

    if puck_dataset is not None:
        puck_tensor = torch.from_numpy(puck_dataset).float()
        save_dict['puck_sequences'] = puck_tensor
        save_dict['has_puck'] = True
    else:
        save_dict['has_puck'] = False
    
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
    if action_dataset is not None:
        print(f"  Includes action sequences: Yes")
    if puck_dataset is not None:
        print(f"  Includes puck sequences: Yes")


def print_dataset_statistics(position_dataset, stats, action_dataset=None, puck_dataset=None):
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
    print(f"  Total position sequences: {position_dataset.shape[0]:,}")
    print(f"  Total timesteps: {stats['total_timesteps']:,}")
    print(f"  Position shape: {position_dataset.shape}")
    print(f"  Position memory size: {position_dataset.nbytes / (1024**2):.2f} MB")
    
    if action_dataset is not None:
        print(f"  Action shape: {action_dataset.shape}")
        print(f"  Action memory size: {action_dataset.nbytes / (1024**2):.2f} MB")
    if puck_dataset is not None:
        print(f"  Puck shape: {puck_dataset.shape}")
        print(f"  Puck memory size: {puck_dataset.nbytes / (1024**2):.2f} MB")
    
    # Compute statistics for each position dimension
    all_positions = position_dataset.reshape(-1, 2)
    dim_names = ['X Position (m)', 'Y Position (m)']
    
    print(f"\nPosition vector statistics:")
    for i, name in enumerate(dim_names):
        values = all_positions[:, i]
        print(f"  {name}:")
        print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"    Mean: {values.mean():.4f}")
        print(f"    Std: {values.std():.4f}")
    
    # Compute statistics for action dimensions if available
    if action_dataset is not None:
        all_actions = action_dataset.reshape(-1, 2)
        action_dim_names = ['Delta X (m)', 'Delta Y (m)']
        
        print(f"\nAction vector statistics:")
        for i, name in enumerate(action_dim_names):
            values = all_actions[:, i]
            print(f"  {name}:")
            print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
            print(f"    Mean: {values.mean():.4f}")
            print(f"    Std: {values.std():.4f}")

    if puck_dataset is not None:
        all_puck = puck_dataset.reshape(-1, 2)
        puck_dim_names = ['Puck X (m)', 'Puck Y (m)']

        print(f"\nPuck vector statistics:")
        for i, name in enumerate(puck_dim_names):
            values = all_puck[:, i]
            print(f"  {name}:")
            print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
            print(f"    Mean: {values.mean():.4f}")
            print(f"    Std: {values.std():.4f}")

        # Shared puck feature processing preview (noise disabled for deterministic stats).
        puck_features = build_puck_discriminator_features_numpy(
            puck_dataset,
            current_index=2,
            vertical_axis=0,
            downward_positive_direction=1.0,
            downward_speed_max=0.75,
            speed_dt=0.05,
            noise_std=0.0,
        )
        speed_bins = puck_features[:, 3]
        unique_bins, counts = np.unique(speed_bins, return_counts=True)
        print("\nPuck discriminator feature preview (noise_std=0.0, axis=0):")
        print(f"  Feature shape: {puck_features.shape}")
        print(f"  Direction sign mean: {puck_features[:, 2].mean():.4f}")
        print(f"  Downward speed bin distribution: {dict(zip(unique_bins.tolist(), counts.tolist()))}")


def validate_dataset(position_dataset, sequence_length=5, action_dataset=None, puck_dataset=None):
    """
    Perform validation checks on the dataset.
    
    Args:
        position_dataset: np.ndarray of positions to validate
        sequence_length: Expected sequence length
        action_dataset: Optional np.ndarray of actions to validate
        puck_dataset: Optional np.ndarray of puck windows to validate
        
    Returns:
        bool: True if all checks pass
    """
    print("\nValidating dataset...")
    
    # Check for NaN/Inf in positions
    if np.any(~np.isfinite(position_dataset)):
        print("  ⚠ WARNING: Position dataset contains NaN or infinite values!")
        return False
    
    # Check position shape
    assert len(position_dataset.shape) == 3, f"Expected 3D array, got {len(position_dataset.shape)}D"
    assert position_dataset.shape[1] == sequence_length, f"Expected {sequence_length} states per sequence, got {position_dataset.shape[1]}"
    assert position_dataset.shape[2] == 2, f"Expected 2D position vector, got {position_dataset.shape[2]}D"
    
    print("  ✓ Position validation checks passed")
    
    # Validate action dataset if provided
    if action_dataset is not None:
        if np.any(~np.isfinite(action_dataset)):
            print("  ⚠ WARNING: Action dataset contains NaN or infinite values!")
            return False
        
        assert len(action_dataset.shape) == 3, f"Expected 3D action array, got {len(action_dataset.shape)}D"
        assert action_dataset.shape[0] == position_dataset.shape[0], "Position and action counts must match"
        expected_action_len = sequence_length - 1
        assert action_dataset.shape[1] == expected_action_len, (
            f"Expected {expected_action_len} transition actions per sequence, got {action_dataset.shape[1]}"
        )
        assert action_dataset.shape[2] == 2, f"Expected 2D action vector, got {action_dataset.shape[2]}D"
        
        print("  ✓ Action validation checks passed")

    if puck_dataset is not None:
        if np.any(~np.isfinite(puck_dataset)):
            print("  ⚠ WARNING: Puck dataset contains NaN or infinite values!")
            return False

        assert len(puck_dataset.shape) == 3, f"Expected 3D puck array, got {len(puck_dataset.shape)}D"
        assert puck_dataset.shape[0] == position_dataset.shape[0], "Position and puck counts must match"
        assert puck_dataset.shape[1] == 5, f"Expected 5 puck states per window, got {puck_dataset.shape[1]}"
        assert puck_dataset.shape[2] == 2, f"Expected 2D puck vector, got {puck_dataset.shape[2]}D"

        print("  ✓ Puck validation checks passed")
    
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
    parser.add_argument(
        '--include-actions',
        action='store_true',
        help='Include action sequences (delta target positions) in the dataset'
    )
    parser.add_argument(
        '--include-puck',
        action='store_true',
        help='Include aligned 5-step puck windows [t..t+4] for each paddle sequence'
    )
    
    return parser.parse_args()


def main():
    """Main function to prepare AMP position dataset."""
    args = parse_args()
    if args.include_puck and args.sequence_length != 5:
        raise ValueError("--include-puck currently requires --sequence-length 5.")
    
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
    print(f"  Include actions: {args.include_actions}")
    print(f"  Include puck windows: {args.include_puck}")
    
    # Load demo list if provided
    demo_list = None
    if args.demo_list:
        print(f"  Demo list file: {args.demo_list}")
        demo_list = load_demo_list(args.demo_list)
        print(f"  Loaded {len(demo_list)} trajectory IDs from demo list")
    
    # Process all trajectories
    result = process_all_trajectories(
        args.data_dir,
        min_length=args.min_length,
        max_trajectories=args.max_trajectories,
        safety_check=not args.no_safety_check,
        demo_list=demo_list,
        sequence_length=args.sequence_length,
        include_actions=args.include_actions,
        include_puck=args.include_puck,
    )
    
    # Unpack results based on whether actions are included
    if args.include_actions and args.include_puck:
        position_dataset, action_dataset, puck_dataset, stats = result
    elif args.include_actions:
        position_dataset, action_dataset, stats = result
        puck_dataset = None
    elif args.include_puck:
        position_dataset, puck_dataset, stats = result
        action_dataset = None
    else:
        position_dataset, stats = result
        action_dataset = None
        puck_dataset = None
    
    # Print statistics
    print_dataset_statistics(position_dataset, stats, action_dataset, puck_dataset)
    
    # Validate
    if not validate_dataset(position_dataset, args.sequence_length, action_dataset, puck_dataset):
        print("\n⚠ WARNING: Validation failed! Proceeding with save anyway...")
    
    # Save dataset
    output_path = Path(args.output_path)
    save_dataset(position_dataset, output_path, stats, action_dataset, puck_dataset)
    
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
    
    if args.include_actions:
        print("\n  # With --include-actions flag:")
        print("  action_sequences = data['action_sequences']  # Shape: (N, 4, 2)")
        print("  # transition actions aligned with state transitions in each 5-state window")
        print("  action12 = action_sequences[:, 0, :]  # (N, 2) - action s1->s2")
        print("  action45 = action_sequences[:, 3, :]  # (N, 2) - action s4->s5")
    if args.include_puck:
        print("\n  # With --include-puck flag:")
        print("  puck_sequences = data['puck_sequences']  # Shape: (N, 5, 2)")
        print("  # aligned with paddle windows [t..t+4]; index 2 is two timesteps before s5")


if __name__ == '__main__':
    main()
