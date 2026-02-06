#!/usr/bin/env python3
"""
Normalize AMP dataset to create relative state representations.

This script processes the dataset created by prepare_amp_dataset.py and applies:
1. Translational normalization: First position moved to (0, 0)
2. Keeps the second state relative to the first (with velocities preserving original direction)

Input format: PyTorch tensor of shape [N, 2, 4]
    - N = number of state pairs
    - 2 = current state and next state  
    - 4 = [x_pos, y_pos, x_vel, y_vel]

Output format: PyTorch tensor of shape [N, 6]
    - N = number of normalized transitions
    - 6 = [vel1_x, vel1_y, relative_x_pos, relative_y_pos, relative_x_vel, relative_y_vel]
         (velocities preserve original direction, no rotation applied)
"""

import torch
import numpy as np
from pathlib import Path
import argparse


def normalize_state_pair(state_pair):
    """
    Normalize a state pair to relative coordinates (translation only).
    
    Process:
    1. Translate both states so first position is at (0, 0)
    2. Return first velocity and second state (preserving original velocity directions)
    
    Args:
        state_pair: Array of shape (2, 4) containing [state1, state2]
                   where each state is [x_pos, y_pos, x_vel, y_vel]
    
    Returns:
        np.ndarray: Normalized state of shape (6,)
                   [vel1_x, vel1_y, pos2_x, pos2_y, vel2_x, vel2_y]
    """
    state1 = state_pair[0].copy()
    state2 = state_pair[1].copy()
    
    # Extract positions and velocities
    pos1 = state1[:2]
    vel1 = state1[2:]
    pos2 = state2[:2]
    vel2 = state2[2:]
    
    # Step 1: Translate so first position is at origin
    pos2_translated = pos2 - pos1
    
    # Step 2: Return first velocity (preserving direction) and second state
    # First position [0, 0] contains no information so not included
    # No rotation applied - velocities maintain original direction
    normalized_state = np.concatenate([vel1, pos2_translated, vel2])
    
    return normalized_state


def normalize_dataset(state_pairs, verbose=True):
    """
    Normalize all state pairs in the dataset.
    
    Args:
        state_pairs: Tensor of shape (N, 2, 4)
        verbose: Whether to print progress
        
    Returns:
        np.ndarray: Normalized states of shape (N, 6)
    """
    if verbose:
        print(f"Normalizing {len(state_pairs)} state pairs...")
    
    # Convert to numpy for processing
    state_pairs_np = state_pairs.numpy()
    
    normalized_states = []
    
    for i in range(len(state_pairs_np)):
        normalized_state = normalize_state_pair(state_pairs_np[i])
        normalized_states.append(normalized_state)
    
    normalized_states = np.array(normalized_states, dtype=np.float32)
    
    if verbose:
        print(f"✓ Normalization complete")
        print(f"  Output shape: {normalized_states.shape}")
    
    return normalized_states


def print_statistics(normalized_states):
    """Print statistics about the normalized dataset."""
    print("\n" + "="*80)
    print("NORMALIZED DATASET STATISTICS")
    print("="*80)
    
    print(f"\nDataset shape: {normalized_states.shape}")
    print(f"Number of samples: {len(normalized_states):,}")
    
    dim_names = [
        'First X Velocity (m/s)',
        'First Y Velocity (m/s)',
        'Relative X Position (m)',
        'Relative Y Position (m)',
        'Relative X Velocity (m/s)',
        'Relative Y Velocity (m/s)'
    ]
    
    print(f"\nState statistics:")
    for i, name in enumerate(dim_names):
        values = normalized_states[:, i]
        print(f"  {name}:")
        print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"    Mean: {values.mean():.4f}")
        print(f"    Std: {values.std():.4f}")


def validate_normalization(normalized_states, original_state_pairs, num_samples=5):
    """
    Validate the normalization by checking a few samples.
    
    Args:
        normalized_states: Normalized output (N, 6)
        original_state_pairs: Original input (N, 2, 4)
        num_samples: Number of samples to check
    """
    print("\n" + "="*80)
    print("VALIDATION SAMPLES")
    print("="*80)
    
    original_np = original_state_pairs.numpy()
    
    for i in range(min(num_samples, len(normalized_states))):
        print(f"\nSample {i}:")
        
        # Original states
        state1 = original_np[i, 0]
        state2 = original_np[i, 1]
        
        print(f"  Original state 1: pos=({state1[0]:.4f}, {state1[1]:.4f}), "
              f"vel=({state1[2]:.4f}, {state1[3]:.4f})")
        print(f"  Original state 2: pos=({state2[0]:.4f}, {state2[1]:.4f}), "
              f"vel=({state2[2]:.4f}, {state2[3]:.4f})")
        
        # Normalized state
        norm_state = normalized_states[i]
        print(f"  Normalized: vel1=({norm_state[0]:.4f}, {norm_state[1]:.4f}), "
              f"pos2=({norm_state[2]:.4f}, {norm_state[3]:.4f}), "
              f"vel2=({norm_state[4]:.4f}, {norm_state[5]:.4f})")
        
        # Verify first velocity is preserved (direction maintained)
        vel1_orig_mag = np.linalg.norm(state1[2:])
        vel1_norm_mag = np.linalg.norm(norm_state[0:2])
        print(f"  First velocity preservation: orig_mag={vel1_orig_mag:.4f}, norm_mag={vel1_norm_mag:.4f} (should match)")
        
        # Verify position distance is preserved
        original_distance = np.linalg.norm(state2[:2] - state1[:2])
        normalized_distance = np.linalg.norm(norm_state[2:4])
        print(f"  Distance check: original={original_distance:.4f}, "
              f"normalized={normalized_distance:.4f} (should match)")
        
        # Verify velocity magnitudes are preserved
        vel1_mag = np.linalg.norm(state1[2:])
        vel2_mag = np.linalg.norm(state2[2:])
        norm_vel1_mag = np.linalg.norm(norm_state[0:2])
        norm_vel2_mag = np.linalg.norm(norm_state[4:6])
        print(f"  Velocity magnitudes: v1_orig={vel1_mag:.4f}, v1_norm={norm_vel1_mag:.4f}")
        print(f"                       v2_orig={vel2_mag:.4f}, v2_norm={norm_vel2_mag:.4f}")


def save_normalized_dataset(normalized_states, output_path, original_stats=None):
    """
    Save normalized dataset as PyTorch tensor file.
    
    Args:
        normalized_states: np.ndarray of shape (N, 4)
        output_path: Where to save the .pt file
        original_stats: Optional statistics from original dataset
    """
    # Convert to torch tensor
    tensor_dataset = torch.from_numpy(normalized_states).float()
    
    # Create save dictionary
    save_dict = {
        'normalized_states': tensor_dataset,
        'dataset_shape': tensor_dataset.shape,
        'description': 'Normalized AMP dataset with relative state representations (translation only)',
        'format': '[vel1_x, vel1_y, relative_x_pos, relative_y_pos, relative_x_vel, relative_y_vel]',
        'normalization': {
            'translation': 'First position moved to (0, 0) - not included in output',
            'rotation': 'None - velocities preserve original direction',
            'scale': 'Preserved (no scaling applied)',
            'kept': 'First velocity (with original direction) and second state'
        }
    }
    
    if original_stats is not None:
        save_dict['original_stats'] = original_stats
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, output_path)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Saved normalized dataset to: {output_path}")
    print(f"  File size: {file_size_mb:.2f} MB")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Normalize AMP dataset to relative state representations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to the original dataset file (created by prepare_amp_dataset.py)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for normalized dataset (default: adds _normalized suffix)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Print validation samples showing the normalization'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of validation samples to show'
    )
    
    return parser.parse_args()


def main():
    """Main function to normalize the dataset."""
    args = parse_args()
    
    print("="*80)
    print("AMP DATASET NORMALIZATION")
    print("="*80)
    
    # Determine output path
    input_path = Path(args.input_path)
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        # Add _normalized suffix before extension
        output_path = input_path.parent / f"{input_path.stem}_normalized{input_path.suffix}"
    
    print(f"\nConfiguration:")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    
    # Load original dataset
    print(f"\nLoading dataset from: {input_path}")
    data = torch.load(input_path)
    
    if 'state_pairs' not in data:
        raise ValueError("Input file does not contain 'state_pairs' key. "
                        "Make sure it was created by prepare_amp_dataset.py")
    
    state_pairs = data['state_pairs']
    print(f"  Loaded shape: {state_pairs.shape}")
    print(f"  Expected format: (N, 2, 4) = (num_pairs, [state1, state2], [x, y, vx, vy])")
    
    # Validate input shape
    if len(state_pairs.shape) != 3 or state_pairs.shape[1] != 2 or state_pairs.shape[2] != 4:
        raise ValueError(f"Invalid input shape {state_pairs.shape}. Expected (N, 2, 4)")
    
    # Normalize dataset
    print("\n" + "="*80)
    print("NORMALIZATION PROCESS")
    print("="*80)
    print("\nApplying transformations:")
    print("  1. Translation: Moving first position to (0, 0)")
    print("  2. Extraction: Keeping first velocity and second state (velocities preserve original direction)")
    
    normalized_states = normalize_dataset(state_pairs, verbose=True)
    
    # Print statistics
    print_statistics(normalized_states)
    
    # Validation
    if args.validate:
        validate_normalization(normalized_states, state_pairs, args.num_samples)
    
    # Save normalized dataset
    original_stats = data.get('stats', None)
    save_normalized_dataset(normalized_states, output_path, original_stats)
    
    print("\n" + "="*80)
    print("✓ Normalization complete!")
    print("="*80)
    
    # Print usage example
    print("\nTo load the normalized dataset:")
    print("  import torch")
    print(f"  data = torch.load('{output_path}')")
    print("  normalized_states = data['normalized_states']  # Shape: (N, 6)")
    print("  # Format: [vel1_x, vel1_y, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]")
    print("  # vel1 preserves original direction, second state is relative to first")


if __name__ == '__main__':
    main()
