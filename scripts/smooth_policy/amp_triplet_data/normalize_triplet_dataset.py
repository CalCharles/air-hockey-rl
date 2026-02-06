#!/usr/bin/env python3
"""
Normalize AMP triplet dataset to create relative state representations.

This script processes the dataset created by prepare_amp_triplet_dataset.py and applies
translation-only normalization (no rotation):
1. Translational normalization: First position moved to (0, 0)
2. Keeps the first velocity and all relative positions/velocities

Input format: PyTorch tensor of shape [N, 3, 4]
    - N = number of state triplets
    - 3 = state1, state2, state3
    - 4 = [x_pos, y_pos, x_vel, y_vel]

Output format: PyTorch tensor of shape [N, 10]
    - N = number of normalized transitions
    - 10 = [vel1_x, vel1_y, rel_pos2_x, rel_pos2_y, rel_vel2_x, rel_vel2_y, 
            rel_pos3_x, rel_pos3_y, rel_vel3_x, rel_vel3_y]
"""

import torch
import numpy as np
from pathlib import Path
import argparse


def normalize_state_triplet(state_triplet):
    """
    Normalize a state triplet to relative coordinates (translation only, no rotation).
    
    Process:
    1. Translate all states so first position is at (0, 0)
    2. Keep first velocity and all relative positions/velocities
    
    Args:
        state_triplet: Array of shape (3, 4) containing [state1, state2, state3]
                      where each state is [x_pos, y_pos, x_vel, y_vel]
    
    Returns:
        np.ndarray: Normalized state of shape (10,)
                   [vel1_x, vel1_y, rel_pos2_x, rel_pos2_y, rel_vel2_x, rel_vel2_y,
                    rel_pos3_x, rel_pos3_y, rel_vel3_x, rel_vel3_y]
    """
    state1 = state_triplet[0].copy()
    state2 = state_triplet[1].copy()
    state3 = state_triplet[2].copy()
    
    # Extract positions and velocities
    pos1 = state1[:2]
    vel1 = state1[2:]
    pos2 = state2[:2]
    vel2 = state2[2:]
    pos3 = state3[:2]
    vel3 = state3[2:]
    
    # Step 1: Translate so first position is at origin
    pos2_translated = pos2 - pos1
    pos3_translated = pos3 - pos1
    
    # Step 2: Concatenate first velocity and all relative states
    # No rotation is applied - keep velocities as-is
    normalized_state = np.concatenate([
        vel1,              # First velocity (2D)
        pos2_translated,   # Relative position 2 (2D)
        vel2,              # Second velocity (2D)
        pos3_translated,   # Relative position 3 (2D)
        vel3               # Third velocity (2D)
    ])
    
    return normalized_state


def normalize_dataset(state_triplets, verbose=True):
    """
    Normalize all state triplets in the dataset.
    
    Args:
        state_triplets: Tensor of shape (N, 3, 4)
        verbose: Whether to print progress
        
    Returns:
        np.ndarray: Normalized states of shape (N, 10)
    """
    if verbose:
        print(f"Normalizing {len(state_triplets)} state triplets...")
    
    # Convert to numpy for processing
    state_triplets_np = state_triplets.numpy()
    
    normalized_states = []
    
    for i in range(len(state_triplets_np)):
        normalized_state = normalize_state_triplet(state_triplets_np[i])
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
        'Relative X Position 2 (m)',
        'Relative Y Position 2 (m)',
        'Second X Velocity (m/s)',
        'Second Y Velocity (m/s)',
        'Relative X Position 3 (m)',
        'Relative Y Position 3 (m)',
        'Third X Velocity (m/s)',
        'Third Y Velocity (m/s)'
    ]
    
    print(f"\nState statistics:")
    for i, name in enumerate(dim_names):
        values = normalized_states[:, i]
        print(f"  {name}:")
        print(f"    Range: [{values.min():.4f}, {values.max():.4f}]")
        print(f"    Mean: {values.mean():.4f}")
        print(f"    Std: {values.std():.4f}")


def validate_normalization(normalized_states, original_state_triplets, num_samples=5):
    """
    Validate the normalization by checking a few samples.
    
    Args:
        normalized_states: Normalized output (N, 10)
        original_state_triplets: Original input (N, 3, 4)
        num_samples: Number of samples to check
    """
    print("\n" + "="*80)
    print("VALIDATION SAMPLES")
    print("="*80)
    
    original_np = original_state_triplets.numpy()
    
    for i in range(min(num_samples, len(normalized_states))):
        print(f"\nSample {i}:")
        
        # Original states
        state1 = original_np[i, 0]
        state2 = original_np[i, 1]
        state3 = original_np[i, 2]
        
        print(f"  Original state 1: pos=({state1[0]:.4f}, {state1[1]:.4f}), "
              f"vel=({state1[2]:.4f}, {state1[3]:.4f})")
        print(f"  Original state 2: pos=({state2[0]:.4f}, {state2[1]:.4f}), "
              f"vel=({state2[2]:.4f}, {state2[3]:.4f})")
        print(f"  Original state 3: pos=({state3[0]:.4f}, {state3[1]:.4f}), "
              f"vel=({state3[2]:.4f}, {state3[3]:.4f})")
        
        # Normalized state
        norm_state = normalized_states[i]
        print(f"  Normalized:")
        print(f"    vel1=({norm_state[0]:.4f}, {norm_state[1]:.4f})")
        print(f"    rel_pos2=({norm_state[2]:.4f}, {norm_state[3]:.4f})")
        print(f"    vel2=({norm_state[4]:.4f}, {norm_state[5]:.4f})")
        print(f"    rel_pos3=({norm_state[6]:.4f}, {norm_state[7]:.4f})")
        print(f"    vel3=({norm_state[8]:.4f}, {norm_state[9]:.4f})")
        
        # Verify distances are preserved
        original_distance_12 = np.linalg.norm(state2[:2] - state1[:2])
        normalized_distance_12 = np.linalg.norm(norm_state[2:4])
        print(f"  Distance 1->2: original={original_distance_12:.4f}, "
              f"normalized={normalized_distance_12:.4f} (should match)")
        
        original_distance_13 = np.linalg.norm(state3[:2] - state1[:2])
        normalized_distance_13 = np.linalg.norm(norm_state[6:8])
        print(f"  Distance 1->3: original={original_distance_13:.4f}, "
              f"normalized={normalized_distance_13:.4f} (should match)")
        
        # Verify velocities are preserved
        vel1_match = np.allclose(state1[2:], norm_state[0:2])
        vel2_match = np.allclose(state2[2:], norm_state[4:6])
        vel3_match = np.allclose(state3[2:], norm_state[8:10])
        print(f"  Velocity preservation: vel1={vel1_match}, vel2={vel2_match}, vel3={vel3_match}")


def save_normalized_dataset(normalized_states, output_path, original_stats=None):
    """
    Save normalized dataset as PyTorch tensor file.
    
    Args:
        normalized_states: np.ndarray of shape (N, 10)
        output_path: Where to save the .pt file
        original_stats: Optional statistics from original dataset
    """
    # Convert to torch tensor
    tensor_dataset = torch.from_numpy(normalized_states).float()
    
    # Create save dictionary
    save_dict = {
        'normalized_states': tensor_dataset,
        'dataset_shape': tensor_dataset.shape,
        'description': 'Normalized AMP triplet dataset with relative state representations',
        'format': '[vel1_x, vel1_y, rel_pos2_x, rel_pos2_y, rel_vel2_x, rel_vel2_y, rel_pos3_x, rel_pos3_y, rel_vel3_x, rel_vel3_y]',
        'normalization': {
            'translation': 'First position moved to (0, 0) - not included in output',
            'rotation': 'None (translation only)',
            'scale': 'Preserved (no scaling applied)',
            'kept': 'First velocity and all relative positions/velocities for states 2 and 3'
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
        description='Normalize AMP triplet dataset to relative state representations (translation only)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to the original dataset file (created by prepare_amp_triplet_dataset.py)'
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
    print("AMP TRIPLET DATASET NORMALIZATION")
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
    
    if 'state_triplets' not in data:
        raise ValueError("Input file does not contain 'state_triplets' key. "
                        "Make sure it was created by prepare_amp_triplet_dataset.py")
    
    state_triplets = data['state_triplets']
    print(f"  Loaded shape: {state_triplets.shape}")
    print(f"  Expected format: (N, 3, 4) = (num_triplets, [state1, state2, state3], [x, y, vx, vy])")
    
    # Validate input shape
    if len(state_triplets.shape) != 3 or state_triplets.shape[1] != 3 or state_triplets.shape[2] != 4:
        raise ValueError(f"Invalid input shape {state_triplets.shape}. Expected (N, 3, 4)")
    
    # Normalize dataset
    print("\n" + "="*80)
    print("NORMALIZATION PROCESS")
    print("="*80)
    print("\nApplying transformations:")
    print("  1. Translation: Moving first position to (0, 0)")
    print("  2. Keeping all velocities unchanged (no rotation)")
    print("  3. Extraction: Keeping first velocity and all relative positions/velocities")
    
    normalized_states = normalize_dataset(state_triplets, verbose=True)
    
    # Print statistics
    print_statistics(normalized_states)
    
    # Validation
    if args.validate:
        validate_normalization(normalized_states, state_triplets, args.num_samples)
    
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
    print("  normalized_states = data['normalized_states']  # Shape: (N, 10)")
    print("  # Format: [vel1_x, vel1_y, rel_pos2_x, rel_pos2_y, rel_vel2_x, rel_vel2_y,")
    print("  #          rel_pos3_x, rel_pos3_y, rel_vel3_x, rel_vel3_y]")


if __name__ == '__main__':
    main()
