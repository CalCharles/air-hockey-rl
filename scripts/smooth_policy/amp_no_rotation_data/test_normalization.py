#!/usr/bin/env python3
"""
Test script to verify the normalization logic works correctly.
Creates synthetic test data and validates the normalization process.
"""

import torch
import numpy as np
from normalize_dataset import normalize_state_pair, compute_rotation_matrix


def test_basic_normalization():
    """Test basic normalization with simple cases."""
    print("="*80)
    print("TEST: Basic Normalization")
    print("="*80)
    
    # Test case 1: Simple translation
    print("\n1. Testing translation (no rotation needed):")
    state_pair = np.array([
        [1.0, 2.0, 1.0, 0.0],  # State 1: pos=(1,2), vel=(1,0) already aligned
        [3.0, 4.0, 0.5, 0.5]   # State 2: pos=(3,4), vel=(0.5,0.5)
    ], dtype=np.float32)
    
    normalized = normalize_state_pair(state_pair)
    print(f"   Original state 1: {state_pair[0]}")
    print(f"   Original state 2: {state_pair[1]}")
    print(f"   Normalized: {normalized}")
    print(f"   Expected: vel1=(1.0, 0.0), pos2=(2.0, 2.0), vel2=(0.5, 0.5)")
    print(f"   Actual: vel1=({normalized[0]:.4f}, {normalized[1]:.4f}), "
          f"pos2=({normalized[2]:.4f}, {normalized[3]:.4f}), "
          f"vel2=({normalized[4]:.4f}, {normalized[5]:.4f})")
    
    # Verify position relative distance is preserved
    original_dist = np.linalg.norm(state_pair[1, :2] - state_pair[0, :2])
    normalized_dist = np.linalg.norm(normalized[2:4])  # Position is indices 2:4
    print(f"   Distance preserved: {original_dist:.4f} == {normalized_dist:.4f} ✓")
    
    # Verify first velocity magnitude is preserved
    vel1_orig_mag = np.linalg.norm(state_pair[0, 2:])
    vel1_norm_mag = np.linalg.norm(normalized[0:2])  # First velocity is indices 0:2
    print(f"   Vel1 magnitude preserved: {vel1_orig_mag:.4f} == {vel1_norm_mag:.4f} ✓")
    
    # Test case 2: Rotation needed
    print("\n2. Testing rotation:")
    state_pair = np.array([
        [0.0, 0.0, 0.0, 1.0],  # State 1: at origin, vel=(0,1) pointing up
        [1.0, 1.0, 1.0, 1.0]   # State 2: pos=(1,1), vel=(1,1)
    ], dtype=np.float32)
    
    normalized = normalize_state_pair(state_pair)
    print(f"   Original state 1: {state_pair[0]}")
    print(f"   Original state 2: {state_pair[1]}")
    print(f"   Normalized: {normalized}")
    
    # After rotating 90 degrees clockwise (to align y-axis with x-axis)
    # vel1 (0, 1) should become (1, 0), pos (1, 1) should become (1, -1)
    print(f"   Expected: vel1=(1, 0), pos2=(1, -1)")
    print(f"   Actual: vel1=({normalized[0]:.4f}, {normalized[1]:.4f}), "
          f"pos2=({normalized[2]:.4f}, {normalized[3]:.4f})")
    
    # Verify distance
    original_dist = np.linalg.norm(state_pair[1, :2] - state_pair[0, :2])
    normalized_dist = np.linalg.norm(normalized[2:4])  # Position is indices 2:4
    print(f"   Distance preserved: {original_dist:.4f} == {normalized_dist:.4f} ✓")
    
    # Verify velocity magnitudes preserved
    vel1_mag = np.linalg.norm(state_pair[0, 2:])
    vel2_mag = np.linalg.norm(state_pair[1, 2:])
    norm_vel1_mag = np.linalg.norm(normalized[0:2])
    norm_vel2_mag = np.linalg.norm(normalized[4:6])
    print(f"   Vel1 magnitude preserved: {vel1_mag:.4f} == {norm_vel1_mag:.4f} ✓")
    print(f"   Vel2 magnitude preserved: {vel2_mag:.4f} == {norm_vel2_mag:.4f} ✓")


def test_rotation_matrix():
    """Test the rotation matrix computation."""
    print("\n" + "="*80)
    print("TEST: Rotation Matrix")
    print("="*80)
    
    # Test case 1: Velocity pointing right (already aligned)
    print("\n1. Velocity (1, 0) - no rotation needed:")
    R = compute_rotation_matrix(np.array([1.0, 0.0]))
    print(f"   Rotation matrix:\n{R}")
    test_vec = np.array([1.0, 0.0])
    rotated = R @ test_vec
    print(f"   (1,0) rotated: {rotated} (should be ~(1,0))")
    
    # Test case 2: Velocity pointing up
    print("\n2. Velocity (0, 1) - 90° rotation needed:")
    R = compute_rotation_matrix(np.array([0.0, 1.0]))
    print(f"   Rotation matrix:\n{R}")
    test_vec = np.array([0.0, 1.0])
    rotated = R @ test_vec
    print(f"   (0,1) rotated: {rotated} (should be ~(1,0))")
    
    # Test case 3: Velocity pointing at 45 degrees
    print("\n3. Velocity (1, 1) - 45° rotation needed:")
    R = compute_rotation_matrix(np.array([1.0, 1.0]))
    print(f"   Rotation matrix:\n{R}")
    test_vec = np.array([1.0, 1.0])
    rotated = R @ test_vec
    print(f"   (1,1) rotated: {rotated} (should be ~({np.sqrt(2):.3f},0))")


def test_with_synthetic_dataset():
    """Test with a small synthetic dataset."""
    print("\n" + "="*80)
    print("TEST: Synthetic Dataset")
    print("="*80)
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 10
    
    # Generate random state pairs
    state_pairs = []
    for i in range(n_samples):
        # Random first state
        pos1 = np.random.randn(2) * 0.5
        vel1 = np.random.randn(2) * 0.3
        
        # Second state close to first
        pos2 = pos1 + np.random.randn(2) * 0.2
        vel2 = vel1 + np.random.randn(2) * 0.1
        
        state_pair = np.array([
            np.concatenate([pos1, vel1]),
            np.concatenate([pos2, vel2])
        ], dtype=np.float32)
        state_pairs.append(state_pair)
    
    state_pairs = np.array(state_pairs)
    print(f"Created {n_samples} synthetic state pairs")
    print(f"Shape: {state_pairs.shape}")
    
    # Normalize
    from normalize_dataset import normalize_dataset
    state_pairs_tensor = torch.from_numpy(state_pairs).float()
    normalized = normalize_dataset(state_pairs_tensor, verbose=False)
    
    print(f"\nNormalized shape: {normalized.shape}")
    print(f"Expected shape: ({n_samples}, 6)")
    
    # Check distances are preserved
    print("\nValidating distance preservation:")
    all_distances_match = True
    for i in range(n_samples):
        original_dist = np.linalg.norm(state_pairs[i, 1, :2] - state_pairs[i, 0, :2])
        normalized_dist = np.linalg.norm(normalized[i, 2:4])  # Position is indices 2:4
        match = np.abs(original_dist - normalized_dist) < 1e-5
        if not match:
            print(f"  Sample {i}: MISMATCH {original_dist:.6f} != {normalized_dist:.6f}")
            all_distances_match = False
    
    if all_distances_match:
        print("  ✓ All distances preserved!")
    
    # Check first velocity magnitude is preserved
    print("\nValidating first velocity magnitude preservation:")
    all_vel1_match = True
    for i in range(n_samples):
        original_vel1_mag = np.linalg.norm(state_pairs[i, 0, 2:])
        normalized_vel1_mag = np.linalg.norm(normalized[i, 0:2])  # First velocity is indices 0:2
        match = np.abs(original_vel1_mag - normalized_vel1_mag) < 1e-5
        if not match:
            print(f"  Sample {i}: MISMATCH {original_vel1_mag:.6f} != {normalized_vel1_mag:.6f}")
            all_vel1_match = False
    
    if all_vel1_match:
        print("  ✓ All first velocity magnitudes preserved!")
    
    # Check second velocity magnitude is preserved
    print("\nValidating second velocity magnitude preservation:")
    all_vel2_match = True
    for i in range(n_samples):
        original_vel2_mag = np.linalg.norm(state_pairs[i, 1, 2:])
        normalized_vel2_mag = np.linalg.norm(normalized[i, 4:6])  # Second velocity is indices 4:6
        match = np.abs(original_vel2_mag - normalized_vel2_mag) < 1e-5
        if not match:
            print(f"  Sample {i}: MISMATCH {original_vel2_mag:.6f} != {normalized_vel2_mag:.6f}")
            all_vel2_match = False
    
    if all_vel2_match:
        print("  ✓ All second velocity magnitudes preserved!")
    
    return all_distances_match and all_vel1_match and all_vel2_match


def main():
    """Run all tests."""
    print("="*80)
    print("NORMALIZATION TEST SUITE")
    print("="*80)
    
    try:
        test_rotation_matrix()
        test_basic_normalization()
        success = test_with_synthetic_dataset()
        
        print("\n" + "="*80)
        if success:
            print("✓ ALL TESTS PASSED!")
        else:
            print("⚠ SOME TESTS FAILED!")
        print("="*80)
        
        return success
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
