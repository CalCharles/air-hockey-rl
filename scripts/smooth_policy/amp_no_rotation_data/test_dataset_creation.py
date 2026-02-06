#!/usr/bin/env python3
"""
Quick test script to verify the dataset creation works correctly.
Processes a small number of trajectories for testing.
"""

import sys
from pathlib import Path

# Add the script directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

from prepare_amp_dataset import (
    find_trajectory_files,
    process_all_trajectories,
    validate_dataset,
    print_dataset_statistics
)


def test_dataset_creation():
    """Test the dataset creation with a small number of trajectories."""
    print("="*80)
    print("TESTING AMP DATASET CREATION")
    print("="*80)
    
    data_dir = '/nfs/data/airhockey'
    
    # Test with just 10 trajectories
    max_trajectories = 10
    min_length = 50
    
    print(f"\nTest configuration:")
    print(f"  Data directory: {data_dir}")
    print(f"  Max trajectories: {max_trajectories}")
    print(f"  Min length: {min_length}")
    
    try:
        # Find files
        print("\n1. Finding trajectory files...")
        files = find_trajectory_files(data_dir)
        print(f"   ✓ Found {len(files)} total files")
        
        # Process trajectories
        print("\n2. Processing trajectories...")
        dataset, stats = process_all_trajectories(
            data_dir,
            min_length=min_length,
            max_trajectories=max_trajectories,
            safety_check=True
        )
        print(f"   ✓ Dataset created with shape: {dataset.shape}")
        
        # Print statistics
        print("\n3. Dataset statistics:")
        print_dataset_statistics(dataset, stats)
        
        # Validate
        print("\n4. Validating dataset...")
        is_valid = validate_dataset(dataset)
        
        if is_valid:
            print("\n" + "="*80)
            print("✓ ALL TESTS PASSED!")
            print("="*80)
            print("\nThe dataset creation script is working correctly.")
            print("You can now run the full dataset creation with:")
            print("  python scripts/smooth_policy/amp/prepare_amp_dataset.py")
            return True
        else:
            print("\n⚠ VALIDATION FAILED")
            return False
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_dataset_creation()
    sys.exit(0 if success else 1)
