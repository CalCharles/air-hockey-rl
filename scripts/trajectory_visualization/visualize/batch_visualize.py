#!/usr/bin/env python3
"""
Batch process multiple trajectory files to create visualizations.
"""

import subprocess
from pathlib import Path
import sys

def main():
    # Directory containing trajectory data files
    data_dir = Path('/nfs/data/airhockey')
    
    # Get all trajectory files and sort them
    trajectory_files = sorted(data_dir.glob('trajectory_data*.hdf5'))
    
    if not trajectory_files:
        print(f"No trajectory files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(trajectory_files)} trajectory files")
    
    # Process first 20 trajectories
    num_to_process = min(20, len(trajectory_files))
    
    # Path to the visualization script
    script_path = Path(__file__).parent / 'visualize_trajectory.py'
    
    print(f"\nProcessing {num_to_process} trajectories...")
    print("=" * 80)
    
    for i, trajectory_file in enumerate(trajectory_files[:num_to_process], 1):
        print(f"\n[{i}/{num_to_process}] Processing: {trajectory_file.name}")
        print("-" * 80)
        
        try:
            # Run the visualization script
            result = subprocess.run(
                [sys.executable, str(script_path), str(trajectory_file)],
                capture_output=False,
                text=True,
                check=True
            )
            print(f"✓ Successfully processed {trajectory_file.name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error processing {trajectory_file.name}: {e}")
            continue
        except Exception as e:
            print(f"✗ Unexpected error with {trajectory_file.name}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print(f"✓ Batch processing complete! Processed {num_to_process} trajectories.")
    print("=" * 80)


if __name__ == '__main__':
    main()

