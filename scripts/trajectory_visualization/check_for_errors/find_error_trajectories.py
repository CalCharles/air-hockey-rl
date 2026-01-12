#!/usr/bin/env python3
"""
Scan trajectory files to identify those containing estop or unsafe cases.

According to the field documentation:
- Field 3: estop (1 = estopped, 0 = normal)
- Field 4: safety (0 = unsafe, 1 = safe)

This script will:
1. Scan all .hdf5 files in /nfs/data/airhockey/
2. Check for any timesteps with estop=1 or safety=0
3. Output a list of problematic trajectories to a text file
"""

import h5py
import numpy as np
from pathlib import Path
import sys
from datetime import datetime


def check_trajectory_for_errors(filepath):
    """
    Check a single trajectory file for estop or unsafe conditions.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        dict: Dictionary with error information:
            - has_estop: bool
            - has_unsafe: bool
            - estop_count: int (number of timesteps with estop=1)
            - unsafe_count: int (number of timesteps with safety=0)
            - total_timesteps: int
    """
    try:
        with h5py.File(filepath, 'r') as f:
            train_vals = f['train_vals'][:]
            
            # Extract estop and safety fields
            estop = train_vals[:, 3]      # Field 3: estop indicator
            safety = train_vals[:, 4]     # Field 4: safety indicator
            
            # Check for errors
            estop_mask = (estop == 1)
            unsafe_mask = (safety == 0)
            
            estop_count = np.sum(estop_mask)
            unsafe_count = np.sum(unsafe_mask)
            
            return {
                'success': True,
                'has_estop': estop_count > 0,
                'has_unsafe': unsafe_count > 0,
                'estop_count': int(estop_count),
                'unsafe_count': int(unsafe_count),
                'total_timesteps': len(train_vals),
                'error': None
            }
    except Exception as e:
        return {
            'success': False,
            'has_estop': False,
            'has_unsafe': False,
            'estop_count': 0,
            'unsafe_count': 0,
            'total_timesteps': 0,
            'error': str(e)
        }


def scan_all_trajectories(data_dir, output_dir):
    """
    Scan all trajectory files in the data directory.
    
    Args:
        data_dir: Path to directory containing trajectory files
        output_dir: Path to directory where output files will be saved
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .hdf5 files
    trajectory_files = sorted(data_dir.glob('trajectory_data*.hdf5'))
    
    if not trajectory_files:
        print(f"Error: No trajectory files found in {data_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("TRAJECTORY ERROR SCANNER")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(trajectory_files)} trajectory files to scan")
    print("=" * 80)
    print()
    
    # Lists to store results
    files_with_estop = []
    files_with_unsafe = []
    files_with_errors = []
    successful_scans = []
    
    # Scan each file
    for idx, filepath in enumerate(trajectory_files, 1):
        filename = filepath.name
        
        # Print progress
        if idx % 10 == 0 or idx == 1:
            print(f"Scanning {idx}/{len(trajectory_files)}: {filename}...")
        
        result = check_trajectory_for_errors(filepath)
        
        if not result['success']:
            files_with_errors.append({
                'filename': filename,
                'error': result['error']
            })
            print(f"  ERROR reading {filename}: {result['error']}")
            continue
        
        successful_scans.append(result)
        
        # Check for problems
        has_problem = False
        if result['has_estop']:
            files_with_estop.append({
                'filename': filename,
                'estop_count': result['estop_count'],
                'total_timesteps': result['total_timesteps']
            })
            has_problem = True
        
        if result['has_unsafe']:
            files_with_unsafe.append({
                'filename': filename,
                'unsafe_count': result['unsafe_count'],
                'total_timesteps': result['total_timesteps']
            })
            has_problem = True
        
        # Print notification for problematic files
        if has_problem:
            status = []
            if result['has_estop']:
                status.append(f"ESTOP: {result['estop_count']} timesteps")
            if result['has_unsafe']:
                status.append(f"UNSAFE: {result['unsafe_count']} timesteps")
            print(f"  ⚠️  {filename}: {', '.join(status)}")
    
    print()
    print("=" * 80)
    print("SCAN COMPLETE")
    print("=" * 80)
    print(f"Total files scanned: {len(trajectory_files)}")
    print(f"Successfully processed: {len(successful_scans)}")
    print(f"Read errors: {len(files_with_errors)}")
    print(f"Files with ESTOP: {len(files_with_estop)}")
    print(f"Files with UNSAFE: {len(files_with_unsafe)}")
    
    # Calculate overlap (files with both estop and unsafe)
    estop_filenames = {f['filename'] for f in files_with_estop}
    unsafe_filenames = {f['filename'] for f in files_with_unsafe}
    overlap_filenames = estop_filenames & unsafe_filenames
    print(f"Files with BOTH: {len(overlap_filenames)}")
    
    # Calculate total unique problematic files
    all_problem_filenames = estop_filenames | unsafe_filenames
    print(f"Total unique problematic files: {len(all_problem_filenames)}")
    print("=" * 80)
    print()
    
    # Write results to files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Write estop trajectories
    estop_output = output_dir / f'trajectories_with_estop_{timestamp}.txt'
    with open(estop_output, 'w') as f:
        f.write(f"Trajectories with ESTOP (estop=1)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files with estop: {len(files_with_estop)}\n")
        f.write("=" * 80 + "\n\n")
        
        for item in sorted(files_with_estop, key=lambda x: x['filename']):
            pct = (item['estop_count'] / item['total_timesteps']) * 100
            f.write(f"{item['filename']}\n")
            f.write(f"  Estop timesteps: {item['estop_count']} / {item['total_timesteps']} ({pct:.2f}%)\n\n")
    
    print(f"✓ Wrote estop list to: {estop_output}")
    
    # Write unsafe trajectories
    unsafe_output = output_dir / f'trajectories_with_unsafe_{timestamp}.txt'
    with open(unsafe_output, 'w') as f:
        f.write(f"Trajectories with UNSAFE conditions (safety=0)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total files with unsafe: {len(files_with_unsafe)}\n")
        f.write("=" * 80 + "\n\n")
        
        for item in sorted(files_with_unsafe, key=lambda x: x['filename']):
            pct = (item['unsafe_count'] / item['total_timesteps']) * 100
            f.write(f"{item['filename']}\n")
            f.write(f"  Unsafe timesteps: {item['unsafe_count']} / {item['total_timesteps']} ({pct:.2f}%)\n\n")
    
    print(f"✓ Wrote unsafe list to: {unsafe_output}")
    
    # Write combined list (all problematic files)
    combined_output = output_dir / f'trajectories_with_errors_{timestamp}.txt'
    with open(combined_output, 'w') as f:
        f.write(f"All Trajectories with Errors (ESTOP or UNSAFE)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total problematic files: {len(all_problem_filenames)}\n")
        f.write("=" * 80 + "\n\n")
        
        # Combine the information
        combined_dict = {}
        for item in files_with_estop:
            filename = item['filename']
            if filename not in combined_dict:
                combined_dict[filename] = {
                    'estop_count': 0,
                    'unsafe_count': 0,
                    'total_timesteps': item['total_timesteps']
                }
            combined_dict[filename]['estop_count'] = item['estop_count']
        
        for item in files_with_unsafe:
            filename = item['filename']
            if filename not in combined_dict:
                combined_dict[filename] = {
                    'estop_count': 0,
                    'unsafe_count': 0,
                    'total_timesteps': item['total_timesteps']
                }
            combined_dict[filename]['unsafe_count'] = item['unsafe_count']
        
        for filename in sorted(combined_dict.keys()):
            info = combined_dict[filename]
            f.write(f"{filename}\n")
            if info['estop_count'] > 0:
                pct = (info['estop_count'] / info['total_timesteps']) * 100
                f.write(f"  Estop timesteps: {info['estop_count']} / {info['total_timesteps']} ({pct:.2f}%)\n")
            if info['unsafe_count'] > 0:
                pct = (info['unsafe_count'] / info['total_timesteps']) * 100
                f.write(f"  Unsafe timesteps: {info['unsafe_count']} / {info['total_timesteps']} ({pct:.2f}%)\n")
            f.write("\n")
    
    print(f"✓ Wrote combined list to: {combined_output}")
    
    # Write simple list (just filenames) for easy scripting
    simple_output = output_dir / f'error_trajectory_filenames_{timestamp}.txt'
    with open(simple_output, 'w') as f:
        for filename in sorted(all_problem_filenames):
            f.write(f"{filename}\n")
    
    print(f"✓ Wrote simple filename list to: {simple_output}")
    
    # Write summary statistics
    summary_output = output_dir / f'scan_summary_{timestamp}.txt'
    with open(summary_output, 'w') as f:
        f.write(f"Trajectory Error Scan Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total files scanned: {len(trajectory_files)}\n")
        f.write(f"Successfully processed: {len(successful_scans)}\n")
        f.write(f"Read errors: {len(files_with_errors)}\n\n")
        
        f.write(f"Files with ESTOP: {len(files_with_estop)}\n")
        f.write(f"Files with UNSAFE: {len(files_with_unsafe)}\n")
        f.write(f"Files with BOTH: {len(overlap_filenames)}\n")
        f.write(f"Total unique problematic files: {len(all_problem_filenames)}\n\n")
        
        if successful_scans:
            total_estop_timesteps = sum(s['estop_count'] for s in successful_scans)
            total_unsafe_timesteps = sum(s['unsafe_count'] for s in successful_scans)
            total_timesteps = sum(s['total_timesteps'] for s in successful_scans)
            
            f.write(f"Total timesteps across all trajectories: {total_timesteps}\n")
            f.write(f"Total estop timesteps: {total_estop_timesteps} ({(total_estop_timesteps/total_timesteps)*100:.4f}%)\n")
            f.write(f"Total unsafe timesteps: {total_unsafe_timesteps} ({(total_unsafe_timesteps/total_timesteps)*100:.4f}%)\n\n")
        
        if files_with_errors:
            f.write("\nFiles with read errors:\n")
            f.write("-" * 80 + "\n")
            for item in files_with_errors:
                f.write(f"{item['filename']}: {item['error']}\n")
    
    print(f"✓ Wrote summary to: {summary_output}")
    print()
    print("=" * 80)
    print("✓ All output files created successfully!")
    print("=" * 80)


def main():
    """Main entry point."""
    # Configuration
    data_dir = Path('/nfs/data/airhockey')
    output_dir = Path(__file__).parent
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Run the scan
    scan_all_trajectories(data_dir, output_dir)


if __name__ == '__main__':
    main()

