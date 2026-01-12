#!/usr/bin/env python3
"""
Script to explore and inspect the structure of HDF5 trajectory data files.
This helps understand what data is stored in the HDF5 files.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def print_attrs(name, obj):
    """Print attributes of an HDF5 object."""
    if obj.attrs:
        print(f"    Attributes:")
        for key, val in obj.attrs.items():
            print(f"      {key}: {val}")


def explore_hdf5_structure(file_path, max_samples=5):
    """
    Explore and print the structure of an HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        max_samples: Number of sample values to display for each dataset
    """
    print(f"\n{'='*80}")
    print(f"Exploring HDF5 file: {file_path}")
    print(f"{'='*80}\n")
    
    with h5py.File(file_path, 'r') as f:
        print(f"File: {file_path.name}")
        print(f"\nTop-level keys: {list(f.keys())}\n")
        
        def print_structure(name, obj):
            """Recursively print the structure of the HDF5 file."""
            indent = "  " * name.count('/')
            
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}📊 Dataset: {name}")
                print(f"{indent}   Shape: {obj.shape}")
                print(f"{indent}   Dtype: {obj.dtype}")
                print(f"{indent}   Size: {obj.size} elements")
                
                # Print attributes if any
                if obj.attrs:
                    print(f"{indent}   Attributes:")
                    for key, val in obj.attrs.items():
                        print(f"{indent}     - {key}: {val}")
                
                # Print sample data
                if obj.size > 0:
                    print(f"{indent}   Sample data (first {max_samples} elements):")
                    try:
                        if len(obj.shape) == 1:
                            # 1D array
                            samples = obj[:min(max_samples, obj.shape[0])]
                            print(f"{indent}     {samples}")
                        elif len(obj.shape) == 2:
                            # 2D array
                            samples = obj[:min(max_samples, obj.shape[0]), :]
                            for i, row in enumerate(samples):
                                print(f"{indent}     [{i}]: {row}")
                        else:
                            # Higher dimensional arrays
                            samples = obj[:min(max_samples, obj.shape[0])]
                            print(f"{indent}     {samples}")
                    except Exception as e:
                        print(f"{indent}     Error reading data: {e}")
                
                print()
                
            elif isinstance(obj, h5py.Group):
                print(f"{indent}📁 Group: {name}")
                if obj.attrs:
                    print(f"{indent}   Attributes:")
                    for key, val in obj.attrs.items():
                        print(f"{indent}     - {key}: {val}")
                print()
        
        # Visit all items in the file
        f.visititems(print_structure)
        
        # Print summary statistics
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        def count_items(name, obj):
            """Count datasets and groups."""
            if isinstance(obj, h5py.Dataset):
                count_items.datasets += 1
                count_items.total_size += obj.size
            elif isinstance(obj, h5py.Group):
                count_items.groups += 1
        
        count_items.datasets = 0
        count_items.groups = 0
        count_items.total_size = 0
        
        f.visititems(count_items)
        
        print(f"Total Groups: {count_items.groups}")
        print(f"Total Datasets: {count_items.datasets}")
        print(f"Total Elements: {count_items.total_size:,}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Explore the structure of HDF5 trajectory data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore a specific file
  python explore_hdf5.py /nfs/data/airhockey/trajectory_data434.hdf5
  
  # Explore with more sample data points
  python explore_hdf5.py /nfs/data/airhockey/trajectory_data434.hdf5 --max-samples 10
  
  # Explore the first file in the directory
  python explore_hdf5.py /nfs/data/airhockey/
        """
    )
    
    parser.add_argument(
        'path',
        type=str,
        help='Path to HDF5 file or directory containing HDF5 files'
    )
    
    parser.add_argument(
        '--max-samples',
        type=int,
        default=5,
        help='Number of sample values to display for each dataset (default: 5)'
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        # Single file
        explore_hdf5_structure(path, args.max_samples)
    elif path.is_dir():
        # Directory - find first HDF5 file
        hdf5_files = sorted(path.glob('*.hdf5'))
        if hdf5_files:
            print(f"Found {len(hdf5_files)} HDF5 files in directory")
            print(f"Exploring the first one: {hdf5_files[0].name}")
            explore_hdf5_structure(hdf5_files[0], args.max_samples)
        else:
            print(f"No HDF5 files found in {path}")
    else:
        print(f"Error: {path} is neither a file nor a directory")


if __name__ == "__main__":
    main()

