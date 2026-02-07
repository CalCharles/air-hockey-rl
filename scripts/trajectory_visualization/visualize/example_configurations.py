#!/usr/bin/env python3
"""
Example configurations for trajectory visualization.
This file shows different ways to use visualize_trajectory.py
"""

from pathlib import Path
import sys

# Import the visualization module
sys.path.insert(0, str(Path(__file__).parent))
from visualize_trajectory import (
    load_trajectory_data, 
    extract_positions, 
    TrajectoryRenderer, 
    create_trajectory_gif
)


def example_basic():
    """Basic usage - full resolution, all frames."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage (Full Resolution, All Frames)")
    print("="*80)
    
    data_path = Path('/nfs/data/airhockey/trajectory_data434.hdf5')
    output_path = Path(__file__).parent / 'example_basic.gif'
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    train_vals = load_trajectory_data(data_path)
    pos_x, pos_y, timestamps = extract_positions(train_vals)
    
    renderer = TrajectoryRenderer(
        table_width=1.0436,
        table_length=2.1104,
        paddle_radius=0.0508,
        ppm=400,
        robot_x_offset=1.2
    )
    
    create_trajectory_gif(pos_x, pos_y, timestamps, renderer, output_path)


def example_high_res():
    """High resolution rendering for detailed analysis."""
    print("\n" + "="*80)
    print("EXAMPLE 2: High Resolution (600 ppm)")
    print("="*80)
    
    data_path = Path('/nfs/data/airhockey/trajectory_data434.hdf5')
    output_path = Path(__file__).parent / 'example_high_res.gif'
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    train_vals = load_trajectory_data(data_path)
    pos_x, pos_y, timestamps = extract_positions(train_vals)
    
    renderer = TrajectoryRenderer(
        table_width=1.0436,
        table_length=2.1104,
        paddle_radius=0.0508,
        ppm=600,  # Higher resolution
        robot_x_offset=1.2
    )
    
    create_trajectory_gif(pos_x, pos_y, timestamps, renderer, output_path)
    

def example_subsampled():
    """Subsampled version for quick preview."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Subsampled (Every 2nd Frame)")
    print("="*80)
    
    data_path = Path('/nfs/data/airhockey/trajectory_data434.hdf5')
    output_path = Path(__file__).parent / 'example_subsampled.gif'
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    train_vals = load_trajectory_data(data_path)
    pos_x, pos_y, timestamps = extract_positions(train_vals)
    
    renderer = TrajectoryRenderer(
        table_width=1.0436,
        table_length=2.1104,
        paddle_radius=0.0508,
        ppm=400,
        robot_x_offset=1.2
    )
    
    create_trajectory_gif(
        pos_x, pos_y, timestamps, renderer, output_path,
        subsample=2  # Use every 2nd frame
    )


def example_preview():
    """Quick preview of first 50 frames."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Preview (First 50 Frames Only)")
    print("="*80)
    
    data_path = Path('/nfs/data/airhockey/trajectory_data434.hdf5')
    output_path = Path(__file__).parent / 'example_preview.gif'
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    train_vals = load_trajectory_data(data_path)
    pos_x, pos_y, timestamps = extract_positions(train_vals)
    
    renderer = TrajectoryRenderer(
        table_width=1.0436,
        table_length=2.1104,
        paddle_radius=0.0508,
        ppm=400,
        robot_x_offset=1.2
    )
    
    create_trajectory_gif(
        pos_x, pos_y, timestamps, renderer, output_path,
        max_frames=50  # Only first 50 frames
    )


def example_low_res_fast():
    """Low resolution, subsampled for very fast rendering."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Low Res + Subsampled (Fast Rendering)")
    print("="*80)
    
    data_path = Path('/nfs/data/airhockey/trajectory_data434.hdf5')
    output_path = Path(__file__).parent / 'example_low_res_fast.gif'
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        return
    
    train_vals = load_trajectory_data(data_path)
    pos_x, pos_y, timestamps = extract_positions(train_vals)
    
    renderer = TrajectoryRenderer(
        table_width=1.0436,
        table_length=2.1104,
        paddle_radius=0.0508,
        ppm=300,  # Lower resolution
        robot_x_offset=1.2
    )
    
    create_trajectory_gif(
        pos_x, pos_y, timestamps, renderer, output_path,
        subsample=2  # Every 2nd frame
    )


def main():
    """Run all examples or selected ones."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run example trajectory visualizations'
    )
    parser.add_argument(
        '--example',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run specific example (1-5), or all if not specified'
    )
    
    args = parser.parse_args()
    
    examples = {
        1: ('Basic', example_basic),
        2: ('High Resolution', example_high_res),
        3: ('Subsampled', example_subsampled),
        4: ('Preview', example_preview),
        5: ('Low Res Fast', example_low_res_fast),
    }
    
    if args.example:
        name, func = examples[args.example]
        print(f"\nRunning Example {args.example}: {name}")
        func()
    else:
        print("\nRunning all examples...")
        for num, (name, func) in examples.items():
            print(f"\n>>> Example {num}: {name}")
            func()
    
    print("\n" + "="*80)
    print("All examples complete!")
    print("="*80)


if __name__ == '__main__':
    main()

