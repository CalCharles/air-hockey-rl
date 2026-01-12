#!/usr/bin/env python3
"""
Detailed analysis of trajectory data from real-world air hockey robot.
This script analyzes the structure and content of the HDF5 trajectory file.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def analyze_trajectory_data(file_path):
    """
    Analyze the trajectory data in detail.
    
    Args:
        file_path: Path to the HDF5 file
    """
    print(f"\n{'='*80}")
    print(f"DETAILED TRAJECTORY DATA ANALYSIS")
    print(f"File: {file_path}")
    print(f"{'='*80}\n")
    
    with h5py.File(file_path, 'r') as f:
        # Load all data
        num_hits = f['num_hits'][()]
        occlusions = f['occlusions'][()]
        train_img = f['train_img'][:]
        train_vals = f['train_vals'][:]
        
        print("=" * 80)
        print("BASIC INFORMATION")
        print("=" * 80)
        print(f"Number of hits: {num_hits}")
        print(f"Number of occlusions: {occlusions}")
        print(f"Number of trajectory frames: {len(train_vals)}")
        print(f"Number of images: {len(train_img)}")
        print(f"Image shape: {train_img.shape[1:]}")
        print(f"Values per frame: {train_vals.shape[1]}")
        
        # Analyze train_vals structure
        print(f"\n{'='*80}")
        print("TRAIN_VALS ANALYSIS (32 values per frame)")
        print("=" * 80)
        
        # The first value looks like a timestamp
        print("\n--- Potential Field Identification ---")
        print(f"Field 0 (timestamp?): {train_vals[0, 0]:.2f} -> {train_vals[-1, 0]:.2f}")
        print(f"  Range: {train_vals[:, 0].min():.2f} to {train_vals[:, 0].max():.2f}")
        print(f"  This looks like a Unix timestamp (around 2024)")
        
        # Field 1 looks constant
        print(f"\nField 1: {train_vals[0, 1]:.2f} (constant: {np.allclose(train_vals[:, 1], train_vals[0, 1])})")
        
        # Field 2 appears to be incrementing
        print(f"\nField 2 (frame counter?): {train_vals[0, 2]:.1f} -> {train_vals[-1, 2]:.1f}")
        print(f"  Increment: {np.diff(train_vals[:, 2]).mean():.2f}")
        
        # Field 3 appears constant (0)
        print(f"\nField 3: {train_vals[0, 3]:.1f} (constant: {np.allclose(train_vals[:, 3], 0)})")
        
        # Field 4 appears constant (1)
        print(f"\nField 4: {train_vals[0, 4]:.1f} (constant: {np.allclose(train_vals[:, 4], 1)})")
        
        # Analyze value ranges for all fields
        print(f"\n{'='*80}")
        print("VALUE RANGES FOR ALL 32 FIELDS")
        print("=" * 80)
        
        for i in range(32):
            min_val = train_vals[:, i].min()
            max_val = train_vals[:, i].max()
            mean_val = train_vals[:, i].mean()
            std_val = train_vals[:, i].std()
            is_constant = np.allclose(train_vals[:, i], train_vals[0, i])
            
            const_str = " [CONSTANT]" if is_constant else ""
            print(f"Field {i:2d}: min={min_val:12.6f}, max={max_val:12.6f}, "
                  f"mean={mean_val:12.6f}, std={std_val:12.6f}{const_str}")
        
        # Look for patterns - group similar fields
        print(f"\n{'='*80}")
        print("PATTERN ANALYSIS - Potential Field Groups")
        print("=" * 80)
        
        # Fields that look like positions (moderate range, varying)
        position_like = []
        velocity_like = []
        small_values = []
        
        for i in range(5, 32):  # Skip first 5 metadata fields
            range_val = train_vals[:, i].max() - train_vals[:, i].min()
            std_val = train_vals[:, i].std()
            
            if range_val > 0.1:  # Significant variation
                if abs(train_vals[:, i].mean()) > 1:
                    position_like.append(i)
                else:
                    if std_val > 0.01:
                        velocity_like.append(i)
            elif range_val > 1e-6:
                small_values.append(i)
        
        print(f"\nFields with large values (position-like): {position_like}")
        print(f"Fields with small varying values (velocity-like): {velocity_like}")
        print(f"Fields with very small values: {small_values}")
        
        # Hypothesis: Air hockey robot likely has:
        # - Puck position (x, y, z?)
        # - Puck velocity (vx, vy, vz?)
        # - Mallet/striker position (x, y, z?)
        # - Mallet/striker velocity (vx, vy, vz?)
        # - Joint angles/positions for robot arm
        # - Joint velocities for robot arm
        
        print(f"\n{'='*80}")
        print("HYPOTHESIS: Field Structure")
        print("=" * 80)
        print("""
Based on the data ranges and air hockey domain knowledge:
        
Fields 0-4: Metadata
  [0] Timestamp (Unix time)
  [1] Constant identifier (434)
  [2] Frame counter
  [3] Unknown flag (0)
  [4] Unknown flag (1)
  
Fields 5-16: Likely Robot State (12 values)
  Possibly: 7 joint positions + 7 joint velocities (7-DOF robot?)
            Or: 6 joint positions + 6 joint velocities (6-DOF robot?)
  
  Fields with large negative values: {[i for i in range(5, 17) if train_vals[:, i].mean() < -1]}
  Fields oscillating around 0: {[i for i in range(5, 17) if abs(train_vals[:, i].mean()) < 1]}

Fields 17-31: Likely End-Effector/Puck State (15 values)
  Possibly includes:
    - Puck position (x, y, z?)
    - Puck velocity (vx, vy, vz?)
    - Mallet/end-effector position (x, y, z?)
    - Mallet/end-effector velocity (vx, vy, vz?)
    - Or combined state representation
        """)
        
        # Analyze temporal changes
        print(f"\n{'='*80}")
        print("TEMPORAL ANALYSIS")
        print("=" * 80)
        
        # Calculate velocities for position-like fields
        diffs = np.diff(train_vals, axis=0)
        
        print("\nLargest changes between consecutive frames (indicating motion):")
        for i in position_like[:10]:  # Show top 10
            max_diff = np.abs(diffs[:, i]).max()
            mean_diff = np.abs(diffs[:, i]).mean()
            print(f"  Field {i:2d}: max_diff={max_diff:10.6f}, mean_diff={mean_diff:10.6f}")
        
        return train_vals, train_img


def create_visualization(train_vals, train_img, output_dir):
    """
    Create visualizations of the trajectory data.
    
    Args:
        train_vals: Array of trajectory values
        train_img: Array of images
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Plot 1: Show sample images
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    indices = np.linspace(0, len(train_img) - 1, 8, dtype=int)
    
    for idx, ax in zip(indices, axes.flat):
        ax.imshow(train_img[idx])
        ax.set_title(f"Frame {idx}")
        ax.axis('off')
    
    plt.tight_layout()
    img_path = output_dir / "sample_images.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {img_path}")
    plt.close()
    
    # Plot 2: Time series of interesting fields
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flat
    
    # Plot fields with significant variation
    fields_to_plot = [5, 6, 7, 11, 12, 13, 17, 18]  # Select varied fields
    
    for idx, field_idx in enumerate(fields_to_plot):
        axes[idx].plot(train_vals[:, field_idx], linewidth=1)
        axes[idx].set_title(f"Field {field_idx}")
        axes[idx].set_xlabel("Frame")
        axes[idx].set_ylabel("Value")
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    ts_path = output_dir / "time_series.png"
    plt.savefig(ts_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {ts_path}")
    plt.close()
    
    # Plot 3: Correlation matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Focus on fields 5-31 (skip metadata)
    data_fields = train_vals[:, 5:32]
    corr_matrix = np.corrcoef(data_fields.T)
    
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax.set_title("Correlation Matrix (Fields 5-31)")
    ax.set_xlabel("Field Index (offset by 5)")
    ax.set_ylabel("Field Index (offset by 5)")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation")
    
    # Add ticks
    tick_positions = np.arange(0, 27, 2)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_positions + 5)
    ax.set_yticklabels(tick_positions + 5)
    
    plt.tight_layout()
    corr_path = output_dir / "correlation_matrix.png"
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {corr_path}")
    plt.close()
    
    # Plot 4: Statistical distribution
    fig, axes = plt.subplots(4, 7, figsize=(20, 10))
    axes = axes.flat
    
    for i in range(5, 32):  # Skip metadata fields
        axes[i-5].hist(train_vals[:, i], bins=20, edgecolor='black', alpha=0.7)
        axes[i-5].set_title(f"Field {i}", fontsize=8)
        axes[i-5].tick_params(labelsize=6)
        axes[i-5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    hist_path = output_dir / "distributions.png"
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {hist_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze trajectory data from real-world air hockey robot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to HDF5 trajectory file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./trajectory_analysis_output',
        help='Directory to save analysis plots (default: ./trajectory_analysis_output)'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    
    args = parser.parse_args()
    
    # Analyze data
    train_vals, train_img = analyze_trajectory_data(args.file_path)
    
    # Create visualizations
    if not args.no_plots:
        create_visualization(train_vals, train_img, args.output_dir)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

