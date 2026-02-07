#!/usr/bin/env python3
"""
Analyze end-effector positions (x, y, z) from trajectory data.
Fields 5-10 contain pose data according to FIELD_DOCUMENTATION.md
Position: 5-7 (x, y, z)
Orientation: 8-10 (rx, ry, rz)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from mpl_toolkits.mplot3d import Axes3D

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

def load_trajectory_data(filepath):
    """Load trajectory data from HDF5 file."""
    print(f"Loading data from: {filepath}")
    with h5py.File(filepath, 'r') as f:
        train_vals = f['train_vals'][:]
        print(f"Data shape: {train_vals.shape}")
        print(f"Number of timesteps: {train_vals.shape[0]}")
        print(f"Number of fields per timestep: {train_vals.shape[1]}")
    return train_vals

def extract_positions(train_vals):
    """
    Extract position data from train_vals.
    Fields 5-7: pose_x, pose_y, pose_z (meters)
    Fields 8-10: pose_rx, pose_ry, pose_rz (radians)
    """
    # Positions
    pos_x = train_vals[:, 5]   # Field 5: X position
    pos_y = train_vals[:, 6]   # Field 6: Y position
    pos_z = train_vals[:, 7]   # Field 7: Z position
    
    # Orientations
    pos_rx = train_vals[:, 8]  # Field 8: Rotation around X axis
    pos_ry = train_vals[:, 9]  # Field 9: Rotation around Y axis
    pos_rz = train_vals[:, 10] # Field 10: Rotation around Z axis
    
    # Extract timestamps for plotting
    timestamps = train_vals[:, 0]  # Field 0: cur_time (Unix timestamp)
    
    # Convert to relative time (seconds from start)
    relative_time = timestamps - timestamps[0]
    
    return pos_x, pos_y, pos_z, pos_rx, pos_ry, pos_rz, relative_time

def compute_statistics(pos_x, pos_y, pos_z, pos_rx, pos_ry, pos_rz):
    """Compute comprehensive statistics for position data."""
    
    # Compute distance from origin (not really meaningful but interesting)
    distance_from_origin = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
    
    # Compute total path length
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dz = np.diff(pos_z)
    segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
    total_path_length = np.sum(segment_lengths)
    
    stats = {
        'X Position': {
            'mean': np.mean(pos_x),
            'std': np.std(pos_x),
            'min': np.min(pos_x),
            'max': np.max(pos_x),
            'median': np.median(pos_x),
            'range': np.max(pos_x) - np.min(pos_x),
            'p25': np.percentile(pos_x, 25),
            'p75': np.percentile(pos_x, 75),
            'p95': np.percentile(pos_x, 95),
            'p99': np.percentile(pos_x, 99),
        },
        'Y Position': {
            'mean': np.mean(pos_y),
            'std': np.std(pos_y),
            'min': np.min(pos_y),
            'max': np.max(pos_y),
            'median': np.median(pos_y),
            'range': np.max(pos_y) - np.min(pos_y),
            'p25': np.percentile(pos_y, 25),
            'p75': np.percentile(pos_y, 75),
            'p95': np.percentile(pos_y, 95),
            'p99': np.percentile(pos_y, 99),
        },
        'Z Position': {
            'mean': np.mean(pos_z),
            'std': np.std(pos_z),
            'min': np.min(pos_z),
            'max': np.max(pos_z),
            'median': np.median(pos_z),
            'range': np.max(pos_z) - np.min(pos_z),
            'p25': np.percentile(pos_z, 25),
            'p75': np.percentile(pos_z, 75),
            'p95': np.percentile(pos_z, 95),
            'p99': np.percentile(pos_z, 99),
        },
        'RX Orientation': {
            'mean': np.mean(pos_rx),
            'std': np.std(pos_rx),
            'min': np.min(pos_rx),
            'max': np.max(pos_rx),
            'median': np.median(pos_rx),
            'range': np.max(pos_rx) - np.min(pos_rx),
            'p25': np.percentile(pos_rx, 25),
            'p75': np.percentile(pos_rx, 75),
            'p95': np.percentile(pos_rx, 95),
            'p99': np.percentile(pos_rx, 99),
        },
        'RY Orientation': {
            'mean': np.mean(pos_ry),
            'std': np.std(pos_ry),
            'min': np.min(pos_ry),
            'max': np.max(pos_ry),
            'median': np.median(pos_ry),
            'range': np.max(pos_ry) - np.min(pos_ry),
            'p25': np.percentile(pos_ry, 25),
            'p75': np.percentile(pos_ry, 75),
            'p95': np.percentile(pos_ry, 95),
            'p99': np.percentile(pos_ry, 99),
        },
        'RZ Orientation': {
            'mean': np.mean(pos_rz),
            'std': np.std(pos_rz),
            'min': np.min(pos_rz),
            'max': np.max(pos_rz),
            'median': np.median(pos_rz),
            'range': np.max(pos_rz) - np.min(pos_rz),
            'p25': np.percentile(pos_rz, 25),
            'p75': np.percentile(pos_rz, 75),
            'p95': np.percentile(pos_rz, 95),
            'p99': np.percentile(pos_rz, 99),
        }
    }
    
    return stats, distance_from_origin, total_path_length

def print_statistics(stats, total_path_length):
    """Pretty print statistics."""
    print("\n" + "="*80)
    print("END-EFFECTOR POSITION STATISTICS")
    print("="*80)
    
    print("\n### CARTESIAN POSITIONS ###")
    for axis in ['X Position', 'Y Position', 'Z Position']:
        metrics = stats[axis]
        unit = 'meters'
        print(f"\n{axis} ({unit}):")
        print(f"  Mean:       {metrics['mean']:>10.4f}")
        print(f"  Std Dev:    {metrics['std']:>10.4f}")
        print(f"  Median:     {metrics['median']:>10.4f}")
        print(f"  Min:        {metrics['min']:>10.4f}")
        print(f"  Max:        {metrics['max']:>10.4f}")
        print(f"  Range:      {metrics['range']:>10.4f}")
        print(f"  25th %ile:  {metrics['p25']:>10.4f}")
        print(f"  75th %ile:  {metrics['p75']:>10.4f}")
        print(f"  95th %ile:  {metrics['p95']:>10.4f}")
        print(f"  99th %ile:  {metrics['p99']:>10.4f}")
    
    print(f"\nTotal Path Length: {total_path_length:.4f} meters")
    
    print("\n### ORIENTATIONS (EULER ANGLES) ###")
    for axis in ['RX Orientation', 'RY Orientation', 'RZ Orientation']:
        metrics = stats[axis]
        unit = 'radians'
        print(f"\n{axis} ({unit}):")
        print(f"  Mean:       {metrics['mean']:>10.4f}")
        print(f"  Std Dev:    {metrics['std']:>10.4f}")
        print(f"  Median:     {metrics['median']:>10.4f}")
        print(f"  Min:        {metrics['min']:>10.4f}")
        print(f"  Max:        {metrics['max']:>10.4f}")
        print(f"  Range:      {metrics['range']:>10.4f}")
        print(f"  25th %ile:  {metrics['p25']:>10.4f}")
        print(f"  75th %ile:  {metrics['p75']:>10.4f}")
        print(f"  95th %ile:  {metrics['p95']:>10.4f}")
        print(f"  99th %ile:  {metrics['p99']:>10.4f}")
    
    print("\n" + "="*80)

def create_visualizations(pos_x, pos_y, pos_z, pos_rx, pos_ry, pos_rz, 
                         relative_time, output_dir):
    """Create comprehensive visualizations of position data."""
    
    # Create figure with subplots - CARTESIAN POSITIONS
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Time series plot of all three position axes
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(relative_time, pos_x, 'r-', alpha=0.7, linewidth=0.8, label='X')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Position (m)')
    ax1.set_title('X Position Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(relative_time, pos_y, 'g-', alpha=0.7, linewidth=0.8, label='Y')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position (m)')
    ax2.set_title('Y Position Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(relative_time, pos_z, 'b-', alpha=0.7, linewidth=0.8, label='Z')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Position (m)')
    ax3.set_title('Z Position Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 2. Combined position time series
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(relative_time, pos_x, 'r-', alpha=0.6, linewidth=0.8, label='X')
    ax4.plot(relative_time, pos_y, 'g-', alpha=0.6, linewidth=0.8, label='Y')
    ax4.plot(relative_time, pos_z, 'b-', alpha=0.6, linewidth=0.8, label='Z')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position (m)')
    ax4.set_title('All Axes Position Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 3. 2D trajectory XY plane
    ax5 = plt.subplot(3, 3, 5)
    scatter = ax5.scatter(pos_x, pos_y, c=relative_time, cmap='viridis', s=20, alpha=0.7)
    ax5.plot(pos_x, pos_y, 'gray', alpha=0.3, linewidth=0.5)
    ax5.scatter(pos_x[0], pos_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax5.scatter(pos_x[-1], pos_y[-1], c='red', s=100, marker='x', label='End', zorder=5)
    ax5.set_xlabel('X Position (m)')
    ax5.set_ylabel('Y Position (m)')
    ax5.set_title('Trajectory in XY Plane (Table View)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_aspect('equal', adjustable='box')
    plt.colorbar(scatter, ax=ax5, label='Time (s)')
    
    # 4. Histograms
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(pos_x, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax6.set_xlabel('Position (m)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('X Position Distribution')
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(pos_y, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax7.set_xlabel('Position (m)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Y Position Distribution')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(pos_z, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax8.set_xlabel('Position (m)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Z Position Distribution')
    ax8.grid(True, alpha=0.3)
    
    # 5. Box plot comparison
    ax9 = plt.subplot(3, 3, 9)
    box_data = [pos_x, pos_y, pos_z]
    bp = ax9.boxplot(box_data, tick_labels=['X', 'Y', 'Z'],
                      patch_artist=True, showmeans=True)
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax9.set_ylabel('Position (m)')
    ax9.set_title('Position Distribution Comparison')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / 'position_cartesian_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved cartesian position visualization to: {output_path}")
    
    # Create second figure for ORIENTATIONS
    fig2 = plt.figure(figsize=(18, 12))
    
    # 1. Time series plot of all three orientation axes
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(relative_time, pos_rx, 'r-', alpha=0.7, linewidth=0.8, label='RX')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Orientation (rad)')
    ax1.set_title('RX Orientation Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(relative_time, pos_ry, 'g-', alpha=0.7, linewidth=0.8, label='RY')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Orientation (rad)')
    ax2.set_title('RY Orientation Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(relative_time, pos_rz, 'b-', alpha=0.7, linewidth=0.8, label='RZ')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Orientation (rad)')
    ax3.set_title('RZ Orientation Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 2. Combined orientation time series
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(relative_time, pos_rx, 'r-', alpha=0.6, linewidth=0.8, label='RX')
    ax4.plot(relative_time, pos_ry, 'g-', alpha=0.6, linewidth=0.8, label='RY')
    ax4.plot(relative_time, pos_rz, 'b-', alpha=0.6, linewidth=0.8, label='RZ')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Orientation (rad)')
    ax4.set_title('All Axes Orientation Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 3. Convert to degrees for intuition
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(relative_time, np.degrees(pos_rx), 'r-', alpha=0.6, linewidth=0.8, label='RX')
    ax5.plot(relative_time, np.degrees(pos_ry), 'g-', alpha=0.6, linewidth=0.8, label='RY')
    ax5.plot(relative_time, np.degrees(pos_rz), 'b-', alpha=0.6, linewidth=0.8, label='RZ')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Orientation (degrees)')
    ax5.set_title('All Axes Orientation Over Time (Degrees)')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # 4. Histograms
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(pos_rx, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax6.set_xlabel('Orientation (rad)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('RX Orientation Distribution')
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(pos_ry, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax7.set_xlabel('Orientation (rad)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('RY Orientation Distribution')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(pos_rz, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax8.set_xlabel('Orientation (rad)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('RZ Orientation Distribution')
    ax8.grid(True, alpha=0.3)
    
    # 5. Box plot comparison
    ax9 = plt.subplot(3, 3, 9)
    box_data = [pos_rx, pos_ry, pos_rz]
    bp = ax9.boxplot(box_data, tick_labels=['RX', 'RY', 'RZ'],
                      patch_artist=True, showmeans=True)
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax9.set_ylabel('Orientation (rad)')
    ax9.set_title('Orientation Distribution Comparison')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path2 = output_dir / 'position_orientation_analysis.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved orientation visualization to: {output_path2}")
    
    # Create third figure for 3D trajectory
    fig3 = plt.figure(figsize=(18, 10))
    
    # 1. 3D trajectory
    ax1 = fig3.add_subplot(2, 2, 1, projection='3d')
    scatter = ax1.scatter(pos_x, pos_y, pos_z, c=relative_time, cmap='viridis', s=20, alpha=0.7)
    ax1.plot(pos_x, pos_y, pos_z, 'gray', alpha=0.3, linewidth=0.5)
    ax1.scatter(pos_x[0], pos_y[0], pos_z[0], c='green', s=100, marker='o', label='Start')
    ax1.scatter(pos_x[-1], pos_y[-1], pos_z[-1], c='red', s=100, marker='x', label='End')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_zlabel('Z Position (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Time (s)', shrink=0.5)
    
    # 2. XZ plane trajectory
    ax2 = plt.subplot(2, 2, 2)
    scatter = ax2.scatter(pos_x, pos_z, c=relative_time, cmap='viridis', s=20, alpha=0.7)
    ax2.plot(pos_x, pos_z, 'gray', alpha=0.3, linewidth=0.5)
    ax2.scatter(pos_x[0], pos_z[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(pos_x[-1], pos_z[-1], c='red', s=100, marker='x', label='End', zorder=5)
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Z Position (m)')
    ax2.set_title('Trajectory in XZ Plane (Side View)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='Time (s)')
    
    # 3. YZ plane trajectory
    ax3 = plt.subplot(2, 2, 3)
    scatter = ax3.scatter(pos_y, pos_z, c=relative_time, cmap='viridis', s=20, alpha=0.7)
    ax3.plot(pos_y, pos_z, 'gray', alpha=0.3, linewidth=0.5)
    ax3.scatter(pos_y[0], pos_z[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.scatter(pos_y[-1], pos_z[-1], c='red', s=100, marker='x', label='End', zorder=5)
    ax3.set_xlabel('Y Position (m)')
    ax3.set_ylabel('Z Position (m)')
    ax3.set_title('Trajectory in YZ Plane (Front View)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='Time (s)')
    
    # 4. Instantaneous speed
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dz = np.diff(pos_z)
    dt = np.diff(relative_time)
    instantaneous_speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(relative_time[:-1], instantaneous_speed, 'purple', alpha=0.7, linewidth=0.8)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Instantaneous Speed (m/s)')
    ax4.set_title('Speed from Position Changes')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=np.mean(instantaneous_speed), color='r', linestyle='--', 
                label=f'Mean: {np.mean(instantaneous_speed):.3f} m/s')
    ax4.legend()
    
    plt.tight_layout()
    
    output_path3 = output_dir / 'position_3d_trajectory.png'
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"Saved 3D trajectory visualization to: {output_path3}")
    
    plt.close('all')

def main():
    # Use trajectory_data434.hdf5
    data_path = Path('/nfs/data/airhockey/trajectory_data434.hdf5')
    output_dir = Path('/home/air-hockey/daliu/air-hockey-rl/scripts/trajectory_visualization/acceleration_analysis')
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    # Load data
    train_vals = load_trajectory_data(data_path)
    
    # Extract positions
    pos_x, pos_y, pos_z, pos_rx, pos_ry, pos_rz, relative_time = extract_positions(train_vals)
    
    print(f"\nExtracted position data:")
    print(f"  Number of timesteps: {len(pos_x)}")
    print(f"  Duration: {relative_time[-1]:.2f} seconds")
    print(f"  Average sampling rate: {len(pos_x) / relative_time[-1]:.2f} Hz")
    
    # Compute statistics
    stats, distance_from_origin, total_path_length = compute_statistics(pos_x, pos_y, pos_z, pos_rx, pos_ry, pos_rz)
    
    # Print statistics
    print_statistics(stats, total_path_length)
    
    # Create visualizations
    create_visualizations(pos_x, pos_y, pos_z, pos_rx, pos_ry, pos_rz, 
                         relative_time, output_dir)
    
    print("\n✓ Position analysis complete!")

if __name__ == '__main__':
    main()


