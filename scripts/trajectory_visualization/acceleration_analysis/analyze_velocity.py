#!/usr/bin/env python3
"""
Analyze end-effector velocities (x, y, z) from trajectory data.
Fields 11-16 contain velocity data according to FIELD_DOCUMENTATION.md
Linear velocities: 11-13 (vx, vy, vz)
Angular velocities: 14-16 (vrx, vry, vrz)
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

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

def extract_velocities(train_vals):
    """
    Extract velocity data from train_vals.
    Fields 11-13: speed_vx, speed_vy, speed_vz (m/s)
    Fields 14-16: speed_vrx, speed_vry, speed_vrz (rad/s)
    """
    # Linear velocities
    vel_x = train_vals[:, 11]  # Field 11: Linear velocity in X direction
    vel_y = train_vals[:, 12]  # Field 12: Linear velocity in Y direction
    vel_z = train_vals[:, 13]  # Field 13: Linear velocity in Z direction
    
    # Angular velocities
    vel_rx = train_vals[:, 14]  # Field 14: Angular velocity around X axis
    vel_ry = train_vals[:, 15]  # Field 15: Angular velocity around Y axis
    vel_rz = train_vals[:, 16]  # Field 16: Angular velocity around Z axis
    
    # Extract timestamps for plotting
    timestamps = train_vals[:, 0]  # Field 0: cur_time (Unix timestamp)
    
    # Convert to relative time (seconds from start)
    relative_time = timestamps - timestamps[0]
    
    return vel_x, vel_y, vel_z, vel_rx, vel_ry, vel_rz, relative_time

def compute_statistics(vel_x, vel_y, vel_z, vel_rx, vel_ry, vel_rz):
    """Compute comprehensive statistics for velocity data."""
    
    # Compute magnitude of linear velocity vector
    vel_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    
    # Compute magnitude of angular velocity vector
    vel_ang_magnitude = np.sqrt(vel_rx**2 + vel_ry**2 + vel_rz**2)
    
    stats = {
        'X Linear Velocity': {
            'mean': np.mean(vel_x),
            'std': np.std(vel_x),
            'min': np.min(vel_x),
            'max': np.max(vel_x),
            'median': np.median(vel_x),
            'p25': np.percentile(vel_x, 25),
            'p75': np.percentile(vel_x, 75),
            'p95': np.percentile(vel_x, 95),
            'p99': np.percentile(vel_x, 99),
        },
        'Y Linear Velocity': {
            'mean': np.mean(vel_y),
            'std': np.std(vel_y),
            'min': np.min(vel_y),
            'max': np.max(vel_y),
            'median': np.median(vel_y),
            'p25': np.percentile(vel_y, 25),
            'p75': np.percentile(vel_y, 75),
            'p95': np.percentile(vel_y, 95),
            'p99': np.percentile(vel_y, 99),
        },
        'Z Linear Velocity': {
            'mean': np.mean(vel_z),
            'std': np.std(vel_z),
            'min': np.min(vel_z),
            'max': np.max(vel_z),
            'median': np.median(vel_z),
            'p25': np.percentile(vel_z, 25),
            'p75': np.percentile(vel_z, 75),
            'p95': np.percentile(vel_z, 95),
            'p99': np.percentile(vel_z, 99),
        },
        'Linear Speed (Magnitude)': {
            'mean': np.mean(vel_magnitude),
            'std': np.std(vel_magnitude),
            'min': np.min(vel_magnitude),
            'max': np.max(vel_magnitude),
            'median': np.median(vel_magnitude),
            'p25': np.percentile(vel_magnitude, 25),
            'p75': np.percentile(vel_magnitude, 75),
            'p95': np.percentile(vel_magnitude, 95),
            'p99': np.percentile(vel_magnitude, 99),
        },
        'RX Angular Velocity': {
            'mean': np.mean(vel_rx),
            'std': np.std(vel_rx),
            'min': np.min(vel_rx),
            'max': np.max(vel_rx),
            'median': np.median(vel_rx),
            'p25': np.percentile(vel_rx, 25),
            'p75': np.percentile(vel_rx, 75),
            'p95': np.percentile(vel_rx, 95),
            'p99': np.percentile(vel_rx, 99),
        },
        'RY Angular Velocity': {
            'mean': np.mean(vel_ry),
            'std': np.std(vel_ry),
            'min': np.min(vel_ry),
            'max': np.max(vel_ry),
            'median': np.median(vel_ry),
            'p25': np.percentile(vel_ry, 25),
            'p75': np.percentile(vel_ry, 75),
            'p95': np.percentile(vel_ry, 95),
            'p99': np.percentile(vel_ry, 99),
        },
        'RZ Angular Velocity': {
            'mean': np.mean(vel_rz),
            'std': np.std(vel_rz),
            'min': np.min(vel_rz),
            'max': np.max(vel_rz),
            'median': np.median(vel_rz),
            'p25': np.percentile(vel_rz, 25),
            'p75': np.percentile(vel_rz, 75),
            'p95': np.percentile(vel_rz, 95),
            'p99': np.percentile(vel_rz, 99),
        },
        'Angular Speed (Magnitude)': {
            'mean': np.mean(vel_ang_magnitude),
            'std': np.std(vel_ang_magnitude),
            'min': np.min(vel_ang_magnitude),
            'max': np.max(vel_ang_magnitude),
            'median': np.median(vel_ang_magnitude),
            'p25': np.percentile(vel_ang_magnitude, 25),
            'p75': np.percentile(vel_ang_magnitude, 75),
            'p95': np.percentile(vel_ang_magnitude, 95),
            'p99': np.percentile(vel_ang_magnitude, 99),
        }
    }
    
    return stats, vel_magnitude, vel_ang_magnitude

def print_statistics(stats):
    """Pretty print statistics."""
    print("\n" + "="*80)
    print("END-EFFECTOR VELOCITY STATISTICS")
    print("="*80)
    
    print("\n### LINEAR VELOCITIES ###")
    for axis in ['X Linear Velocity', 'Y Linear Velocity', 'Z Linear Velocity', 'Linear Speed (Magnitude)']:
        metrics = stats[axis]
        unit = 'm/s'
        print(f"\n{axis} ({unit}):")
        print(f"  Mean:       {metrics['mean']:>10.4f}")
        print(f"  Std Dev:    {metrics['std']:>10.4f}")
        print(f"  Median:     {metrics['median']:>10.4f}")
        print(f"  Min:        {metrics['min']:>10.4f}")
        print(f"  Max:        {metrics['max']:>10.4f}")
        print(f"  25th %ile:  {metrics['p25']:>10.4f}")
        print(f"  75th %ile:  {metrics['p75']:>10.4f}")
        print(f"  95th %ile:  {metrics['p95']:>10.4f}")
        print(f"  99th %ile:  {metrics['p99']:>10.4f}")
    
    print("\n### ANGULAR VELOCITIES ###")
    for axis in ['RX Angular Velocity', 'RY Angular Velocity', 'RZ Angular Velocity', 'Angular Speed (Magnitude)']:
        metrics = stats[axis]
        unit = 'rad/s'
        print(f"\n{axis} ({unit}):")
        print(f"  Mean:       {metrics['mean']:>10.4f}")
        print(f"  Std Dev:    {metrics['std']:>10.4f}")
        print(f"  Median:     {metrics['median']:>10.4f}")
        print(f"  Min:        {metrics['min']:>10.4f}")
        print(f"  Max:        {metrics['max']:>10.4f}")
        print(f"  25th %ile:  {metrics['p25']:>10.4f}")
        print(f"  75th %ile:  {metrics['p75']:>10.4f}")
        print(f"  95th %ile:  {metrics['p95']:>10.4f}")
        print(f"  99th %ile:  {metrics['p99']:>10.4f}")
    
    print("\n" + "="*80)

def create_visualizations(vel_x, vel_y, vel_z, vel_rx, vel_ry, vel_rz, 
                         vel_magnitude, vel_ang_magnitude, relative_time, output_dir):
    """Create comprehensive visualizations of velocity data."""
    
    # Create figure with subplots - LINEAR VELOCITIES
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Time series plot of all three linear velocity axes
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(relative_time, vel_x, 'r-', alpha=0.7, linewidth=0.8, label='Vx')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Linear Velocity (m/s)')
    ax1.set_title('X-Axis Linear Velocity Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(relative_time, vel_y, 'g-', alpha=0.7, linewidth=0.8, label='Vy')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Linear Velocity (m/s)')
    ax2.set_title('Y-Axis Linear Velocity Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(relative_time, vel_z, 'b-', alpha=0.7, linewidth=0.8, label='Vz')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Linear Velocity (m/s)')
    ax3.set_title('Z-Axis Linear Velocity Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 2. Combined linear velocity time series
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(relative_time, vel_x, 'r-', alpha=0.6, linewidth=0.8, label='Vx')
    ax4.plot(relative_time, vel_y, 'g-', alpha=0.6, linewidth=0.8, label='Vy')
    ax4.plot(relative_time, vel_z, 'b-', alpha=0.6, linewidth=0.8, label='Vz')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Linear Velocity (m/s)')
    ax4.set_title('All Axes Linear Velocity Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 3. Linear speed magnitude over time
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(relative_time, vel_magnitude, 'purple', alpha=0.7, linewidth=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Linear Speed (m/s)')
    ax5.set_title('Linear Speed Magnitude Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 4. Histograms
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(vel_x, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax6.set_xlabel('Linear Velocity (m/s)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('X-Axis Linear Velocity Distribution')
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(vel_y, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax7.set_xlabel('Linear Velocity (m/s)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Y-Axis Linear Velocity Distribution')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(vel_z, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax8.set_xlabel('Linear Velocity (m/s)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Z-Axis Linear Velocity Distribution')
    ax8.grid(True, alpha=0.3)
    
    # 5. Box plot comparison
    ax9 = plt.subplot(3, 3, 9)
    box_data = [vel_x, vel_y, vel_z, vel_magnitude]
    bp = ax9.boxplot(box_data, tick_labels=['Vx', 'Vy', 'Vz', 'Magnitude'],
                      patch_artist=True, showmeans=True)
    colors = ['lightcoral', 'lightgreen', 'lightblue', 'plum']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax9.set_ylabel('Linear Velocity (m/s)')
    ax9.set_title('Linear Velocity Distribution Comparison')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / 'velocity_linear_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved linear velocity visualization to: {output_path}")
    
    # Create second figure for ANGULAR VELOCITIES
    fig2 = plt.figure(figsize=(18, 12))
    
    # 1. Time series plot of all three angular velocity axes
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(relative_time, vel_rx, 'r-', alpha=0.7, linewidth=0.8, label='ωx')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Angular Velocity (rad/s)')
    ax1.set_title('RX Angular Velocity Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(relative_time, vel_ry, 'g-', alpha=0.7, linewidth=0.8, label='ωy')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Angular Velocity (rad/s)')
    ax2.set_title('RY Angular Velocity Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(relative_time, vel_rz, 'b-', alpha=0.7, linewidth=0.8, label='ωz')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Angular Velocity (rad/s)')
    ax3.set_title('RZ Angular Velocity Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 2. Combined angular velocity time series
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(relative_time, vel_rx, 'r-', alpha=0.6, linewidth=0.8, label='ωx')
    ax4.plot(relative_time, vel_ry, 'g-', alpha=0.6, linewidth=0.8, label='ωy')
    ax4.plot(relative_time, vel_rz, 'b-', alpha=0.6, linewidth=0.8, label='ωz')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Angular Velocity (rad/s)')
    ax4.set_title('All Axes Angular Velocity Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 3. Angular speed magnitude over time
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(relative_time, vel_ang_magnitude, 'purple', alpha=0.7, linewidth=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Angular Speed (rad/s)')
    ax5.set_title('Angular Speed Magnitude Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 4. Histograms
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(vel_rx, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax6.set_xlabel('Angular Velocity (rad/s)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('RX Angular Velocity Distribution')
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(vel_ry, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax7.set_xlabel('Angular Velocity (rad/s)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('RY Angular Velocity Distribution')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(vel_rz, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax8.set_xlabel('Angular Velocity (rad/s)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('RZ Angular Velocity Distribution')
    ax8.grid(True, alpha=0.3)
    
    # 5. Box plot comparison
    ax9 = plt.subplot(3, 3, 9)
    box_data = [vel_rx, vel_ry, vel_rz, vel_ang_magnitude]
    bp = ax9.boxplot(box_data, tick_labels=['ωx', 'ωy', 'ωz', 'Magnitude'],
                      patch_artist=True, showmeans=True)
    colors = ['lightcoral', 'lightgreen', 'lightblue', 'plum']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax9.set_ylabel('Angular Velocity (rad/s)')
    ax9.set_title('Angular Velocity Distribution Comparison')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path2 = output_dir / 'velocity_angular_analysis.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved angular velocity visualization to: {output_path2}")
    
    # Create third figure for detailed analysis
    fig3 = plt.figure(figsize=(18, 10))
    
    # 1. Rolling statistics for linear velocities
    window = min(50, len(vel_x) // 10)
    if window > 1:
        vel_x_rolling = np.convolve(vel_x, np.ones(window)/window, mode='valid')
        vel_y_rolling = np.convolve(vel_y, np.ones(window)/window, mode='valid')
        vel_z_rolling = np.convolve(vel_z, np.ones(window)/window, mode='valid')
        time_rolling = relative_time[:len(vel_x_rolling)]
        
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(relative_time, vel_x, 'r-', alpha=0.3, linewidth=0.5, label='Raw')
        ax1.plot(time_rolling, vel_x_rolling, 'darkred', linewidth=2, label=f'Rolling Mean (w={window})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Linear Velocity (m/s)')
        ax1.set_title('Vx: Raw vs Smoothed')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(relative_time, vel_y, 'g-', alpha=0.3, linewidth=0.5, label='Raw')
        ax2.plot(time_rolling, vel_y_rolling, 'darkgreen', linewidth=2, label=f'Rolling Mean (w={window})')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Linear Velocity (m/s)')
        ax2.set_title('Vy: Raw vs Smoothed')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(relative_time, vel_z, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
        ax3.plot(time_rolling, vel_z_rolling, 'darkblue', linewidth=2, label=f'Rolling Mean (w={window})')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Linear Velocity (m/s)')
        ax3.set_title('Vz: Raw vs Smoothed')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 2. Scatter plots - velocity relationships
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(vel_x, vel_y, c=relative_time, cmap='viridis', alpha=0.5, s=10)
    ax4.set_xlabel('Vx (m/s)')
    ax4.set_ylabel('Vy (m/s)')
    ax4.set_title('Vx vs Vy (colored by time)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Time (s)')
    
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(vel_x, vel_z, c=relative_time, cmap='viridis', alpha=0.5, s=10)
    ax5.set_xlabel('Vx (m/s)')
    ax5.set_ylabel('Vz (m/s)')
    ax5.set_title('Vx vs Vz (colored by time)')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Time (s)')
    
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(vel_y, vel_z, c=relative_time, cmap='viridis', alpha=0.5, s=10)
    ax6.set_xlabel('Vy (m/s)')
    ax6.set_ylabel('Vz (m/s)')
    ax6.set_title('Vy vs Vz (colored by time)')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Time (s)')
    
    plt.tight_layout()
    
    output_path3 = output_dir / 'velocity_detailed_analysis.png'
    plt.savefig(output_path3, dpi=150, bbox_inches='tight')
    print(f"Saved detailed velocity visualization to: {output_path3}")
    
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
    
    # Extract velocities
    vel_x, vel_y, vel_z, vel_rx, vel_ry, vel_rz, relative_time = extract_velocities(train_vals)
    
    print(f"\nExtracted velocity data:")
    print(f"  Number of timesteps: {len(vel_x)}")
    print(f"  Duration: {relative_time[-1]:.2f} seconds")
    print(f"  Average sampling rate: {len(vel_x) / relative_time[-1]:.2f} Hz")
    
    # Compute statistics
    stats, vel_magnitude, vel_ang_magnitude = compute_statistics(vel_x, vel_y, vel_z, vel_rx, vel_ry, vel_rz)
    
    # Print statistics
    print_statistics(stats)
    
    # Create visualizations
    create_visualizations(vel_x, vel_y, vel_z, vel_rx, vel_ry, vel_rz, 
                         vel_magnitude, vel_ang_magnitude, relative_time, output_dir)
    
    print("\n✓ Velocity analysis complete!")

if __name__ == '__main__':
    main()


