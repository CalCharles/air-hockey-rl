#!/usr/bin/env python3
"""
Generate a comprehensive summary report for trajectory analysis.
Combines acceleration, velocity, and position statistics.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sns.set_style("whitegrid")

def load_trajectory_data(filepath):
    """Load trajectory data from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        train_vals = f['train_vals'][:]
    return train_vals

def extract_all_data(train_vals):
    """Extract position, velocity, and acceleration data."""
    # Positions (fields 5-7)
    pos_x = train_vals[:, 5]
    pos_y = train_vals[:, 6]
    pos_z = train_vals[:, 7]
    
    # Velocities (fields 11-13)
    vel_x = train_vals[:, 11]
    vel_y = train_vals[:, 12]
    vel_z = train_vals[:, 13]
    
    # Accelerations (fields 23-25)
    acc_x = train_vals[:, 23]
    acc_y = train_vals[:, 24]
    acc_z = train_vals[:, 25]
    
    # Timestamps
    timestamps = train_vals[:, 0]
    relative_time = timestamps - timestamps[0]
    
    return pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, relative_time

def create_summary_plot(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, 
                       acc_x, acc_y, acc_z, relative_time, output_dir):
    """Create a comprehensive summary plot."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Position over time
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(relative_time, pos_x, 'r-', linewidth=1.5, label='X')
    ax1.plot(relative_time, pos_y, 'g-', linewidth=1.5, label='Y')
    ax1.plot(relative_time, pos_z, 'b-', linewidth=1.5, label='Z')
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Position (m)', fontsize=10)
    ax1.set_title('Position Over Time', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Velocity over time
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(relative_time, vel_x, 'r-', linewidth=1.5, label='Vx')
    ax2.plot(relative_time, vel_y, 'g-', linewidth=1.5, label='Vy')
    ax2.plot(relative_time, vel_z, 'b-', linewidth=1.5, label='Vz')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Velocity (m/s)', fontsize=10)
    ax2.set_title('Velocity Over Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Acceleration over time
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(relative_time, acc_x, 'r-', linewidth=1.5, label='Ax')
    ax3.plot(relative_time, acc_y, 'g-', linewidth=1.5, label='Ay')
    ax3.plot(relative_time, acc_z, 'b-', linewidth=1.5, label='Az')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Acceleration (m/s²)', fontsize=10)
    ax3.set_title('Acceleration Over Time', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Magnitudes
    pos_mag = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
    vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    ax4 = plt.subplot(3, 4, 4)
    ax4.plot(relative_time, vel_mag, 'purple', linewidth=2, label='Speed')
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Speed (m/s)', fontsize=10)
    ax4.set_title('Speed Magnitude Over Time', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. XY trajectory
    ax5 = plt.subplot(3, 4, 5)
    scatter = ax5.scatter(pos_x, pos_y, c=relative_time, cmap='viridis', s=30)
    ax5.plot(pos_x, pos_y, 'gray', alpha=0.3, linewidth=1)
    ax5.scatter(pos_x[0], pos_y[0], c='green', s=150, marker='o', label='Start', zorder=5)
    ax5.scatter(pos_x[-1], pos_y[-1], c='red', s=150, marker='X', label='End', zorder=5)
    ax5.set_xlabel('X Position (m)', fontsize=10)
    ax5.set_ylabel('Y Position (m)', fontsize=10)
    ax5.set_title('XY Trajectory (Table View)', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal', adjustable='box')
    plt.colorbar(scatter, ax=ax5, label='Time (s)')
    
    # 6. Position distributions
    ax6 = plt.subplot(3, 4, 6)
    box_data = [pos_x, pos_y, pos_z]
    bp = ax6.boxplot(box_data, tick_labels=['X', 'Y', 'Z'], patch_artist=True, showmeans=True)
    colors = ['lightcoral', 'lightgreen', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax6.set_ylabel('Position (m)', fontsize=10)
    ax6.set_title('Position Distributions', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Velocity distributions
    ax7 = plt.subplot(3, 4, 7)
    box_data = [vel_x, vel_y, vel_z]
    bp = ax7.boxplot(box_data, tick_labels=['Vx', 'Vy', 'Vz'], patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax7.set_ylabel('Velocity (m/s)', fontsize=10)
    ax7.set_title('Velocity Distributions', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Acceleration distributions
    ax8 = plt.subplot(3, 4, 8)
    box_data = [acc_x, acc_y, acc_z]
    bp = ax8.boxplot(box_data, tick_labels=['Ax', 'Ay', 'Az'], patch_artist=True, showmeans=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax8.set_ylabel('Acceleration (m/s²)', fontsize=10)
    ax8.set_title('Acceleration Distributions', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # 9. X-axis comparison
    ax9 = plt.subplot(3, 4, 9)
    ax9_twin1 = ax9.twinx()
    ax9_twin2 = ax9.twinx()
    ax9_twin2.spines['right'].set_position(('outward', 60))
    
    p1 = ax9.plot(relative_time, pos_x, 'r-', linewidth=1.5, alpha=0.7, label='Position')
    p2 = ax9_twin1.plot(relative_time, vel_x, 'g-', linewidth=1.5, alpha=0.7, label='Velocity')
    p3 = ax9_twin2.plot(relative_time, acc_x, 'b-', linewidth=1.5, alpha=0.7, label='Acceleration')
    
    ax9.set_xlabel('Time (s)', fontsize=10)
    ax9.set_ylabel('Position (m)', fontsize=10, color='r')
    ax9_twin1.set_ylabel('Velocity (m/s)', fontsize=10, color='g')
    ax9_twin2.set_ylabel('Acceleration (m/s²)', fontsize=10, color='b')
    ax9.set_title('X-Axis: Pos/Vel/Acc', fontsize=12, fontweight='bold')
    ax9.tick_params(axis='y', labelcolor='r')
    ax9_twin1.tick_params(axis='y', labelcolor='g')
    ax9_twin2.tick_params(axis='y', labelcolor='b')
    ax9.grid(True, alpha=0.3)
    
    # 10. Y-axis comparison
    ax10 = plt.subplot(3, 4, 10)
    ax10_twin1 = ax10.twinx()
    ax10_twin2 = ax10.twinx()
    ax10_twin2.spines['right'].set_position(('outward', 60))
    
    ax10.plot(relative_time, pos_y, 'r-', linewidth=1.5, alpha=0.7, label='Position')
    ax10_twin1.plot(relative_time, vel_y, 'g-', linewidth=1.5, alpha=0.7, label='Velocity')
    ax10_twin2.plot(relative_time, acc_y, 'b-', linewidth=1.5, alpha=0.7, label='Acceleration')
    
    ax10.set_xlabel('Time (s)', fontsize=10)
    ax10.set_ylabel('Position (m)', fontsize=10, color='r')
    ax10_twin1.set_ylabel('Velocity (m/s)', fontsize=10, color='g')
    ax10_twin2.set_ylabel('Acceleration (m/s²)', fontsize=10, color='b')
    ax10.set_title('Y-Axis: Pos/Vel/Acc', fontsize=12, fontweight='bold')
    ax10.tick_params(axis='y', labelcolor='r')
    ax10_twin1.tick_params(axis='y', labelcolor='g')
    ax10_twin2.tick_params(axis='y', labelcolor='b')
    ax10.grid(True, alpha=0.3)
    
    # 11. Z-axis comparison
    ax11 = plt.subplot(3, 4, 11)
    ax11_twin1 = ax11.twinx()
    ax11_twin2 = ax11.twinx()
    ax11_twin2.spines['right'].set_position(('outward', 60))
    
    ax11.plot(relative_time, pos_z, 'r-', linewidth=1.5, alpha=0.7, label='Position')
    ax11_twin1.plot(relative_time, vel_z, 'g-', linewidth=1.5, alpha=0.7, label='Velocity')
    ax11_twin2.plot(relative_time, acc_z, 'b-', linewidth=1.5, alpha=0.7, label='Acceleration')
    
    ax11.set_xlabel('Time (s)', fontsize=10)
    ax11.set_ylabel('Position (m)', fontsize=10, color='r')
    ax11_twin1.set_ylabel('Velocity (m/s)', fontsize=10, color='g')
    ax11_twin2.set_ylabel('Acceleration (m/s²)', fontsize=10, color='b')
    ax11.set_title('Z-Axis: Pos/Vel/Acc', fontsize=12, fontweight='bold')
    ax11.tick_params(axis='y', labelcolor='r')
    ax11_twin1.tick_params(axis='y', labelcolor='g')
    ax11_twin2.tick_params(axis='y', labelcolor='b')
    ax11.grid(True, alpha=0.3)
    
    # 12. Statistics table
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    stats_text = f"""
    TRAJECTORY STATISTICS
    Duration: {relative_time[-1]:.2f} s
    Timesteps: {len(relative_time)}
    Sample Rate: {len(relative_time)/relative_time[-1]:.1f} Hz
    
    POSITION RANGES:
    X: [{np.min(pos_x):.3f}, {np.max(pos_x):.3f}] m
    Y: [{np.min(pos_y):.3f}, {np.max(pos_y):.3f}] m
    Z: [{np.min(pos_z):.3f}, {np.max(pos_z):.3f}] m
    
    VELOCITY STATS (mean ± std):
    Vx: {np.mean(vel_x):.3f} ± {np.std(vel_x):.3f} m/s
    Vy: {np.mean(vel_y):.3f} ± {np.std(vel_y):.3f} m/s
    Vz: {np.mean(vel_z):.3f} ± {np.std(vel_z):.3f} m/s
    Speed: {np.mean(vel_mag):.3f} ± {np.std(vel_mag):.3f} m/s
    
    ACCELERATION STATS (mean ± std):
    Ax: {np.mean(acc_x):.2f} ± {np.std(acc_x):.2f} m/s²
    Ay: {np.mean(acc_y):.2f} ± {np.std(acc_y):.2f} m/s²
    Az: {np.mean(acc_z):.2f} ± {np.std(acc_z):.2f} m/s²
    """
    
    ax12.text(0.1, 0.95, stats_text, transform=ax12.transAxes,
              fontsize=9, verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Robot End-Effector Trajectory Analysis - Complete Summary', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = output_dir / 'SUMMARY_complete_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved summary visualization to: {output_path}")
    
    plt.close()

def main():
    data_path = Path('/nfs/data/airhockey/trajectory_data434.hdf5')
    output_dir = Path('/home/air-hockey/daliu/air-hockey-rl/scripts/trajectory_visualization/acceleration_analysis')
    
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Generating comprehensive summary report...")
    print(f"Data source: {data_path}")
    
    # Load data
    train_vals = load_trajectory_data(data_path)
    
    # Extract all data
    pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z, relative_time = \
        extract_all_data(train_vals)
    
    # Create summary plot
    create_summary_plot(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, 
                       acc_x, acc_y, acc_z, relative_time, output_dir)
    
    print("\n" + "="*80)
    print("COMPLETE TRAJECTORY ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nFile: {data_path.name}")
    print(f"Duration: {relative_time[-1]:.2f} seconds")
    print(f"Timesteps: {len(relative_time)}")
    print(f"Sampling rate: {len(relative_time)/relative_time[-1]:.2f} Hz")
    
    # Compute path length
    dx = np.diff(pos_x)
    dy = np.diff(pos_y)
    dz = np.diff(pos_z)
    segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
    total_path_length = np.sum(segment_lengths)
    print(f"Total path length: {total_path_length:.3f} meters")
    
    # Speed stats
    vel_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    print(f"\nSpeed: mean={np.mean(vel_mag):.3f} m/s, max={np.max(vel_mag):.3f} m/s")
    
    # Acceleration stats
    acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    print(f"Acceleration magnitude: mean={np.mean(acc_mag):.2f} m/s², max={np.max(acc_mag):.2f} m/s²")
    
    print("\n✓ Summary report complete!")
    print("="*80)

if __name__ == '__main__':
    main()


