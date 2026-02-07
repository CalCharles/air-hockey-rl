#!/usr/bin/env python3
"""
Analyze all trajectory data files and generate comprehensive statistics.

This script processes all trajectory HDF5 files and computes:
- Velocity magnitude statistics
- Acceleration magnitude statistics
- Force magnitude statistics
- Paddle-to-target distance statistics

It logs the maximum values of each metric and creates visualizations
showing the relative frequencies of all magnitudes across all trajectories.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def find_trajectory_files(data_dir):
    """Find all trajectory HDF5 files in the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find all trajectory_data*.hdf5 files
    trajectory_files = sorted(data_path.glob('trajectory_data*.hdf5'))
    
    if not trajectory_files:
        raise FileNotFoundError(f"No trajectory files found in {data_dir}")
    
    return trajectory_files


def load_trajectory_data(filepath):
    """Load trajectory data from HDF5 file."""
    with h5py.File(filepath, 'r') as f:
        train_vals = f['train_vals'][:]
    return train_vals


def extract_metrics(train_vals):
    """
    Extract velocity, acceleration, force, and distance metrics from trajectory data.
    
    According to FIELD_DOCUMENTATION:
    - Fields 5-10: pose (xyz rxryrz)
    - Fields 11-16: speed (xyz rxryrz)
    - Fields 17-22: force (xyz rxryrz)
    - Fields 23-25: acc (xyz)
    - Fields 26-31: desired_pose (xyz rxryrz)
    
    Returns:
        dict: Dictionary containing all metrics arrays
    """
    # Extract position (current and desired)
    pos_x, pos_y, pos_z = train_vals[:, 5], train_vals[:, 6], train_vals[:, 7]
    target_x, target_y, target_z = train_vals[:, 26], train_vals[:, 27], train_vals[:, 28]
    
    # Extract velocity (linear and angular)
    vel_x, vel_y, vel_z = train_vals[:, 11], train_vals[:, 12], train_vals[:, 13]
    vel_rx, vel_ry, vel_rz = train_vals[:, 14], train_vals[:, 15], train_vals[:, 16]
    
    # Extract force (linear and torque)
    force_x, force_y, force_z = train_vals[:, 17], train_vals[:, 18], train_vals[:, 19]
    torque_x, torque_y, torque_z = train_vals[:, 20], train_vals[:, 21], train_vals[:, 22]
    
    # Extract acceleration (linear only)
    acc_x, acc_y, acc_z = train_vals[:, 23], train_vals[:, 24], train_vals[:, 25]
    
    # Compute magnitudes
    velocity_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    angular_velocity_magnitude = np.sqrt(vel_rx**2 + vel_ry**2 + vel_rz**2)
    
    acceleration_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    force_magnitude = np.sqrt(force_x**2 + force_y**2 + force_z**2)
    torque_magnitude = np.sqrt(torque_x**2 + torque_y**2 + torque_z**2)
    
    # Compute distance to target
    distance_to_target = np.sqrt(
        (target_x - pos_x)**2 + 
        (target_y - pos_y)**2 + 
        (target_z - pos_z)**2
    )
    
    return {
        'velocity_magnitude': velocity_magnitude,
        'angular_velocity_magnitude': angular_velocity_magnitude,
        'acceleration_magnitude': acceleration_magnitude,
        'force_magnitude': force_magnitude,
        'torque_magnitude': torque_magnitude,
        'distance_to_target': distance_to_target,
    }


def aggregate_statistics(trajectory_files):
    """
    Process all trajectory files and aggregate statistics.
    
    Args:
        trajectory_files: List of paths to trajectory HDF5 files
        
    Returns:
        dict: Aggregated statistics and all data points
    """
    # Storage for all data points
    all_velocity_magnitudes = []
    all_angular_velocity_magnitudes = []
    all_acceleration_magnitudes = []
    all_force_magnitudes = []
    all_torque_magnitudes = []
    all_distances_to_target = []
    
    # Statistics per trajectory
    trajectory_stats = []
    
    print(f"\nProcessing {len(trajectory_files)} trajectory files...")
    
    for traj_file in tqdm(trajectory_files, desc="Analyzing trajectories"):
        try:
            # Load trajectory data
            train_vals = load_trajectory_data(traj_file)
            
            # Extract metrics
            metrics = extract_metrics(train_vals)
            
            # Store all data points
            all_velocity_magnitudes.extend(metrics['velocity_magnitude'])
            all_angular_velocity_magnitudes.extend(metrics['angular_velocity_magnitude'])
            all_acceleration_magnitudes.extend(metrics['acceleration_magnitude'])
            all_force_magnitudes.extend(metrics['force_magnitude'])
            all_torque_magnitudes.extend(metrics['torque_magnitude'])
            all_distances_to_target.extend(metrics['distance_to_target'])
            
            # Compute statistics for this trajectory
            traj_stat = {
                'filename': traj_file.name,
                'num_frames': len(train_vals),
                'velocity_max': float(np.max(metrics['velocity_magnitude'])),
                'velocity_mean': float(np.mean(metrics['velocity_magnitude'])),
                'angular_velocity_max': float(np.max(metrics['angular_velocity_magnitude'])),
                'angular_velocity_mean': float(np.mean(metrics['angular_velocity_magnitude'])),
                'acceleration_max': float(np.max(metrics['acceleration_magnitude'])),
                'acceleration_mean': float(np.mean(metrics['acceleration_magnitude'])),
                'force_max': float(np.max(metrics['force_magnitude'])),
                'force_mean': float(np.mean(metrics['force_magnitude'])),
                'torque_max': float(np.max(metrics['torque_magnitude'])),
                'torque_mean': float(np.mean(metrics['torque_magnitude'])),
                'distance_max': float(np.max(metrics['distance_to_target'])),
                'distance_mean': float(np.mean(metrics['distance_to_target'])),
            }
            trajectory_stats.append(traj_stat)
            
        except Exception as e:
            print(f"\nWarning: Failed to process {traj_file.name}: {e}")
            continue
    
    # Convert to numpy arrays for easier computation
    all_velocity_magnitudes = np.array(all_velocity_magnitudes)
    all_angular_velocity_magnitudes = np.array(all_angular_velocity_magnitudes)
    all_acceleration_magnitudes = np.array(all_acceleration_magnitudes)
    all_force_magnitudes = np.array(all_force_magnitudes)
    all_torque_magnitudes = np.array(all_torque_magnitudes)
    all_distances_to_target = np.array(all_distances_to_target)
    
    # Compute overall statistics
    overall_stats = {
        'num_trajectories': len(trajectory_stats),
        'total_frames': len(all_velocity_magnitudes),
        
        'velocity_magnitude': {
            'max': float(np.max(all_velocity_magnitudes)),
            'min': float(np.min(all_velocity_magnitudes)),
            'mean': float(np.mean(all_velocity_magnitudes)),
            'median': float(np.median(all_velocity_magnitudes)),
            'std': float(np.std(all_velocity_magnitudes)),
            'p95': float(np.percentile(all_velocity_magnitudes, 95)),
            'p99': float(np.percentile(all_velocity_magnitudes, 99)),
        },
        
        'angular_velocity_magnitude': {
            'max': float(np.max(all_angular_velocity_magnitudes)),
            'min': float(np.min(all_angular_velocity_magnitudes)),
            'mean': float(np.mean(all_angular_velocity_magnitudes)),
            'median': float(np.median(all_angular_velocity_magnitudes)),
            'std': float(np.std(all_angular_velocity_magnitudes)),
            'p95': float(np.percentile(all_angular_velocity_magnitudes, 95)),
            'p99': float(np.percentile(all_angular_velocity_magnitudes, 99)),
        },
        
        'acceleration_magnitude': {
            'max': float(np.max(all_acceleration_magnitudes)),
            'min': float(np.min(all_acceleration_magnitudes)),
            'mean': float(np.mean(all_acceleration_magnitudes)),
            'median': float(np.median(all_acceleration_magnitudes)),
            'std': float(np.std(all_acceleration_magnitudes)),
            'p95': float(np.percentile(all_acceleration_magnitudes, 95)),
            'p99': float(np.percentile(all_acceleration_magnitudes, 99)),
        },
        
        'force_magnitude': {
            'max': float(np.max(all_force_magnitudes)),
            'min': float(np.min(all_force_magnitudes)),
            'mean': float(np.mean(all_force_magnitudes)),
            'median': float(np.median(all_force_magnitudes)),
            'std': float(np.std(all_force_magnitudes)),
            'p95': float(np.percentile(all_force_magnitudes, 95)),
            'p99': float(np.percentile(all_force_magnitudes, 99)),
        },
        
        'torque_magnitude': {
            'max': float(np.max(all_torque_magnitudes)),
            'min': float(np.min(all_torque_magnitudes)),
            'mean': float(np.mean(all_torque_magnitudes)),
            'median': float(np.median(all_torque_magnitudes)),
            'std': float(np.std(all_torque_magnitudes)),
            'p95': float(np.percentile(all_torque_magnitudes, 95)),
            'p99': float(np.percentile(all_torque_magnitudes, 99)),
        },
        
        'distance_to_target': {
            'max': float(np.max(all_distances_to_target)),
            'min': float(np.min(all_distances_to_target)),
            'mean': float(np.mean(all_distances_to_target)),
            'median': float(np.median(all_distances_to_target)),
            'std': float(np.std(all_distances_to_target)),
            'p95': float(np.percentile(all_distances_to_target, 95)),
            'p99': float(np.percentile(all_distances_to_target, 99)),
        },
    }
    
    # Return all data
    return {
        'overall_stats': overall_stats,
        'trajectory_stats': trajectory_stats,
        'all_data': {
            'velocity_magnitude': all_velocity_magnitudes,
            'angular_velocity_magnitude': all_angular_velocity_magnitudes,
            'acceleration_magnitude': all_acceleration_magnitudes,
            'force_magnitude': all_force_magnitudes,
            'torque_magnitude': all_torque_magnitudes,
            'distance_to_target': all_distances_to_target,
        }
    }


def print_statistics(stats):
    """Print comprehensive statistics in a readable format."""
    overall = stats['overall_stats']
    
    print("\n" + "="*100)
    print("COMPREHENSIVE TRAJECTORY ANALYSIS - ALL TRAJECTORIES")
    print("="*100)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  Number of trajectories analyzed: {overall['num_trajectories']}")
    print(f"  Total number of frames:          {overall['total_frames']:,}")
    
    print("\n" + "-"*100)
    print("LINEAR VELOCITY MAGNITUDE (m/s)")
    print("-"*100)
    v = overall['velocity_magnitude']
    print(f"  Maximum:     {v['max']:>12.6f} m/s  ⚡")
    print(f"  99th %ile:   {v['p99']:>12.6f} m/s")
    print(f"  95th %ile:   {v['p95']:>12.6f} m/s")
    print(f"  Mean:        {v['mean']:>12.6f} m/s")
    print(f"  Median:      {v['median']:>12.6f} m/s")
    print(f"  Std Dev:     {v['std']:>12.6f} m/s")
    print(f"  Minimum:     {v['min']:>12.6f} m/s")
    
    print("\n" + "-"*100)
    print("ANGULAR VELOCITY MAGNITUDE (rad/s)")
    print("-"*100)
    av = overall['angular_velocity_magnitude']
    print(f"  Maximum:     {av['max']:>12.6f} rad/s  ⚡")
    print(f"  99th %ile:   {av['p99']:>12.6f} rad/s")
    print(f"  95th %ile:   {av['p95']:>12.6f} rad/s")
    print(f"  Mean:        {av['mean']:>12.6f} rad/s")
    print(f"  Median:      {av['median']:>12.6f} rad/s")
    print(f"  Std Dev:     {av['std']:>12.6f} rad/s")
    print(f"  Minimum:     {av['min']:>12.6f} rad/s")
    
    print("\n" + "-"*100)
    print("ACCELERATION MAGNITUDE (m/s²)")
    print("-"*100)
    a = overall['acceleration_magnitude']
    print(f"  Maximum:     {a['max']:>12.6f} m/s²  ⚡")
    print(f"  99th %ile:   {a['p99']:>12.6f} m/s²")
    print(f"  95th %ile:   {a['p95']:>12.6f} m/s²")
    print(f"  Mean:        {a['mean']:>12.6f} m/s²")
    print(f"  Median:      {a['median']:>12.6f} m/s²")
    print(f"  Std Dev:     {a['std']:>12.6f} m/s²")
    print(f"  Minimum:     {a['min']:>12.6f} m/s²")
    
    print("\n" + "-"*100)
    print("FORCE MAGNITUDE (N)")
    print("-"*100)
    f = overall['force_magnitude']
    print(f"  Maximum:     {f['max']:>12.6f} N  ⚡")
    print(f"  99th %ile:   {f['p99']:>12.6f} N")
    print(f"  95th %ile:   {f['p95']:>12.6f} N")
    print(f"  Mean:        {f['mean']:>12.6f} N")
    print(f"  Median:      {f['median']:>12.6f} N")
    print(f"  Std Dev:     {f['std']:>12.6f} N")
    print(f"  Minimum:     {f['min']:>12.6f} N")
    
    print("\n" + "-"*100)
    print("TORQUE MAGNITUDE (N⋅m)")
    print("-"*100)
    t = overall['torque_magnitude']
    print(f"  Maximum:     {t['max']:>12.6f} N⋅m  ⚡")
    print(f"  99th %ile:   {t['p99']:>12.6f} N⋅m")
    print(f"  95th %ile:   {t['p95']:>12.6f} N⋅m")
    print(f"  Mean:        {t['mean']:>12.6f} N⋅m")
    print(f"  Median:      {t['median']:>12.6f} N⋅m")
    print(f"  Std Dev:     {t['std']:>12.6f} N⋅m")
    print(f"  Minimum:     {t['min']:>12.6f} N⋅m")
    
    print("\n" + "-"*100)
    print("PADDLE-TO-TARGET DISTANCE (m)")
    print("-"*100)
    d = overall['distance_to_target']
    print(f"  Maximum:     {d['max']:>12.6f} m  ⚡")
    print(f"  99th %ile:   {d['p99']:>12.6f} m")
    print(f"  95th %ile:   {d['p95']:>12.6f} m")
    print(f"  Mean:        {d['mean']:>12.6f} m")
    print(f"  Median:      {d['median']:>12.6f} m")
    print(f"  Std Dev:     {d['std']:>12.6f} m")
    print(f"  Minimum:     {d['min']:>12.6f} m")
    
    print("\n" + "="*100)


def create_frequency_plots(all_data, output_dir):
    """
    Create comprehensive frequency distribution plots for all metrics.
    
    Args:
        all_data: Dictionary containing all data arrays
        output_dir: Path to save output plots
    """
    output_path = output_dir / 'magnitude_frequency_distributions.png'
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Relative Frequency Distributions of All Magnitudes\n(Across All Trajectories)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Linear Velocity Magnitude
    ax = axes[0, 0]
    velocity = all_data['velocity_magnitude']
    # Remove zeros for better visualization (log scale)
    velocity_nonzero = velocity[velocity > 1e-6]
    ax.hist(velocity_nonzero, bins=100, alpha=0.7, color='steelblue', edgecolor='black', density=True)
    ax.set_xlabel('Linear Velocity Magnitude (m/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Linear Velocity Distribution\nMax: {np.max(velocity):.4f} m/s', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(velocity), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(velocity):.4f}')
    ax.axvline(np.median(velocity), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(velocity):.4f}')
    ax.legend()
    
    # 2. Angular Velocity Magnitude
    ax = axes[0, 1]
    angular_velocity = all_data['angular_velocity_magnitude']
    angular_velocity_nonzero = angular_velocity[angular_velocity > 1e-6]
    ax.hist(angular_velocity_nonzero, bins=100, alpha=0.7, color='orange', edgecolor='black', density=True)
    ax.set_xlabel('Angular Velocity Magnitude (rad/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Angular Velocity Distribution\nMax: {np.max(angular_velocity):.4f} rad/s', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(angular_velocity), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(angular_velocity):.4f}')
    ax.axvline(np.median(angular_velocity), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(angular_velocity):.4f}')
    ax.legend()
    
    # 3. Acceleration Magnitude
    ax = axes[1, 0]
    acceleration = all_data['acceleration_magnitude']
    acceleration_nonzero = acceleration[acceleration > 1e-6]
    ax.hist(acceleration_nonzero, bins=100, alpha=0.7, color='crimson', edgecolor='black', density=True)
    ax.set_xlabel('Acceleration Magnitude (m/s²)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Acceleration Distribution\nMax: {np.max(acceleration):.4f} m/s²', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(acceleration), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(acceleration):.4f}')
    ax.axvline(np.median(acceleration), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(acceleration):.4f}')
    ax.legend()
    
    # 4. Force Magnitude
    ax = axes[1, 1]
    force = all_data['force_magnitude']
    force_nonzero = force[force > 1e-6]
    ax.hist(force_nonzero, bins=100, alpha=0.7, color='purple', edgecolor='black', density=True)
    ax.set_xlabel('Force Magnitude (N)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Force Distribution\nMax: {np.max(force):.4f} N', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(force), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(force):.4f}')
    ax.axvline(np.median(force), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(force):.4f}')
    ax.legend()
    
    # 5. Torque Magnitude
    ax = axes[2, 0]
    torque = all_data['torque_magnitude']
    torque_nonzero = torque[torque > 1e-6]
    ax.hist(torque_nonzero, bins=100, alpha=0.7, color='teal', edgecolor='black', density=True)
    ax.set_xlabel('Torque Magnitude (N⋅m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Torque Distribution\nMax: {np.max(torque):.4f} N⋅m', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(torque), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(torque):.4f}')
    ax.axvline(np.median(torque), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(torque):.4f}')
    ax.legend()
    
    # 6. Distance to Target
    ax = axes[2, 1]
    distance = all_data['distance_to_target']
    distance_nonzero = distance[distance > 1e-6]
    ax.hist(distance_nonzero, bins=100, alpha=0.7, color='forestgreen', edgecolor='black', density=True)
    ax.set_xlabel('Paddle-to-Target Distance (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Distance-to-Target Distribution\nMax: {np.max(distance):.4f} m', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(distance), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distance):.4f}')
    ax.axvline(np.median(distance), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(distance):.4f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved frequency distribution plot to: {output_path}")
    plt.close()


def create_log_scale_plots(all_data, output_dir):
    """
    Create log-scale plots to better visualize the full range of magnitudes.
    
    Args:
        all_data: Dictionary containing all data arrays
        output_dir: Path to save output plots
    """
    output_path = output_dir / 'magnitude_frequency_distributions_log_scale.png'
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Relative Frequency Distributions (Log Scale)\n(Across All Trajectories)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Linear Velocity Magnitude (log scale)
    ax = axes[0, 0]
    velocity = all_data['velocity_magnitude']
    velocity_nonzero = velocity[velocity > 1e-6]
    ax.hist(velocity_nonzero, bins=np.logspace(np.log10(velocity_nonzero.min()), 
                                                np.log10(velocity_nonzero.max()), 100),
            alpha=0.7, color='steelblue', edgecolor='black', density=True)
    ax.set_xlabel('Linear Velocity Magnitude (m/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Linear Velocity (Log Scale)\nMax: {np.max(velocity):.4f} m/s', fontsize=12)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, which='both')
    
    # 2. Angular Velocity Magnitude (log scale)
    ax = axes[0, 1]
    angular_velocity = all_data['angular_velocity_magnitude']
    angular_velocity_nonzero = angular_velocity[angular_velocity > 1e-6]
    if len(angular_velocity_nonzero) > 0:
        ax.hist(angular_velocity_nonzero, bins=np.logspace(np.log10(angular_velocity_nonzero.min()), 
                                                            np.log10(angular_velocity_nonzero.max()), 100),
                alpha=0.7, color='orange', edgecolor='black', density=True)
        ax.set_xscale('log')
    ax.set_xlabel('Angular Velocity Magnitude (rad/s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Angular Velocity (Log Scale)\nMax: {np.max(angular_velocity):.4f} rad/s', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    # 3. Acceleration Magnitude (log scale)
    ax = axes[1, 0]
    acceleration = all_data['acceleration_magnitude']
    acceleration_nonzero = acceleration[acceleration > 1e-6]
    if len(acceleration_nonzero) > 0:
        ax.hist(acceleration_nonzero, bins=np.logspace(np.log10(acceleration_nonzero.min()), 
                                                        np.log10(acceleration_nonzero.max()), 100),
                alpha=0.7, color='crimson', edgecolor='black', density=True)
        ax.set_xscale('log')
    ax.set_xlabel('Acceleration Magnitude (m/s²)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Acceleration (Log Scale)\nMax: {np.max(acceleration):.4f} m/s²', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    # 4. Force Magnitude (log scale)
    ax = axes[1, 1]
    force = all_data['force_magnitude']
    force_nonzero = force[force > 1e-6]
    if len(force_nonzero) > 0:
        ax.hist(force_nonzero, bins=np.logspace(np.log10(force_nonzero.min()), 
                                                 np.log10(force_nonzero.max()), 100),
                alpha=0.7, color='purple', edgecolor='black', density=True)
        ax.set_xscale('log')
    ax.set_xlabel('Force Magnitude (N)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Force (Log Scale)\nMax: {np.max(force):.4f} N', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    # 5. Torque Magnitude (log scale)
    ax = axes[2, 0]
    torque = all_data['torque_magnitude']
    torque_nonzero = torque[torque > 1e-6]
    if len(torque_nonzero) > 0:
        ax.hist(torque_nonzero, bins=np.logspace(np.log10(torque_nonzero.min()), 
                                                  np.log10(torque_nonzero.max()), 100),
                alpha=0.7, color='teal', edgecolor='black', density=True)
        ax.set_xscale('log')
    ax.set_xlabel('Torque Magnitude (N⋅m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Torque (Log Scale)\nMax: {np.max(torque):.4f} N⋅m', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    # 6. Distance to Target (log scale)
    ax = axes[2, 1]
    distance = all_data['distance_to_target']
    distance_nonzero = distance[distance > 1e-6]
    if len(distance_nonzero) > 0:
        ax.hist(distance_nonzero, bins=np.logspace(np.log10(distance_nonzero.min()), 
                                                    np.log10(distance_nonzero.max()), 100),
                alpha=0.7, color='forestgreen', edgecolor='black', density=True)
        ax.set_xscale('log')
    ax.set_xlabel('Paddle-to-Target Distance (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Relative Frequency (Probability Density)', fontsize=11, fontweight='bold')
    ax.set_title(f'Distance-to-Target (Log Scale)\nMax: {np.max(distance):.4f} m', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved log-scale frequency distribution plot to: {output_path}")
    plt.close()


def save_statistics_json(stats, output_dir):
    """Save statistics to JSON file for later use."""
    output_path = output_dir / 'statistics.json'
    
    # Save only the overall stats and trajectory-level stats (not raw data)
    save_data = {
        'overall_stats': stats['overall_stats'],
        'trajectory_stats': stats['trajectory_stats'],
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"✓ Saved statistics JSON to: {output_path}")


def save_statistics_summary(stats, output_dir):
    """Save a human-readable summary text file."""
    output_path = output_dir / 'statistics_summary.txt'
    
    overall = stats['overall_stats']
    
    with open(output_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE TRAJECTORY ANALYSIS - SUMMARY REPORT\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Dataset Overview:\n")
        f.write(f"  Number of trajectories: {overall['num_trajectories']}\n")
        f.write(f"  Total frames:           {overall['total_frames']:,}\n\n")
        
        f.write("-"*100 + "\n")
        f.write("MAXIMUM VALUES (Critical for Bounds)\n")
        f.write("-"*100 + "\n")
        f.write(f"Linear Velocity:         {overall['velocity_magnitude']['max']:.6f} m/s\n")
        f.write(f"Angular Velocity:        {overall['angular_velocity_magnitude']['max']:.6f} rad/s\n")
        f.write(f"Acceleration:            {overall['acceleration_magnitude']['max']:.6f} m/s²\n")
        f.write(f"Force:                   {overall['force_magnitude']['max']:.6f} N\n")
        f.write(f"Torque:                  {overall['torque_magnitude']['max']:.6f} N⋅m\n")
        f.write(f"Distance to Target:      {overall['distance_to_target']['max']:.6f} m\n\n")
        
        f.write("-"*100 + "\n")
        f.write("DETAILED STATISTICS\n")
        f.write("-"*100 + "\n\n")
        
        for metric_name, metric_label, unit in [
            ('velocity_magnitude', 'Linear Velocity', 'm/s'),
            ('angular_velocity_magnitude', 'Angular Velocity', 'rad/s'),
            ('acceleration_magnitude', 'Acceleration', 'm/s²'),
            ('force_magnitude', 'Force', 'N'),
            ('torque_magnitude', 'Torque', 'N⋅m'),
            ('distance_to_target', 'Distance to Target', 'm'),
        ]:
            m = overall[metric_name]
            f.write(f"{metric_label} ({unit}):\n")
            f.write(f"  Maximum:     {m['max']:>12.6f}\n")
            f.write(f"  99th %ile:   {m['p99']:>12.6f}\n")
            f.write(f"  95th %ile:   {m['p95']:>12.6f}\n")
            f.write(f"  Mean:        {m['mean']:>12.6f}\n")
            f.write(f"  Median:      {m['median']:>12.6f}\n")
            f.write(f"  Std Dev:     {m['std']:>12.6f}\n")
            f.write(f"  Minimum:     {m['min']:>12.6f}\n\n")
    
    print(f"✓ Saved statistics summary to: {output_path}")


def main():
    """Main function to analyze all trajectory data."""
    # Configuration
    data_dir = '/nfs/data/airhockey'
    output_dir = Path('/home/air-hockey/daliu/air-hockey-rl/scripts/trajectory_visualization/bounds')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*100)
    print("TRAJECTORY DATA ANALYSIS - ALL FILES")
    print("="*100)
    
    # Find all trajectory files
    print(f"\nSearching for trajectory files in: {data_dir}")
    trajectory_files = find_trajectory_files(data_dir)
    print(f"Found {len(trajectory_files)} trajectory files")
    
    # Aggregate statistics
    stats = aggregate_statistics(trajectory_files)
    
    # Print statistics to console
    print_statistics(stats)
    
    # Save statistics
    print("\nSaving results...")
    save_statistics_json(stats, output_dir)
    save_statistics_summary(stats, output_dir)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_frequency_plots(stats['all_data'], output_dir)
    create_log_scale_plots(stats['all_data'], output_dir)
    
    print("\n" + "="*100)
    print("✓ ANALYSIS COMPLETE!")
    print("="*100)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - statistics.json (machine-readable)")
    print(f"  - statistics_summary.txt (human-readable)")
    print(f"  - magnitude_frequency_distributions.png (linear scale)")
    print(f"  - magnitude_frequency_distributions_log_scale.png (log scale)")
    print()


if __name__ == '__main__':
    main()


