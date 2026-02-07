#!/usr/bin/env python3
"""
Analyze all trajectory data files to compute detailed statistics on:
- Velocity magnitude
- Acceleration magnitude  
- Force magnitude
- Paddle-target distance

Outputs maximum values and creates frequency distribution plots.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm


def load_trajectory_data(filepath):
    """
    Load trajectory data from HDF5 file.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        numpy.ndarray: Train values array
    """
    with h5py.File(filepath, 'r') as f:
        train_vals = f['train_vals'][:]
    return train_vals


def extract_data(train_vals):
    """
    Extract relevant data from trajectory.
    
    According to field documentation:
    - Field 0: cur_time (Unix timestamp)
    - Fields 5-7: pose (x, y, z)
    - Fields 17-19: force (x, y, z)
    - Fields 26-28: desired_pose (x, y, z)
    
    Args:
        train_vals: Array of trajectory data
        
    Returns:
        dict: Dictionary with extracted data
    """
    data = {
        'pos_x': train_vals[:, 5],
        'pos_y': train_vals[:, 6],
        'pos_z': train_vals[:, 7],
        'target_x': train_vals[:, 26],
        'target_y': train_vals[:, 27],
        'target_z': train_vals[:, 28],
        'force_x': train_vals[:, 17],
        'force_y': train_vals[:, 18],
        'force_z': train_vals[:, 19],
        'timestamps': train_vals[:, 0]
    }
    return data


def compute_metrics(data):
    """
    Compute velocity, acceleration, force, and distance metrics.
    
    Args:
        data: Dictionary with extracted trajectory data
        
    Returns:
        dict: Dictionary with computed metrics
    """
    # Compute force magnitude
    force_mag = np.sqrt(
        data['force_x']**2 + 
        data['force_y']**2 + 
        data['force_z']**2
    )
    
    # Compute paddle-target distance
    paddle_target_dist = np.sqrt(
        (data['pos_x'] - data['target_x'])**2 +
        (data['pos_y'] - data['target_y'])**2 +
        (data['pos_z'] - data['target_z'])**2
    )
    
    # Compute velocity from position differences
    dt = np.diff(data['timestamps'])
    dt = np.maximum(dt, 1e-6)  # Avoid division by zero
    
    dx = np.diff(data['pos_x'])
    dy = np.diff(data['pos_y'])
    dz = np.diff(data['pos_z'])
    
    vel_x = dx / dt
    vel_y = dy / dt
    vel_z = dz / dt
    
    velocity_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    
    # Compute acceleration from velocity differences
    dvel_x = np.diff(vel_x)
    dvel_y = np.diff(vel_y)
    dvel_z = np.diff(vel_z)
    dt_acc = dt[:-1]  # One less element due to second diff
    dt_acc = np.maximum(dt_acc, 1e-6)
    
    acc_x = dvel_x / dt_acc
    acc_y = dvel_y / dt_acc
    acc_z = dvel_z / dt_acc
    
    acceleration_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    metrics = {
        'velocity_magnitude': velocity_mag,
        'acceleration_magnitude': acceleration_mag,
        'force_magnitude': force_mag,
        'paddle_target_distance': paddle_target_dist
    }
    
    return metrics


def analyze_all_trajectories(data_dir):
    """
    Analyze all trajectory files in the data directory.
    
    Args:
        data_dir: Path to directory containing trajectory files
        
    Returns:
        dict: Aggregated metrics across all trajectories
    """
    data_dir = Path(data_dir)
    trajectory_files = sorted(data_dir.glob('trajectory_data*.hdf5'))
    
    if not trajectory_files:
        print(f"No trajectory files found in {data_dir}")
        sys.exit(1)
    
    print(f"Found {len(trajectory_files)} trajectory files")
    print("Processing trajectories...")
    
    # Aggregate all metrics
    all_velocity = []
    all_acceleration = []
    all_force = []
    all_distance = []
    
    # Track maximums per file
    file_stats = []
    
    for traj_file in tqdm(trajectory_files, desc="Analyzing trajectories"):
        try:
            # Load and process trajectory
            train_vals = load_trajectory_data(traj_file)
            data = extract_data(train_vals)
            metrics = compute_metrics(data)
            
            # Append to aggregated lists
            all_velocity.extend(metrics['velocity_magnitude'].tolist())
            all_acceleration.extend(metrics['acceleration_magnitude'].tolist())
            all_force.extend(metrics['force_magnitude'].tolist())
            all_distance.extend(metrics['paddle_target_distance'].tolist())
            
            # Track per-file maximums
            file_stats.append({
                'filename': traj_file.name,
                'max_velocity': np.max(metrics['velocity_magnitude']),
                'max_acceleration': np.max(metrics['acceleration_magnitude']),
                'max_force': np.max(metrics['force_magnitude']),
                'max_distance': np.max(metrics['paddle_target_distance'])
            })
            
        except Exception as e:
            print(f"\nError processing {traj_file.name}: {e}")
            continue
    
    # Convert to numpy arrays
    aggregated_metrics = {
        'velocity_magnitude': np.array(all_velocity),
        'acceleration_magnitude': np.array(all_acceleration),
        'force_magnitude': np.array(all_force),
        'paddle_target_distance': np.array(all_distance)
    }
    
    return aggregated_metrics, file_stats


def print_statistics(metrics, file_stats):
    """
    Print detailed statistics about the metrics.
    
    Args:
        metrics: Dictionary with aggregated metrics
        file_stats: List of per-file statistics
    """
    print("\n" + "=" * 80)
    print("TRAJECTORY BOUNDS ANALYSIS")
    print("=" * 80)
    
    print("\nAGGREGATED STATISTICS ACROSS ALL TRAJECTORIES:")
    print("-" * 80)
    
    for metric_name, values in metrics.items():
        print(f"\n{metric_name.upper().replace('_', ' ')}:")
        print(f"  Count:       {len(values):,} samples")
        print(f"  Maximum:     {np.max(values):.6f}")
        print(f"  Mean:        {np.mean(values):.6f}")
        print(f"  Median:      {np.median(values):.6f}")
        print(f"  Std Dev:     {np.std(values):.6f}")
        print(f"  95th %-ile:  {np.percentile(values, 95):.6f}")
        print(f"  99th %-ile:  {np.percentile(values, 99):.6f}")
        print(f"  99.9th %-ile: {np.percentile(values, 99.9):.6f}")
    
    print("\n" + "=" * 80)
    print("MAXIMUM VALUES PER TRAJECTORY FILE:")
    print("-" * 80)
    
    # Sort by different metrics and show top 5 for each
    metrics_to_check = ['max_velocity', 'max_acceleration', 'max_force', 'max_distance']
    
    for metric in metrics_to_check:
        print(f"\nTop 5 files by {metric.replace('max_', '').replace('_', ' ').upper()}:")
        sorted_stats = sorted(file_stats, key=lambda x: x[metric], reverse=True)
        for i, stat in enumerate(sorted_stats[:5], 1):
            print(f"  {i}. {stat['filename']}: {stat[metric]:.6f}")
    
    # Overall maximum across all files
    print("\n" + "=" * 80)
    print("GLOBAL MAXIMUM VALUES:")
    print("-" * 80)
    for metric in metrics_to_check:
        max_stat = max(file_stats, key=lambda x: x[metric])
        print(f"{metric.replace('max_', '').replace('_', ' ').title():30s}: {max_stat[metric]:.6f} (from {max_stat['filename']})")


def create_frequency_plots(metrics, output_dir):
    """
    Create frequency distribution plots for all metrics.
    
    Args:
        metrics: Dictionary with aggregated metrics
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CREATING FREQUENCY PLOTS...")
    print("-" * 80)
    
    # Create a 2x2 subplot for all metrics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metric_names = [
        'velocity_magnitude',
        'acceleration_magnitude', 
        'force_magnitude',
        'paddle_target_distance'
    ]
    
    titles = [
        'Velocity Magnitude Distribution',
        'Acceleration Magnitude Distribution',
        'Force Magnitude Distribution',
        'Paddle-Target Distance Distribution'
    ]
    
    xlabels = [
        'Velocity (m/s)',
        'Acceleration (m/s²)',
        'Force (N)',
        'Distance (m)'
    ]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
    
    for idx, (metric_name, title, xlabel, color) in enumerate(zip(metric_names, titles, xlabels, colors)):
        ax = axes[idx]
        values = metrics[metric_name]
        
        # Remove extreme outliers for better visualization (keep 99.9th percentile)
        max_val = np.percentile(values, 99.9)
        filtered_values = values[values <= max_val]
        
        # Create histogram
        n, bins, patches = ax.hist(filtered_values, bins=100, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add vertical lines for statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        p95_val = np.percentile(values, 95)
        p99_val = np.percentile(values, 99)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
        ax.axvline(p95_val, color='orange', linestyle='--', linewidth=2, label=f'95th: {p95_val:.4f}')
        ax.axvline(p99_val, color='purple', linestyle='--', linewidth=2, label=f'99th: {p99_val:.4f}')
        
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Log scale for y-axis if data is very skewed
        if np.max(n) / np.median(n[n > 0]) > 100:
            ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save combined plot
    combined_path = output_dir / 'magnitude_distributions.png'
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved combined plot: {combined_path}")
    
    # Create individual plots with more detail
    for metric_name, title, xlabel, color in zip(metric_names, titles, xlabels, colors):
        fig, ax = plt.subplots(figsize=(12, 8))
        values = metrics[metric_name]
        
        # Full range histogram
        max_val = np.percentile(values, 99.9)
        filtered_values = values[values <= max_val]
        
        n, bins, patches = ax.hist(filtered_values, bins=150, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Statistics
        mean_val = np.mean(values)
        median_val = np.median(values)
        p95_val = np.percentile(values, 95)
        p99_val = np.percentile(values, 99)
        p999_val = np.percentile(values, 99.9)
        max_overall = np.max(values)
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.6f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.6f}')
        ax.axvline(p95_val, color='orange', linestyle='--', linewidth=2, label=f'95th: {p95_val:.6f}')
        ax.axvline(p99_val, color='purple', linestyle='--', linewidth=2, label=f'99th: {p99_val:.6f}')
        ax.axvline(p999_val, color='brown', linestyle='--', linewidth=2, label=f'99.9th: {p999_val:.6f}')
        
        ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.set_title(f'{title}\n(Max: {max_overall:.6f}, Samples: {len(values):,})', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Use log scale if very skewed
        if np.max(n) / np.median(n[n > 0]) > 100:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save individual plot
        individual_path = output_dir / f'{metric_name}_distribution.png'
        plt.savefig(individual_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {metric_name} plot: {individual_path}")
        plt.close()
    
    plt.close('all')


def save_statistics_to_file(metrics, file_stats, output_dir):
    """
    Save statistics to a text file.
    
    Args:
        metrics: Dictionary with aggregated metrics
        file_stats: List of per-file statistics
        output_dir: Directory to save the file
    """
    output_dir = Path(output_dir)
    output_file = output_dir / 'bounds_statistics.txt'
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAJECTORY BOUNDS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("AGGREGATED STATISTICS ACROSS ALL TRAJECTORIES:\n")
        f.write("-" * 80 + "\n\n")
        
        for metric_name, values in metrics.items():
            f.write(f"{metric_name.upper().replace('_', ' ')}:\n")
            f.write(f"  Count:        {len(values):,} samples\n")
            f.write(f"  Maximum:      {np.max(values):.6f}\n")
            f.write(f"  Mean:         {np.mean(values):.6f}\n")
            f.write(f"  Median:       {np.median(values):.6f}\n")
            f.write(f"  Std Dev:      {np.std(values):.6f}\n")
            f.write(f"  95th %-ile:   {np.percentile(values, 95):.6f}\n")
            f.write(f"  99th %-ile:   {np.percentile(values, 99):.6f}\n")
            f.write(f"  99.9th %-ile: {np.percentile(values, 99.9):.6f}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("MAXIMUM VALUES PER TRAJECTORY FILE:\n")
        f.write("=" * 80 + "\n\n")
        
        # Per-file statistics
        f.write(f"{'Filename':<40} {'Velocity':>12} {'Accel':>12} {'Force':>12} {'Distance':>12}\n")
        f.write("-" * 80 + "\n")
        
        for stat in file_stats:
            f.write(f"{stat['filename']:<40} {stat['max_velocity']:>12.6f} {stat['max_acceleration']:>12.6f} "
                   f"{stat['max_force']:>12.6f} {stat['max_distance']:>12.6f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("GLOBAL MAXIMUM VALUES:\n")
        f.write("-" * 80 + "\n")
        
        metrics_to_check = ['max_velocity', 'max_acceleration', 'max_force', 'max_distance']
        for metric in metrics_to_check:
            max_stat = max(file_stats, key=lambda x: x[metric])
            f.write(f"{metric.replace('max_', '').replace('_', ' ').title():30s}: {max_stat[metric]:.6f} (from {max_stat['filename']})\n")
    
    print(f"\n✓ Saved statistics to: {output_file}")


def main():
    """Main function to analyze trajectory bounds."""
    
    # Configuration
    data_dir = Path('/nfs/data/airhockey')
    output_dir = Path(__file__).parent
    
    print("=" * 80)
    print("TRAJECTORY BOUNDS ANALYSIS")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"\nError: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Analyze all trajectories
    metrics, file_stats = analyze_all_trajectories(data_dir)
    
    # Print statistics
    print_statistics(metrics, file_stats)
    
    # Save statistics to file
    save_statistics_to_file(metrics, file_stats, output_dir)
    
    # Create plots
    create_frequency_plots(metrics, output_dir)
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()


