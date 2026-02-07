#!/usr/bin/env python3
"""
Analyze end-effector accelerations (x, y, z) from trajectory data.
Fields 23-25 contain acceleration data according to FIELD_DOCUMENTATION.md
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

def extract_accelerations(train_vals):
    """
    Extract acceleration data from train_vals.
    Fields 23-25: acc_ax, acc_ay, acc_az (m/s²)
    """
    acc_x = train_vals[:, 23]  # Field 23: Acceleration in X direction
    acc_y = train_vals[:, 24]  # Field 24: Acceleration in Y direction
    acc_z = train_vals[:, 25]  # Field 25: Acceleration in Z direction
    
    # Also extract timestamps for plotting
    timestamps = train_vals[:, 0]  # Field 0: cur_time (Unix timestamp)
    
    # Convert to relative time (seconds from start)
    relative_time = timestamps - timestamps[0]
    
    return acc_x, acc_y, acc_z, relative_time

def compute_statistics(acc_x, acc_y, acc_z):
    """Compute comprehensive statistics for acceleration data."""
    
    # Compute magnitude of acceleration vector
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    stats = {
        'X Acceleration': {
            'mean': np.mean(acc_x),
            'std': np.std(acc_x),
            'min': np.min(acc_x),
            'max': np.max(acc_x),
            'median': np.median(acc_x),
            'p25': np.percentile(acc_x, 25),
            'p75': np.percentile(acc_x, 75),
            'p95': np.percentile(acc_x, 95),
            'p99': np.percentile(acc_x, 99),
        },
        'Y Acceleration': {
            'mean': np.mean(acc_y),
            'std': np.std(acc_y),
            'min': np.min(acc_y),
            'max': np.max(acc_y),
            'median': np.median(acc_y),
            'p25': np.percentile(acc_y, 25),
            'p75': np.percentile(acc_y, 75),
            'p95': np.percentile(acc_y, 95),
            'p99': np.percentile(acc_y, 99),
        },
        'Z Acceleration': {
            'mean': np.mean(acc_z),
            'std': np.std(acc_z),
            'min': np.min(acc_z),
            'max': np.max(acc_z),
            'median': np.median(acc_z),
            'p25': np.percentile(acc_z, 25),
            'p75': np.percentile(acc_z, 75),
            'p95': np.percentile(acc_z, 95),
            'p99': np.percentile(acc_z, 99),
        },
        'Magnitude': {
            'mean': np.mean(acc_magnitude),
            'std': np.std(acc_magnitude),
            'min': np.min(acc_magnitude),
            'max': np.max(acc_magnitude),
            'median': np.median(acc_magnitude),
            'p25': np.percentile(acc_magnitude, 25),
            'p75': np.percentile(acc_magnitude, 75),
            'p95': np.percentile(acc_magnitude, 95),
            'p99': np.percentile(acc_magnitude, 99),
        }
    }
    
    return stats, acc_magnitude

def print_statistics(stats):
    """Pretty print statistics."""
    print("\n" + "="*80)
    print("END-EFFECTOR ACCELERATION STATISTICS")
    print("="*80)
    
    for axis, metrics in stats.items():
        print(f"\n{axis} (m/s²):")
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

def create_visualizations(acc_x, acc_y, acc_z, acc_magnitude, relative_time, output_dir):
    """Create comprehensive visualizations of acceleration data."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Time series plot of all three axes
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(relative_time, acc_x, 'r-', alpha=0.7, linewidth=0.8, label='X')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.set_title('X-Axis Acceleration Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(relative_time, acc_y, 'g-', alpha=0.7, linewidth=0.8, label='Y')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.set_title('Y-Axis Acceleration Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(relative_time, acc_z, 'b-', alpha=0.7, linewidth=0.8, label='Z')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Acceleration (m/s²)')
    ax3.set_title('Z-Axis Acceleration Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 2. Combined time series
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(relative_time, acc_x, 'r-', alpha=0.6, linewidth=0.8, label='X')
    ax4.plot(relative_time, acc_y, 'g-', alpha=0.6, linewidth=0.8, label='Y')
    ax4.plot(relative_time, acc_z, 'b-', alpha=0.6, linewidth=0.8, label='Z')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Acceleration (m/s²)')
    ax4.set_title('All Axes Acceleration Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 3. Magnitude over time
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(relative_time, acc_magnitude, 'purple', alpha=0.7, linewidth=0.8)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Acceleration Magnitude (m/s²)')
    ax5.set_title('Acceleration Magnitude Over Time')
    ax5.grid(True, alpha=0.3)
    
    # 4. Histograms
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(acc_x, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax6.set_xlabel('Acceleration (m/s²)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('X-Axis Acceleration Distribution')
    ax6.grid(True, alpha=0.3)
    
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(acc_y, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax7.set_xlabel('Acceleration (m/s²)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Y-Axis Acceleration Distribution')
    ax7.grid(True, alpha=0.3)
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.hist(acc_z, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax8.set_xlabel('Acceleration (m/s²)')
    ax8.set_ylabel('Frequency')
    ax8.set_title('Z-Axis Acceleration Distribution')
    ax8.grid(True, alpha=0.3)
    
    # 5. Box plot comparison
    ax9 = plt.subplot(3, 3, 9)
    box_data = [acc_x, acc_y, acc_z, acc_magnitude]
    bp = ax9.boxplot(box_data, tick_labels=['X', 'Y', 'Z', 'Magnitude'],
                      patch_artist=True, showmeans=True)
    colors = ['lightcoral', 'lightgreen', 'lightblue', 'plum']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax9.set_ylabel('Acceleration (m/s²)')
    ax9.set_title('Acceleration Distribution Comparison')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / 'acceleration_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_path}")
    
    # Create a second figure for more detailed analysis
    fig2 = plt.figure(figsize=(18, 10))
    
    # 1. Rolling statistics
    window = min(50, len(acc_x) // 10)  # Adaptive window size
    if window > 1:
        acc_x_rolling = np.convolve(acc_x, np.ones(window)/window, mode='valid')
        acc_y_rolling = np.convolve(acc_y, np.ones(window)/window, mode='valid')
        acc_z_rolling = np.convolve(acc_z, np.ones(window)/window, mode='valid')
        time_rolling = relative_time[:len(acc_x_rolling)]
        
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(relative_time, acc_x, 'r-', alpha=0.3, linewidth=0.5, label='Raw')
        ax1.plot(time_rolling, acc_x_rolling, 'darkred', linewidth=2, label=f'Rolling Mean (w={window})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.set_title('X-Axis: Raw vs Smoothed')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(relative_time, acc_y, 'g-', alpha=0.3, linewidth=0.5, label='Raw')
        ax2.plot(time_rolling, acc_y_rolling, 'darkgreen', linewidth=2, label=f'Rolling Mean (w={window})')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Acceleration (m/s²)')
        ax2.set_title('Y-Axis: Raw vs Smoothed')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(relative_time, acc_z, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
        ax3.plot(time_rolling, acc_z_rolling, 'darkblue', linewidth=2, label=f'Rolling Mean (w={window})')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.set_title('Z-Axis: Raw vs Smoothed')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
    
    # 2. Scatter plots - acceleration relationships
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(acc_x, acc_y, c=relative_time, cmap='viridis', alpha=0.5, s=10)
    ax4.set_xlabel('X Acceleration (m/s²)')
    ax4.set_ylabel('Y Acceleration (m/s²)')
    ax4.set_title('X vs Y Acceleration (colored by time)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Time (s)')
    
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(acc_x, acc_z, c=relative_time, cmap='viridis', alpha=0.5, s=10)
    ax5.set_xlabel('X Acceleration (m/s²)')
    ax5.set_ylabel('Z Acceleration (m/s²)')
    ax5.set_title('X vs Z Acceleration (colored by time)')
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Time (s)')
    
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(acc_y, acc_z, c=relative_time, cmap='viridis', alpha=0.5, s=10)
    ax6.set_xlabel('Y Acceleration (m/s²)')
    ax6.set_ylabel('Z Acceleration (m/s²)')
    ax6.set_title('Y vs Z Acceleration (colored by time)')
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Time (s)')
    
    plt.tight_layout()
    
    output_path2 = output_dir / 'acceleration_analysis_detailed.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Saved detailed visualization to: {output_path2}")
    
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
    
    # Extract accelerations
    acc_x, acc_y, acc_z, relative_time = extract_accelerations(train_vals)
    
    print(f"\nExtracted acceleration data:")
    print(f"  Number of timesteps: {len(acc_x)}")
    print(f"  Duration: {relative_time[-1]:.2f} seconds")
    print(f"  Average sampling rate: {len(acc_x) / relative_time[-1]:.2f} Hz")
    
    # Compute statistics
    stats, acc_magnitude = compute_statistics(acc_x, acc_y, acc_z)
    
    # Print statistics
    print_statistics(stats)
    
    # Create visualizations
    create_visualizations(acc_x, acc_y, acc_z, acc_magnitude, relative_time, output_dir)
    
    print("\n✓ Analysis complete!")

if __name__ == '__main__':
    main()


