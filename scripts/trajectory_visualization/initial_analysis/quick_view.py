#!/usr/bin/env python3
"""
Quick visualization tool to view trajectory data interactively.
Shows key information at a glance.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse


def quick_view(file_path):
    """
    Create a single comprehensive figure showing all key information.
    
    Args:
        file_path: Path to HDF5 trajectory file
    """
    with h5py.File(file_path, 'r') as f:
        num_hits = f['num_hits'][()]
        occlusions = f['occlusions'][()]
        train_img = f['train_img'][:]
        train_vals = f['train_vals'][:]
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(4, 6, figure=fig, hspace=0.4, wspace=0.4)
    
    # Title
    fig.suptitle(f'Trajectory Data Quick View: {file_path}\n'
                 f'Duration: {train_vals[-1, 0] - train_vals[0, 0]:.2f}s | '
                 f'Frames: {len(train_vals)} | Hits: {num_hits} | Occlusions: {occlusions}',
                 fontsize=16, fontweight='bold')
    
    # Row 1: Sample images
    for i in range(6):
        ax = fig.add_subplot(gs[0, i])
        idx = int(i * (len(train_img) - 1) / 5)
        ax.imshow(train_img[idx])
        ax.set_title(f'Frame {int(train_vals[idx, 2])}', fontsize=9)
        ax.axis('off')
    
    # Row 2: Robot joint positions
    ax1 = fig.add_subplot(gs[1, 0:3])
    for i in range(5, 11):
        ax1.plot(train_vals[:, i], label=f'Joint {i-5}', linewidth=1.5)
    ax1.set_title('Robot Joint Positions (Fields 5-10)', fontweight='bold')
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Position (rad)')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Row 2: Robot joint velocities
    ax2 = fig.add_subplot(gs[1, 3:6])
    for i in range(11, 17):
        ax2.plot(train_vals[:, i], label=f'Joint {i-11}', linewidth=1.5)
    ax2.set_title('Robot Joint Velocities (Fields 11-16)', fontweight='bold')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Velocity (rad/s)')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Row 3: Task space positions (likely puck and mallet)
    ax3 = fig.add_subplot(gs[2, 0:3])
    # Plot high-variance position fields
    position_fields = [17, 18, 19, 23, 24, 25]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    labels = ['Puck X?', 'Puck Y?', 'Coord Z?', 'Mallet X?', 'Mallet Y?', 'Ref Z?']
    for field, color, label in zip(position_fields, colors, labels):
        ax3.plot(train_vals[:, field], label=f'F{field}: {label}', 
                linewidth=1.5, color=color, alpha=0.8)
    ax3.set_title('Task Space Positions (Fields 17-25)', fontweight='bold')
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Position (units unknown)')
    ax3.legend(loc='best', fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # Row 3: Task space velocities
    ax4 = fig.add_subplot(gs[2, 3:6])
    velocity_fields = [20, 21, 22, 26, 27]
    v_colors = ['red', 'blue', 'green', 'orange', 'purple']
    v_labels = ['Puck vX?', 'Puck vY?', 'Vel Z?', 'Mallet vX?', 'Mallet vY?']
    for field, color, label in zip(velocity_fields, v_colors, v_labels):
        ax4.plot(train_vals[:, field], label=f'F{field}: {label}', 
                linewidth=1.5, color=color, alpha=0.8)
    ax4.set_title('Task Space Velocities (Fields 20-27)', fontweight='bold')
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Velocity')
    ax4.legend(loc='best', fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    # Row 4: 2D trajectory visualization
    ax5 = fig.add_subplot(gs[3, 0:3])
    # Assuming fields 17, 18 are X, Y positions
    ax5.plot(train_vals[:, 17], train_vals[:, 18], 'b-', linewidth=2, alpha=0.7, label='Puck trajectory')
    ax5.scatter(train_vals[0, 17], train_vals[0, 18], color='green', s=100, marker='o', 
               label='Start', zorder=5)
    ax5.scatter(train_vals[-1, 17], train_vals[-1, 18], color='red', s=100, marker='X', 
               label='End', zorder=5)
    
    # Also plot mallet if fields 23, 24 are mallet positions
    if train_vals[:, 23].std() > 0.1:  # Check if there's variation
        ax5.plot(train_vals[:, 23], train_vals[:, 24], 'orange', linewidth=2, 
                alpha=0.5, linestyle='--', label='Mallet trajectory')
    
    ax5.set_title('2D Trajectory (X-Y plane)', fontweight='bold')
    ax5.set_xlabel('Field 17 (X?)')
    ax5.set_ylabel('Field 18 (Y?)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axis('equal')
    
    # Row 4: Statistics table
    ax6 = fig.add_subplot(gs[3, 3:6])
    ax6.axis('off')
    
    stats_text = [
        "FIELD STATISTICS",
        "=" * 50,
        "",
        "Robot Joint Positions (rad):",
        f"  Joint 0 (F5):  {train_vals[:, 5].min():7.3f} to {train_vals[:, 5].max():7.3f}",
        f"  Joint 1 (F6):  {train_vals[:, 6].min():7.3f} to {train_vals[:, 6].max():7.3f}",
        "",
        "Task Space Positions:",
        f"  Field 17 (Puck X?): {train_vals[:, 17].min():7.2f} to {train_vals[:, 17].max():7.2f}",
        f"  Field 18 (Puck Y?): {train_vals[:, 18].min():7.2f} to {train_vals[:, 18].max():7.2f}",
        f"  Field 19 (Z?):      {train_vals[:, 19].min():7.2f} to {train_vals[:, 19].max():7.2f}",
        "",
        "Task Space Velocities:",
        f"  Field 20 (vX?): {train_vals[:, 20].min():7.3f} to {train_vals[:, 20].max():7.3f}",
        f"  Field 21 (vY?): {train_vals[:, 21].min():7.3f} to {train_vals[:, 21].max():7.3f}",
        "",
        "Metadata:",
        f"  Timestamp: {train_vals[0, 0]:.2f} to {train_vals[-1, 0]:.2f}",
        f"  Duration: {train_vals[-1, 0] - train_vals[0, 0]:.2f} seconds",
        f"  Frame rate: {len(train_vals) / (train_vals[-1, 0] - train_vals[0, 0]):.1f} Hz",
        f"  Trajectory ID: {int(train_vals[0, 1])}",
    ]
    
    ax6.text(0.05, 0.95, '\n'.join(stats_text), 
            transform=ax6.transAxes,
            fontsize=9,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Quick view of trajectory data - single comprehensive figure"
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to HDF5 trajectory file'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Save figure to file instead of showing interactively'
    )
    
    args = parser.parse_args()
    
    fig = quick_view(args.file_path)
    
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

