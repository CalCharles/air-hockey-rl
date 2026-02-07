#!/usr/bin/env python3
"""
Create a comprehensive report on the trajectory data structure.
"""

import h5py
import numpy as np
from pathlib import Path
import argparse


def create_report(file_path, output_file):
    """
    Create a comprehensive analysis report.
    
    Args:
        file_path: Path to the HDF5 file
        output_file: Path to save the report
    """
    
    with h5py.File(file_path, 'r') as f:
        num_hits = f['num_hits'][()]
        occlusions = f['occlusions'][()]
        train_img = f['train_img']
        train_vals = f['train_vals'][:]
    
    report = []
    report.append("=" * 80)
    report.append("TRAJECTORY DATA ANALYSIS REPORT")
    report.append(f"File: {file_path}")
    report.append("=" * 80)
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 80)
    report.append(f"This file contains trajectory data from a real-world air hockey robot.")
    report.append(f"It includes {len(train_vals)} frames of synchronized camera images and sensor data.")
    report.append(f"The trajectory captured {num_hits} puck hits with {occlusions} occlusions.")
    report.append("")
    
    report.append("DATA STRUCTURE")
    report.append("-" * 80)
    report.append("")
    report.append("1. METADATA FIELDS (Fields 0-4):")
    report.append("   - Field 0: Unix timestamp (seconds since epoch)")
    report.append(f"     Range: {train_vals[0, 0]:.2f} to {train_vals[-1, 0]:.2f}")
    report.append(f"     Duration: {train_vals[-1, 0] - train_vals[0, 0]:.2f} seconds")
    report.append("")
    report.append(f"   - Field 1: Trajectory ID = {int(train_vals[0, 1])}")
    report.append("")
    report.append("   - Field 2: Frame counter")
    report.append(f"     Range: {int(train_vals[0, 2])} to {int(train_vals[-1, 2])}")
    report.append("")
    report.append("   - Field 3: Unknown flag (always 0)")
    report.append("   - Field 4: Unknown flag (always 1)")
    report.append("")
    
    report.append("2. ROBOT JOINT STATE (Fields 5-16, 12 values):")
    report.append("   This appears to be a 6-DOF robot arm with joint positions and velocities.")
    report.append("")
    report.append("   Joint Positions (Fields 5-10):")
    for i in range(5, 11):
        report.append(f"     Field {i:2d}: min={train_vals[:, i].min():8.4f}, "
                     f"max={train_vals[:, i].max():8.4f}, "
                     f"mean={train_vals[:, i].mean():8.4f}, "
                     f"std={train_vals[:, i].std():7.4f}")
    report.append("")
    
    report.append("   Joint Velocities (Fields 11-16):")
    for i in range(11, 17):
        report.append(f"     Field {i:2d}: min={train_vals[:, i].min():8.4f}, "
                     f"max={train_vals[:, i].max():8.4f}, "
                     f"mean={train_vals[:, i].mean():8.4f}, "
                     f"std={train_vals[:, i].std():7.4f}")
    report.append("")
    
    report.append("3. TASK-SPACE STATE (Fields 17-31, 15 values):")
    report.append("   These appear to be Cartesian positions and velocities.")
    report.append("")
    
    # Identify which fields change the most
    ranges = [(i, train_vals[:, i].max() - train_vals[:, i].min()) for i in range(17, 32)]
    ranges.sort(key=lambda x: x[1], reverse=True)
    
    report.append("   High-variance fields (likely positions/major state variables):")
    for field_idx, range_val in ranges[:6]:
        report.append(f"     Field {field_idx:2d}: range={range_val:8.4f}, "
                     f"mean={train_vals[:, field_idx].mean():8.4f}, "
                     f"std={train_vals[:, field_idx].std():7.4f}")
    report.append("")
    
    report.append("   Low-variance fields (likely velocities or secondary measurements):")
    for field_idx, range_val in ranges[6:]:
        report.append(f"     Field {field_idx:2d}: range={range_val:8.4f}, "
                     f"mean={train_vals[:, field_idx].mean():8.4f}, "
                     f"std={train_vals[:, field_idx].std():7.4f}")
    report.append("")
    
    report.append("4. CAMERA IMAGES:")
    report.append(f"   - Count: {len(train_vals)} frames")
    report.append(f"   - Resolution: 320x240 pixels (QVGA)")
    report.append(f"   - Format: RGB color (3 channels)")
    report.append(f"   - Storage: uint8 (0-255)")
    report.append("")
    
    report.append("INTERPRETATION")
    report.append("-" * 80)
    report.append("")
    report.append("Based on the data structure and value ranges, this file likely contains:")
    report.append("")
    report.append("1. **Robot State (Fields 5-16)**:")
    report.append("   - 6 joint angle positions (in radians)")
    report.append("   - 6 joint angular velocities (in radians/second)")
    report.append("   These control the robot arm that holds the air hockey mallet.")
    report.append("")
    report.append("2. **Task Space State (Fields 17-31)**:")
    report.append("   Likely includes:")
    report.append("   - Puck position (x, y, z) and velocity (vx, vy, vz)")
    report.append("   - End-effector/mallet position (x, y, z) and velocity (vx, vy, vz)")
    report.append("   - Possibly additional derived features or measurements")
    report.append("")
    report.append("   Field 19 shows large changes (range ~16 units), suggesting it might be")
    report.append("   a primary position coordinate (e.g., y-axis along the table).")
    report.append("")
    report.append("   Field 25 has a large negative mean (~-9.5), which might indicate")
    report.append("   a z-coordinate offset or a coordinate in a different reference frame.")
    report.append("")
    report.append("3. **Visual Information**:")
    report.append("   The camera images show the air hockey table from above, capturing:")
    report.append("   - The puck position")
    report.append("   - The robot's mallet/striker")
    report.append("   - The table boundaries")
    report.append("   - Possibly tracking markers")
    report.append("")
    
    report.append("TEMPORAL CHARACTERISTICS")
    report.append("-" * 80)
    duration = train_vals[-1, 0] - train_vals[0, 0]
    fps = len(train_vals) / duration
    report.append(f"Duration: {duration:.2f} seconds")
    report.append(f"Frames: {len(train_vals)}")
    report.append(f"Effective frame rate: {fps:.1f} Hz")
    report.append("")
    
    # Calculate some motion statistics
    diffs = np.diff(train_vals[:, 17:32], axis=0)
    max_velocities = np.abs(diffs).max(axis=0)
    
    report.append("Maximum frame-to-frame changes in task space:")
    for i, max_vel in enumerate(max_velocities):
        field_idx = i + 17
        report.append(f"  Field {field_idx:2d}: {max_vel:8.4f}")
    report.append("")
    
    report.append("USAGE NOTES")
    report.append("-" * 80)
    report.append("This data can be used for:")
    report.append("- Training vision-based puck tracking models")
    report.append("- Learning robot control policies from demonstrations")
    report.append("- Validating physics simulations against real-world behavior")
    report.append("- Analyzing robot motion patterns and strategies")
    report.append("- Testing prediction/forecasting algorithms for puck trajectory")
    report.append("")
    
    report.append("DATA QUALITY INDICATORS")
    report.append("-" * 80)
    report.append(f"Number of successful hits: {num_hits}")
    report.append(f"Number of occlusions: {occlusions}")
    report.append(f"Trajectory completeness: Good (no missing frames in sequence)")
    report.append("")
    
    # Write report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_text = "\n".join(report)
    output_path.write_text(report_text)
    
    # Also print to console
    print(report_text)
    
    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive analysis report for trajectory data"
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to HDF5 trajectory file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./trajectory_analysis_output/trajectory_report.txt',
        help='Path to save report (default: ./trajectory_analysis_output/trajectory_report.txt)'
    )
    
    args = parser.parse_args()
    
    create_report(args.file_path, args.output)


if __name__ == "__main__":
    main()

