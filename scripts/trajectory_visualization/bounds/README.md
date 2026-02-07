# Trajectory Bounds Analysis

This directory contains the analysis of all trajectory data to determine bounds and distributions for key metrics.

## Overview

The `analyze_bounds.py` script processes all trajectory HDF5 files from `/nfs/data/airhockey` and computes detailed statistics on:

1. **Velocity Magnitude** - Computed from position differences over time
2. **Acceleration Magnitude** - Computed from velocity differences over time  
3. **Force Magnitude** - Extracted from force fields in trajectory data
4. **Paddle-Target Distance** - Distance between actual paddle position and desired target position

## Results Summary

### Global Maximum Values

Based on analysis of **418 trajectory files** with **207,023 samples** total:

| Metric | Maximum Value | Source File |
|--------|--------------|-------------|
| Velocity | 1.064 m/s | trajectory_data793.hdf5 |
| Acceleration | 9.969 m/s² | trajectory_data565.hdf5 |
| Force | 88.729 N | trajectory_data829.hdf5 |
| Distance | 0.285 m | trajectory_data716.hdf5 |

### Statistical Summary

#### Velocity Magnitude (m/s)
- **Mean**: 0.236 m/s
- **Median**: 0.206 m/s  
- **95th percentile**: 0.554 m/s
- **99th percentile**: 0.762 m/s
- **Maximum**: 1.064 m/s

#### Acceleration Magnitude (m/s²)
- **Mean**: 0.891 m/s²
- **Median**: 0.611 m/s²
- **95th percentile**: 2.702 m/s²
- **99th percentile**: 4.009 m/s²
- **Maximum**: 9.969 m/s²

#### Force Magnitude (N)
- **Mean**: 6.233 N
- **Median**: 5.706 N
- **95th percentile**: 10.686 N
- **99th percentile**: 17.926 N
- **Maximum**: 88.729 N

#### Paddle-Target Distance (m)
- **Mean**: 0.068 m
- **Median**: 0.060 m
- **95th percentile**: 0.155 m
- **99th percentile**: 0.206 m
- **Maximum**: 0.285 m

## Output Files

- **`bounds_statistics.txt`** - Detailed text report with all statistics and per-file maximums
- **`magnitude_distributions.png`** - Combined 2x2 plot showing all four distributions
- **`velocity_magnitude_distribution.png`** - Detailed velocity distribution plot
- **`acceleration_magnitude_distribution.png`** - Detailed acceleration distribution plot
- **`force_magnitude_distribution.png`** - Detailed force distribution plot
- **`paddle_target_distance_distribution.png`** - Detailed distance distribution plot

## Usage

To re-run the analysis:

```bash
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate
python scripts/trajectory_visualization/bounds/analyze_bounds.py
```

## Implementation Details

The script:
1. Loads all `trajectory_data*.hdf5` files from `/nfs/data/airhockey`
2. Extracts position, target position, and force data from HDF5 fields
3. Computes velocities using finite differences: `v = Δpos / Δt`
4. Computes accelerations using finite differences: `a = Δvel / Δt`
5. Computes force and distance magnitudes using Euclidean norm
6. Aggregates all samples across trajectories
7. Generates frequency distribution plots with statistical markers
8. Saves detailed statistics to text file

### Data Fields Used

From the HDF5 `train_vals` array:
- Field 0: `cur_time` (Unix timestamp)
- Fields 5-7: `pose` (x, y, z position)
- Fields 17-19: `force` (x, y, z force)
- Fields 26-28: `desired_pose` (x, y, z target position)

## Notes

- One file (`trajectory_data752.hdf5`) had an error due to zero-size array
- Plots use 99.9th percentile filtering to handle extreme outliers
- Histograms use log scale when data is highly skewed
- Statistical markers show mean, median, 95th, and 99th percentiles


