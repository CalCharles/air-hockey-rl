# Trajectory Data Bounds Analysis

This directory contains a comprehensive analysis of all trajectory data files, focusing on the bounds and distributions of key metrics.

## Overview

The analysis processes all trajectory HDF5 files (`trajectory_data*.hdf5`) from `/nfs/data/airhockey/` and computes detailed statistics for:

1. **Linear Velocity Magnitude** - Speed of the paddle in 3D space (m/s)
2. **Angular Velocity Magnitude** - Rotational speed around all axes (rad/s)
3. **Acceleration Magnitude** - Linear acceleration in 3D space (m/s²)
4. **Force Magnitude** - Applied forces at the end-effector (N)
5. **Torque Magnitude** - Applied torques at the end-effector (N⋅m)
6. **Paddle-to-Target Distance** - Distance between actual and desired position (m)

## Key Results

### Maximum Values (Critical Bounds)

These maximum values represent the absolute bounds observed across **417 trajectories** with **207,023 total frames**:

| Metric | Maximum Value | Unit |
|--------|---------------|------|
| **Linear Velocity** | 1.074393 | m/s |
| **Angular Velocity** | 0.010541 | rad/s |
| **Acceleration** | 30.985776 | m/s² |
| **Force** | 88.728680 | N |
| **Torque** | 11.541066 | N⋅m |
| **Distance to Target** | 0.284864 | m |

### Statistical Summary

#### Linear Velocity (m/s)
- **Maximum**: 1.074393 m/s ⚡
- **99th percentile**: 0.763779 m/s
- **95th percentile**: 0.553996 m/s
- **Mean**: 0.235866 m/s
- **Median**: 0.206099 m/s

#### Angular Velocity (rad/s)
- **Maximum**: 0.010541 rad/s ⚡
- **99th percentile**: 0.002667 rad/s
- **95th percentile**: 0.001527 rad/s
- **Mean**: 0.000395 rad/s
- **Median**: 0.000180 rad/s

#### Acceleration (m/s²)
- **Maximum**: 30.985776 m/s² ⚡
- **99th percentile**: 11.448252 m/s²
- **95th percentile**: 10.404147 m/s²
- **Mean**: 9.627737 m/s²
- **Median**: 9.573646 m/s²

#### Force (N)
- **Maximum**: 88.728680 N ⚡
- **99th percentile**: 17.926205 N
- **95th percentile**: 10.685754 N
- **Mean**: 6.232969 N
- **Median**: 5.705943 N

#### Torque (N⋅m)
- **Maximum**: 11.541066 N⋅m ⚡
- **99th percentile**: 1.804399 N⋅m
- **95th percentile**: 0.728383 N⋅m
- **Mean**: 0.342463 N⋅m
- **Median**: 0.277238 N⋅m

#### Distance to Target (m)
- **Maximum**: 0.284864 m ⚡
- **99th percentile**: 0.206198 m
- **95th percentile**: 0.155353 m
- **Mean**: 0.068220 m
- **Median**: 0.059665 m

## Generated Files

### Data Files
1. **`statistics.json`** - Machine-readable JSON containing all statistics and per-trajectory breakdowns
2. **`statistics_summary.txt`** - Human-readable text summary of all statistics

### Visualization Files
1. **`magnitude_frequency_distributions.png`** - Relative frequency distributions (linear scale) with mean and median markers
2. **`magnitude_frequency_distributions_log_scale.png`** - Log-scale distributions for better visualization of the full range

## Usage

### Running the Analysis

```bash
# Activate virtual environment
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate

# Run the analysis script
python scripts/trajectory_visualization/bounds/analyze_all_trajectories.py
```

The script will:
1. Scan `/nfs/data/airhockey/` for all `trajectory_data*.hdf5` files
2. Extract velocity, acceleration, force, and distance metrics
3. Compute comprehensive statistics
4. Generate visualizations
5. Save results to this directory

### Interpreting the Results

#### For Setting Safety Bounds
Use the **maximum values** as absolute safety bounds for your system. These represent the highest values observed in real robot operation.

#### For Normalization
Use the **95th or 99th percentile** values for normalization in machine learning models, as they filter out extreme outliers while capturing most of the data range.

#### For Understanding Typical Behavior
Use the **mean** and **median** values to understand typical operating conditions. Note that for some metrics (like force), the median is significantly lower than the mean, indicating a right-skewed distribution with occasional high forces.

## Technical Details

### Data Source
- **Path**: `/nfs/data/airhockey/trajectory_data*.hdf5`
- **Format**: HDF5 with `train_vals` dataset
- **Fields**: See `FIELD_DOCUMENTATION.md` for complete field specification

### Field Mapping
According to the trajectory data format:
- **Fields 5-7**: Position (x, y, z) in meters
- **Fields 11-13**: Linear velocity (vx, vy, vz) in m/s
- **Fields 14-16**: Angular velocity (vrx, vry, vrz) in rad/s
- **Fields 17-19**: Force (fx, fy, fz) in Newtons
- **Fields 20-22**: Torque (τx, τy, τz) in N⋅m
- **Fields 23-25**: Acceleration (ax, ay, az) in m/s²
- **Fields 26-28**: Desired position (target_x, target_y, target_z) in meters

### Computation Methods

#### Magnitude Calculations
All magnitudes are computed as Euclidean norms:
```python
velocity_magnitude = sqrt(vx² + vy² + vz²)
acceleration_magnitude = sqrt(ax² + ay² + az²)
force_magnitude = sqrt(fx² + fy² + fz²)
```

#### Distance to Target
```python
distance_to_target = sqrt((target_x - x)² + (target_y - y)² + (target_z - z)²)
```

## Observations

### Velocity Behavior
- Most operations occur at moderate speeds (mean ~0.24 m/s)
- Maximum observed speed is ~1.07 m/s, showing the robot can achieve high velocities when needed
- Angular velocities are very small (max ~0.01 rad/s), indicating the wrist orientation remains relatively stable

### Force Distribution
- Heavy right-skew: median (5.7 N) is much lower than mean (6.2 N)
- Maximum force of 88.7 N likely represents puck impacts or collisions
- 95th percentile at 10.7 N suggests most operations involve modest forces

### Acceleration
- Remarkably consistent: mean (9.63 m/s²) very close to median (9.57 m/s²)
- This is approximately 1g (9.8 m/s²), suggesting gravity compensation
- Maximum of 31 m/s² represents rapid direction changes

### Control Accuracy
- Median tracking error is ~6 cm, showing good but not perfect tracking
- Maximum error of ~28 cm indicates occasional large deviations
- 95th percentile at ~15.5 cm suggests most errors are moderate

## Script Details

### Dependencies
```python
h5py          # HDF5 file reading
numpy         # Numerical computations
matplotlib    # Plotting
seaborn       # Plot styling
tqdm          # Progress bars
```

### Performance
- Processes 418 trajectories in < 1 second
- Memory efficient: processes one file at a time
- Handles corrupted/empty files gracefully

### Error Handling
The script includes robust error handling:
- Skips corrupted or empty trajectory files
- Reports failed files but continues processing
- One file (`trajectory_data752.hdf5`) was found to be empty and was skipped

## Related Documentation

- **Field Documentation**: `../initial_analysis/FIELD_DOCUMENTATION.md`
- **Acceleration Analysis**: `../acceleration_analysis/README.md`
- **Trajectory Visualization**: `../visualize/README.md`

## Citation

If you use this analysis in your research, please cite:
```
Air Hockey Robot Trajectory Analysis
Generated: November 2025
Data: 417 trajectories, 207,023 frames
Source: UR5e Robot Real-Time Data Exchange (RTDE)
```

---

*Last updated: November 29, 2025*


