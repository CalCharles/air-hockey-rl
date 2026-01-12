# Trajectory Analysis Results

## Overview

This directory contains comprehensive statistical and visual analysis of robot end-effector **acceleration**, **velocity**, and **position** data from trajectory file `trajectory_data434.hdf5`.

**Data Source**: `/nfs/data/airhockey/trajectory_data434.hdf5`  
**Trajectory Duration**: 5.95 seconds  
**Number of Timesteps**: 122  
**Average Sampling Rate**: ~20.51 Hz

---

## Analysis Scripts

### 1. `analyze_acceleration.py`
Analyzes end-effector accelerations (fields 23-25: ax, ay, az)
- Extracts X, Y, Z linear accelerations
- Computes statistics and magnitude
- Generates time series, distributions, and correlation plots

### 2. `analyze_velocity.py`
Analyzes end-effector velocities (fields 11-16: vx, vy, vz, ωx, ωy, ωz)
- Extracts linear velocities (X, Y, Z)
- Extracts angular velocities (RX, RY, RZ)
- Analyzes both linear and rotational motion

### 3. `analyze_position.py`
Analyzes end-effector positions (fields 5-10: x, y, z, rx, ry, rz)
- Extracts Cartesian positions (X, Y, Z)
- Extracts orientations (RX, RY, RZ Euler angles)
- Computes path length and creates 3D trajectory visualizations

---

## Generated Visualizations

### Acceleration Analysis
- **`acceleration_analysis.png`** (380 KB)
  - 9 subplots showing time series, combined plots, distributions, and box plots
  - Separate views for X, Y, Z axes and magnitude
  
- **`acceleration_analysis_detailed.png`** (357 KB)
  - Raw vs smoothed (rolling mean) comparisons
  - 2D scatter plots showing acceleration relationships colored by time

### Velocity Analysis
- **`velocity_linear_analysis.png`** (353 KB)
  - Linear velocities (Vx, Vy, Vz) time series and distributions
  - Combined views and magnitude analysis
  
- **`velocity_angular_analysis.png`** (387 KB)
  - Angular velocities (ωx, ωy, ωz) time series and distributions
  - Rotational motion analysis
  
- **`velocity_detailed_analysis.png`** (314 KB)
  - Raw vs smoothed velocity comparisons
  - Velocity relationship scatter plots

### Position Analysis
- **`position_cartesian_analysis.png`** (321 KB)
  - X, Y, Z position time series
  - 2D trajectory view (XY plane - table view)
  - Position distributions and statistics
  
- **`position_orientation_analysis.png`** (257 KB)
  - Orientation (RX, RY, RZ) time series in radians and degrees
  - Euler angle distributions
  
- **`position_3d_trajectory.png`** (469 KB)
  - 3D trajectory visualization
  - Multiple viewing angles (XY, XZ, YZ planes)
  - Instantaneous speed computed from position changes

---

## Key Statistics Summary

### Acceleration (m/s²)

| Axis | Mean | Std Dev | Min | Max | Median |
|------|------|---------|-----|-----|--------|
| **X** | 0.70 | 1.21 | -5.00 | 4.48 | 0.79 |
| **Y** | 0.20 | 0.53 | -1.17 | 2.03 | 0.19 |
| **Z** | -9.53 | 0.36 | -11.82 | -8.71 | -9.50 |
| **Magnitude** | 9.64 | 0.40 | 8.72 | 11.94 | 9.58 |

**⚠️ Note**: Z-axis dominated by gravity (~-9.8 m/s²)

### Linear Velocity (m/s)

| Axis | Mean | Std Dev | Min | Max | Median |
|------|------|---------|-----|-----|--------|
| **Vx** | -0.006 | 0.223 | -0.710 | 0.415 | 0.014 |
| **Vy** | 0.008 | 0.128 | -0.310 | 0.329 | 0.011 |
| **Vz** | 0.000 | 0.018 | -0.033 | 0.057 | -0.002 |
| **Speed** | 0.205 | 0.157 | 0.007 | 0.728 | 0.180 |

### Angular Velocity (rad/s)

| Axis | Mean | Std Dev | Max |
|------|------|---------|-----|
| **ωx** | 0.0000 | 0.0001 | 0.0003 |
| **ωy** | 0.0000 | 0.0005 | 0.0019 |
| **ωz** | 0.0000 | 0.0000 | 0.0001 |
| **Magnitude** | 0.0003 | 0.0004 | 0.0019 |

**Note**: Angular velocities are extremely small, indicating minimal wrist rotation during this trajectory.

### Position (meters)

| Axis | Mean | Std Dev | Range | Min | Max |
|------|------|---------|-------|-----|-----|
| **X** | -0.563 | 0.074 | 0.241 | -0.732 | -0.491 |
| **Y** | -0.057 | 0.061 | 0.252 | -0.202 | 0.050 |
| **Z** | 0.313 | 0.006 | 0.019 | 0.307 | 0.326 |

**Total Path Length**: 1.215 meters

### Orientation (radians)

| Axis | Mean | Std Dev | Notes |
|------|------|---------|-------|
| **RX** | -0.0015 | 0.0000 | Nearly constant |
| **RY** | -3.0648 | 0.0001 | Nearly constant (~-175.5°) |
| **RZ** | -0.0000 | 0.0000 | Nearly constant |

**Note**: End-effector orientation remains nearly constant throughout the trajectory.

---

## Key Insights

### 1. **Motion Characteristics**
- **Horizontal motion dominates**: Most dynamic movement occurs in X and Y axes
- **Vertical stability**: Z position varies minimally (range: 19mm)
- **Constant orientation**: Robot maintains nearly fixed wrist orientation
- **Low speeds**: Average linear speed ~0.2 m/s with peaks ~0.73 m/s

### 2. **Acceleration Patterns**
- X-axis shows highest variability (std: 1.21 m/s²)
- Z-axis measurements include gravitational component
- Peak accelerations reach ~5 m/s² in horizontal plane

### 3. **Path Characteristics**
- Total distance traveled: 1.21 meters over 5.95 seconds
- Motion primarily in XY plane (table surface)
- Start and end points are different (not a closed loop)

### 4. **Control Observations**
- Very minimal angular motion suggests position-only control
- Smooth velocity profiles indicate controlled movements
- Consistent sampling rate throughout trajectory

---

## Usage

To regenerate any analysis:

```bash
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate

# Run individual analyses
python3 scripts/trajectory_visualization/acceleration_analysis/analyze_acceleration.py
python3 scripts/trajectory_visualization/acceleration_analysis/analyze_velocity.py
python3 scripts/trajectory_visualization/acceleration_analysis/analyze_position.py
```

To analyze a different trajectory file, edit the `data_path` variable in each script:

```python
data_path = Path('/nfs/data/airhockey/trajectory_dataXXX.hdf5')
```

---

## Related Documentation

- **Field Specification**: `scripts/trajectory_visualization/initial_analysis/FIELD_DOCUMENTATION.md`
- **Data Source**: `airhockey/sims/real/proprioceptive_state.py`

---

*Analysis generated: November 24, 2025*  
*Trajectory file: `trajectory_data434.hdf5`*


