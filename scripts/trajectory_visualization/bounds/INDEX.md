# Trajectory Bounds Analysis - Index

This directory contains a comprehensive analysis of trajectory data bounds and statistics.

## 📋 Quick Start

**Want the key numbers?** → See [`KEY_FINDINGS.md`](KEY_FINDINGS.md)

**Want detailed documentation?** → See [`ANALYSIS_README.md`](ANALYSIS_README.md)

**Want the raw data?** → See [`statistics.json`](statistics.json) or [`statistics_summary.txt`](statistics_summary.txt)

## 📊 Maximum Values (TL;DR)

Across **417 trajectories** with **207,023 frames**:

| Metric | Max Value | Unit |
|--------|-----------|------|
| Linear Velocity | **1.074** | m/s |
| Angular Velocity | **0.011** | rad/s |
| Acceleration | **30.986** | m/s² |
| Force | **88.729** | N |
| Torque | **11.541** | N⋅m |
| Distance to Target | **0.285** | m |

## 📁 File Organization

### Documentation
- **`INDEX.md`** (this file) - Navigation and overview
- **`KEY_FINDINGS.md`** - Summary of key insights and practical applications
- **`ANALYSIS_README.md`** - Detailed methodology and complete statistics
- **`README.md`** - Legacy README (from previous analysis)

### Data Files
- **`statistics.json`** - Complete statistics in machine-readable format
- **`statistics_summary.txt`** - Human-readable summary report
- **`bounds_statistics.txt`** - Legacy statistics file

### Scripts
- **`analyze_all_trajectories.py`** - Main analysis script (NEW - comprehensive)
- **`analyze_bounds.py`** - Legacy analysis script

### Visualizations

#### New Comprehensive Visualizations
- **`magnitude_frequency_distributions.png`** - All 6 metrics in one figure (linear scale)
  - Linear velocity, angular velocity, acceleration
  - Force, torque, distance to target
  - Each with mean/median markers and probability density

- **`magnitude_frequency_distributions_log_scale.png`** - Same metrics in log scale
  - Better visualization of full range
  - Shows low-frequency regions

#### Legacy Individual Visualizations
- **`velocity_magnitude_distribution.png`** - Linear velocity only
- **`acceleration_magnitude_distribution.png`** - Acceleration only
- **`force_magnitude_distribution.png`** - Force only
- **`paddle_target_distance_distribution.png`** - Distance only
- **`magnitude_distributions.png`** - Combined view

## 🚀 Usage

### View Statistics
```bash
# Quick summary in terminal
cat statistics_summary.txt

# Or view JSON
cat statistics.json | jq .overall_stats
```

### Re-run Analysis
```bash
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate
python scripts/trajectory_visualization/bounds/analyze_all_trajectories.py
```

### View Visualizations
Open any PNG file in an image viewer:
```bash
# Example: view comprehensive distributions
xdg-open magnitude_frequency_distributions.png
```

## 📈 What's Analyzed

For each trajectory, the script extracts and analyzes:

1. **Linear Velocity** (`sqrt(vx² + vy² + vz²)`)
   - From fields 11-13 in HDF5 data
   - Measures paddle speed in 3D space

2. **Angular Velocity** (`sqrt(ωx² + ωy² + ωz²)`)
   - From fields 14-16
   - Measures rotational speed

3. **Acceleration** (`sqrt(ax² + ay² + az²)`)
   - From fields 23-25
   - Measures linear acceleration magnitude

4. **Force** (`sqrt(fx² + fy² + fz²)`)
   - From fields 17-19
   - Measures applied force at end-effector

5. **Torque** (`sqrt(τx² + τy² + τz²)`)
   - From fields 20-22
   - Measures applied torque at end-effector

6. **Distance to Target** (`sqrt((xt-x)² + (yt-y)² + (zt-z)²)`)
   - Computed from fields 5-7 (position) and 26-28 (desired position)
   - Measures control tracking error

## 🎯 Use Cases

### For Machine Learning
```python
# Normalization using 99th percentile
velocity_norm = velocity / 0.764  # m/s
force_norm = force / 17.93        # N
acceleration_norm = accel / 11.45 # m/s²
```

### For Simulation
```python
# Safety bounds (max values + 10% margin)
assert velocity < 1.2      # m/s
assert force < 100.0       # N
assert acceleration < 35.0 # m/s²
```

### For Anomaly Detection
```python
# Flag outliers beyond 95th percentile
if velocity > 0.554:      # m/s
    flag_high_velocity()
if force > 10.686:        # N
    flag_high_force()
```

## 🔍 Data Quality Notes

- **Total trajectories processed**: 417 out of 418 files
- **Failed files**: 1 (`trajectory_data752.hdf5` - empty file)
- **Data source**: `/nfs/data/airhockey/trajectory_data*.hdf5`
- **Robot**: UR5e with RTDE interface
- **Sampling rate**: ~20 Hz (varies by trajectory)

## 📚 Related Analyses

- **Velocity Analysis**: `../acceleration_analysis/analyze_velocity.py`
- **Acceleration Analysis**: `../acceleration_analysis/analyze_acceleration.py`
- **Field Documentation**: `../initial_analysis/FIELD_DOCUMENTATION.md`
- **Trajectory Visualization**: `../visualize/visualize_trajectory.py`

## 🆕 Recent Updates

**November 29, 2025**
- Created comprehensive analysis script (`analyze_all_trajectories.py`)
- Generated new visualizations with all 6 metrics
- Added detailed documentation (`ANALYSIS_README.md`, `KEY_FINDINGS.md`)
- Processed all 417 valid trajectory files
- Generated statistics for 207,023 total frames

## 💡 Tips

1. **Start with KEY_FINDINGS.md** for a quick overview
2. **Use statistics.json** for programmatic access to data
3. **Check visualizations** to understand distributions
4. **Read ANALYSIS_README.md** for methodology details
5. **Re-run the script** if new trajectory data is added

## 📞 Support

For questions about:
- **Field definitions**: See `../initial_analysis/FIELD_DOCUMENTATION.md`
- **Methodology**: See `ANALYSIS_README.md`
- **Quick reference**: See `KEY_FINDINGS.md`
- **Raw statistics**: See `statistics.json` or `statistics_summary.txt`

---

*Generated: November 29, 2025*  
*Data: 417 trajectories, 207,023 frames*  
*Source: UR5e Real Robot Trajectories*


