# Analysis Files Index

## 📊 Main Summary
- **`SUMMARY_complete_analysis.png`** (660 KB) - **START HERE!** Comprehensive overview showing all metrics in one visualization

## 📖 Documentation
- **`README.md`** - Complete analysis report with statistics tables and insights
- **`INDEX.md`** - This file

## 🔧 Analysis Scripts
1. **`analyze_acceleration.py`** - Analyzes X, Y, Z accelerations (fields 23-25)
2. **`analyze_velocity.py`** - Analyzes linear and angular velocities (fields 11-16)
3. **`analyze_position.py`** - Analyzes positions and orientations (fields 5-10)
4. **`summary_report.py`** - Generates comprehensive summary visualization

## 📈 Visualization Files

### Acceleration Analysis (2 files)
- `acceleration_analysis.png` (380 KB) - Time series, distributions, box plots
- `acceleration_analysis_detailed.png` (357 KB) - Raw vs smoothed, correlation plots

### Velocity Analysis (3 files)
- `velocity_linear_analysis.png` (353 KB) - Linear velocities (Vx, Vy, Vz)
- `velocity_angular_analysis.png` (387 KB) - Angular velocities (ωx, ωy, ωz)
- `velocity_detailed_analysis.png` (314 KB) - Smoothed and correlation plots

### Position Analysis (3 files)
- `position_cartesian_analysis.png` (321 KB) - X, Y, Z positions and XY trajectory
- `position_orientation_analysis.png` (257 KB) - Euler angle orientations
- `position_3d_trajectory.png` (469 KB) - 3D trajectory with multiple view angles

## 🎯 Quick Reference

### To View Results
```bash
# View the main summary
open acceleration_analysis/SUMMARY_complete_analysis.png

# View detailed README
cat acceleration_analysis/README.md
```

### To Regenerate Analysis
```bash
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate

# Run all analyses
python3 scripts/trajectory_visualization/acceleration_analysis/analyze_acceleration.py
python3 scripts/trajectory_visualization/acceleration_analysis/analyze_velocity.py
python3 scripts/trajectory_visualization/acceleration_analysis/analyze_position.py
python3 scripts/trajectory_visualization/acceleration_analysis/summary_report.py
```

## 📋 Data Source
- **Trajectory File**: `/nfs/data/airhockey/trajectory_data434.hdf5`
- **Duration**: 5.95 seconds
- **Timesteps**: 122
- **Sampling Rate**: ~20.51 Hz
- **Total Path Length**: 1.215 meters

## 🔍 Key Findings Summary

1. **Motion Pattern**: Primarily horizontal (XY plane) with minimal vertical movement
2. **Speed**: Average 0.20 m/s, max 0.73 m/s
3. **Orientation**: Nearly constant throughout trajectory (minimal wrist rotation)
4. **Acceleration**: Dominated by gravity on Z-axis (~-9.8 m/s²)
5. **Control**: Smooth, controlled movements with consistent sampling

---

*Total size: 3.6 MB | 14 files | Generated: November 24, 2025*


