# Air Hockey Robot Trajectory Data Analysis

## File Analyzed
`/nfs/data/airhockey/trajectory_data434.hdf5`

## Overview

This HDF5 file contains **synchronized sensor and visual data** from a real-world air hockey robot system. The data represents approximately **6 seconds** of gameplay captured at **~20 Hz**, containing 122 frames of robot state, task-space measurements, and camera images.

---

## Data Structure

The file contains 4 datasets:

### 1. `num_hits` (scalar)
- **Value**: 2
- **Meaning**: Number of times the robot successfully hit the puck during this trajectory

### 2. `occlusions` (scalar)
- **Value**: 0  
- **Meaning**: Number of times the puck was occluded (not visible to tracking system)

### 3. `train_img` (122, 240, 320, 3)
- **Shape**: 122 frames of 240×320 RGB images
- **Format**: uint8 (0-255)
- **Content**: Top-down camera view of the air hockey table showing:
  - The puck
  - The robot's mallet/striker
  - Table boundaries
  - Arena markings

### 4. `train_vals` (122, 32)
- **Shape**: 122 frames × 32 values per frame
- **Type**: float64
- **Content**: Complete state information for each frame

---

## Detailed Field Breakdown (`train_vals`)

### **Fields 0-4: Metadata** (5 values)

| Field | Name | Description | Example Range |
|-------|------|-------------|---------------|
| 0 | Timestamp | Unix timestamp (seconds since epoch) | 1713831647.97 → 1713831653.92 |
| 1 | Trajectory ID | Identifier for this trajectory | 434 (constant) |
| 2 | Frame Counter | Sequential frame number | 20 → 141 |
| 3 | Unknown Flag | Purpose unclear | 0 (constant) |
| 4 | Unknown Flag | Purpose unclear | 1 (constant) |

**Duration**: 5.95 seconds  
**Effective Frame Rate**: 20.5 Hz

---

### **Fields 5-16: Robot Joint State** (12 values)

These represent the **6-DOF (6 degrees of freedom) robot arm** state:

#### Joint Positions (Fields 5-10)
Likely the angular positions of 6 robot joints in radians.

| Field | Min | Max | Mean | Std Dev | Notes |
|-------|-----|-----|------|---------|-------|
| 5 | -0.732 | -0.491 | -0.563 | 0.074 | Large motion |
| 6 | -0.202 | 0.050 | -0.057 | 0.061 | Significant variation |
| 7 | 0.307 | 0.326 | 0.313 | 0.006 | Small changes |
| 8 | -0.0015 | -0.0015 | -0.0015 | <0.001 | Nearly constant |
| 9 | -3.065 | -3.065 | -3.065 | <0.001 | Nearly constant |
| 10 | ~0 | ~0 | ~0 | <0.001 | Nearly constant |

**Interpretation**: 
- Fields 5-6 show the most variation → likely shoulder/elbow joints doing most of the work
- Fields 7-10 have minimal variation → likely wrist joints or end-effector orientation staying relatively stable

#### Joint Velocities (Fields 11-16)
Angular velocities of the 6 joints in radians/second.

| Field | Min | Max | Mean | Std Dev |
|-------|-----|-----|------|---------|
| 11 | -0.710 | 0.415 | -0.006 | 0.223 |
| 12 | -0.310 | 0.329 | 0.007 | 0.128 |
| 13 | -0.033 | 0.057 | 0.000 | 0.018 |
| 14-16 | ~0 | ~0 | ~0 | <0.001 |

---

### **Fields 17-31: Task-Space State** (15 values)

These represent **Cartesian space** information - positions and velocities of objects and end-effector.

#### High-Variance Fields (Likely Positions)

| Field | Min | Max | Range | Mean | Std Dev | Interpretation |
|-------|-----|-----|-------|------|---------|----------------|
| 17 | -7.70 | 12.31 | 20.01 | 0.65 | 4.04 | **Puck X position** (along table width) |
| 18 | -11.74 | 8.34 | 20.07 | -0.92 | 4.44 | **Puck Y position** (along table length) |
| 19 | -6.42 | 10.00 | 16.43 | 4.80 | 2.27 | **Likely Z or another spatial coord** |
| 23 | -5.00 | 4.48 | 9.48 | 0.70 | 1.21 | **Possibly mallet X position** |
| 24 | -1.17 | 2.03 | 3.20 | 0.20 | 0.53 | **Possibly mallet Y position** |
| 25 | -11.82 | -8.71 | 3.10 | -9.53 | 0.36 | **Z offset or reference frame coord** |

**Key Observations**:
- Fields 17-18 have the largest ranges (~20 units) → likely puck X,Y tracking the full table
- Maximum frame-to-frame change in field 19 is 10.6 units → very fast puck motion
- Field 25's large negative mean suggests a coordinate offset

#### Medium/Low-Variance Fields (Likely Velocities)

| Field | Range | Mean | Std Dev | Interpretation |
|-------|-------|------|---------|----------------|
| 20 | 1.76 | -0.06 | 0.48 | **Puck X velocity** |
| 21 | 2.10 | 0.28 | 0.52 | **Puck Y velocity** |
| 22 | 0.37 | 0.06 | 0.11 | **Velocity component** |
| 26 | 0.29 | -0.56 | 0.09 | **Mallet velocity or orientation** |
| 27 | 0.28 | -0.06 | 0.07 | **Mallet velocity or orientation** |

#### Very Small Value Fields (Nearly Constant)

| Field | Range | Interpretation |
|-------|-------|----------------|
| 28-31 | <0.001 | Likely derived features, sensor noise, or unused fields |

---

## Likely Data Interpretation

Based on domain knowledge of air hockey and the value patterns:

### Probable State Vector Structure

```
Fields 0-4:   Metadata (timestamp, ID, frame, flags)
Fields 5-10:  6 Robot joint positions (radians)
Fields 11-16: 6 Robot joint velocities (rad/s)
Fields 17-19: Puck position (x, y, z) in meters or cm
Fields 20-22: Puck velocity (vx, vy, vz) in m/s or cm/s
Fields 23-25: Robot mallet/end-effector position (x, y, z)
Fields 26-27: Mallet velocity or orientation
Fields 28-31: Additional features or unused
```

### Reference Frame
- The air hockey table appears to be the reference frame
- X-Y plane = table surface
- Z = height above table
- Large negative value in field 25 might indicate the table is at z ≈ -9.5 in robot base frame

---

## Visual Data Analysis

### Image Characteristics
- **Resolution**: 320×240 (QVGA) - standard for real-time robot vision
- **Color**: RGB, well-balanced channels (mean ~93 across R,G,B)
- **View**: Overhead/top-down camera of the table
- **Frame Rate**: ~20.5 Hz (synchronized with sensor data)

### Motion Analysis
- **Average motion intensity**: 2.88 (pixel difference between frames)
- **Peak motion intensity**: 5.14 (during fast puck movement)
- **Frames with significant motion**: 52 out of 122 (~43%)

---

## Applications

This data is suitable for:

1. **Vision-Based Tracking**
   - Training puck detection/tracking models
   - Developing visual servoing algorithms
   - Testing computer vision pipelines

2. **Control Learning**
   - Imitation learning from expert demonstrations
   - Inverse kinematics model training
   - Policy learning for air hockey strategies

3. **Simulation Validation**
   - Comparing simulated vs. real physics
   - Validating puck dynamics models
   - Testing trajectory prediction algorithms

4. **Performance Analysis**
   - Studying robot motion patterns
   - Analyzing hitting strategies
   - Measuring reaction times and accuracy

---

## Data Quality

✅ **Strengths**:
- Complete trajectory (no missing frames)
- Synchronized sensor and visual data
- Clean data (0 occlusions)
- Successful gameplay (2 hits recorded)
- Good temporal resolution (20 Hz)

⚠️ **Limitations**:
- Relatively short duration (~6 seconds)
- Some fields appear unused or constant
- Field documentation not included in file
- Camera resolution modest (320×240)

---

## File Collection

The trajectory is part of a larger dataset:
- Located in: `/nfs/data/airhockey/`
- Files: `trajectory_data434.hdf5` through `trajectory_data452.hdf5` and beyond
- File sizes: 7 MB to 98 MB
- Total dataset: ~23 GB

Each file likely represents a single rally or game sequence.

---

## Generated Visualizations

This analysis produced the following plots:

1. **sample_images.png** - 8 sample frames showing trajectory progression
2. **full_trajectory_sequence.png** - 30 frames showing complete trajectory
3. **key_frames.png** - 5 key moments with timestamps
4. **time_series.png** - Temporal evolution of 8 important state variables
5. **correlation_matrix.png** - Correlation between all state fields
6. **distributions.png** - Statistical distributions of all 27 data fields
7. **motion_intensity.png** - Frame-to-frame visual motion over time
8. **frame_differences.png** - Visual differences between consecutive frames
9. **detailed_single_frame.png** - Multi-view analysis of a single frame

---

## Scripts Created

All analysis scripts are located in `scripts/trajectory_visualization/`:

1. **explore_hdf5.py** - Basic HDF5 structure exploration
2. **analyze_trajectory_data.py** - Detailed statistical analysis with visualizations
3. **examine_images.py** - In-depth image content analysis
4. **create_report.py** - Comprehensive text report generation

### Usage Examples

```bash
# Activate virtual environment
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate

# Explore structure
python scripts/trajectory_visualization/explore_hdf5.py /nfs/data/airhockey/trajectory_data434.hdf5

# Full analysis with plots
python scripts/trajectory_visualization/analyze_trajectory_data.py /nfs/data/airhockey/trajectory_data434.hdf5

# Image analysis
python scripts/trajectory_visualization/examine_images.py /nfs/data/airhockey/trajectory_data434.hdf5

# Generate report
python scripts/trajectory_visualization/create_report.py /nfs/data/airhockey/trajectory_data434.hdf5
```

---

## Conclusions

The file `/nfs/data/airhockey/trajectory_data434.hdf5` contains **high-quality real-world robot trajectory data** from an air hockey system. It captures:

- **Robot state**: 6-DOF arm joint positions and velocities
- **Task state**: Puck and mallet positions/velocities in Cartesian space
- **Visual data**: Synchronized overhead camera images
- **Metadata**: Timing, frame numbers, and event counts

The data is **well-structured and suitable for machine learning**, control system development, and simulation validation. The synchronized nature of the sensor and visual data makes it particularly valuable for multi-modal learning approaches.

---

*Analysis completed: November 24, 2025*

