# Trajectory Data Analysis Tools

This directory contains tools for analyzing real-world air hockey robot trajectory data stored in HDF5 format.

## Overview

The trajectory data files (located in `/nfs/data/airhockey/`) contain synchronized sensor and visual data from a 6-DOF robot arm playing air hockey. Each file includes:

- **Robot joint states** (positions and velocities)
- **Task-space states** (puck and mallet positions/velocities)
- **Camera images** (320×240 RGB, ~20 Hz)
- **Event metadata** (hits, occlusions, timestamps)

## Data Structure

Each HDF5 file contains 4 datasets:

```
trajectory_dataXXX.hdf5
├── num_hits      : scalar - number of successful hits
├── occlusions    : scalar - number of vision occlusions
├── train_img     : (N, 240, 320, 3) - RGB camera images
└── train_vals    : (N, 32) - state vectors with:
    ├── Fields 0-4   : Metadata (timestamp, ID, frame counter, flags)
    ├── Fields 5-10  : Robot joint positions (6-DOF, radians)
    ├── Fields 11-16 : Robot joint velocities (rad/s)
    └── Fields 17-31 : Task-space state (positions & velocities)
```

## Scripts

### 1. `explore_hdf5.py` - Basic Structure Explorer

Quickly inspect the structure and contents of any HDF5 file.

```bash
# Activate environment first
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate

# Basic exploration
python scripts/trajectory_visualization/explore_hdf5.py /nfs/data/airhockey/trajectory_data434.hdf5

# Show more samples
python scripts/trajectory_visualization/explore_hdf5.py /nfs/data/airhockey/trajectory_data434.hdf5 --max-samples 10
```

**Output**: Console text showing dataset shapes, types, and sample values.

---

### 2. `analyze_trajectory_data.py` - Statistical Analysis

Perform detailed statistical analysis and create comprehensive visualizations.

```bash
python scripts/trajectory_visualization/analyze_trajectory_data.py \
    /nfs/data/airhockey/trajectory_data434.hdf5 \
    --output-dir ./my_analysis

# Skip plot generation (analysis only)
python scripts/trajectory_visualization/analyze_trajectory_data.py \
    /nfs/data/airhockey/trajectory_data434.hdf5 \
    --no-plots
```

**Outputs**:
- `sample_images.png` - Grid of 8 sample frames
- `time_series.png` - Temporal plots of key state variables
- `correlation_matrix.png` - Correlation between all fields
- `distributions.png` - Histograms of all data fields

---

### 3. `examine_images.py` - Visual Data Analysis

Deep dive into the camera image data, motion detection, and visual characteristics.

```bash
python scripts/trajectory_visualization/examine_images.py \
    /nfs/data/airhockey/trajectory_data434.hdf5 \
    --output-dir ./my_analysis
```

**Outputs**:
- `full_trajectory_sequence.png` - 30 frames showing full trajectory
- `key_frames.png` - 5 key moments with timestamps
- `motion_intensity.png` - Frame-to-frame motion graph
- `frame_differences.png` - Visual diff between consecutive frames
- `detailed_single_frame.png` - Multi-view analysis (color, grayscale, edges, threshold)

---

### 4. `create_report.py` - Comprehensive Text Report

Generate a detailed text report with interpretations and statistics.

```bash
python scripts/trajectory_visualization/create_report.py \
    /nfs/data/airhockey/trajectory_data434.hdf5 \
    --output trajectory_report.txt
```

**Output**: `trajectory_report.txt` - Comprehensive analysis including:
- Data structure breakdown
- Field interpretations
- Statistical summaries
- Usage recommendations

---

### 5. `quick_view.py` - Single-Page Visual Summary

Create a comprehensive single-figure visualization showing all key information.

```bash
# Display interactively
python scripts/trajectory_visualization/quick_view.py \
    /nfs/data/airhockey/trajectory_data434.hdf5

# Save to file
python scripts/trajectory_visualization/quick_view.py \
    /nfs/data/airhockey/trajectory_data434.hdf5 \
    --save summary.png
```

**Output**: Single figure with:
- Sample images across trajectory
- Joint position/velocity plots
- Task-space position/velocity plots
- 2D trajectory visualization
- Statistical summary table

---

## Example Workflow

```bash
# 1. Activate environment
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate

# 2. Quick exploration
python scripts/trajectory_visualization/explore_hdf5.py /nfs/data/airhockey/trajectory_data434.hdf5

# 3. Full analysis with all visualizations
python scripts/trajectory_visualization/analyze_trajectory_data.py \
    /nfs/data/airhockey/trajectory_data434.hdf5 \
    --output-dir ./analysis_output

# 4. Detailed image analysis
python scripts/trajectory_visualization/examine_images.py \
    /nfs/data/airhockey/trajectory_data434.hdf5 \
    --output-dir ./analysis_output

# 5. Generate comprehensive report
python scripts/trajectory_visualization/create_report.py \
    /nfs/data/airhockey/trajectory_data434.hdf5 \
    --output ./analysis_output/report.txt

# 6. Create quick summary figure
python scripts/trajectory_visualization/quick_view.py \
    /nfs/data/airhockey/trajectory_data434.hdf5 \
    --save ./analysis_output/summary.png
```

## Data Interpretation

### Fields 0-4: Metadata
- **0**: Unix timestamp (seconds)
- **1**: Trajectory ID number
- **2**: Frame counter
- **3-4**: Unknown flags

### Fields 5-16: Robot State (6-DOF arm)
- **5-10**: Joint positions in radians
  - Joints 0-1 show most variation (shoulder/elbow)
  - Joints 2-5 relatively stable (wrist/end-effector)
- **11-16**: Joint velocities in rad/s

### Fields 17-31: Task Space (likely interpretation)
- **17-18**: Puck X,Y position (large range ~20 units)
- **19**: Z coordinate or secondary spatial coordinate
- **20-21**: Puck X,Y velocity
- **22**: Velocity component
- **23-24**: Mallet X,Y position (medium range ~3-9 units)
- **25**: Z offset or reference frame coordinate (large negative value ~-9.5)
- **26-27**: Mallet velocity components
- **28-31**: Very small values (derived features or unused)

### Camera Images
- **Resolution**: 320×240 (QVGA)
- **View**: Overhead/top-down of air hockey table
- **Content**: Puck, mallet, table boundaries, markers
- **Frame rate**: ~20 Hz (synchronized with sensor data)

## Sample Analysis Results

For `trajectory_data434.hdf5`:
- **Duration**: 5.95 seconds
- **Frames**: 122 (20.5 Hz)
- **Hits**: 2
- **Occlusions**: 0
- **Puck motion range**: ~20 units in X and Y
- **Maximum puck velocity change**: 10.6 units/frame in field 19

## Applications

This data is suitable for:

1. **Vision-based tracking** - Train puck/mallet detection models
2. **Imitation learning** - Learn control policies from demonstrations
3. **Simulation validation** - Compare real vs. simulated physics
4. **Trajectory prediction** - Forecast puck motion
5. **Strategy analysis** - Study robot gameplay patterns

## Dependencies

All scripts require:
- `h5py` - HDF5 file I/O
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `scipy` - Image processing (for `examine_images.py`)

These are included in the project's UV virtual environment.

## Notes

- Always activate the UV virtual environment before running scripts
- Output directories are created automatically if they don't exist
- All scripts use the same HDF5 file format
- Visualizations are saved at 150 DPI for good quality
- The exact meaning of some fields is inferred from patterns and domain knowledge

## Generated Analysis Files

A complete analysis for `trajectory_data434.hdf5` is available in:
```
trajectory_analysis_output/
├── ANALYSIS_SUMMARY.md            - This comprehensive markdown summary
├── trajectory_report.txt          - Detailed text report
├── quick_view_summary.png         - Single-page visual summary
├── sample_images.png              - Sample image frames
├── full_trajectory_sequence.png   - Complete sequence visualization
├── key_frames.png                 - Key moments
├── time_series.png                - State variable time series
├── correlation_matrix.png         - Field correlations
├── distributions.png              - Statistical distributions
├── motion_intensity.png           - Visual motion analysis
├── frame_differences.png          - Frame diff visualization
└── detailed_single_frame.png      - Single frame analysis
```

## Questions?

For questions about:
- **Data format**: Consult `ANALYSIS_SUMMARY.md` in the output directory
- **Field meanings**: See the interpretation sections above
- **Tool usage**: Run any script with `--help` flag

---

*Tools created: November 24, 2025*

