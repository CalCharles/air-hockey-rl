# Real-World Air Hockey Robot Trajectory Data - Field Documentation

## Overview

This document provides the **official specification** for the 32-field state vector stored in the `train_vals` dataset of HDF5 trajectory files.

Data is at: /nfs/data/airhockey

**Source**: `airhockey/sims/real/proprioceptive_state.py`  
**Data Source**: UR5e Robot RTDE (Real-Time Data Exchange) Interface  
**File Format**: HDF5 with synchronized camera images and robot state

---

## Field Breakdown (32 Total Fields)

### Metadata Fields (0-4): 5 values

| Field | Name | Type | Units | Description |
|-------|------|------|-------|-------------|
| **0** | `cur_time` | float | seconds | Unix timestamp / Wall clock time |
| **1** | `tidx` | int | - | Trajectory index / dataset identifier |
| **2** | `i` | int | - | Frame/timestep number within trajectory |
| **3** | `estop` | bool | - | Emergency stop indicator (1 = estopped, 0 = normal) |
| **4** | `safety` | bool | - | Safety status (0 = unsafe, 1 = safe) |

---

### End-Effector Pose (5-10): 6 values

**Robot end-effector pose in Cartesian space**

| Field | Component | Type | Units | Description |
|-------|-----------|------|-------|-------------|
| **5** | pose_x | float | meters | End-effector X position |
| **6** | pose_y | float | meters | End-effector Y position |
| **7** | pose_z | float | meters | End-effector Z position |
| **8** | pose_rx | float | radians | End-effector rotation around X axis |
| **9** | pose_ry | float | radians | End-effector rotation around Y axis |
| **10** | pose_rz | float | radians | End-effector rotation around Z axis |

**Note**: These are Cartesian coordinates in the robot base frame, not joint angles.

---

### End-Effector Speed (11-16): 6 values

**Robot end-effector velocity in Cartesian space**

| Field | Component | Type | Units | Description |
|-------|-----------|------|-------|-------------|
| **11** | speed_vx | float | m/s | Linear velocity in X direction |
| **12** | speed_vy | float | m/s | Linear velocity in Y direction |
| **13** | speed_vz | float | m/s | Linear velocity in Z direction |
| **14** | speed_vrx | float | rad/s | Angular velocity around X axis |
| **15** | speed_vry | float | rad/s | Angular velocity around Y axis |
| **16** | speed_vrz | float | rad/s | Angular velocity around Z axis |

---

### End-Effector Force (17-22): 6 values

**Forces and torques at the robot end-effector**

| Field | Component | Type | Units | Description |
|-------|-----------|------|-------|-------------|
| **17** | force_fx | float | Newtons | Force in X direction |
| **18** | force_fy | float | Newtons | Force in Y direction |
| **19** | force_fz | float | Newtons | Force in Z direction |
| **20** | force_τx | float | N⋅m | Torque around X axis |
| **21** | force_τy | float | N⋅m | Torque around Y axis |
| **22** | force_τz | float | N⋅m | Torque around Z axis |

**Note**: These values capture contact forces during puck interactions.

---

### End-Effector Acceleration (23-25): 3 values

**Linear acceleration of the robot end-effector**

| Field | Component | Type | Units | Description |
|-------|-----------|------|-------|-------------|
| **23** | acc_ax | float | m/s² | Acceleration in X direction |
| **24** | acc_ay | float | m/s² | Acceleration in Y direction |
| **25** | acc_az | float | m/s² | Acceleration in Z direction |

**Note**: Only linear acceleration is recorded, not rotational.

---

### Desired End-Effector Pose (26-31): 6 values

**Commanded/target pose for the robot controller**

| Field | Component | Type | Units | Description |
|-------|-----------|------|-------|-------------|
| **26** | desired_x | float | meters | Desired X position |
| **27** | desired_y | float | meters | Desired Y position |
| **28** | desired_z | float | meters | Desired Z position |
| **29** | desired_rx | float | radians | Desired rotation around X |
| **30** | desired_ry | float | radians | Desired rotation around Y |
| **31** | desired_rz | float | radians | Desired rotation around Z |

**Note**: This represents the control input/action. The difference between desired and actual pose is the control error.

---

## Additional Fields (Not in Standard 32-Field Format)

Some data processing pipelines append puck state:

### Puck State (32-34): 3 values

| Field | Component | Type | Units | Description |
|-------|-----------|------|-------|-------------|
| **32** | puck_x | float | meters | Puck X position on table |
| **33** | puck_y | float | meters | Puck Y position on table |
| **34** | puck_occlusion | bool | - | Occlusion flag (1 = occluded/not visible, 0 = visible) |

**Note**: The standard HDF5 files contain only fields 0-31 (32 total values). Puck data may be stored separately or added during post-processing.

---

## Robot System Information

### Hardware
- **Robot**: Universal Robots UR5e
- **DOF**: 6 degrees of freedom
- **Type**: Collaborative robot arm (cobot)
- **End-effector**: Custom air hockey mallet attachment

### Coordinate System
- **Base frame**: Robot base (mounted on table edge)
- **Offset**: Approximately 1.2 meters in X between robot frame and table center
- **Convention**: Right-handed Cartesian coordinates
- **Rotation**: Euler angles in rxyz convention

### Data Interface
- **Protocol**: RTDE (Real-Time Data Exchange)
- **Frequency**: ~20 Hz (variable, see field 0 timestamps)
- **Synchronization**: Camera images synchronized with robot state

---

## Usage Examples

### Extracting Specific Fields

```python
import h5py
import numpy as np
from airhockey.sims.real.proprioceptive_state import slicer

# Load data
with h5py.File('trajectory_data434.hdf5', 'r') as f:
    train_vals = f['train_vals'][:]

# Use the official slicer function
data_dict = slicer(train_vals[0])  # Parse first frame

# Access specific components
print(f"Time: {data_dict['cur_time']}")
print(f"End-effector position: {data_dict['pose'][:3]}")
print(f"End-effector velocity: {data_dict['speed'][:3]}")
print(f"Forces: {data_dict['force'][:3]}")
print(f"Desired position: {data_dict['desired_pose'][:3]}")
```

### Computing Control Error

```python
# Position error
position_error = data_dict['desired_pose'][:3] - data_dict['pose'][:3]

# Velocity from desired motion
desired_velocity = data_dict['desired_pose'][:3] - data_dict['pose'][:3]

# Actual velocity
actual_velocity = data_dict['speed'][:3]
```

### Detecting Puck Hits

```python
# Large forces indicate puck contact
force_magnitude = np.linalg.norm(data_dict['force'][:3])
if force_magnitude > threshold:
    print("Puck hit detected!")
```

---

## Data Quality Notes

1. **Sampling rate**: Varies slightly but averages ~20 Hz
2. **Missing data**: Some trajectories may have occlusions (check field 4 for safety)
3. **Coordinate transforms**: May need offset adjustment for table coordinates
4. **Force sensitivity**: Force readings can be noisy; consider filtering
5. **Rotational components**: Often show minimal variation (wrist remains stable)

---

## Related Files

- **Data loading**: `dataset_management/repair_data.py`
- **State parsing**: `airhockey/sims/real/proprioceptive_state.py`
- **Robot control**: `airhockey/sims/air_hockey_real.py`
- **Trajectory writing**: `airhockey/sims/real/trajectory_merging.py`

---

## References

**Official field specification** (from `proprioceptive_state.py`):
```python
DATA_RANGES = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,11], [11,17], [17,23], [23,26], [26,32], [32,35]]
DATA_NAMES = ["cur_time", "tidx", "i", "estop", "safety", "pose", "speed", "force", "acc", "desired_pose", "puck"]
```

**Documentation comments** (from `repair_data.py` lines 97-110):
```python
'''
Expected output keys:
cur_time: wall time (1,)
tidx: trajectory index (1,)
i: timestep in trajectory index (1,)
estop: estop indicator, 1 is estopped (1,)
safety: safety indicator, 0 if unsafe (1,)
pose: robot end effector pose xyz rxryrx (6,)
speed: robot end effector speed xyz rxryrz (6,)
force: robot end effector force xyz rxryrz (6,)
acc: robot end effector acceleration xyz (3,)
desired_pose: `action` desired pose xyz rxryrz (6,)
puck: location of the puck, with occlusion 1 if occluded xy occluded (3,)
'''
```

---

*Documentation generated: November 24, 2025*  
*Last verified against codebase commit: [current]*

