"""Load real robot trajectory data from HDF5 files.

Pipeline call stack (puck system identification):
  real_data_puck_pipeline.py  RealDataPuckPipeline.load_real_data()
    └─ real_to_sim_observations.py  load_multiple_real_trajectories_for_sysid()
        └─ data_loading.py  load_all_trajectories() / load_trajectory()  ◄── YOU ARE HERE
            └─ reads trajectory_data<idx>.hdf5 → dict with paddle, puck, speed, action, ...

  - traj["pose"]   : raw end-effector pose (NOT base coords, needs +1.2 x offset)
  - traj["paddle"] : pose with offset applied = base coordinates
  - traj["puck"]   : (x, y, occluded) already in base coordinates
  - traj["speed"]  : velocity (same in both frames, offset is constant)
  - traj["action"] : relative deltas (desired_pose - pose), no offset needed
"""

import os
import glob
import h5py
import numpy as np
import argparse


# Keys available in the processed state_data HDF5 files:
#   cur_time: (N, 1)       wall time
#   tidx: (N, 1)           trajectory index
#   i: (N, 1)              timestep within trajectory
#   estop: (N, 1)          estop indicator (1 = estopped)
#   safety: (N, 1)         safety indicator (0 = unsafe)
#   pose: (N, 6)           end effector pose xyz rxryrz
#   speed: (N, 6)          end effector speed xyz rxryrz
#   force: (N, 6)          end effector force xyz rxryrz
#   acc: (N, 3)            end effector acceleration xyz
#   desired_pose: (N, 6)   action desired pose xyz rxryrz
#   puck: (N, 3)           puck location xy + occlusion flag
#   paddle: (N, 6)         paddle pose (with offset applied)
#   action: (N, 2)         computed action = desired_pose[:2] - pose[:2]
#   image: (N, 240, 320, 3) camera image (uint8)

STATE_KEYS = [
    "cur_time", "tidx", "i", "estop", "safety",
    "pose", "speed", "force", "acc",
    "desired_pose", "puck", "paddle", "action",
]


def load_trajectory(data_dir, filename, load_images=False):
    """Load a single trajectory from an HDF5 file.

    Supports both per-key format (pose, puck, paddle, ...) and legacy
    train_vals 35-column format. Legacy files are auto-detected and
    converted to per-key format via load_legacy_train_vals_trajectory().

    Args:
        data_dir: Directory containing the HDF5 files.
        filename: Name of the HDF5 file (e.g. 'trajectory_data100.hdf5').
        load_images: Whether to load image data. Default False to save memory.

    Returns:
        dict of numpy arrays keyed by dataset name.
    """
    filepath = os.path.join(data_dir, filename)
    traj = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            if key == "image" and not load_images:
                continue
            traj[key] = np.array(f[key])

    if "puck" not in traj and "train_vals" in traj:
        trajs = load_legacy_train_vals_trajectory(filepath)
        traj = next(iter(trajs.values()))

    return traj


def load_all_trajectories(data_dir, load_images=False, num_load=-1):
    """Load all trajectory HDF5 files from a directory.

    Args:
        data_dir: Directory containing trajectory_data*.hdf5 files.
        load_images: Whether to load image data. Default False.
        num_load: Max number of trajectories to load. -1 for all.

    Returns:
        dict mapping trajectory index (int) to trajectory dict.
    """
    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".hdf5")],
        key=lambda f: int(f[len("trajectory_data") : -len(".hdf5")]),
    )
    if num_load > 0 and num_load < len(files):
        rng = np.random.default_rng()
        indices = rng.choice(len(files), size=num_load, replace=False)
        files = [files[i] for i in sorted(indices)]

    trajectories = {}
    for filename in files:
        tidx = int(filename[len("trajectory_data") : -len(".hdf5")])
        trajectories[tidx] = load_trajectory(data_dir, filename, load_images=load_images)
    return trajectories


# Raw train_vals column mapping (new slicer format, 32 common columns):
#   col 0: cur_time, col 1: tidx, col 2: i, col 3: estop, col 4: safety
#   cols 5-10: pose (xyz rxryrz)
#   cols 11-16: speed (xyz rxryrz)
#   cols 17-22: force (xyz rxryrz)
#   cols 23-25: acc (xyz)
#   cols 26-31: desired_pose (xyz rxryrz)
# 50-col files have extra cols 32-49: joint_pos(6), joint_vel(6), joint_acc(6)

OFFSET_CONSTANT = 1.2

# Box2D table dimensions (from configs/baseline_configs/box2d/puck_height.yaml)
BOX2D_TABLE_LENGTH = 1.9304   # Y-axis (meters)
BOX2D_TABLE_WIDTH = 0.8636    # X-axis (meters)
BOX2D_PADDLE_RADIUS = 0.0508
BOX2D_PUCK_RADIUS = 0.03175


def load_legacy_train_vals_trajectory(filepath):
    """Load a legacy train_vals HDF5 file and convert to standard per-key format.

    Handles the 35-column format where:
        col 0: cur_time, col 1: tidx, col 2: i, col 3: estop, col 4: safety
        cols 5-10: pose (xyz rxryrz), cols 11-16: speed, cols 17-22: force
        cols 23-25: acc, cols 26-31: desired_pose, cols 32-34: puck (x, y, occluded)

    Splits by tidx into separate trajectories and adds derived fields
    (paddle, action) to match the output of load_trajectory().

    Args:
        filepath: Path to the legacy HDF5 file.

    Returns:
        dict mapping trajectory index (int) to trajectory dict with keys:
            cur_time, tidx, i, estop, safety, pose, speed, force, acc,
            desired_pose, puck, paddle, action
    """
    with h5py.File(filepath, "r") as f:
        train_vals = np.array(f["train_vals"])

    n_cols = train_vals.shape[1]
    if n_cols < 35:
        raise ValueError(f"Expected at least 35 columns, got {n_cols}")

    # Split by trajectory index
    tidx_col = train_vals[:, 1]
    unique_tidx = np.unique(tidx_col).astype(int)

    trajectories = {}
    for tidx in unique_tidx:
        mask = tidx_col == tidx
        rows = train_vals[mask]

        traj = {
            "cur_time": rows[:, 0:1],
            "tidx": rows[:, 1:2],
            "i": rows[:, 2:3],
            "estop": rows[:, 3:4],
            "safety": rows[:, 4:5],
            "pose": rows[:, 5:11],
            "speed": rows[:, 11:17],
            "force": rows[:, 17:23],
            "acc": rows[:, 23:26],
            "desired_pose": rows[:, 26:32],
            "puck": rows[:, 32:35],
        }

        # Derived fields
        traj["paddle"] = traj["pose"].copy()
        traj["paddle"][:, 0] += OFFSET_CONSTANT
        traj["action"] = traj["desired_pose"][:, :2] - traj["pose"][:, :2]

        trajectories[int(tidx)] = traj

    return trajectories


def load_raw_train_vals(data_dir, num_load=-1):
    """Load raw train_vals arrays from all trajectory HDF5 files.

    Args:
        data_dir: Directory containing trajectory_data*.hdf5 files.
        num_load: Max number of files to load. -1 for all.

    Returns:
        list of numpy arrays, one per trajectory file.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "trajectory_data*.hdf5")))
    if num_load > 0:
        files = files[:num_load]

    all_vals = []
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            if "train_vals" in f:
                all_vals.append(np.array(f["train_vals"]))
    return all_vals


def compute_trajectory_statistics(data_dir, num_load=-1):
    """Compute summary statistics on trajectory data.

    Supports both per-key HDF5 files (pose, speed, etc.) and legacy
    train_vals format. Extracts paddle position (with offset), speed,
    action, force, acceleration, and pose. Also compares against Box2D
    table dimensions.

    Args:
        data_dir: Directory containing trajectory_data*.hdf5 files.
        num_load: Max number of files to load. -1 for all.

    Returns:
        dict with keys for each field, each containing a dict of
        {min, max, mean, std, p5, p95} per dimension.
    """
    trajectories = load_all_trajectories(data_dir, load_images=False, num_load=num_load)
    if len(trajectories) == 0:
        print("No trajectory files found.")
        return {}

    traj_lengths = [t["pose"].shape[0] for t in trajectories.values()]

    # Concatenate per-key arrays across all trajectories
    pose = np.concatenate([t["pose"] for t in trajectories.values()], axis=0)
    speed = np.concatenate([t["speed"] for t in trajectories.values()], axis=0)
    force = np.concatenate([t["force"] for t in trajectories.values()], axis=0)
    acc = np.concatenate([t["acc"] for t in trajectories.values()], axis=0)
    desired_pose = np.concatenate([t["desired_pose"] for t in trajectories.values()], axis=0)

    # Derived fields
    paddle_pos = pose[:, :2].copy()
    paddle_pos[:, 0] += OFFSET_CONSTANT
    action = desired_pose[:, :2] - pose[:, :2]

    def _stats(arr, labels):
        result = {}
        for i, label in enumerate(labels):
            col = arr[:, i]
            p5, p95 = np.percentile(col, [5, 95])
            result[label] = {
                "min": float(col.min()),
                "max": float(col.max()),
                "mean": float(col.mean()),
                "std": float(col.std()),
                "p5": float(p5),
                "p95": float(p95),
            }
        return result

    stats = {
        "meta": {
            "total_timesteps": int(pose.shape[0]),
            "num_files": len(trajectories),
            "traj_length_min": int(min(traj_lengths)),
            "traj_length_max": int(max(traj_lengths)),
            "traj_length_mean": float(np.mean(traj_lengths)),
            "traj_length_median": float(np.median(traj_lengths)),
        },
        "paddle_position": _stats(paddle_pos, ["x", "y"]),
        "raw_pose_xy": _stats(pose[:, :2], ["x", "y"]),
        "pose_z": _stats(pose[:, 2:3], ["z"]),
        "paddle_speed_xy": _stats(speed[:, :2], ["vx", "vy"]),
        "paddle_speed_z": _stats(speed[:, 2:3], ["vz"]),
        "action": _stats(action, ["ax", "ay"]),
        "desired_pose_xy": _stats(desired_pose[:, :2], ["x", "y"]),
        "acceleration": _stats(acc, ["ax", "ay", "az"]),
        "force_xyz": _stats(force[:, :3], ["fx", "fy", "fz"]),
    }

    return stats


def print_trajectory_statistics(data_dir, num_load=-1):
    """Load trajectory data, compute statistics, and print a report.

    Args:
        data_dir: Directory containing trajectory_data*.hdf5 files.
        num_load: Max number of files to load. -1 for all.
    """
    stats = compute_trajectory_statistics(data_dir, num_load=num_load)
    if not stats:
        return

    meta = stats["meta"]
    print(f"Loading from: {data_dir}")
    print(f"Total timesteps: {meta['total_timesteps']}")
    print(f"Files: {meta['num_files']}")
    print(f"Trajectory lengths: min={meta['traj_length_min']}, max={meta['traj_length_max']}, "
          f"mean={meta['traj_length_mean']:.1f}, median={meta['traj_length_median']:.1f}")
    print()

    def _print_field(name, field_stats):
        print(f"--- {name} ---")
        for dim, s in field_stats.items():
            print(f"  {dim:>10s}: min={s['min']:11.5f}  max={s['max']:11.5f}  "
                  f"mean={s['mean']:10.5f}  std={s['std']:9.5f}  "
                  f"[p5={s['p5']:10.5f}, p95={s['p95']:10.5f}]")
        print()

    print("=" * 120)
    print("SUMMARY STATISTICS FOR REAL TRAJECTORY DATA")
    print("=" * 120)
    print()

    _print_field("PADDLE POSITION (pose[:2] + offset 1.2)", stats["paddle_position"])
    _print_field("RAW POSE XY (end-effector, no offset)", stats["raw_pose_xy"])
    _print_field("POSE Z (height)", stats["pose_z"])
    _print_field("PADDLE SPEED XY", stats["paddle_speed_xy"])
    _print_field("PADDLE SPEED Z", stats["paddle_speed_z"])
    _print_field("ACTION (desired_pose[:2] - pose[:2])", stats["action"])
    _print_field("DESIRED POSE XY", stats["desired_pose_xy"])
    _print_field("ACCELERATION XYZ", stats["acceleration"])
    _print_field("FORCE XYZ", stats["force_xyz"])

    print("--- PUCK POSITION ---")
    print("  Puck data is NOT in the raw train_vals columns.")
    print("  Puck position comes from separate state_trajectory_data*.hdf5 files")
    print("  (via image detection pipeline) or defaults to occluded.")
    print()

    # Box2D comparison
    half_w = BOX2D_TABLE_WIDTH / 2
    half_l = BOX2D_TABLE_LENGTH / 2
    pp = stats["paddle_position"]
    act = stats["action"]
    spd = stats["paddle_speed_xy"]

    print("=" * 120)
    print("BOX2D TABLE DIMENSIONS (from configs/baseline_configs/box2d/)")
    print("=" * 120)
    print(f"  Table length (Y-axis): {BOX2D_TABLE_LENGTH} m")
    print(f"  Table width  (X-axis): {BOX2D_TABLE_WIDTH} m")
    print(f"  Paddle radius: {BOX2D_PADDLE_RADIUS} m")
    print(f"  Puck radius:   {BOX2D_PUCK_RADIUS} m")
    print(f"  Table X bounds: [{-half_w:.4f}, {half_w:.4f}]")
    print(f"  Table Y bounds: [{-half_l:.4f}, {half_l:.4f}]")
    print()
    print(f"  {'':30s} {'Real Data':>25s}   {'Box2D Bounds':>25s}")
    print(f"  {'Paddle X':30s} [{pp['x']['min']:.5f}, {pp['x']['max']:.5f}]   [{-half_w:.5f}, {half_w:.5f}]")
    print(f"  {'Paddle Y':30s} [{pp['y']['min']:.5f}, {pp['y']['max']:.5f}]   [{-half_l:.5f}, {0.0:.5f}] (ego half)")
    print(f"  {'Action X':30s} [{act['ax']['min']:.5f}, {act['ax']['max']:.5f}]")
    print(f"  {'Action Y':30s} [{act['ay']['min']:.5f}, {act['ay']['max']:.5f}]")
    print(f"  {'Speed X':30s} [{spd['vx']['min']:.5f}, {spd['vx']['max']:.5f}]")
    print(f"  {'Speed Y':30s} [{spd['vy']['min']:.5f}, {spd['vy']['max']:.5f}]")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trajectory data statistics")
    parser.add_argument("--data-dir", type=str, default="/data2/air_hockey/air_hockey_state_data/datastor1/calebc/public/data/mouse/state_data_all_new/",
                        help="Directory containing trajectory_data*.hdf5 files")
    parser.add_argument("--num-load", type=int, default=5,
                        help="Max number of files to load (-1 for all)")
    args = parser.parse_args()

    print_trajectory_statistics(args.data_dir, num_load=args.num_load)
