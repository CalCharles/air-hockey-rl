"""Convert real trajectory HDF5 data into the observation format
used by the system identification pipeline.

The sim observation vector is:
    [paddle_x, paddle_y, paddle_vx, paddle_vy, puck_x, puck_y, puck_vx, puck_vy]

Coordinate note: paddle positions use the 'paddle' key (pose + offset 1.2)
which is in base coordinates compatible with base_coord_to_box2d.
Puck positions from the 'puck' key are already in the same base frame.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_loading import (
    load_trajectory,
    BOX2D_PADDLE_RADIUS,
    BOX2D_PUCK_RADIUS,
    BOX2D_TABLE_WIDTH,
)


def real_trajectory_to_sim_format(traj, dt=0.05):
    """Convert a loaded real trajectory dict into sim-compatible format.

    Args:
        traj: dict from load_trajectory() with keys 'paddle', 'speed', 'puck', 'action'.
        dt: timestep duration (seconds) for finite difference velocity.

    Returns:
        dict with keys:
            observations: (N, 8) array
            actions: (N, 2) action deltas
            hits_array: list of 0/1 hit indicators
    """
    # Paddle: use 'paddle' key which has offset applied (base coordinates)
    paddle_pos = traj["paddle"][:, :2]   # (N, 2) base coords
    paddle_vel = traj["speed"][:, :2]    # (N, 2) velocity (same in base frame)

    # Puck position and occlusion flag (already in base coords)
    puck_pos = traj["puck"][:, :2]            # (N, 2)
    puck_occluded = traj["puck"][:, 2] > 0.5  # (N,)

    # Puck velocity via finite differences
    N = len(puck_pos)
    puck_vel = np.zeros((N, 2))
    if N > 1:
        puck_vel[1:] = (puck_pos[1:] - puck_pos[:-1]) / dt
        puck_vel[0] = puck_vel[1]
    # Zero out velocity when puck is occluded
    puck_vel[puck_occluded] = 0.0

    # 8D observation: [paddle_x, paddle_y, paddle_vx, paddle_vy,
    #                  puck_x, puck_y, puck_vx, puck_vy]
    observations = np.column_stack([
        paddle_pos,   # cols 0-1
        paddle_vel,   # cols 2-3
        puck_pos,     # cols 4-5
        puck_vel,     # cols 6-7
    ])

    # Actions (relative deltas, no offset needed)
    actions = traj["action"][:, :2]  # (N, 2)

    # Hit detection: paddle-puck distance < threshold OR wall collision
    hit_threshold = BOX2D_PADDLE_RADIUS + BOX2D_PUCK_RADIUS + 0.02
    distances = np.linalg.norm(puck_pos - paddle_pos, axis=1)
    wall_distance_boundaries = (BOX2D_TABLE_WIDTH / 2) - BOX2D_PUCK_RADIUS
    wall_hit_threshold = 0.01
    paddle_hit = distances < hit_threshold
    wall_hit = np.abs(puck_pos[:, 1]) + wall_hit_threshold >= wall_distance_boundaries
    hits_array = (paddle_hit | wall_hit).astype(int).tolist()

    # Mark occluded timesteps as hits (conservative: avoid sampling them)
    for i in range(N):
        if puck_occluded[i]:
            hits_array[i] = 1

    return {
        "observations": observations,
        "actions": actions,
        "hits_array": hits_array,
    }


def load_real_trajectory_for_sysid(data_dir, filename, dt=0.05):
    """Load and convert a real trajectory file for system identification.

    Args:
        data_dir: Directory containing trajectory HDF5 files.
        filename: HDF5 filename (e.g. 'trajectory_data100.hdf5').
        dt: Timestep duration for puck velocity computation.

    Returns:
        dict formatted for CMAPlanner: {0: {observations, actions, hits_array}}
    """
    traj = load_trajectory(data_dir, filename, load_images=False)
    converted = real_trajectory_to_sim_format(traj, dt=dt)
    # Wrap in episode-keyed dict expected by CMAPlanner
    return {0: converted}


def load_multiple_real_trajectories_for_sysid(data_dir, num_load=-1, dt=0.05):
    """Load and convert multiple real trajectory files for system identification.

    Loads all trajectory HDF5 files from the directory, converts each to
    sim-compatible format, and returns a list of episode dicts. This enables
    sampling segments across many trajectories to avoid overfitting to a
    single episode.

    Args:
        data_dir: Directory containing trajectory_data*.hdf5 files.
        num_load: Max number of trajectories to load. -1 for all.
        dt: Timestep duration for puck velocity computation.

    Returns:
        list of dicts, each with keys {observations, actions, hits_array}.
    """
    from data_loading import load_all_trajectories

    raw_trajectories = load_all_trajectories(data_dir, load_images=False, num_load=num_load)
    print(f"Loaded {len(raw_trajectories)} raw trajectory files from {data_dir}")

    episodes = []
    skipped = 0
    for tidx in sorted(raw_trajectories.keys()):
        try:
            converted = real_trajectory_to_sim_format(raw_trajectories[tidx], dt=dt)
            if len(converted["observations"]) < 10:
                skipped += 1
                continue
            episodes.append(converted)
        except Exception as e:
            print(f"  Skipping trajectory {tidx}: {e}")
            skipped += 1

    print(f"Converted {len(episodes)} trajectories ({skipped} skipped)")
    return episodes
