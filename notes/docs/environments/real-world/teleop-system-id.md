# Teleop System-ID Trajectories

Mouse teleoperation recordings for sim-to-real system identification. Trajectories
are stored under `sysid/teleop/` in the split-schema HDF5 format.

## Recording trajectories

```bash
conda activate air
python scripts/real/teleoperate.py \
    --cfg configs/real_configs/mouse_config.yaml \
    --save-path sysid/teleop
```

Keys during recording:

- `y` — save trajectory and reset
- `q` — reset without saving
- `x` — exit

Output: `sysid/teleop/trajectory_dataN.hdf5` (N auto-increments from the highest
existing index in the output directory).

### Optional flags

| Flag | Effect |
|------|--------|
| `--policy-limits` | Constrain mouse to per-step policy magnitude |
| `--action-scale F` | Scale factor for `--policy-limits` (default 1.0) |
| `--legacy-schema` | Use old flat `train_vals` format instead of split-schema |

## Rendering segment GIFs

`render_teleop_segments.py` splits a trajectory into fixed-length intervals and
renders a GIF per interval, with frame numbers and timestamps overlaid.

```bash
PYTHONPATH=/home/pearl/air-hockey-rl \
python scripts/smooth_policy/visualize_demo/render_teleop_segments.py \
    sysid/teleop/trajectory_dataN.hdf5 \
    -o sysid/teleop/trajectory_dataN_segments \
    --interval 100
```

> The `PYTHONPATH` override is needed because the ROS `scripts` package on this
> machine shadows the repo's `scripts/` directory.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `input` | *(required)* | Path to split-schema HDF5 |
| `-o` / `--output-dir` | `<stem>_segments` next to input | Output root directory |
| `--interval` | `100` | Frames per segment |

### Output structure

Each segment directory contains a sliced HDF5 and a rendered GIF:

```
trajectory_dataN_segments/
├── frames_0_100/
│   ├── segment_0_100.hdf5
│   └── trajectory_visualization.gif
├── frames_100_200/
│   ├── segment_100_200.hdf5
│   └── trajectory_visualization.gif
...
└── frames_3500_3527/
    ├── segment_3500_3527.hdf5
    └── trajectory_visualization.gif
```

## HDF5 split-schema fields

| Dataset | Width | Description |
|---------|-------|-------------|
| `cur_time` | 1 | Unix timestamp |
| `tidx` | 1 | Trajectory / episode index |
| `i` | 1 | Step index within episode |
| `estop` | 1 | E-stop flag |
| `safety` | 1 | Safety flag |
| `pose` | 6 | Paddle pose (x, y, + 4 unused) |
| `speed` | 6 | Paddle velocity (vx, vy, + 4 unused) |
| `force` | 6 | Force (zeros for teleop) |
| `acc` | 3 | Acceleration (zeros for teleop) |
| `desired_pose` | 6 | Commanded target pose |
| `puck` | 3 | Puck position (x, y, occluded) |

These are the same fields used by the online TD3 real-world pipeline
(`async_td3_real.py`), so downstream tools (visualization, replay-in-sim) work
identically.

## Related scripts

| Script | Purpose |
|--------|---------|
| `scripts/real/teleoperate.py` | Record teleop trajectories |
| `scripts/smooth_policy/visualize_demo/render_teleop_segments.py` | Split + render segment GIFs |
| `scripts/smooth_policy/visualize_demo/visualize_real_trajectory_split.py` | Render a full split-schema HDF5 as one GIF |
| `scripts/smooth_policy/visualize_demo/replay_real_in_sim.py` | Side-by-side real vs sim replay |
