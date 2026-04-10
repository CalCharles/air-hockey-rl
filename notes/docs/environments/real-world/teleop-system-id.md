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

## Named category splitting

For system identification, trajectories are split into named motion categories
(e.g., side-to-side, up-and-down, circular, random) with specific frame ranges.
Each category gets its own folder with a full-range HDF5 plus 100-frame chunks,
each containing a trajectory GIF and a side-by-side sim-vs-real replay GIF.

### trajectory_data1 categories

Script: `scripts/smooth_policy/visualize_demo/split_teleop_categories.py`

```bash
PYTHONPATH=/home/pearl/air-hockey-rl \
python scripts/smooth_policy/visualize_demo/split_teleop_categories.py \
    sysid/teleop/trajectory_data1.hdf5 \
    -o sysid/teleop/system_id
```

Categories: `side_to_side_fast` (120–300), `side_to_side_slow` (310–650),
`side_to_side_dynamic` (675–1150), `up_and_down_slow` (1200–1525),
`up_and_down_fast` (1550–1725), `up_and_down_dynamic` (1780–2050),
`circle_fast` (2100–2500), `circle_slow` (2575–2850), `random` (3150–3350).

### trajectory_data3 categories (with sim-vs-real replay)

Script: `scripts/smooth_policy/visualize_demo/split_teleop_categories_data3.py`

```bash
PYTHONPATH=/home/pearl/air-hockey-rl \
python scripts/smooth_policy/visualize_demo/split_teleop_categories_data3.py
```

Output: `sysid/teleop/system_id3/`

Categories: `side_to_side_fast` (101–300), `side_to_side_slow` (325–600),
`side_to_side_dynamic` (650–900), `up_and_down_fast` (960–1200),
`up_and_down_slow` (1275–1550), `up_and_down_dynamic` (1590–1760),
`diagonal_fast` (1790–2000), `diagonal_dynamic` (2025–2200),
`circle_fast` (2235–2650), `circle_slow` (2680–3170), `random` (3225–3525).

Unlike data1, the data3 script also renders side-by-side sim-vs-real replay GIFs
(`sim_vs_real.gif` + `sim_vs_real.json` metrics) for every 100-frame chunk, with
the sim **reset fresh at each chunk boundary** (not carried forward).

### Category output structure

```
system_id3/
├── side_to_side_fast/
│   ├── side_to_side_fast.hdf5          (full range)
│   ├── frames_101_201/
│   │   ├── segment_101_201.hdf5
│   │   ├── trajectory_visualization.gif
│   │   ├── sim_vs_real.gif             (data3 only)
│   │   └── sim_vs_real.json            (data3 only)
│   └── frames_201_300/
│       └── ...
├── circle_slow/
│   └── ...
```

## PID controller tuning (system identification)

The Box2D simulator uses a PID controller for paddle motion when
`use_pid: true` is set in the sim config. PID gains directly affect how
closely the sim paddle tracks action commands, and thus how well the sim
matches real-world paddle dynamics.

### Current PID parameters (in canonical configs)

| Parameter | Standard config | Heavy config |
|-----------|----------------|--------------|
| `pid_kp` | 5000 | 5000 |
| `pid_kd` | 200 | 200 |
| `pid_ki` | 0.0 | 0.0 |

Configs: `scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params.yaml`
(and `…_heavy.yaml` for heavier paddle/puck densities).

### Grid search

A grid search over `pid_kp` (2000–6000) and `pid_kd` (200–1000) was run using
8 representative 100-frame trajectory segments from data3 categories:

- `circle_fast`, `circle_slow`, `side_to_side_dynamic`, `side_to_side_slow`,
  `up_and_down_dynamic`, `up_and_down_slow`, `diagonal_fast`, `random`

Results: `sysid/teleop/system_id3/grid_search_results/`
- `grid_search_results.json` — full results (45 Kp/Kd combos)
- `paddle_error_heatmap.png` — paddle tracking error heatmap
- `puck_error_heatmap.png` — puck tracking error heatmap

Best found: **Kp=6000, Kd=200** (lowest mean paddle error).

### PID test script

`scripts/test_pid_controller.py` — standalone test comparing legacy force-based
controller vs PID controller on a step-input scenario. Plots position, velocity,
acceleration, and jerk.

## Related scripts

| Script | Purpose |
|--------|---------|
| `scripts/real/teleoperate.py` | Record teleop trajectories |
| `scripts/smooth_policy/visualize_demo/render_teleop_segments.py` | Split + render segment GIFs (fixed interval) |
| `scripts/smooth_policy/visualize_demo/split_teleop_categories.py` | Named category split for data1 |
| `scripts/smooth_policy/visualize_demo/split_teleop_categories_data3.py` | Named category split + replay for data3 |
| `scripts/smooth_policy/visualize_demo/visualize_real_trajectory_split.py` | Render a full split-schema HDF5 as one GIF |
| `scripts/smooth_policy/visualize_demo/replay_real_in_sim.py` | Side-by-side real vs sim replay |
| `scripts/test_pid_controller.py` | PID vs legacy controller comparison |
