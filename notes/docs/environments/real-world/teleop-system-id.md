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
(entrypoint `async_td3_real_modular.py`; row format defined by
`_build_split_episode_row` in the shared library `async_td3_real.py`),
so downstream tools (visualization, replay-in-sim) work identically.

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

## Paddle system identification

The Box2D simulator uses a PID controller for paddle motion when
`use_pid: true` is set in the sim config. PID gains and paddle density directly
affect how closely the sim paddle tracks action commands, and thus how well the
sim matches real-world paddle dynamics.

All grid searches used the same 8 representative 100-frame trajectory segments
from data3 categories: `circle_fast`, `circle_slow`, `side_to_side_dynamic`,
`side_to_side_slow`, `up_and_down_dynamic`, `up_and_down_slow`, `diagonal_fast`,
`random`.

### Grid search progression

Multiple rounds of grid search were run, each expanding or refining the
parameter space. All results live under `sysid/teleop/system_id3/`.

#### 1. 2D PID grid (initial)

Grid: `pid_kp` 2000–6000 (step 500), `pid_kd` 200–1000 (step 200). Fixed density.

Results: `grid_search_results/grid_search_results.json`
Best: **Kp=6000, Kd=200**.

#### 2. 3D coarse: PID + paddle_density

Grid: `pid_kp` 2000–8000 (step 1000), `pid_kd` {200, 600, 1000}, `paddle_density` 1000–3000 (step 500). Puck active.

Results: `grid_search_results_3d/grid_search_3d_results.json`
Best: **Kp=8000, Kd=200, density=3000** (paddle err 0.069).

#### 3. 3D fine: PID + density (puck active)

Grid: `pid_kp` {5500, 6500, 7500, 8500, 9000}, `pid_kd` {100, 300, 400}, `paddle_density` {1750, 2250, 2750, 3250}. Puck active.

Results: `grid_search_results_3d_fine/grid_search_3d_fine_results.json`
Best: **Kp=7500, Kd=100, density=2750** (paddle err 0.069).

#### 4. 3D fine: puck parked

Same grid as (3); puck parked at (-0.9, 0.0) to isolate paddle tracking.

Results: `grid_search_results_3d_fine_no_puck/grid_search_3d_fine_no_puck_results.json`
Best: **Kp=9000, Kd=100, density=3250** (paddle err 0.069).

#### 5. Windowed replay (reset every 20 frames, puck parked)

Sim resets to real state every 20 frames, reducing compounding drift.

Results: `grid_search_results_3d_fine_windowed/grid_search_3d_fine_windowed_results.json`
Best: **Kp=8500, Kd=100, density=2750** (paddle err 0.042).

#### 6. Windowed replay (reset every 10 frames, puck parked)

Results: `grid_search_results_3d_fine_windowed_10/grid_search_3d_fine_windowed_results.json`
Best: **Kp=8500, Kd=100, density=2750** (paddle err 0.024).

#### 7. Finer grid, windowed-10, puck parked

Grid: `pid_kp` 7000–10000 (incl. 9500, 10000), `pid_kd` {50, 75, 100, 125, 150}, `paddle_density` {2250, 2500, 2750, 3000, 3250}.

Results: `grid_search_results_3d_finer_windowed_10/grid_search_3d_fine_windowed_results.json`
Best: **Kp=10000, Kd=50, density=3250** (paddle err 0.024).

#### 8. Ki sweep (full segment, puck active)

Fixed Kp=7500, Kd=100, density=2750. Ki ∈ {0, 10, 25, 50, 100, 150, 200, 300, 500}.

Results: `grid_search_results_ki/ki_sweep_results.json`
Best: **Ki=0** (for full-segment replay with puck).

#### 9. Ki sweep (windowed-10, puck parked)

Fixed Kp=9000, Kd=50, density=3000. Same Ki values.

Results: `grid_search_results_ki_windowed_10/ki_sweep_results.json`
Best: **Ki=500** (paddle err 0.023), but improvement over Ki=0 is marginal.

### Summary of best parameters

| Protocol | Kp | Kd | Ki | Density | Paddle err |
|----------|-----|-----|-----|---------|------------|
| 2D PID only | 6000 | 200 | 0 | (fixed) | — |
| 3D fine, puck active | 7500 | 100 | 0 | 2750 | 0.069 |
| 3D fine, puck parked | 9000 | 100 | 0 | 3250 | 0.069 |
| Windowed-10, puck parked | 8500 | 100 | 0 | 2750 | 0.024 |
| Finer windowed-10 | 10000 | 50 | 0 | 3250 | 0.024 |
| **Chosen practical best** | **9000** | **50** | **0** | **3000** | — |

The **chosen practical best** (Kp=9000, Kd=50, Ki=0, density=3000) is used by
`render_best_config_all.py` for visualization and is captured in the sysid
config: `scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params.yaml`.

### Key findings

1. **Higher Kp is better**: all searches push toward the upper end of the grid. The canonical config's Kp=5000 significantly under-tracks.
2. **Lower Kd is better**: Kd=50–100 consistently beats Kd=200+. Less derivative damping allows the paddle to respond more aggressively to position error.
3. **Density ≈ 3000 matches reality**: real paddle inertia is much closer to `density=3000` (as used in the sysid canonical config) than the legacy `density=1000`.
4. **Ki ≈ 0**: integral gain provides negligible benefit and can cause oscillation in some segments.

### PID test script

`scripts/test_pid_controller.py` — standalone test comparing legacy force-based
controller vs PID controller on a step-input scenario. Plots position, velocity,
acceleration, and jerk.

### Visualization of best config

`sysid/teleop/system_id3/render_best_config_all.py` renders side-by-side
sim-vs-real GIFs for all segments using the chosen best config. Output:
`sysid/teleop/system_id3/visualizations_best_config/`.

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
