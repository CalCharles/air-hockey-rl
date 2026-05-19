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
python scripts/visualization/render_teleop_segments.py \
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
(entrypoint `extras/async_td3_real.py`; row format defined by
`_build_split_episode_row` in the runtime library `helper/real_td3_runtime.py`),
so downstream tools (visualization, replay-in-sim) work identically.

## Named category splitting

For system identification, trajectories are split into named motion categories
(e.g., side-to-side, up-and-down, circular, random) with specific frame ranges.
Each category gets its own folder with a full-range HDF5 plus 100-frame chunks,
each containing a trajectory GIF and a side-by-side sim-vs-real replay GIF.

### trajectory_data1 categories

Script: `scripts/visualization/split_teleop_categories.py` — **deleted in commit `13a050a` (2026-05-11)**. The data1 split is already on disk under `sysid/teleop/system_id/`; the script can be recovered with `git show 13a050a^:scripts/smooth_policy/visualize_demo/split_teleop_categories.py` if a re-split is ever needed.

Categories (frame ranges): `side_to_side_fast` (120–300), `side_to_side_slow` (310–650),
`side_to_side_dynamic` (675–1150), `up_and_down_slow` (1200–1525),
`up_and_down_fast` (1550–1725), `up_and_down_dynamic` (1780–2050),
`circle_fast` (2100–2500), `circle_slow` (2575–2850), `random` (3150–3350).

### trajectory_data3 categories (with sim-vs-real replay)

Script: `scripts/visualization/split_teleop_categories_data3.py` (restored 2026-05-19 from `13a050a^`; patched to import from `scripts.visualization.*` modules and to load `configs/new_juggle/sysid_best_params_hist2.yaml` as the replay sim config).

```bash
PYTHONPATH=/home/pearl/air-hockey-rl \
python scripts/visualization/split_teleop_categories_data3.py
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
config: `configs/new_juggle/sysid_best_params.yaml`.

### Key findings

1. **Higher Kp is better**: all searches push toward the upper end of the grid. The canonical config's Kp=5000 significantly under-tracks.
2. **Lower Kd is better**: Kd=50–100 consistently beats Kd=200+. Less derivative damping allows the paddle to respond more aggressively to position error.
3. **Density ≈ 3000 matches reality**: real paddle inertia is much closer to `density=3000` (as used in the sysid canonical config) than the legacy `density=1000`.
4. **Ki ≈ 0**: integral gain provides negligible benefit and can cause oscillation in some segments.

### PID test script

`scripts/test_pid_controller.py` — **deleted in commit `13a050a` (2026-05-11)**. Standalone test that compared the legacy force-based controller vs the PID controller on a step input. Recoverable with `git show 13a050a^:scripts/test_pid_controller.py` if needed (not on the critical path for sysid).

### Visualization of best config

`scripts/sysid/render_best_config_all.py` renders side-by-side sim-vs-real GIFs
for all segments under `$AIRHOCKEY_SYSID_DATA_DIR` (default
`sysid/teleop/system_id3/`) using the constants set inside the script. Output:
`$AIRHOCKEY_SYSID_DATA_DIR/visualizations_best_config/`.

## Related scripts

| Script | Purpose |
|--------|---------|
| `scripts/real/teleoperate.py` | Record teleop trajectories |
| `scripts/visualization/render_teleop_segments.py` | Split + render segment GIFs (fixed interval) |
| `scripts/visualization/split_teleop_categories.py` | Named category split for data1 — *deleted, recover from `13a050a^`* |
| `scripts/visualization/split_teleop_categories_data3.py` | Named category split + replay for data3 |
| `scripts/visualization/visualize_real_trajectory_split.py` | Render a full split-schema HDF5 as one GIF |
| `scripts/visualization/replay_real_in_sim.py` | Core replay primitive — side-by-side real vs sim, plus `replay_errors_windowed` used by every grid search |
| `scripts/sysid/grid_search_*.py` | PID / density / Ki grid searches (5 protocols). Default data dir / config overridable via `AIRHOCKEY_SYSID_DATA_DIR`, `AIRHOCKEY_SYSID_CONFIG`, `AIRHOCKEY_SYSID_SUBSET` (see [`_sysid_paths.py`](../../../../scripts/sysid/_sysid_paths.py)). |
| `scripts/sysid/render_best_config_all.py` | Render sim-vs-real GIFs for the chosen best config |
| `scripts/test_pid_controller.py` | PID vs legacy controller comparison — *deleted, recover from `13a050a^`* |

## Re-running sysid for a new sim variant (e.g., `hist_len: 4` action smoothing)

The sim's commanded-pose smoothing window is `simulator_params.hist_len` — applied in `_filter_update` in [`airhockey/sims/airhockey_box2d.py`](../../../../airhockey/sims/airhockey_box2d.py) (mean of `(desired_i − current_i)` over the last `hist_len` steps, added to the current pose). The real-side env applies an equivalent `filter_update` in [`airhockey/sims/air_hockey_real.py`](../../../../airhockey/sims/air_hockey_real.py). Changing `hist_len` changes the effective velocity command, which shifts the PID/density optimum — the canonical `Kp=9000, Kd=50, density=3000` in `sysid_best_params_hist2.yaml` was fit at `hist_len: 2` and is **not** valid for other values.

This section documents the full protocol for re-running the paddle sysid against a new `hist_len` (worked example: `hist_len: 4`). It assumes data1/data3 trajectories were collected under a *different* `hist_len` and therefore must be re-recorded.

All commands below are copy-paste-ready. Run from the repo root (`/home/pearl/air-hockey-rl`) in the `air` conda env. The grid-search scripts read three env vars — `AIRHOCKEY_SYSID_DATA_DIR`, `AIRHOCKEY_SYSID_CONFIG`, `AIRHOCKEY_SYSID_SUBSET` — so a new variant requires zero script edits.

### 0. Prerequisites — draft the variant config and confirm the real side

```bash
conda activate air
cd /home/pearl/air-hockey-rl

# Draft the new sim config (PID/density values will get overwritten by the sweep).
cp configs/new_juggle/sysid_best_params_hist2.yaml \
   configs/new_juggle/sysid_best_params_hist4.yaml

# Then edit configs/new_juggle/sysid_best_params_hist4.yaml:
#   simulator_params.hist_len: 4    # was 2
# Leave pid_kp / pid_kd / paddle_density at the hist2 best values; the sweep replaces them.
```

Verify the **real side** also uses `hist_len: 4` before recording — same `hist_len` is read by `air_hockey_real.py` (line 578). Inspect `configs/real_configs/mouse_config.yaml` and any args file used by `scripts/real/teleoperate.py`; if `hist_len` is set elsewhere in the launch path, set it to `4` there too. Recording with mismatched `hist_len` between sim config and real env invalidates the sweep.

Hardware checklist: UR5 powered & calibrated, mouse plugged in, camera homography current, same paddle as the original sysid.

### 1. Record new teleop trajectories

```bash
python scripts/real/teleoperate.py \
    --cfg configs/real_configs/mouse_config.yaml \
    --save-path sysid/teleop_hist4
```

Keys: `y` save+reset, `q` reset without saving, `x` exit.

Aim for ≈3500 frames covering, in order with short pauses between each: `side_to_side_fast` → `side_to_side_slow` → `side_to_side_dynamic` → `up_and_down_fast` → `up_and_down_slow` → `up_and_down_dynamic` → `diagonal_fast` → `diagonal_dynamic` → `circle_fast` → `circle_slow` → `random`. Frame ranges don't have to match the original data3 — you'll annotate them in step 3.

Output: `sysid/teleop_hist4/trajectory_data1.hdf5` (N auto-increments).

### 2. Verify the recording (optional)

```bash
PYTHONPATH=$(pwd) \
python scripts/visualization/visualize_real_trajectory_split.py \
    sysid/teleop_hist4/trajectory_data1.hdf5
```

Look for: noticeably smoother paddle motion vs `hist_len: 2` (4-step moving average lags more), all categories present, no e-stop spikes.

### 3. Annotate category frame ranges

```bash
PYTHONPATH=$(pwd) \
python scripts/visualization/render_teleop_segments.py \
    sysid/teleop_hist4/trajectory_data1.hdf5 \
    -o sysid/teleop_hist4/trajectory_data1_segments --interval 100
```

Open each segment GIF and write down start/end frames per category. Keep the 11 category names from data3 (`side_to_side_fast`, `side_to_side_slow`, `side_to_side_dynamic`, `up_and_down_fast`, `up_and_down_slow`, `up_and_down_dynamic`, `diagonal_fast`, `diagonal_dynamic`, `circle_fast`, `circle_slow`, `random`) — the downstream subset list uses these names verbatim.

### 4. Split into named categories

Make a variant of `split_teleop_categories_data3.py` for the new recording:

```bash
cp scripts/visualization/split_teleop_categories_data3.py \
   scripts/visualization/split_teleop_categories_hist4.py
```

Edit the four constants at the top of the new file:

```python
INPUT_PATH = _REPO_ROOT / "sysid" / "teleop_hist4" / "trajectory_data1.hdf5"
OUTPUT_ROOT = _REPO_ROOT / "sysid" / "teleop_hist4" / "system_id"
SIM_CONFIG = str(_REPO_ROOT / "configs/new_juggle/sysid_best_params_hist4.yaml")
CATEGORIES = [
    ("side_to_side_fast",    <start>, <end>),
    ...  # use the ranges from step 3
]
```

Then run:

```bash
PYTHONPATH=$(pwd) \
python scripts/visualization/split_teleop_categories_hist4.py
```

Output: `sysid/teleop_hist4/system_id/<category>/frames_<a>_<b>/{segment_<a>_<b>.hdf5, trajectory_visualization.gif, sim_vs_real.gif, sim_vs_real.json}`. Skim a few `sim_vs_real.gif`s — they replay against the hist2 PID gains, so divergence is expected; that's the gap the grid search will close.

### 5. Write the subset JSON

Pick the **first 100-frame chunk** of each of the 8 representative categories and write the paths (relative to `sysid/teleop_hist4/system_id/`) into a JSON list:

```bash
cat > sysid/teleop_hist4/system_id/subset.json <<'EOF'
[
  "circle_fast/frames_<a>_<b>/segment_<a>_<b>.hdf5",
  "circle_slow/frames_<a>_<b>/segment_<a>_<b>.hdf5",
  "side_to_side_dynamic/frames_<a>_<b>/segment_<a>_<b>.hdf5",
  "side_to_side_slow/frames_<a>_<b>/segment_<a>_<b>.hdf5",
  "up_and_down_dynamic/frames_<a>_<b>/segment_<a>_<b>.hdf5",
  "up_and_down_slow/frames_<a>_<b>/segment_<a>_<b>.hdf5",
  "diagonal_fast/frames_<a>_<b>/segment_<a>_<b>.hdf5",
  "random/frames_<a>_<b>/segment_<a>_<b>.hdf5"
]
EOF
```

Replace each `<a>_<b>` with the actual frame numbers — these will match the chunk names that `split_teleop_categories_hist4.py` wrote to disk.

### 6. Run the grid searches

Export the three env vars once per shell — every script under `scripts/sysid/` picks them up:

```bash
export AIRHOCKEY_SYSID_DATA_DIR="$(pwd)/sysid/teleop_hist4/system_id"
export AIRHOCKEY_SYSID_CONFIG="$(pwd)/configs/new_juggle/sysid_best_params_hist4.yaml"
export AIRHOCKEY_SYSID_SUBSET="$(pwd)/sysid/teleop_hist4/system_id/subset.json"
```

Each grid search uses `replay_errors_windowed` from `scripts/visualization/replay_real_in_sim.py` — sim resets to the real paddle state every 10 frames so errors don't compound; `park_puck=True` (inside the windowed scripts) isolates paddle tracking from puck-interaction noise. Output JSONs land under `$AIRHOCKEY_SYSID_DATA_DIR/grid_search_results_*/`.

Run in order (each writes its own results directory; the next reads from JSON if it needs the previous best):

```bash
# 6a. 3D coarse — establish the basin.
#     Grid: pid_kp 2000–8000 step 1000, pid_kd {200,600,1000}, paddle_density 1000–3000 step 500.
python scripts/sysid/grid_search_pid_density.py

# 6b. 3D fine, puck parked, windowed-10 — the protocol that drove err to 0.024 at hist_len=2.
#     Grid: pid_kp {5500,6500,7500,8500,9000}, pid_kd {100,300,400}, paddle_density {1750,2250,2750,3250}.
python scripts/sysid/grid_search_pid_density_fine_windowed.py

# 6c. Finer windowed-10 — only if (6b) lands at a grid edge.
#     Grid: pid_kp 7000–10000, pid_kd {50,75,100,125,150}, paddle_density {2250,2500,2750,3000,3250}.
python scripts/sysid/grid_search_pid_density_fine.py

# 6d. Ki sweep — fix best Kp/Kd/density from (6c), sweep pid_ki ∈ {0,10,25,50,100,150,200,300,500}.
#     Expect Ki≈0 to win, but verify.
python scripts/sysid/grid_search_ki.py
```

Each script prints progress to stdout and writes `*_results.json` under `$AIRHOCKEY_SYSID_DATA_DIR/grid_search_results_*/`. Sort by `mean_paddle_err` (averaged across the 8 segments) and take the top combo.

Each results JSON already contains a `"best"` key recording the winning combo. Quick peek:

```bash
python -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['best'], indent=2))" \
    sysid/teleop_hist4/system_id/grid_search_results_3d_fine_windowed_10/grid_search_3d_fine_windowed_results.json
```

### 7. Render the chosen best config

Edit the four `BEST_*` constants in `scripts/sysid/render_best_config_all.py` to the values you selected in step 6, then:

```bash
python scripts/sysid/render_best_config_all.py
```

Output: `$AIRHOCKEY_SYSID_DATA_DIR/visualizations_best_config/<category>/frames_*/sim_vs_real.gif`. Spot-check that the sim paddle now tracks the real paddle within the per-segment tolerance the grid reported. If a category looks wrong, the grid's best combo overfit the other 7 — go back to step 6 with a narrower grid around the winning combo and add that category to the subset.

### 8. Commit the new sysid config

Update `configs/new_juggle/sysid_best_params_hist4.yaml` with the winning `pid_kp`, `pid_kd`, `pid_ki`, `paddle_density`. Header comment: cite the grid-search JSON path so future readers can audit the fit. Note that this config is paired with `hist_len: 4` and is **not** interchangeable with `sysid_best_params_hist2.yaml`.

The puck parameters (`gravity`, `puck_damping`) don't depend on `hist_len` and can be carried over from `sysid_best_params.yaml`. To re-verify them, see [`puck-system-id.md`](puck-system-id.md).

### Smoke-test the wiring before recording

Sanity-check that the env-var override actually swaps paths, before you tie up the robot for 30 minutes of teleop:

```bash
AIRHOCKEY_SYSID_DATA_DIR=/tmp/fake_data \
AIRHOCKEY_SYSID_CONFIG=/tmp/fake_cfg.yaml \
python -c "from scripts.sysid._sysid_paths import SYSID_DIR, DEFAULT_CONFIG; print(SYSID_DIR); print(DEFAULT_CONFIG)"
# Expect: /tmp/fake_data
#         /tmp/fake_cfg.yaml
```

### Pitfalls

- **Real-side `hist_len` mismatch**: if `scripts/real/teleoperate.py` runs with `hist_len: 2` while the sim grid search is configured with `hist_len: 4`, the recorded `desired_pose` stream reflects the wrong filter and the fit will land on bad values. Verify both sides before recording.
- **Reusing data3 segments**: the existing 8-segment splits under `sysid/teleop/system_id3/` were recorded at `hist_len: 2`. Pointing `AIRHOCKEY_SYSID_DATA_DIR` at `system_id3` while running with `hist_len: 4` in the config does **not** cancel out — the recorded `desired_pose` is what the *real* robot received at the time, with whatever smoother was active then. You must re-record.
- **Forgetting `AIRHOCKEY_SYSID_SUBSET`**: with `AIRHOCKEY_SYSID_DATA_DIR` pointed at a new directory but no `AIRHOCKEY_SYSID_SUBSET`, the scripts fall back to the hardcoded data3-hist2 paths (e.g. `circle_fast/frames_2235_2335/segment_2235_2335.hdf5`) — which don't exist in the new directory and crash with `FileNotFoundError`. The error message will name the missing path; that's the symptom to look for.
- **`replay_real_in_sim.py` was deleted from `scripts/smooth_policy/visualize_demo/` and restored to `scripts/visualization/replay_real_in_sim.py` on 2026-05-19**. The grid scripts also moved from `sysid/teleop/system_id3/` to `scripts/sysid/` on the same date. Old code or docs pointing at the deleted paths should be updated to `scripts.visualization.replay_real_in_sim` / `scripts/sysid/*.py`.
- **Heavy puck noise during the grid search masks paddle errors**. `load_sim_config(config_path, enable_noise=False)` (already what every grid script does) disables `puck_noise`, occlusions, observation/action delay, and attenuation. Don't turn them back on for the sweep.
