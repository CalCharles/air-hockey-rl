# System Identification & Real Data Pipeline

Full pipeline: load real robot trajectories → compute inflection logs → generate GIFs → run two-stage CMA-ES system identification.

```
HDF5 trajectory files → data_loading.py (load & inspect)
                       → puck_inflection.py (peaks, contacts → JSON logs)
                       → generate_*_gifs.py (visualization)
                       → run_system_id.py (CMA-ES: Stage 1 paddle, Stage 2 puck)
```

---

## Step 1 — Load & Inspect Trajectories

**File:** `scripts/real_data_transforms/data_loading.py`

```bash
python scripts/real_data_transforms/data_loading.py --data-dir /path/to/data --num-load 10
```

```python
from scripts.real_data_transforms.data_loading import load_all_trajectories, load_trajectory
trajs = load_all_trajectories("/path/to/data", load_images=False, num_load=-1)
traj = load_trajectory("/path/to/data/trajectory_data100.hdf5", load_images=False)
```

---

## Step 2 — Compute Inflection Logs

**File:** `scripts/real_data_transforms/puck_inflection.py`

Pre-analyzes puck events per trajectory: peaks (x-minima, vx ≈ 0), valleys, contacts (paddle hits / wall collisions), approach intervals, free-flight segments. **Required before batch GIFs or puck sys-ID.**

```bash
# Batch-process all trajectories (run this first!)
python scripts/real_data_transforms/puck_inflection.py --batch --max-traj-idx 864

# Single trajectory
python scripts/real_data_transforms/puck_inflection.py --traj-idx 100

# Force-regenerate
python scripts/real_data_transforms/puck_inflection.py --batch --force
```

**Output:** `scripts/real_data_transforms/logs/inflection_<idx>.json` with keys: `peaks`, `valleys`, `contacts`, `approach_intervals`, `free_flight_segments`, `paddle_to_paddle_intervals`.

---

## Step 3 — Generate GIFs

### `trajectory_to_gif.py` (central GIF tool)

| Subcommand | Description | Key Args |
|------------|-------------|----------|
| `render` | Real trajectory GIF | `--traj-idx`, `--show-velocity` |
| `rollout` | Sim rollout GIF | `--traj-idx`, `--cfg` |
| `sync` | Side-by-side real vs sim | `--traj-idx`, `--cfg` |
| `inflection` | GIF with peak/contact markers | `--traj-idx`, `--log-path` |
| `parabolic` | Fitted velocity curves | `--traj-idx`, `--x-degree`, `--y-degree` |
| `puck_segments` | Short free-flight segments | `--cfg`, `--traj-indices`, `--max-segments` |

Common args: `--data-dir`, `--output-dir` (default `gifs/`), `--fps` (default 20).

```bash
python scripts/real_data_transforms/trajectory_to_gif.py render --traj-idx 100
python scripts/real_data_transforms/trajectory_to_gif.py sync --traj-idx 100 \
    --cfg scripts/domain_adaptation/realworld_paddle_config/estimate.yaml
```

### Batch GIF scripts (require inflection logs + sim config)

| Script | What it generates |
|--------|-------------------|
| `generate_approach_gifs.py` | Peak → paddle contact intervals (stratified by duration) |
| `generate_contact_gifs.py` | Windows around paddle hits (`--pre-steps`, `--post-steps`) |
| `generate_paddle_gifs.py` | Longer-horizon paddle segments (`--segment-length`) |

All accept `--cfg`, `--output-dir`, `--num-gifs`. Each interval/event produces `real.gif`, `sim_rollout.gif`, `sync.gif`.

---

## Step 4 — System Identification

Two-stage CMA-ES to fit Box2D physics to real data.

- **Stage 1 — Paddle:** Optimizes `force_scaling`, `paddle_damping`, `pid_kp`, `pid_kd` (sim vs real paddle positions)
- **Stage 2 — Puck:** With paddle params fixed, optimizes `gravity`, `puck_damping` (sim vs real puck free-flight)

### Running via `run_system_id.py`

```bash
# Full pipeline
python scripts/domain_adaptation/system_identification/run_system_id.py \
    --data-dir /path/to/data --paddle-num-iterations 100 --puck-num-iterations 80

# Paddle only
python scripts/domain_adaptation/system_identification/run_system_id.py \
    --data-dir /path/to/data --skip-puck

# Puck only (with pre-computed paddle params)
python scripts/domain_adaptation/system_identification/run_system_id.py \
    --data-dir /path/to/data --skip-paddle \
    --paddle-params-path .../optimal_params.yaml --puck-num-iterations 100
```

**Pipeline control:** `--skip-paddle`, `--skip-puck`, `--paddle-params-path`

**Paddle args** (prefix `--paddle-*`): `--paddle-sys-id-path`, `--paddle-comp-type` (default `posl2`), `--paddle-num-iterations` (100), `--paddle-num-population` (14), `--paddle-variance` (0.25), `--paddle-traj-length` (8), `--paddle-run-dir`, `--paddle-run-id`

**Puck args** (prefix `--puck-*`): `--puck-sys-id-path`, `--puck-interval-mode` (`peak_start`/`approach`/`paddle_to_paddle`), `--puck-velocity-mode` (`parabolic`/`finite_diff`), `--puck-comp-type` (`l2`), `--puck-num-iterations` (100), `--puck-num-population` (6), `--puck-variance` (0.3), `--puck-traj-length` (30), `--inflection-log-dir`

### Running stages individually

**Paddle:** `scripts/domain_adaptation/system_identification/paddle/real_data_paddle_pipeline.py`
Samples random trajectory segments, replays actions in sim, compares paddle positions via CMA-ES.

**Puck:** `scripts/domain_adaptation/system_identification/puck/real_data_puck_pipeline.py`
Loads fixed paddle params, samples free-flight segments (avoiding paddle-hit timesteps), optimizes puck physics.

**Grid search alternative:** `scripts/domain_adaptation/system_identification/paddle/grid_search_paddle.py` — exhaustive parameter sweep, outputs `grid_results.csv`, `best_params.yaml`, heatmaps.

### CMA-ES internals (`planners.py`)

`CMAPlanner` normalizes params to [0,1] via `MinMaxNormalizer`, denormalizes for sim evaluation. Paddle: random segments across episodes. Puck: stratified sampling within non-contact intervals (when `peak_start=True`, segments start at puck x-peaks). Maintains held-out validation set.

### Parameter bounds

**Paddle** (`sys_id_configs/real2sim/paddle_id_params.yaml`): force_scaling [0.01, 1.0], paddle_damping [10, 20], pid_kp [2000, 4000], pid_kd [10, 200]

**Puck** (`sys_id_configs/real2sim/puck_id_params.yaml`): gravity [-0.671, -0.669] (tight — widen if re-running from scratch), puck_damping [0.1, 2.0]

### Loss metrics

`l2` (full state), `posl2` (position-only, recommended for paddle), `l1`, `posl1`, `last` (final frame only), `dtw` (dynamic time warping)

---

## Step 5 — Analysis Scripts

**Directory:** `scripts/real_data_transforms/analysis_scripts/` — read inflection logs, produce stats/figures.

| Script | Purpose |
|--------|---------|
| `analyze_approaches.py` | Approach interval counts, durations, y-displacement |
| `analyze_peak_velocities.py` | Puck vx/vy histograms at peaks |
| `analyze_approach_distributions.py` | Distribution subplots (fall height, y-displacement) |
| `analyze_zero_vx_approaches.py` | Near-stationary (vx ≈ 0) approach analysis |

---

## Config: `estimate.yaml`

**File:** `scripts/domain_adaptation/realworld_paddle_config/estimate.yaml`

Base Box2D config for sim rollouts and sys-ID. Key params:

```yaml
simulator_params:
  force_scaling: 0.99        # Paddle (Stage 1)
  paddle_damping: 17
  pid_kp: 5000
  pid_kd: 200
  gravity: -0.65             # Puck (Stage 2)
  puck_damping: 0.25
  length: 1.9304       
  width: 0.8636
  paddle_radius: 0.0508
  puck_radius: 0.03175
```

---

## Script Reference

| Script | Purpose |
|--------|---------|
| `real_data_transforms/data_loading.py` | Load HDF5 trajectories |
| `real_data_transforms/puck_inflection.py` | Peaks/contacts → JSON logs |
| `real_data_transforms/puck_velocity_fit.py` | Polynomial velocity estimation |
| `real_data_transforms/real_to_sim_observations.py` | Real traj → 8D obs for sys-ID |
| `real_data_transforms/rendering_utils.py` | Sprite rendering, pixel coords, GIF saving |
| `real_data_transforms/trajectory_to_gif.py` | Multi-subcommand GIF generation |
| `real_data_transforms/generate_*_gifs.py` | Batch GIFs (approach, contact, paddle) |
| `domain_adaptation/system_identification/run_system_id.py` | Two-stage sys-ID entry point |
| `domain_adaptation/system_identification/paddle/real_data_paddle_pipeline.py` | Stage 1: paddle CMA-ES |
| `domain_adaptation/system_identification/puck/real_data_puck_pipeline.py` | Stage 2: puck CMA-ES |
| `domain_adaptation/system_identification/paddle/grid_search_paddle.py` | Grid search (paddle) |
| `domain_adaptation/planners.py` | CMA-ES engine & trajectory sampler |
| `real_data_transforms/analysis_scripts/` | Post-hoc statistical analysis |
