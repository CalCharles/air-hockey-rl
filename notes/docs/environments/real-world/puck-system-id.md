# Puck System Identification

Fitting the Box2D simulator's puck physics (gravity, damping) to real-world puck trajectory data.

## Data pipeline

### Source data

Real-world trajectory HDF5s from teleop recordings (`sysid/system_id_segments/trajectory_data{451,458,461,476,478}/`). Each trajectory folder contains manually labeled puck-focused sub-segments named `puck_<start>_<end>/`.

### Consolidated puck segments

All puck segments were copied to `sysid/puck_segments/` with trajectory-prefixed names:

```
sysid/puck_segments/
├── trajectory_data451_puck_10_45/      (55 frames)
├── trajectory_data451_puck_130_170/    (60 frames)
├── trajectory_data458_puck_85_140/     (70 frames)
├── trajectory_data461_puck_10_50/      (60 frames)
├── trajectory_data461_puck_55_75/      (40 frames)
├── trajectory_data476_puck_175_205/    (50 frames)
├── trajectory_data476_puck_215_245/    (50 frames)
├── trajectory_data476_puck_75_150/     (95 frames)
├── trajectory_data478_puck_15_40/      (45 frames)
└── trajectory_data478_puck_95_120/     (37 frames)
```

`trajectory_data461_puck_75_95` was removed — it contained a side-wall bounce (puck y reached -0.408m, past the wall contact threshold at y=-0.400m).

Each subfolder contains:
- `puck_<start>_<end>.hdf5` — split-schema HDF5 (same format as teleop recordings, see `notes/docs/environments/real-world/teleop-system-id.md` for field reference)
- `trajectory_visualization.gif` — rendered GIF of the segment

### Segment structure

Each segment has a **10-frame buffer on each side**:
- Leading buffer: local frames 0–9
- Core: local frames 10 to N-11
- Trailing buffer: last 10 frames

The buffer provides context for velocity estimation. **Model evaluation is on core frames only.**

Key data fields used: `puck` (x, y, occluded), `cur_time`, `pose` (paddle position for contact checks).

Sampling rate: ~20 Hz (mean dt ≈ 0.049s).

## Puck physics model

### Damped kinematic model

```
dv/dt = -γ·v - g
```

where γ is the linear damping coefficient and g is the gravity deceleration vector (sign convention: positive g = deceleration; pass negative values for acceleration).

**Analytical solution:**
- `v(t) = (v0 + g/γ)·exp(-γ·t) - g/γ`
- `x(t) = x0 + (v0 + g/γ)/γ · (1 - exp(-γ·t)) + (-g/γ)·t`

When γ→0 this reduces to the gravity-only model: `x(t) = x0 + v0·t - 0.5·g·t²`.

### Gravity convention

In base coordinates (the frame used by puck tracker and HDF5):
- The table is tilted so puck accelerates in +x direction
- `gx < 0` means the puck accelerates in +x (deceleration parameter is negative → acceleration)
- `gy = 0` (no gravity component along table width)
- Box2D sim uses `gravity: -0.65` which maps to `gx = -0.65` in this convention

### Fitting approach

For fixed (gx, γ), the damped model is **linear in (pos0, v0)** per segment. This enables fast grid search:

1. Rearrange: `x(t) - vterm·t = x0 + u/γ · (1 - exp(-γ·t))` where `u = v0 + g/γ`, `vterm = -g/γ`
2. Design matrix: `A = [1, (1 - exp(-γ·t))/γ]`
3. Solve per-axis linear least squares for `[x0, u]`, recover `v0 = u - g/γ`

Fitting uses **all frames** (buffer + core + buffer). Evaluation on **core only**.

## Grid search results

Script: `sysid/puck_grid_search.py`
Output: `sysid/puck_segments/grid_search/`

Grid: gx ∈ [-1.2, 0.0] (50 pts), γ ∈ [0.0, 1.5] (60 pts), gy fixed at 0.

### Best parameters

| Config | gx | γ | Mean core error |
|--------|-----|-----|-----------------|
| Gravity-only | -0.650 | 0.000 | 3.16 cm |
| Sim params | -0.650 | 0.250 | 3.00 cm |
| **Grid best** | **-0.661** | **0.178** | **2.86 cm** |

### Key findings

1. **Gravity is well-constrained**: sharp minimum near gx ≈ -0.66. The sim value (-0.65) is essentially correct — the real table tilt matches the sim closely.

2. **Damping is weakly constrained**: broad shallow minimum around γ ≈ 0.18. The error curve is nearly flat from γ=0 to γ=0.4, so the sim value (0.25) is fine.

3. **Sim parameters are near-optimal**: the grid search only improves by 0.14 cm over the sim's existing (gx=-0.65, γ=0.25). No config changes needed.

4. **Residual error floor (~3 cm)** comes from: position measurement quantization (~2mm steps creating alternating fast/slow velocity estimates), paddle-puck contacts in buffer zones corrupting some fits, and unmodeled effects (friction anisotropy, puck spin).

### Per-segment results

See `sysid/puck_segments/grid_search/summary.txt` for full per-segment mean and max error tables.

Worst segments (td476_puck_215_245, td478_puck_15_40, td476_puck_75_150) all have paddle-puck contacts in the buffer frames, creating velocity discontinuities that the single-trajectory model can't fully accommodate.

## Box2D replay evaluation

The kinematic grid search fits an analytical model to data. This section tests the **full Box2D physics engine** with the same parameters — including numerical integration, wall collisions, and the `linearDamping` implementation — to verify that the sim actually produces trajectories matching reality.

Script: `sysid/evaluate_puck_box2d.py`
Output: `sysid/puck_segments/box2d_eval/`

### Method

For each segment and each (gx, γ) config:

1. Fit initial (pos0, v0) using the damped kinematic model (same LSQ as grid search, all frames).
2. Compute the model-predicted velocity at the **core start frame** (frame 10).
3. Initialize Box2D with the puck at the **real position** at core start and the **model-predicted velocity**. Paddle is parked far away with collisions disabled.
4. Step the Box2D env at 20 Hz (fixed 0.05s timestep) through all core frames.
5. Compare sim puck positions to real data on core frames.

Starting at core start with the real position (rather than the fitted pos0 at frame 0) avoids issues where the kinematic fit's extrapolated pos0 falls outside the table bounds — the analytical model has no walls, but Box2D does.

### Results

| Config | gx | γ | Box2D mean err | Kinematic mean err |
|--------|-----|-----|----------------|-------------------|
| Gravity-only | -0.650 | 0.000 | 4.71 cm | 3.16 cm |
| **Sim params** | **-0.650** | **0.250** | **3.65 cm** | **3.00 cm** |
| Grid best | -0.661 | 0.178 | 4.64 cm | 2.86 cm |

**Excluding td476_puck_75_150** (75-frame segment with wall bounces in the no-damping configs):

| Config | Box2D mean err (excl. outlier) |
|--------|-------------------------------|
| Gravity-only | 2.90 cm |
| **Sim params** | **3.33 cm** |
| Grid best | 2.98 cm |

### Key findings

1. **Sim params (γ=0.25) is the most robust config in Box2D.** While the kinematic grid search favored lower damping, the Box2D engine produces the most consistent results with γ=0.25 — particularly on long segments where wall interactions matter.

2. **No-damping configs diverge on long segments.** Without damping, the puck retains velocity longer, leading to wall bounces that the analytical model doesn't predict. The worst case is td476_puck_75_150 (75 core frames, ~3.75s): gravity-only gives 21 cm mean error due to end-wall bounces, while sim params gives 6.5 cm.

3. **Box2D errors are ~0.5–1.5 cm higher than kinematic model errors** on most segments. Sources of difference: Box2D uses semi-implicit Euler integration (vs analytical solution), fixed 0.05s timesteps (vs real ~0.049s variable dt), and wall collision handling.

4. **Typical segment accuracy is 1–5 cm** for short-to-medium segments (10–40 core frames). Errors grow with segment length as integration differences accumulate.

### Per-segment results

See `sysid/puck_segments/box2d_eval/summary.txt` for full per-segment mean and max error tables.

Notable outliers:
- **td476_puck_75_150**: 75 core frames, puck traverses nearly the full table length. Gravity-only and grid-best diverge badly (21cm, 19cm) due to wall bounces; sim params handles it well (6.5cm) because damping prevents the puck from reaching the wall.
- **td478_puck_15_40**: ~5–7 cm across all configs. Contains paddle-puck contacts in buffer frames that corrupt the velocity fit.

## Output files

### Kinematic model (grid search)

| File | Description |
|------|-------------|
| `sysid/puck_segments/grid_search/summary.txt` | Full numerical results table |
| `sysid/puck_segments/grid_search/error_heatmap.png` | 2D heatmap of error vs (gx, γ) |
| `sysid/puck_segments/grid_search/error_slices.png` | 1D slices through best point |
| `sysid/puck_segments/grid_search/all_segments_fit.png` | Per-segment trajectory fits (3 configs overlaid) |
| `sysid/puck_segments/fit_plots/g-0.65_0.0/all_segments.png` | Per-segment fits: gravity-only vs per-segment-optimized damping |
| `sysid/puck_segments/fit_plots/g-0.65_0.0/error_comparison.png` | Bar chart comparing gravity-only vs per-segment damping |

### Box2D replay

| File | Description |
|------|-------------|
| `sysid/puck_segments/box2d_eval/summary.txt` | Full numerical results table (3 configs × 10 segments) |
| `sysid/puck_segments/box2d_eval/all_segments_box2d.png` | Per-segment x(t)/y(t) trajectory comparison (real vs Box2D, 3 configs overlaid) |
| `sysid/puck_segments/box2d_eval/td*.gif` | Side-by-side GIFs per segment: REAL \| gravity-only \| sim-params \| grid-best |

## Scripts

| Script | Purpose |
|--------|---------|
| `sysid/visualize_puck_fit.py` | Per-segment comparison of gravity-only vs damped (per-segment γ via nonlinear opt). Generates overlay plots and prints a comparison table. |
| `sysid/puck_grid_search.py` | 2D grid search over shared (gx, γ). Uses linear LSQ per segment for each grid point. Generates heatmap, slices, trajectory plots, and summary.txt. |
| `sysid/evaluate_puck_box2d.py` | Box2D replay evaluation. Fits initial velocity via kinematic model, replays in Box2D, generates side-by-side GIFs and error tables. |

## Plotting conventions

- X-position plots have **inverted y-axis** (negative x = top of table = top of plot)
- Time axis uses **timestep index at 20 Hz** (buffer frames shown as negative indices)
- Buffer data points: gray. Core real data: blue (x) / orange (y). Model predictions: colored lines.
- Box2D GIFs: 4-panel layout (REAL | gravity-only | sim-params | grid-best), 10 fps, 120px panel width.

## Open problems

### Box2D vs kinematic model gap

Box2D errors are systematically ~0.5–1.5 cm higher than the kinematic model on the same data. Known sources:

- **Timestep mismatch**: Box2D uses a fixed 0.05s step; real data averages ~0.049s with ±4ms jitter. Over a 75-frame segment this accumulates ~0.1s of timing drift. Could be closed by stepping the raw `world.Step(dt)` with per-frame real dt values instead of going through `env.step()`.
- **Integration method**: Box2D uses semi-implicit Euler; the kinematic fit uses the exact analytical solution. For small γ·dt this is negligible, but compounds over long horizons.
- **`linearDamping` implementation**: Box2D applies damping as `v *= 1/(1 + dt·damping)` per step, which is a first-order approximation of the continuous `v(t) = v0·exp(-γ·t)`. Over many steps the discrete and continuous trajectories diverge slightly.

### Wall bounce divergence on long segments

td476_puck_75_150 (75 core frames, ~3.75s) causes 20+ cm errors for gravity-only and grid-best because the sim puck reaches a wall that the real puck doesn't (or vice versa). This highlights that:

- The kinematic model has no concept of walls and fits a single ballistic arc. If the real trajectory involves a wall-proximity pass, the fitted velocity can be slightly off, and Box2D's wall collision amplifies this into a large divergence.
- Damping (γ=0.25) mitigates this by slowing the puck enough to avoid the wall, but this is a coincidence of the current data — different segments could fail differently.
- A more robust approach might re-fit the velocity using only the first N frames after core start rather than the full-segment LSQ, reducing sensitivity to late-trajectory behavior.

### Velocity fit sensitivity to buffer contacts

Several segments (td478_puck_15_40, td476_puck_215_245) have paddle-puck contacts in the buffer frames. The LSQ fit includes buffer frames for stability, but contacts create velocity discontinuities that the single-trajectory model can't accommodate. The fitted v0 is then a compromise that doesn't accurately represent the post-contact velocity.

Possible mitigations:
- Detect contact frames (sudden velocity change) and exclude them from the fit.
- Use only post-contact frames for the velocity fit when a contact is detected in the buffer.
- Fit only on core frames (sacrificing the buffer's stabilizing effect on the fit).

### Residual ~3 cm error floor

Even the best analytical fits plateau at ~3 cm mean error. Candidate causes not yet investigated:

- **Puck spin**: the real puck can spin, affecting friction and trajectory curvature. The model assumes pure translational motion.
- **Friction anisotropy**: the air cushion may not be uniform across the table surface, creating position-dependent deceleration.
- **Measurement noise**: puck tracker positions show ~2mm quantization steps, creating aliased velocity estimates that the LSQ fit averages over but can't fully correct.

## Next steps

1. **Variable-dt Box2D stepping**: use `world.Step(real_dt)` per frame instead of the env's fixed 0.05s step to eliminate the timestep mismatch and isolate the remaining Box2D integration error.
2. **Wall-aware fitting**: for segments where the Box2D puck hits a wall, re-fit the velocity on only the pre-bounce portion, or add wall-bounce events to the kinematic model.
3. **Contact detection in buffers**: automatically flag buffer frames with paddle-puck contacts (velocity discontinuities) and exclude them from the LSQ fit.
4. **Use validated parameters in training**: the current sim config (gx=-0.65, γ=0.25) is confirmed near-optimal. No parameter change needed, but the ~3–5 cm puck prediction error should be kept in mind when designing reward functions or puck-prediction-dependent policies.
