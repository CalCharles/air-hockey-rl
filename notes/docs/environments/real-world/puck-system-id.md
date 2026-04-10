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

## Output files

| File | Description |
|------|-------------|
| `sysid/puck_segments/grid_search/summary.txt` | Full numerical results table |
| `sysid/puck_segments/grid_search/error_heatmap.png` | 2D heatmap of error vs (gx, γ) |
| `sysid/puck_segments/grid_search/error_slices.png` | 1D slices through best point |
| `sysid/puck_segments/grid_search/all_segments_fit.png` | Per-segment trajectory fits (3 configs overlaid) |
| `sysid/puck_segments/fit_plots/g-0.65_0.0/all_segments.png` | Per-segment fits: gravity-only vs per-segment-optimized damping |
| `sysid/puck_segments/fit_plots/g-0.65_0.0/error_comparison.png` | Bar chart comparing gravity-only vs per-segment damping |

## Scripts

| Script | Purpose |
|--------|---------|
| `sysid/visualize_puck_fit.py` | Per-segment comparison of gravity-only vs damped (per-segment γ via nonlinear opt). Generates overlay plots and prints a comparison table. |
| `sysid/puck_grid_search.py` | 2D grid search over shared (gx, γ). Uses linear LSQ per segment for each grid point. Generates heatmap, slices, trajectory plots, and summary.txt. |

## Plotting conventions

- X-position plots have **inverted y-axis** (negative x = top of table = top of plot)
- Time axis uses **timestep index at 20 Hz** (buffer frames shown as negative indices)
- Buffer data points: gray. Core real data: blue (x) / orange (y). Model predictions: colored lines.

## Next steps

This work validates that the sim's puck physics parameters are close to reality. Possible follow-ups:

- Replay puck trajectories in the full Box2D simulator (like `replay_real_in_sim.py` does for paddle), comparing sim puck positions against real — this tests the full physics engine including damping implementation, wall bounces, and integration method, not just the kinematic model.
- Investigate whether the ~3 cm error floor can be reduced by modeling puck spin, friction anisotropy, or air-cushion non-uniformity.
- Use the validated puck parameters to improve puck prediction in the online TD3 policy.
