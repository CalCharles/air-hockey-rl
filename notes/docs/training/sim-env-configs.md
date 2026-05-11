# Sim Environment Configs (`configs/new_juggle/`)

These files define the Box2D simulator parameters, task, and spawn settings for the juggling environment. They live at [`configs/new_juggle/`](../../../configs/new_juggle/) and are referenced by TD3 training configs via their `config:` field.

## Source-sim configs

### `sysid_best_params.yaml` — System-ID best-fit
Base system-ID physics tuned to match real-world dynamics. Defaults to `hist_len: 1` (no PID-target smoothing).
- **Puck**: `gravity: -0.661`, `puck_damping: 0.178`, `puck_density: 3000` (grid search over 10 real-world puck trajectory segments; see [`real-world/puck-system-id.md`](../environments/real-world/puck-system-id.md)).
- **Paddle**: `pid_kp: 9000`, `pid_kd: 50`, `pid_ki: 0`, `paddle_density: 3000` (multi-round 3D grid search over 8 teleop categories; see [`real-world/teleop-system-id.md`](../environments/real-world/teleop-system-id.md)).
- Sim-to-real gap features enabled: puck position noise, random occlusions, 25 ms observation delay with jitter, action force attenuation, near-paddle puck spawn, plus the three collision-randomization knobs (paddle-puck strength, paddle-puck direction, wall direction).

### `sysid_best_params_hist2.yaml` — Sysid + hist_len=2 (canonical, **active**)
Identical to `sysid_best_params.yaml` except `simulator_params.hist_len: 2`, which enables a 2-timestep low-pass filter on the PID target (see `_filter_update` in `airhockey/sims/airhockey_box2d.py`). This is the sim config wired into the active [`configs/td3/td3_recommended_top50_hist2.yaml`](../../../configs/td3/td3_recommended_top50_hist2.yaml).

## Sim2sim targets

### `sim2sim_warp075_p30.yaml` — Canonical big-gap target
Paddle −30% (mass-preserved) + edge-preserving sine y-warp 0.075 on the puck observation. All delays / hist_len / restitutions held at source. Zero-shot return ≈ 48. Used by the canonical `phaseC_actor2_1M` residual recipe.

### `sim2sim_warp075_p10.yaml` — Mild-paddle big-gap target
Same warp, paddle −10%. zs ≈ 49. Used by `phaseD_actor2_p10_1M`.

### `sim2sim_warp100_p30.yaml` — Harder-warp target
Sine y-warp 0.10, paddle −30%. Used by `phaseD_actor4_w10_1M`.

### `sim2sim_combined.yaml` — Small-gap target
Paddle and dynamics deltas without the sine warp. Used by the small-gap recipe (`td3_sim2sim_residual.yaml`).

See [`sim2sim.md`](sim2sim.md) for how sim2sim targets compose with the residual recipes.

## Conventions for new sim configs

Each target sim config should start with `# Source: configs/new_juggle/<source>.yaml` for provenance and annotate every perturbed key with `# PERTURBED: <reason>`. Keep all unperturbed physics keys identical to the source so target authors can audit at a glance.
