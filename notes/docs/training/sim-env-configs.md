# Sim Environment Configs (`configs/new_juggle/`)

These files define the Box2D simulator parameters, task, and spawn settings for the juggling environment. They live at [`configs/new_juggle/`](../../../configs/new_juggle/) and are referenced by TD3 training configs via their `config:` field.

## Source-sim configs

### `sysid_best_params.yaml` — System-ID best-fit
Base system-ID physics tuned to match real-world dynamics. Defaults to `hist_len: 1` (no PID-target smoothing).
- **Puck**: `gravity: -0.661`, `puck_damping: 0.178`, `puck_density: 3000` (grid search over 10 real-world puck trajectory segments; see [`real-world/puck-system-id.md`](../environments/real-world/puck-system-id.md)).
- **Paddle**: `pid_kp: 9000`, `pid_kd: 50`, `pid_ki: 0`, `paddle_density: 3000` (multi-round 3D grid search over 8 teleop categories; see [`real-world/teleop-system-id.md`](../environments/real-world/teleop-system-id.md)).
- Sim-to-real gap features enabled: puck position noise (σ = 0.01 m), plain spatially-uniform random occlusions (5 % per-step start probability, run length ≤ 7 frames), fixed 25 ms observation delay, near-paddle puck spawn 15 % of resets.
- **The older engineered randomization stack** (per-collision strength/direction jitter, wall-direction jitter, action force attenuation, delay jitter, paddle-density fluctuation, spatially-varying occlusion zones) was removed from the env on 2026-05-11. For sim2sim / sim2real transfer, layer environment-parameter randomization on top of this baseline — see `sim_paramrand_pm25.yaml` below.

### `sysid_best_params_hist2.yaml` — Sysid + hist_len=2 (canonical source-sim, **active**)
Identical to `sysid_best_params.yaml` except `simulator_params.hist_len: 2`, which enables a 2-timestep low-pass filter on the PID target (see `_filter_update` in `airhockey/sims/airhockey_box2d.py`). This is the sim config wired into the active [`configs/td3/td3_recommended_top50_hist2.yaml`](../../../configs/td3/td3_recommended_top50_hist2.yaml) for source-sim-only training.

### `zeroshot_ablations/sim_paramrand_pm25.yaml` — Canonical sim2sim / sim2real training env (**active**)
Same baseline as `sysid_best_params_hist2.yaml` plus per-reset environment-parameter randomization: `paddle_density`, `puck_damping`, `gravity` each drawn uniform within ±25 % of their sysid values. Paired with [`configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml`](../../../configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml) and launched via `scripts/td3/td3_training_dr.py`. **This is the recommended training env for any new source policy that needs to transfer** (see [`sim2sim.md`](sim2sim.md)).

## Sim2sim targets

### `sim2sim_warp075_p30.yaml` — Canonical big-gap target
Paddle −30% (mass-preserved) + edge-preserving sine y-warp 0.075 on the puck observation. All delays / hist_len / restitutions held at source. Zero-shot return ≈ 48. Used by the canonical `phaseC_actor2_1M` residual recipe.

### `sim2sim_warp075_p10.yaml` — Mild-paddle big-gap target
Same warp, paddle −10%. zs ≈ 49. Used by `phaseD_actor2_p10_1M`.

### `sim2sim_warp100_p30.yaml` — Harder-warp target
Sine y-warp 0.10, paddle −30%. Used by `phaseD_actor4_w10_1M`.

### `sim2sim_combined.yaml` — Small-gap target
Paddle and dynamics deltas without the sine warp. Used by the small-gap recipe (`td3_sim2sim_residual.yaml`).

See [`sim2sim.md`](sim2sim.md) for how sim2sim targets compose with the residual recipes — and for the broader strategy of training the source policy with environment-parameter randomization (`sim_paramrand_pm25.yaml`) rather than the deprecated engineered-DR stack.

## Conventions for new sim configs

Each target sim config should start with `# Source: configs/new_juggle/<source>.yaml` for provenance and annotate every perturbed key with `# PERTURBED: <reason>`. Keep all unperturbed physics keys identical to the source so target authors can audit at a glance.
