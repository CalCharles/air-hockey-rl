# Sim Environment Configs (`new_juggle/`)

These files define the Box2D simulator parameters, task, and spawn settings for the juggling environment. They live in `configs/new_juggle/` and are referenced by TD3/PPO training configs via their `config:` field.

## Current configs

### `sysid_best_params.yaml` — System-ID best-fit
Base system-ID physics tuned to match real-world dynamics. Used by the historical ablation runs (`td3-ablations-updates-and-depth.md`, `td3-exploration-ablations.md`).
- **Puck**: `gravity: -0.661`, `puck_damping: 0.178`, `puck_density: 3000` (grid search over 10 real-world puck trajectory segments; see [`real-world/puck-system-id.md`](../environments/real-world/puck-system-id.md))
- **Paddle**: `pid_kp: 9000`, `pid_kd: 50`, `pid_ki: 0`, `paddle_density: 3000` (multi-round 3D grid search over 8 teleop categories; see [`real-world/teleop-system-id.md`](../environments/real-world/teleop-system-id.md))
- Same sim-to-real gap features (noise, occlusions, obs delay, force attenuation, near-paddle spawn) as the legacy config.

### `sysid_best_params_hist2.yaml` — Sysid + hist_len=2 (canonical, **active**)
**Canonical sim config for new runs.** Identical to `sysid_best_params.yaml` except adds `simulator_params.hist_len: 2`, which enables a 2-timestep low-pass filter on the PID target (see `_filter_update` in `airhockey/sims/airhockey_box2d.py` — this matches the env's default `hist_len`). This is the sim config wired into the active `td3_recommended_top50_hist2.yaml` and matches the v1 `latest_models/canonical/hist2_motion0/config.yaml` checkpoint on **all** physics knobs after the 2026-05-05 restitution-revert edit: gravity, densities, PID gains, restitution coefficients, occlusion, and observation delay are identical to v1. The **only** difference from v1 is the per-collision randomization knobs that were added later (`enable_paddle_puck_strength_randomization`, `enable_paddle_puck_direction_randomization`, `enable_wall_direction_randomization` and their cone/range params), which are documented at the bottom of the file. Restitution-revert rationale: keep v1↔canonical comparable on bounce coefficients so the canonical→v1 delta is purely the collision randomization, not also a restitution shift. See [`scratch/experiments/2026-05-05_17-38_zero-shot-sim2real-ablations.md`](../../scratch/experiments/2026-05-05_17-38_zero-shot-sim2real-ablations.md) for the decision context.

### `legacy/sysid_best_params_hist{3,4,5}.yaml` — Hist-length ablations (legacy)
Variants with `hist_len` ∈ {3, 4, 5} for the temporal-smoothing ablation. The hist4 variant was the canonical sim config for the original `td3_recommended.yaml` (now also in `legacy/`). Preserved for reproducing past ablations; new runs should use `sysid_best_params_hist2.yaml`.

### `pid_noise_constant_upper_half_custom_sim_params.yaml` — Legacy (pre-sysid)
Still referenced by existing TD3 args YAMLs (`td3_standard.yaml`, `td3_no_alignment.yaml`). Task: `puck_juggle_upper_half_reward`. Pre-sysid physics: `paddle_density: 1000`, `puck_density: 250`, `puck_restitution: 1.09145`. Sim-to-real gap features enabled:
- Puck position noise: `puck_noise_std: 0.01`
- Random occlusions: target rate 0.025, with near-paddle boost (3x multiplier within 0.05m)
- Observation delay: 25ms ± 25% randomization
- Action force attenuation: 30% chance, 25–75% scaling
- Near-paddle puck spawn: 15% chance, tight `horizontal_std_m: 0.015`
