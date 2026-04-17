# Known Issues

## 1. Asynchronous z-force clamping: too infrequent, too strong

The async z-force worker (`_async_z_force_worker` in
`airhockey/sims/air_hockey_real.py`) pushes the paddle onto the table surface via
periodic `forceMode` commands. The original settings (100 Hz, wrench_z = 1.0)
applied force too infrequently at too high a magnitude, causing jerky contact
behaviour and occasional bounce/slip.

**Fix applied:** increased loop rate from 100 Hz to 150 Hz (force magnitude kept at
original 1.0). This produces smoother, more continuous table contact. See edit in:

- `airhockey/sims/air_hockey_real.py` — default config `async_z_force_target_hz`

## 2. Async z-force worker fails at startup (RTDE register conflict)

The `_async_z_force_worker` spawned as a `multiprocessing.Process` fails
every startup with `One of the RTDE input registers are already in use!`
because the main process already holds `RTDEControl(..., FLAG_USE_EXT_UR_CAP)`
and the UR controller only allows one such client. The sync
`apply_negative_z_force` calls on the main loop keep the paddle pinned during
`env.step()` / reset, but nothing clamps during long idle phases (e.g. the
wait-for-puck gate in `scripts/real/rollout_new.py`).

As of 2026-04-17 the two paths are mutually exclusive under the
`async_z_force_enabled` flag — sync sites are skipped when the flag is
`True`, so with the default config + broken worker **no clamping ran at
all**. This directly caused a "position deviates from path (SHOULDER)"
protective stop during the reset `moveL` (rigid paddle contact with no
force-mode compliance). See
[`environments/real-world/protective-stops.md`](environments/real-world/protective-stops.md)
for the full incident write-up.

**Workaround applied 2026-04-17**: `async_z_force_enabled: false` has been
set under `simulator_params:` in every real-sim config so the sync clamp
sites fire until the worker is redesigned:

- `configs/real_configs/rollout_td3_config.yaml`
- `configs/real_configs/rollout_config.yaml`
- `configs/real_configs/mouse_config.yaml`
- `configs/real_configs/primitive_exploration_config.yaml`
- `configs/baseline_configs/random_configs/puck_vel_real.yaml`
- `configs/baseline_configs/random_configs/paddle_pos_neg_regions_real_preset.yaml`
- `configs/baseline_configs/random_configs/puck_height_real.yaml`

See [`environments/real-world/async-z-force-future-steps.md`](environments/real-world/async-z-force-future-steps.md)
for the threads+lock redesign plan that will let us remove these
overrides.
