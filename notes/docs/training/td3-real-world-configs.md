# TD3 Real-World Config Files

Config files live in `scripts/smooth_policy/amp_history/configs/td3_real_world/`.
These are used for training on or mirroring the real robot, as opposed to pure sim configs in `configs/td3/`.

## Current configs

### `td3_no_alignment.yaml` — Real-world no-alignment baseline
1M timesteps, `q_updates: 200`, `target_network_frequency: 10`. Exploration primitives are **fully disabled** (`primitive_chance: 0.00`) — pure policy rollouts. Defines fine-grained angle/magnitude bounds on exploration primitives (kept for reference but unused at chance=0). Motion reward is small but nonzero (`motion_reward_weight: 0.03`) with only `velocity` and `jerk` terms active. Uses `pid_noise_constant_upper_half_custom_sim_params.yaml`. Purpose: real-world-oriented no-alignment training without any primitive exploration.

### `td3_no_alignment_explore.yaml` — Real-world no-alignment with exploration
Identical to `td3_no_alignment` except exploration primitives are active (`primitive_chance: 0.05`, constant — no annealing). Uses `y_aligned` (weight 1.0), `same_direction` and `target_position_directional` (weight 0.25 each); `stand_still` disabled. Detailed angle/magnitude comments explaining coordinate system conventions. `exploration_target_position_steps: 7` (vs. 5 in sim configs). Purpose: same no-alignment regime but with structured exploration for data collection on the real robot.

### `td3_online.yaml` — Real-world online training (async TD3)
The primary config for live on-robot training via the async TD3 pipeline. Key differences from sim configs:
- Very low learning rates: `policy_lr: 0.00005`, `q_lr: 0.0001` (5–10x lower)
- `min_replay_size_before_learning: 0` — begins updating as soon as replay has any data (the legacy `learning_starts` alias is no longer remapped; use `learning_starts_fresh_steps` for the residual fresh-rollout fill phase)
- Minimal update budget: `q_updates: 20`, `actor_updates_per_iteration: 5`
- Warm-start HDF5 replay loading from `real_runs/warm_start_trajectories`
- Periodic checkpointing every 20 successful online episodes
- `replay_source_priority: checkpoint_only` by default
- Fixed exploration at `primitive_chance: 0.025` (no annealing)
- `motion_reward_weight: 0.025`
- Many sim-specific fields (`per_enabled`, `buffer_size`, `total_timesteps`, etc.) are present but marked as ignored by the async TD3 real pipeline

### `td3_reset_online.yaml` — Online reset policy training
Used by `extras/async_td3_real_reset_policy.py` — a separate policy trained to reset the puck to a valid state. Structurally simpler: single-stream TD3 (no PER, no success/failure buffer split), `buffer_size: 300000`, `q_updates: 10`, `actor_updates_per_iteration: 2`, `target_network_frequency: 2`. Loads reset episodes from `real_runs/online_run/reset_hdf5` (legacy flat layout). Adds reset-specific knobs: `max_reset_window_steps: 120`, margin/failure-count thresholds for detecting a bad reset. No motion reward or exploration primitives.

> **Note on warm-start path layout.** `extras/async_td3_real.py` now writes per-episode artifacts under `<data_root_dir>/<model_path_parent_dir>/data_<YYYYMMDD-HHMMSS>/{episode_hdf5,reset_hdf5,episode_gifs,episode_camera_videos}/` (see [`td3-async-replay.md`](../environments/real-world/td3-async-replay.md#launch-commands)). The flat `real_runs/online_run/reset_hdf5` path above refers to a previously collected dataset; for fresh runs you'll want to point `warm_start_hdf5_dirs` at the new nested location (the loader is recursive when `warm_start_hdf5_recursive: true`, so pointing at `real_runs/online_run/` and letting it walk works).

## Key axes
- **Exploration on/off**: `td3_no_alignment` (off) vs. `td3_no_alignment_explore` (on)
- **Sim vs. live robot**: `td3_no_alignment*` (sim-style loop) vs. `td3_online` (async real-robot loop)
- **Task policy vs. reset policy**: `td3_online` vs. `td3_reset_online`
