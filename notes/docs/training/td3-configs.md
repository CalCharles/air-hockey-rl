# TD3 Simulator Config Files

Config files live in `scripts/smooth_policy/amp_history/configs/td3/`.
All are used with `scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py`.

## Current configs

### `td3_standard.yaml` — Baseline
The reference config. 1M timesteps, `q_updates: 100`, `target_network_frequency: 20`, `actor_updates_per_iteration: 50`. Motion reward weight is `0.0` (task-only) but all motion component weights are set to `0.5`. Uses sim config `pid_noise_constant_upper_half_custom_sim_params.yaml`.

### `td3_no_alignment.yaml` — No temporal/axis alignment
350K timesteps, more aggressive critic updates (`q_updates: 200`, `target_network_frequency: 10`). Zeros out `temporal_alignment_reward_weight` and `axis_alignment_reward_weight`; keeps `stand_still: 0.1`, `velocity: 0.5`, `jerk: 0.5`. Same sim config as standard. Purpose: ablate alignment reward terms.

### `td3_no_alignment_heavy.yaml` — No alignment + heavy physics
Identical to `td3_no_alignment` except uses `pid_noise_constant_upper_half_custom_sim_params_heavy.yaml` (3x paddle density, 5x puck density). Purpose: test no-alignment regime under heavier sim params.

### `td3_no_alignment_real_world_mirror.yaml` — Sim mirror of real-world online training
Very short run (120K steps), minimal update budget (`q_updates: 20`, `actor_updates_per_iteration: 5`). Exploration is fixed with no annealing (`anneal_steps: 0`, constant `primitive_chance: 0.025`), no warm-start policy takeover. Adds fine-grained angle/magnitude bounds on exploration primitives. Uses `sim_real_world_adaptation.yaml` sim config and `full_checkpoint_load: fine_tune`. Purpose: mirror the online real-world training loop in simulation.
