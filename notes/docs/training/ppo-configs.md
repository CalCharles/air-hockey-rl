# PPO Config Files

> **Legacy.** TD3 is the active training path; see [`td3-configs.md`](td3-configs.md). This doc is kept for reference to the PPO+AMP runs in the tree.

Config files live in `scripts/smooth_policy/amp_history/configs/ppo/`.
All are used with the PPO/AMP training pipeline ([`amp_training.py`](../../../scripts/smooth_policy/amp_history/amp_training/amp_training.py)).

## Configs

### `amp_larger.yaml` — AMP + task reward baseline
8 envs, 5000 iterations (`num_steps: 512`). Runs the full AMP discriminator pipeline with reward split 50/50 between task and AMP discriminator (`task_reward_weight: 0.5`, `disc_reward_weight: 0.5`). State-only discriminator (`use_action_discriminator: false`), 3 hidden layers of size 64. Small `temporal_alignment` and `action_magnitude` reward scales (0.4 each). Actor: 5 hidden layers of size 64. Uses `pid_noise_constant_upper_half_custom_sim_params.yaml`. Purpose: standard AMP training with diverse demo data.

### `amp_motion_penalty_vj.yaml` — PPO task-only + velocity/jerk penalties
8 envs, 500 iterations (10x fewer than `amp_larger`). AMP discriminator fully disabled (`disc_reward_weight: 0.0`, `num_discriminator_updates: 0`). Pure task reward (`task_reward_weight: 1.0`) with explicit timestep motion penalties: `velocity_penalty_weight: 0.75`, `jerk_penalty_weight: 0.25`, plus running normalization with warmup. Much deeper actor: 20 hidden layers of size 64. Purpose: replace AMP smoothness signal with direct motion penalties.

> **Broken reference:** this YAML's `config:` field points to `pid_noise_constant_upper_half.yaml`, which does not exist in `configs/new_juggle/`. The run is non-functional as-written; if reviving it, repoint to `pid_noise_constant_upper_half_custom_sim_params.yaml` or an appropriate variant.

## Key axis
**AMP discriminator vs. explicit motion penalties** as the smoothness mechanism.
