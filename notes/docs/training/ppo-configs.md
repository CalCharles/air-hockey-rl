# PPO Config Files

Config files live in `scripts/smooth_policy/amp_history/configs/ppo/`.
All are used with the PPO/AMP training pipeline.

## Current configs

### `amp_larger.yaml` — AMP + task reward baseline
8 envs, 5000 iterations (`num_steps: 512`). Runs the full AMP discriminator pipeline with reward split 50/50 between task and AMP discriminator (`task_reward_weight: 0.5`, `disc_reward_weight: 0.5`). State-only discriminator (`use_action_discriminator: false`), 3 hidden layers of size 64. Small `temporal_alignment` and `action_magnitude` reward scales (0.4 each). Actor: 5 hidden layers of size 64. Uses `pid_noise_constant_upper_half_custom_sim_params.yaml`. Purpose: standard AMP training with diverse demo data.

### `amp_motion_penalty_vj.yaml` — PPO task-only + velocity/jerk penalties
8 envs, 500 iterations (10x fewer than `amp_larger`). AMP discriminator fully disabled (`disc_reward_weight: 0.0`, `num_discriminator_updates: 0`). Pure task reward (`task_reward_weight: 1.0`) with explicit timestep motion penalties: `velocity_penalty_weight: 0.75`, `jerk_penalty_weight: 0.25`, plus running normalization with warmup. Much deeper actor: 20 hidden layers of size 64. Uses `pid_noise_constant_upper_half.yaml` (no custom sim params variant). Purpose: replace AMP smoothness signal with direct motion penalties.

## Key axis
**AMP discriminator vs. explicit motion penalties** as the smoothness mechanism.
