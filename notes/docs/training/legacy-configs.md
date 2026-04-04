# Legacy Config Directories

The following subdirectories under `scripts/smooth_policy/amp_history/configs/` are **legacy** — they predate the current TD3-based pipeline and are kept for reference only. No active training uses them.

---

## `pid/` — Early sim environment configs

These are environment (sim) configs consumed by the early SAC and SSL training scripts via their `config:` field. They predate the `new_juggle/` configs that current TD3 training uses.

| File | Purpose |
|------|---------|
| `pid_default_config.yaml` | Minimal `puck_juggle` sim. Lighter physics: `paddle_density: 1000`, `puck_density: 250`, `gravity: -0.75`, `wall_bounce_scale: 0.4`. All shaping rewards zeroed except `wall_bumping_rew: -1`. `base_reward_scaling: 0.3`. 250-step episodes. |
| `puck_juggle_original_config.yaml` | Original `puck_juggle` sim with active shaping. Heavier paddle (`density: 2500`, `damping: 3`), lower gravity (`-0.5`), tight `wall_bounce_scale: 0.02`. Enables diagonal/horizontal/direction-change penalties and `stand_still_rew: -0.05`. `base_reward_scaling: 1.0`. |
| `puck_touch_original_config.yaml` | Same physics as `puck_juggle_original_config` but switches task to `puck_touch`. Shorter episodes (100 steps). Terminates on puck-hit-paddle and puck-stop. All shaping rewards zeroed. |

These were superseded by the `new_juggle/` configs which add noise injection, custom sim params, and support for the current TD3 pipeline.

---

## `sac/` — Soft Actor-Critic experiments

Three configs for an early SAC + AMP exploration. All share the same skeleton: 1M timesteps, 8 envs, buffer 100K, hidden size 128.

| File | Purpose |
|------|---------|
| `sac_amp.yaml` | SAC with AMP discriminator enabled. Balanced task + disc reward (`0.5 / 0.5`). Uses `pid_default_config.yaml`. |
| `sac_puck_juggle.yaml` | SAC task-only (`disc_reward_weight: 0.0`). Uses `puck_juggle_original_config.yaml`. `autotune: false`. |
| `sac_task_only.yaml` | SAC task-only with `autotune: true` and disc infrastructure present but inactive. Uses `pid_default_config.yaml`. |

SAC was abandoned in favour of TD3 (better sample efficiency for continuous control on this task).

---

## `ssl/` — PPO + AMP + Self-Supervised Latent (SSL)

PPO-based runs that added a learned SSL encoder (latent dim 10) with auxiliary reward/dynamics prediction heads on top of the AMP discriminator. All use `amp_training/amp_training_ppo_ssl.py` (or equivalent).

| File | Purpose |
|------|---------|
| `amp_lsgan_ssl_default.yaml` | Baseline SSL run: discriminator **enabled** (`use_action_discriminator: false`, `use_puck_discriminator: false` but disc reward weight `0.5`), exploration reward `0.5`, new juggle sim. 1000 iterations. |
| `amp_lsgan_ssl_disc_off_new_juggle.yaml` | Disc off (`disc_reward_weight: 0.0`), new juggle env. 500 iterations. All alignment rewards zeroed. |
| `amp_lsgan_ssl_disc_off_original_juggle.yaml` | Disc off, original juggle env. Larger heads (10 hidden layers). 500 iterations. |
| `amp_lsgan_ssl_jerk_velocity.yaml` | Disc off, new juggle env. Adds velocity (`0.4`) and jerk (`0.2`) penalty terms; all other motion rewards zero. |

The SSL approach was not pursued further — the learned latent did not provide a clear benefit over direct reward shaping.

---

## `rma/` — Rapid Motor Adaptation (two-stage)

Two-stage RMA pipeline: stage 1 trains a policy + privileged encoder over randomised physics; stage 2 distils the encoder into a causal adaptation module from observation history.

| File | Purpose |
|------|---------|
| `amp_default.yaml` | Stage-1 RMA + AMP. 64 envs, 10K iterations, 500 randomised env pool, hidden size 512, latent dim 12. |
| `amp_larger.yaml` | Stage-1 RMA + AMP with a second "long-horizon" discriminator (`long_history_len: 30`). Hidden size 1024, 1000 env pool. |
| `adaptation_supervised_default.yaml` | Stage-2 supervised adaptation. Freezes stage-1 encoder; trains a conv adaptation net (causal, history 50–100 steps) via MSE. |
| `env_randomization_25pct.yaml` | Shared randomisation spec (±25 %) for paddle/puck density, damping, force scaling, PID gains, wall bounce. |
| `eval_rma_joint_default.yaml` | Joint stage-1 + stage-2 evaluation harness. Runs 25 env specs × 5 episodes, saves GIFs. |

RMA was replaced by the current online TD3 real-world adaptation approach (`td3_real_world/`).

---

## `distillation/` — PPO-to-TD3 policy distillation

One config for converting a trained stochastic PPO policy into a deterministic policy suitable as a TD3 warm-start.

| File | Purpose |
|------|---------|
| `refactor.yaml` | Loads a PPO checkpoint (`checkpoint_150/model.pth` from `new_architecture_ppo_disc/`) and writes a deterministic policy to `td3/extras/warm_start/model1.pth`. Architecture: 5 hidden layers of size 64, `use_last_action_in_policy_state: true`. |

This was a one-time migration step; the resulting `model1.pth` is the warm-start used by current TD3 configs.

---

## `new_juggle/` legacy sim configs

Two configs in `new_juggle/` that are no longer in active use.

### `pid_noise_no_base_reward_config.yaml`
Task: `puck_juggle_no_base_reward` (no base reward signal). Simpler noise setup — no action force attenuation, no observation delay, no near-paddle spawn boost. Lower `puck_noise_std: 0.005`, higher target occlusion rate (0.05). Adds `wall_bumping_rew: -1`. Slightly higher `puck_damping: 0.3`. Superseded by the standard `pid_noise_constant_upper_half_custom_sim_params.yaml`.

### `sim_real_world_adaptation.yaml`
Heavily modified config for sim-to-real gap adaptation, used by `td3_no_alignment_real_world_mirror.yaml`. Key differences from standard:
- Lighter, more sensitive paddle: `paddle_density: 800`
- Lower damping: `puck_damping: 0.1`
- Lower restitution: `puck_restitution: 0.9`, `side_wall_restitution: 0.8`, `end_wall_restitution: 0.9`
- Observation position homography: fixed random homographic distortion of observed positions
- Fixed state velocity/jerk initialization: paddle spawns with nonzero velocity/jerk; puck velocity masked
- Jerk e-stop simulation: terminates episode if jerk exceeds thresholds over a rolling window
- Longer observation delay: 30ms ± 30%
- Custom table boundary limits

Superseded by the online real-world training approach (`td3_real_world/td3_online.yaml`).
