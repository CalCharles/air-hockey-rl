# baseline

## High-level description

Canonical TD3 recipe with every randomization knob at its default sysid+DR setting. Reference baseline for the +200k continuation sweep — every other single-knob ablation should be compared against this one.

---

## Standardized configuration

| Field | Value |
|---|---|
| Ablation type | baseline |
| Source checkpoint | `runs/td3/zeroshot_ablations_700k/baseline/seed0/checkpoint_675000/training_state.pth` |
| Training step at this ckpt | 675,000 |
| Source run dir | `runs/td3/zeroshot_ablations_700k/baseline/seed0` |
| Sim config (Box2D env) | `scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist2.yaml` |
| TD3 args (recipe) | `scripts/smooth_policy/amp_history/configs/td3/zeroshot_ablations_700k/td3_zeroshot_baseline_extend.yaml` |
| Deployment file (here) | `training_state.pth` |

### Always-on defaults (project standard, all 16 ablations)

- 85/15 starting distribution (`puck_spawn_near_paddle_prob: 0.15`) — wide data distribution
- Sysid params (gravity=-0.661, puck_damping=0.178, paddle_density=3000, pid_kp=9000, pid_kd=50)
- `enable_observation_delay: true` (project default — see `feedback_obs_delay_default_on` memory; do NOT flip)
- TD3 recipe (`td3_hist2_motion0_v2.yaml`): 2-layer 64-wide actor + Q, q_updates=25, actor_updates_per_iteration=6, hist_len=2
- Per-checkpoint single-env eval (4 episodes) — except `paramrand_pm25` which uses 5×4 multi-env eval

### Knobs ON in this ablation (beyond defaults)

- `puck_noise: true` (additive Gaussian puck-position noise, std=0.01)
- `enable_random_occlusions: true`
- `randomize_delay: true` (±25% per-step jitter on the 25 ms observation delay)
- `enable_action_force_attenuation: true` (30% chance to attenuate the commanded force by 25-75%)
- `enable_paddle_puck_strength_randomization: true` (paddle-puck collision impulse magnitude × U[0.5, 1.0])
- `enable_paddle_puck_direction_randomization: true` (paddle-puck impulse direction ± 10° cone)
- `enable_wall_direction_randomization: true` (wall-collision direction ± 10° cone)

### Knobs OFF in this ablation (vs canonical)

- (none — canonical setup)

### Training metric at the picked checkpoint

- Step 676000: Rolling(2k) Avg Return = 84.36, Success Rate = 0.93, Avg Episode Length = 135.21 (single-env training-time metric)

---

## Deployment

Real-world rollout via `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_eval.py`
(or `async_td3_real_modular.py` for further fine-tuning). The `training_state.pth`
contains actor + Q networks + replay buffers + optimizer state + RNG + the saved
`args` dict — everything needed to load or resume the policy. The original sim
config and TD3 args YAMLs are referenced above for full reproducibility.
