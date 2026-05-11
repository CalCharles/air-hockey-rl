# only_action_attenuation

## High-level description

Isolation study: only `enable_action_force_attenuation` ON; all collision randomization (×3), puck noise, occlusions, and delay-jitter OFF. Tests whether actuator-side randomization alone suffices. (At 700k, this single-knob isolation achieved the highest training-end mean of all 15 ablations — 134 vs 88-122 for the single-knob removals.)

---

## Standardized configuration

| Field | Value |
|---|---|
| Ablation type | isolation |
| Source checkpoint | `runs/td3/zeroshot_ablations_700k/only_action_attenuation/seed0/checkpoint_675000/training_state.pth` |
| Training step at this ckpt | 675,000 |
| Source run dir | `runs/td3/zeroshot_ablations_700k/only_action_attenuation/seed0` |
| Sim config (Box2D env) | `scripts/smooth_policy/amp_history/configs/new_juggle/zeroshot_ablations/sim_only_action_attenuation.yaml` |
| TD3 args (recipe) | `scripts/smooth_policy/amp_history/configs/td3/zeroshot_ablations_700k/td3_zeroshot_only_action_attenuation.yaml` |
| Deployment file (here) | `training_state.pth` |

### Always-on defaults (project standard, all 16 ablations)

- 85/15 starting distribution (`puck_spawn_near_paddle_prob: 0.15`) — wide data distribution
- Sysid params (gravity=-0.661, puck_damping=0.178, paddle_density=3000, pid_kp=9000, pid_kd=50)
- `enable_observation_delay: true` (project default — see `feedback_obs_delay_default_on` memory; do NOT flip)
- TD3 recipe (`td3_hist2_motion0_v2.yaml`): 2-layer 64-wide actor + Q, q_updates=25, actor_updates_per_iteration=6, hist_len=2
- Per-checkpoint single-env eval (4 episodes) — except `paramrand_pm25` which uses 5×4 multi-env eval

### Knobs ON in this ablation (beyond defaults)

- `enable_action_force_attenuation: true`

### Knobs OFF in this ablation (vs canonical)

- `puck_noise: false`
- `enable_random_occlusions: false`
- `enable_paddle_puck_strength_randomization: false`
- `enable_paddle_puck_direction_randomization: false`
- `enable_wall_direction_randomization: false`
- `randomize_delay: false`

### Training metric at the picked checkpoint

- Step 676000: Rolling(2k) Avg Return = 143.40, Success Rate = 0.90, Avg Episode Length = 202.00 (single-env training-time metric)

---

## Deployment

Real-world rollout via `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_eval.py`
(or `async_td3_real_modular.py` for further fine-tuning). The `training_state.pth`
contains actor + Q networks + replay buffers + optimizer state + RNG + the saved
`args` dict — everything needed to load or resume the policy. The original sim
config and TD3 args YAMLs are referenced above for full reproducibility.
