# paramrand_pm25

## High-level description

Physics-parameter domain randomization: paddle_density / puck_damping / gravity are drawn uniform within ±25% of their sysid values per episode reset. Engineered randomization (collision×3, action attenuation, delay-jitter) is OFF — paramrand is meant as an ALTERNATIVE. Picked at 1M steps (rolling-5 mean = 132.7, single-ckpt = 145.5, per-env spread compressed to 6.5 across 5 dynamics envs) — the paramrand trajectory plateaus around 1M and does not meaningfully improve through 2M.

---

## Standardized configuration

| Field | Value |
|---|---|
| Ablation type | paramrand |
| Source checkpoint | `runs/td3/zeroshot_paramrand/paramrand_pm25/seed0/checkpoint_1000000/training_state.pth` |
| Training step at this ckpt | 1,000,000 |
| Source run dir | `runs/td3/zeroshot_paramrand/paramrand_pm25/seed0` |
| Sim config (Box2D env) | `scripts/smooth_policy/amp_history/configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml` |
| TD3 args (recipe) | `scripts/smooth_policy/amp_history/configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml` |
| Deployment file (here) | `training_state.pth` |

### Always-on defaults (project standard, all 16 ablations)

- 85/15 starting distribution (`puck_spawn_near_paddle_prob: 0.15`) — wide data distribution
- Sysid params (gravity=-0.661, puck_damping=0.178, paddle_density=3000, pid_kp=9000, pid_kd=50)
- `enable_observation_delay: true` (project default — see `feedback_obs_delay_default_on` memory; do NOT flip)
- TD3 recipe (`td3_hist2_motion0_v2.yaml`): 2-layer 64-wide actor + Q, q_updates=25, actor_updates_per_iteration=6, hist_len=2
- Per-checkpoint single-env eval (4 episodes) — except `paramrand_pm25` which uses 5×4 multi-env eval

### Knobs ON in this ablation (beyond defaults)

- `puck_noise: true`
- `enable_random_occlusions: true`

### Knobs OFF in this ablation (vs canonical)

- `enable_paddle_puck_strength_randomization: false`
- `enable_paddle_puck_direction_randomization: false`
- `enable_wall_direction_randomization: false`
- `enable_action_force_attenuation: false`
- `randomize_delay: false`

### Domain randomization (per-reset)

Per `env.reset()`, draws each variable uniform in `[low, high]` from `random_variable_ranges`, reassigns to `simulator_params`, then rebuilds the Box2D simulator. The agent has consistent dynamics within an episode but they shift between episodes — implicit meta-learning over the 5-step paddle/puck history.

| variable | range | reference |
|---|---|---|
| `paddle_density` | [2250.0, 3750.0] | sysid 3000, ±25% |
| `puck_damping` | [0.1335, 0.2225] | sysid 0.178, ±25% |
| `gravity` | [-0.826, -0.496] | sysid -0.661, ±25% |

### Eval-time multi-env settings

5 fixed param-dicts seed-sampled from the same ±25% ranges using `np.random.RandomState(eval_param_seed=12345)` at training start; held constant for all evaluations. Per checkpoint: 4 episodes × 5 envs = 20 episodes, aggregated to a single mean. Eval-env starts vary per ckpt via a per-call seed shift (otherwise the deterministic env replays identical trajectories).

### Trainer

Custom entrypoint `td3_training_dr.py` (wraps `td3_training.py` via monkey-patch on `evaluate_agent`).

### Training metric at the picked checkpoint

- Step 1001000: Rolling(2k) Avg Return = 95.08, Success Rate = 0.83, Avg Episode Length = 150.58 (single-env training-time metric)
- 5-env eval at this ckpt: mean_return = 145.50, mean_success = 1.000, per_env_returns = [144.5, 148.5, 148.2, 144.2, 142.0] (spread = 6.5)

---

## Deployment

Real-world rollout via `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_eval.py`
(or `async_td3_real_modular.py` for further fine-tuning). The `training_state.pth`
contains actor + Q networks + replay buffers + optimizer state + RNG + the saved
`args` dict — everything needed to load or resume the policy. The original sim
config and TD3 args YAMLs are referenced above for full reproducibility.
