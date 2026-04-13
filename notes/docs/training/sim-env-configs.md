# Sim Environment Configs (`new_juggle/`)

These files define the Box2D simulator parameters, task, and spawn settings for the juggling environment. They live in `scripts/smooth_policy/amp_history/configs/new_juggle/` and are referenced by TD3/PPO training configs via their `config:` field.

## Current configs

### `pid_noise_constant_upper_half_custom_sim_params.yaml` — Standard (default)
Used by the majority of training runs. Task: `puck_juggle_upper_half_reward`. Standard physics: `paddle_density: 1000`, `puck_density: 250`, `puck_restitution: 1.09145`. Sim-to-real gap features enabled:
- Puck position noise: `puck_noise_std: 0.01`
- Random occlusions: target rate 0.025, with near-paddle boost (3x multiplier within 0.05m)
- Observation delay: 25ms ± 25% randomization
- Action force attenuation: 30% chance, 25–75% scaling
- Near-paddle puck spawn: 15% chance, tight `horizontal_std_m: 0.015`

### `pid_noise_constant_upper_half_custom_sim_params_heavy.yaml` — Heavy physics variant
Identical to the standard config except:
- **3x heavier paddle**: `paddle_density: 3000`
- **5x heavier puck**: `puck_density: 1250`
- Slightly wider near-paddle spawn: `horizontal_std_m: 0.1`, `speed_max: 0.25 m/s`

Used by `td3_no_alignment_heavy.yaml` to test policy robustness under heavier real-world-like inertia.

### `sysid_best_params.yaml` — System-ID best-fit parameters
Physics parameters tuned to best match real-world dynamics via system identification:
- **Puck**: `gravity: -0.661`, `puck_damping: 0.178` (grid search over 10 real-world puck trajectory segments; see [`real-world/puck-system-id.md`](../environments/real-world/puck-system-id.md))
- **Paddle**: `pid_kp: 9000`, `pid_kd: 50`, `pid_ki: 0`, `paddle_density: 3000` (multi-round 3D grid search over 8 teleop categories; see [`real-world/teleop-system-id.md`](../environments/real-world/teleop-system-id.md))

All other settings (noise, occlusions, delays, rewards) are identical to the standard config.
