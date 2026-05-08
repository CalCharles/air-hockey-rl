# Observation and Action Spaces

Grounded in `airhockey/utils.py` (`get_observation_by_type`), `airhockey/airhockey_base.py` (`init_observation`), and `scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py` (extract helpers).

---

## Active training obs type: `history` (30-dim)

Set via `obs_type: history` in the sim config (e.g., `configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params.yaml`).

### Layout

```
[  0: 15]  paddle history  —  5 × [x, y, valid]  (oldest t-4 … newest t)
[ 15: 30]  puck   history  —  5 × [x, y, valid]  (oldest t-4 … newest t)
```

Each triplet is `[x_pos, y_pos, valid_flag]`:
- `x_pos`, `y_pos` — position in metres in the table coordinate frame
- `valid_flag` — `1.0` = valid reading, `0.0` = occluded / missing (Box2D random-occlusion or real-world detection gap)

Named slice indices used by **training code** (reward shaping; not part of the policy network input):

| Slice | Content | Source |
|-------|---------|--------|
| `[12:14]` | current paddle position (x, y) | `extract_current_paddle_position` |
| `[15:17]` | oldest puck position (t-4) | velocity estimate denominator |
| `[27:29]` | current puck position (x, y) | `extract_current_puck_position` |
| `[27:29] − [15:17]` | estimated puck velocity (proxy) — **used by reward shaping (`velocity_reward_from_magnitude`, `jerk_reward_from_magnitude`); NOT inserted into the obs vector seen by the actor/critic.** The policy reads only the raw 30 (or 32) dims and is expected to learn its own velocity sense from the position history. | `extract_current_puck_velocity` (`td3_training.py:333`) |

### Temporal density caveat

The "oldest t-4 … newest t" labelling above is index-based, not
real-time-based. The **time between consecutive history entries
depends on the simulator and on `enable_observation_delay`**:

| Source | Spacing | 5-entry window |
|---|---:|---:|
| Box2D `enable_observation_delay: true` (canonical baseline) | ~25 ms | ~125 ms |
| Box2D `enable_observation_delay: false` | ~50 ms | ~250 ms |
| Real-world (`air_hockey_real.py`) | ~50 ms (20 Hz, no sub-step loop) | ~250 ms |

This is a side effect of `puck_history.append` living inside the
breakpoints sub-step loop in `airhockey_box2d.py:1830` — see
[`box2d/simulator-essentials.md`](box2d/simulator-essentials.md#-subtle-side-effect-enable_observation_delay-changes-puck_history-sampling-rate)
for the mechanism.

Implications:

- The "puck velocity proxy" magnitude (`obs[27:29] − obs[15:17]`) is
  computed as a **raw position delta** with no time-normalisation. The
  same physical puck velocity produces a 2× larger raw delta when the
  history density is 20 Hz (delay-off / real-world) than when it is
  40 Hz (canonical training).
- The reward-shaping thresholds `velocity_at_one`, `velocity_at_zero`,
  `jerk_at_one`, `jerk_at_zero` (set in TD3 args) are calibrated for
  the canonical 40 Hz training density. They are **not** valid as-is
  for any other density.
- The canonical `hist2_motion0_v2`-style training therefore uses a
  history density that does **not** match real-world deployment — a
  silent sim-to-real obs distribution gap independent of any explicit
  domain-randomization knob. See
  [`scratch/experiments/2026-05-05_17-38_zero-shot-sim2real-ablations.md`](../../scratch/experiments/2026-05-05_17-38_zero-shot-sim2real-ablations.md)
  for the failed `no_obs_delay` ablation that surfaced this.

### Policy observation (with last action)

When `use_last_action_in_policy_state: true` (default in all TD3 configs), the actor receives a **32-dim** vector:

```
[  0: 30]  raw obs (history)
[ 30: 32]  last action [ax, ay]  (zero-initialised at episode start)
```

The replay buffer stores the raw 30-dim obs; the last-action augmentation is applied at actor/critic call time via `augment_policy_observation`.

---

## Other obs types (legacy / alternative tasks)

| `obs_type` | Dim | Layout |
|------------|-----|--------|
| `vel` | 8 | `[paddle_x, paddle_y, paddle_vx, paddle_vy, puck_x, puck_y, puck_vx, puck_vy]` |
| `pos` | 4 | `[paddle_x, paddle_y, puck_x, puck_y]` |
| `paddle` | 4 | `[paddle_x, paddle_y, paddle_vx, paddle_vy]` |
| `paddle_acceleration_vel` | 12 | paddle pos+vel+acc+force (8) + puck pos+vel (4) |
| `paddle_acceleration_history` | 23 | paddle pos+vel+acc+force (8) + 5×puck history (15) |

The extract helpers in `td3_training.py` branch on `obs_dim` to support both `vel` (8-dim) and `history` (30-dim) observations in the same codebase.

---

## Coordinate frame

- **Origin**: centre of the table.
- **x-axis**: along the table length (1.9304 m). Positive x = toward the agent's own goal (bottom half). Negative x = upper half / opponent side.
- **y-axis**: across the table width (0.8636 m). Symmetric around 0.
- Units: **metres** throughout.
- The Box2D simulator uses an internal coordinate frame; `_box2d_to_base_coords` / `_base_to_box2d_coords` convert between them. Observations are always returned in the **base frame**.

---

## Action space

**Shape**: `(2,)` — continuous, `Box(low=-1, high=1)`

```
action[0]  →  normalised x-displacement target
action[1]  →  normalised y-displacement target
```

### PID target computation (Box2D)

The action is **not** a direct force. With `use_pid: true` the simulator converts it to a PID setpoint each step:

```
target_pos = current_pos + action * move_lims
```

where `move_lims = [rmax_x, rmax_y]`:

| Parameter | Default value | Meaning |
|-----------|--------------|---------|
| `rmax_x` | **0.26 m** | max x-displacement per step at `action[0] = ±1` |
| `rmax_y` | **0.12 m** | max y-displacement per step at `action[1] = ±1` |

The raw target is then:
1. Projected onto a per-step rectangular movement bound (ellipse clamp via `_get_edge`).
2. Clipped to the paddle workspace bounds.
3. Fed to the PD/PID controller (`pid_kp: 5000`, `pid_kd: 200`, `pid_ki: 0.0`) which outputs a force applied to the paddle body.

### Training-level scaling

`action_scale: 1.0` in all current TD3 configs — no additional RL-level scaling. The `DeterministicAgent` outputs values in `[-1, 1]` via `tanh`; these pass directly to `env.step`.

### Boundary enforcement

`AirHockeyBaseEnv.single_agent_step` clips the action **before** passing it to the simulator: if the paddle is already at a boundary edge, the component pointing further out-of-bounds is zeroed. This is applied in environment space, not Box2D space.

### Real-world differences

On the real UR5 robot the action is converted to an **absolute TCP target** in robot base-frame coordinates via the homography pipeline, not a relative displacement. The effective `move_lims` differ and are set via `real_configs/rollout_td3_config.yaml`. See `notes/docs/environments/real-world/overview.md` for details.
