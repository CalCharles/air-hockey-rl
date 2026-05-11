# Reward shaping

How task and motion rewards are composed for the dual-head TD3 critics.

Simulation code: [`td3_training.py`](../../../scripts/td3/td3_training.py).
Real-world code: [`real_motion_rewards.py`](../../../scripts/td3/helper/real_motion_rewards.py).

## Two reward streams

Each transition carries two scalar rewards, consumed by separate critic heads with independent discount factors:

| Stream | Critic head | Discount | Source |
|--------|-------------|----------|--------|
| **Task reward** | `task_head` | `task_gamma` (0.975) | Environment reward (base reward, shaping, survival bonus) |
| **Motion reward** | `motion_head` | `motion_gamma` (0.8) | Computed from paddle kinematics (velocity, jerk, alignment) |

The actor objective combines both via `task_reward_weight` and `motion_reward_weight`. See [td3-algorithm.md](td3-algorithm.md) for the actor loss formula.

## Task reward

The task reward comes from the environment's `get_base_reward` (scaled by `base_reward_scaling`), plus optional reward shaping and a per-step survival bonus:

```
task_reward = base_reward * base_reward_scaling
            + reward_shaping          (if use_reward_shaping)
            + survival_bonus_per_step (if enable_survival_bonus and not done)
```

On truncation (time limit), the task reward is `truncate_rew` instead.

## Motion reward components

Motion reward is a weighted sum of five components, each in roughly [0, 1] range (some can go negative):

### 1. Stand-still reward

Binary: `1.0` if the paddle's net displacement over the temporal horizon is below `stand_still_threshold`, `0.0` otherwise. Only valid after `temporal_horizon` steps since the last reset.

When stand-still fires, temporal alignment and axis alignment are overridden to `1.0` (standing still is maximally aligned by convention).

### 2. Temporal alignment reward

Cosine similarity between the paddle's realized movement vector (over the temporal window) and the direction toward the puck at the start of the window:

```
cos = dot(realized_movement, puck_direction) / (|realized_movement| * |puck_direction|)
reward = clip((cos + 1) / 2, 0, 1)
```

Ranges from 0 (moving away from puck) to 1 (moving directly toward puck).

### 3. Axis alignment reward

How closely the movement direction aligns with a cardinal axis (x or y). Encourages clean horizontal or vertical motion rather than diagonal drift:

```
max_axis_cos = max(|unit_x|, |unit_y|)
min_axis_cos = 1/sqrt(2)   (diagonal baseline)
reward = clip((max_axis_cos - min_axis_cos) / (1 - min_axis_cos), 0, 1)
```

### 4. Velocity reward

Linear ramp between two magnitude thresholds:

```
reward = 1 - (velocity_mag - velocity_at_one) / (velocity_at_zero - velocity_at_one)
reward = clamp_max(reward, 1.0)
```

- At `velocity_at_one` or below: reward = 1.0 (calm motion)
- At `velocity_at_zero`: reward = 0.0 (too fast)
- Above `velocity_at_zero`: reward goes negative (penalty)

### 5. Jerk reward

Same linear ramp formula as velocity, using `jerk_at_one` and `jerk_at_zero` thresholds. Unlike velocity, jerk reward is **not** clamped at 1.0, so very low jerk can score above 1.

E-stop / protective-stop events do not contribute any special reward term: the e-stop transition is stored as a truncation with `done=0` and no motion-reward penalty. See [episode-lifecycle.md](../environments/real-world/episode-lifecycle.md#e-stop-transitions-are-stored-as-truncations).

## Component weights

Each component is multiplied by its weight before summation:

| Component | Weight parameter | Typical range |
|-----------|-----------------|---------------|
| Stand-still | `stand_still_reward_weight` | 0.0-0.3 |
| Temporal alignment | `temporal_alignment_reward_weight` | 0.0-0.3 |
| Axis alignment | `axis_alignment_reward_weight` | 0.0-0.2 |
| Velocity | `velocity_reward_weight` | 0.1-0.5 |
| Jerk | `jerk_reward_weight` | 0.1-0.5 |

```
motion_reward = sum(weight_i * component_i)
```

## Sim vs real paths

The reward formulas are identical, but the code paths differ:

- **Simulation** (`td3_training.py`): magnitude values come from `parse_motion_magnitudes_from_infos`; reward is computed inline in the training loop.
- **Real-world** (`real_motion_rewards.py`): magnitudes are extracted from `step_info` or `state_info` dicts; a `MotionRewardState` dataclass tracks paddle/puck position history and current kinematic magnitudes across steps. The `_compute_motion_reward_components` function returns a dict of all raw and weighted components for logging.

## Temporal validity

All alignment-based rewards (stand-still, temporal, axis) require at least `temporal_horizon` steps since the last episode reset before they become active. Before that, `temporal_valid = 0.0` and those components contribute nothing. This prevents noisy initial-step kinematics from producing misleading rewards.

## Related docs

- [TD3 algorithm](td3-algorithm.md) -- how rewards feed into dual-head Bellman targets
- [Episode lifecycle (real)](../environments/real-world/episode-lifecycle.md) -- where motion rewards are computed during collection
