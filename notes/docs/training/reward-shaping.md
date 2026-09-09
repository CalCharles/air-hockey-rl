# Reward shaping

How the task reward is composed and what the env returns to the TD3 critic.

Simulation code: [`td3_training.py`](../../../scripts/td3/td3_training.py); env-side base reward: [`airhockey/airhockey_base.py`](../../../airhockey/airhockey_base.py).

## Single reward stream

TD3 trains a single-head critic against the environment's scalar reward. The Bellman target uses one `gamma` (default 0.975); see [td3-algorithm.md](td3-algorithm.md) for the actor objective.

## Task reward

The reward comes from the environment's `get_base_reward` (scaled by `base_reward_scaling`), plus an optional per-step survival bonus:

```
reward = base_reward * base_reward_scaling
       + survival_bonus_per_step  (if enable_survival_bonus and not done)
```

Per-task base rewards of the five canonical tasks (`configs/td3/tasks/`). The scales are part of the task definitions (class constants in the reward classes), not config knobs:

| Task | Reward class | Base reward |
|---|---|---|
| `puck_juggle_upper_half_reward` | juggle reward (dense, ~0.5/step) | unchanged |
| `puck_touch` | `AirHockeyPuckTouchReward` | +1 on the (terminal) paddle–puck contact, else 0 |
| `paddle_reach_position` | `AirHockeyPaddleReachPositionSparseReward` | `GOAL_REWARD = 10` on the (terminal) step within `goal_radius`, else 0 |
| `paddle_reach_position_velocity` | `AirHockeyPaddleReachPositionVelocityReward` | `GOAL_REWARD = 10` when position *and* velocity tolerances are met, else 0 |
| `puck_velocity` | `AirHockeyPuckVelReward` | `DISPLACEMENT_SCALE (10) × max(prev_x − x, 0)`: upward puck travel per step, positions only |

The ×10 on the three sparse tasks is load-bearing: at ×1 the critic's Q sits at 0.01–0.1, the actor saturates and the policy collapses to a constant action (see `notes/scratch/experiments/2026-09-04_01-05_sparse-task-collapse-diagnosis.md`).

On truncation (time limit) the reward is `truncate_rew` instead. E-stop / protective-stop transitions are stored as truncations with `done=0` (no special penalty); see [episode-lifecycle.md](../environments/real-world/episode-lifecycle.md#e-stop-transitions-are-stored-as-truncations).

## Related docs

- [TD3 algorithm](td3-algorithm.md)
- [Episode lifecycle (real)](../environments/real-world/episode-lifecycle.md)
