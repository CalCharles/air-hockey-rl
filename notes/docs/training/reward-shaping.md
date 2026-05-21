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

On truncation (time limit) the reward is `truncate_rew` instead. E-stop / protective-stop transitions are stored as truncations with `done=0` (no special penalty); see [episode-lifecycle.md](../environments/real-world/episode-lifecycle.md#e-stop-transitions-are-stored-as-truncations).

## Related docs

- [TD3 algorithm](td3-algorithm.md)
- [Episode lifecycle (real)](../environments/real-world/episode-lifecycle.md)
