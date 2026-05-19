# Training architecture

The training stack lives entirely under [`scripts/td3/`](../../../scripts/td3). TD3 with dual-head critics and transformed Bellman targets is the only active algorithm; SAC, PPO, AMP, RMA, and SSL variants have been removed.

## Entrypoints

| Mode | Entrypoint |
|------|------------|
| Sim training | [`scripts/td3/td3_training.py`](../../../scripts/td3/td3_training.py) |
| Real-world async training | [`scripts/td3/extras/async_td3_real.py`](../../../scripts/td3/extras/async_td3_real.py) |
| Real-world frozen-policy eval (TD3 / SGCRL / …) | [`scripts/td3/extras/async_td3_real_eval.py`](../../../scripts/td3/extras/async_td3_real_eval.py) — see [`real-world-eval-pipeline.md`](real-world-eval-pipeline.md) for the agent-dispatch + task-hooks abstractions |
| Human-baseline teleop eval (user study) | [`scripts/td3/extras/async_td3_real_teleop_eval.py`](../../../scripts/td3/extras/async_td3_real_teleop_eval.py) |
| Real-world reset-policy training | [`scripts/td3/extras/async_td3_real_reset_policy.py`](../../../scripts/td3/extras/async_td3_real_reset_policy.py) |

## Code layout

```
scripts/td3/
├── td3_training.py        # sim TD3 trainer (single-env collection + learner loop)
├── agent.py               # TD3 actor network (stochastic head, ResidualMLPTrunk)
├── deterministic_agent.py # frozen / deployment actor
├── residual_agent.py      # residual-head actor wrapping a frozen base
├── encoder.py             # actor encoder used by agent.py
├── evaluate.py            # evaluate_agent() — sync episode rollouts for logging
├── eval_utils.py          # eval helpers (load / unroll a checkpoint)
├── helper/                # runtime support
│   ├── real_td3_runtime.py             # Args, LearnerRuntimeState, learner step
│   ├── real_policy_runner.py           # collector-side rollout loop
│   ├── real_reset_runner.py            # reset-FSM execution loop
│   ├── real_collector_factories.py     # episode boundaries / artifacts
│   ├── real_collector_metrics.py       # per-episode metric capture
│   ├── real_collector_reset.py
│   ├── real_episode_buffers.py
│   ├── real_eval_agents.py             # --agent dispatch (td3 / sgcrl / …) for the eval entrypoint
│   ├── real_eval_stats.py
│   ├── real_motion_rewards.py
│   ├── real_stop_state.py
│   ├── real_transition_hold.py
│   ├── real_warm_start.py              # HDF5 replay seeding for real runs
│   ├── replay_buffer.py                # uniform replay
│   ├── prioritized_replay_buffer.py    # PER
│   ├── shared_replay.py
│   ├── td3_replay_sampling.py
│   ├── td3_episode_collection.py
│   ├── td3_checkpointing.py            # save / load training_state.pth
│   ├── td3_metrics.py
│   ├── dual_head_q.py                  # task + motion Q heads
│   ├── exploration_primitives.py
│   ├── exploration_selector.py
│   ├── motion_magnitudes.py            # paddle vel/accel/jerk parsing
│   ├── juggle_counter.py
│   ├── real_task_eval_hooks.py         # per-task eval metrics + min_timesteps (juggle vs generic)
│   ├── episode_artifacts.py
│   └── run_event_log.py
├── extras/                # CLI entrypoints (real-world)
└── tests/                 # pytest suite
```

## Configs

All canonical YAMLs are at the repo root under [`configs/`](../../../configs/):

| Dir | What |
|-----|------|
| [`configs/new_juggle/`](../../../configs/new_juggle/) | Sim env configs (sysid_best_params*, sim2sim warp targets) |
| [`configs/td3/`](../../../configs/td3/) | TD3 sim training args + residual recipes |
| [`configs/td3_real_world/`](../../../configs/td3_real_world/) | Real-robot residual fine-tune args |
| [`configs/real_configs/`](../../../configs/real_configs/) | Real-robot rollout / mouse-teleop configs |

See [`td3-configs.md`](td3-configs.md) and [`sim-env-configs.md`](sim-env-configs.md) for per-file details.

## Detailed topics

| Topic | Doc |
|-------|-----|
| TD3 algorithm (h-transform, dual-head critics, actor objective) | [`td3-algorithm.md`](td3-algorithm.md) |
| Network architecture (ResidualMLPTrunk, DualHeadQ, DeterministicAgent) | [`network-architecture.md`](network-architecture.md) |
| Reward shaping (task + motion reward composition) | [`reward-shaping.md`](reward-shaping.md) |
| Replay buffers and episode handling (PER, success/failure, staging) | [`replay-and-episodes.md`](replay-and-episodes.md) |
| Checkpoint system (schema, resume vs fine-tune, migrations) | [`checkpointing.md`](checkpointing.md) |
| Residual RL fine-tuning recipe | [`residual-rl-recipe.md`](residual-rl-recipe.md) |
| Sim-to-sim transfer testing | [`sim2sim.md`](sim2sim.md) |
