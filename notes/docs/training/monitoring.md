# Monitoring training runs

What you can observe while a TD3 run is in progress, where the data lives, and the exact TensorBoard scalar / console keys produced by each entrypoint.

Both active TD3 entrypoints log to **TensorBoard** (no wandb). There is no wandb dashboard; if you don't run TensorBoard, the only signal you see is the console megaprint.

---

## TL;DR — viewing a live run

```bash
# Sim training (td3_training.py): everything under <log_parent_dir>.
tensorboard --logdir <log_parent_dir> --port 6006 --bind_all

# Async real-world (async_td3_real_modular.py):
# every artifact (TB logs, episode HDF5/GIFs/camera videos, checkpoints,
# latency profiles) lands in ONE folder per run:
#   <data_root_dir>/<model_path_parent_dir>/data_<TIMESTAMP>/
# Point TensorBoard at <data_root_dir> and it discovers collector_tb/
# and learner_tb/ inside each run folder automatically.
tensorboard --logdir <data_root_dir> --port 6006 --bind_all
```

### Async unified-run layout

Every async-real run produces this single folder, created by `_setup_run_data_dir` (defined in `async_td3_real.py`, called from the modular entrypoint's `main()`):

```
<data_root_dir>/<model_path_parent_dir>/data_<TIMESTAMP>/
    episode_hdf5/             # per-step trajectories
    reset_hdf5/               # reset-FSM trajectories
    episode_gifs/             # side-by-side Box2D + camera GIFs
    episode_camera_videos/    # raw camera MP4s
    collector_tb/             # rolling50, safety, exploration, artifacts
    learner_tb/               # losses, charts/SPS, replay sizes (only if learner is updating)
    checkpoint_*/             # periodic checkpoints (when --enable-periodic-checkpointing)
    latency_profiles/         # only with --enable-latency-profiling
    run_note.txt              # optional human note
```

`_setup_run_data_dir` is the single source of truth: it creates the timestamped run folder and **forces** `args.checkpoint_root_dir = run_data_dir` and `args.log_parent_dir = None`, regardless of what the args-file or CLI set those to. This intentionally overrides the legacy `checkpoint_root_dir` / `log_parent_dir` flags so a run cannot end up split across multiple folders.

If the args-file (e.g. `td3_online.yaml`, which sets both keys) had non-empty values, you'll see this at startup:

```
[run_data] all artifacts unified under: <run_data_dir>
[run_data] ignoring prior checkpoint_root_dir=… log_parent_dir=… (unified under run_data_dir; change --data-root-dir to relocate)
TensorBoard logs: <run_data_dir>
```

To direct artifacts to a different location, change **`--data-root-dir`**. The `--checkpoint-root-dir` / `--log-parent-dir` flags still exist on `Args` but are no longer authoritative — `_setup_run_data_dir` overrides them so the run folder always holds everything.

`--no-enable-periodic-checkpointing` only suppresses `*.pth` writes; TB logs still land in the run folder.

---

## Sim: `td3_training.py`

Single TensorBoard writer at `log_parent_dir` (`td3_training.py:765`). Two cadences:

### Per-update (every gradient step)

`log_scalar_metrics(...)` (`td3_training.py:1961`, helper at `helper/td3_metrics.py`) writes:

| Group | Scalars |
|------|---------|
| `losses/` | `q_task_loss`, `q_motion_loss`, `q_total_loss`, `actor_loss`, `actor_norm_task_mean`, `actor_norm_motion_mean` |
| Sampled-batch reward stats | `sampled_task_reward_mean/std`, `sampled_motion_reward_mean/std`, `sampled_combined_reward_mean/std`, plus mean/std for each motion component (`stand_still`, `temporal_alignment`, `axis_alignment`, `velocity`, `jerk`) raw and weighted |
| Replay state | PER importance-weight stats, priority TD-error means, success/failure buffer sizes, episode-window counts |
| `charts/` | `exploration_primitive_chance`, `SPS` |

### Every 500 env steps (`td3_training.py:1967`)

Console print + TB scalars:

| Scalar | Console prefix | What it is |
|--------|---------------|------------|
| `charts/avg_episodic_return`, `charts/min_episodic_return`, `charts/max_episodic_return` | `Rolling(2k) Avg/Min/Max Return:` | Last ~2k env-steps of finished episodes |
| `charts/avg_success_rate` | `Success Rate:` | Mean success flag over same window |
| `charts/rolling2k_avg_episode_return`, `…_avg_episode_length`, `…_episode_count` | (same line) | Same window stats |
| `charts/rolling2k_puck_hits_total`, `…_estop_events_total`, `…_puck_hits_per_env_step`, `…_estop_rate` | `Rolling(2k) Puck Hits / E-Stop Events / per env-step / E-Stop Rate` | Contact + safety rates |
| `motion/avg_velocity_magnitude`, `…acceleration…`, `…jerk…` | `Avg Velocity / Acceleration / Jerk` | Smoothness diagnostics |
| `contacts/interval_paddle_puck_collisions_total`, `…_per_env_step` | `Paddle-Puck Collisions (last interval)` | Contact frequency |
| `exploration/interval_primitive_*`, `…_policy_takeover_*`, `…_target_position_directional_*` | `Primitive / Policy Takeover / Target-Position Directional Actions` | Action source breakdown — only sim has takeover / horizontal-dominant fractions |

### Eval loop & artifacts

Every `checkpoint_interval` (`td3_training.py:2133–2222`):

- `evaluate_agent(n_eps=4, n_gifs=1)` — held-out rollouts in a separate eval env
- `rollout_data_like_0.gif` snapshot from training
- Model checkpoint (`actor.pth`, critics, full training state)

Final eval at script end: `td3_training.py:2291–2299`.

### Practical: scalars to watch first

1. `charts/rolling2k_avg_episode_return` — main learning curve.
2. `charts/avg_success_rate` and `charts/rolling2k_estop_rate` — task and safety together.
3. `losses/actor_loss` / `losses/q_total_loss` — divergence early-warning.
4. `motion/avg_jerk_magnitude` — smoothness, especially before sim2real fine-tune.

---

## Async real: `async_td3_real_modular.py`

Two TB writers, one per process (collector writer is created in `async_td3_real_modular.py`, learner writer in `_init_sync_learner_state` in `async_td3_real.py`). Cadence is **wall-clock**, default 60 s on each side (`collector_log_interval_sec`, `learner_log_interval_sec`, defaults defined on the `Args` dataclass in `async_td3_real.py`).

### Collector TB (`collector_tb/`)

Per finished episode (`_write_per_episode_tb`, `:515`):

| Scalar | What it is |
|--------|------------|
| `charts/episodic_return`, `charts/episodic_length`, `charts/episodic_success` | Single-episode values |
| `rewards/<motion_component>_mean` | Per-episode mean of each motion-reward component, when motion metrics are present |

Every `collector_log_interval_sec` (`_periodic_log`, `:310–512`), in addition to clearing-and-averaging the per-episode lists:

| Group | Scalars | Source |
|-------|---------|--------|
| **rolling50** (last ≤50 episodes) | `rolling50/task_reward_avg`, `rolling50/motion_reward_avg`, `rolling50/episode_length_avg`, `rolling50/estop_episode_count`, `rolling50/window_count` | `helper/real_collector_metrics.py:66–75` |
| Window aggregates | `charts/avg_episodic_return`, `charts/min_episodic_return`, `charts/max_episodic_return`, `charts/avg_success_rate`, `charts/avg_episodic_length` | `:457–467` (only when episodes occurred in the interval) |
| Replay | `replay/success_buffer_size`, `replay/failure_buffer_size` | `:383–384` |
| Exploration | `exploration/primitive_chance`, `exploration/primitive_env_steps`, `exploration/primitive_horizontal_env_steps`, `exploration/target_position_directional_env_steps` | `:385–406` |
| Artifacts | `artifacts/episodes_saved`, `…_removed_short`, `…_removed_invalid`, `…_gif_generated`, `…_gif_failed`, `…_camera_video_generated`, `…_camera_video_failed` | `:407–426` |
| Safety | `safety/estop_steps`, `safety/estop_episodes`, `safety/controller_disconnect_steps`, `safety/controller_disconnect_episodes`, `safety/readiness_fail_steps`, `safety/readiness_fail_estop_episodes`, `safety/readiness_fail_dropped_steps` | `:428–449` |
| Transitions | `transitions/hold_active`, `transitions/hold_steps_remaining`, `transitions/hold_events_total` | `:443–449` |
| Throughput | `charts/SPS`, `runtime/elapsed_total_s` | `:450–454` |

Console megaprint (`:472–503`) prints the same set on one `[collector] …` line plus a `[collector_progress] …` line per episode (`:572–591`) with rolling50 echoed for terminal visibility. If the transition-hold FSM fired, a `[collector_transition] reason_counts={…}` line follows (`:504–508`).

### Learner TB (`learner_tb/`)

Only meaningful when the learner is actually updating (i.e. the replay has more than `--min-replay-size-before-learning` transitions). Every `learner_log_interval_sec` (logging branch in `_run_sync_learner_iteration`, `async_td3_real.py`):

- Each key in `state.latest_train_metrics` (mirrors the sim `losses/…` and `sampled_*` set, plus `losses/residual_action_l2` when residual RL is on)
- `charts/SPS` (gradient-step rate)
- `replay/success_buffer_size`, `replay/failure_buffer_size`

Console: `[learner] q_updates=… actor_updates=… replay_size=…` per interval.

> **Important for the rollout-only command in [residual-rl-recipe.md](residual-rl-recipe.md)** with `--min-replay-size-before-learning 999999999`: the learner never updates, so `learner_tb/` will be empty. All your monitoring lives in `collector_tb/` and the `[collector] …` console line.

### What's *not* there compared to sim

The async path collects on hardware so several sim diagnostics are intentionally absent:

| Missing in async | Sim location | Why / workaround |
|------------------|--------------|------------------|
| Per-update sampled-batch reward statistics | `td3_training.py:1953–1961` | Async learner runs in another process; only `losses/…` are surfaced to `learner_tb/` |
| Held-out evaluation loop (`evaluate_agent`) | `td3_training.py:2195–2205, :2291–2299` | No eval env on hardware; rolling50 over live episodes is the proxy |
| Min/max return over a sliding window | `td3_training.py:1983–1985` | `charts/{min,max}_episodic_return` exists but is reset each interval, not a sliding window |
| Policy-takeover fraction, horizontal-dominant fraction | `td3_training.py:2086–2125` | Async logs primitive-step counts but not the takeover / horizontal split |
| Velocity / acceleration / jerk averages | `td3_training.py:2029–2031` | Compute from saved episode HDF5 if needed |
| Per-env-step puck hit rate | `td3_training.py:2012–2018` | `safety/estop_*` is the closest signal; for hits, post-process episode HDF5 |
| PER importance-weight / TD-error visibility | `helper/td3_metrics.py` | Fine when learner is idle; if active, it goes to `learner_tb/` |

### Episode artifacts on disk

Created per episode by the orchestrator via `helper/episode_artifacts.py` into the same unified run folder shown in the [layout above](#async-unified-run-layout):

- `episode_hdf5/` — full per-step trajectories (obs, actions, rewards, metadata)
- `reset_hdf5/` — reset-FSM trajectories (separate)
- `episode_gifs/` — side-by-side Box2D-replay + real-camera GIFs
- `episode_camera_videos/` — raw camera MP4s

Example: `--data-root-dir real_runs/online_run_modular --model-path latest_model/hist2_motion0/training_state.pth` produces `real_runs/online_run_modular/latest_model/hist2_motion0/data_<TIMESTAMP>/` containing all of the above plus `collector_tb/`, `learner_tb/`, and (when enabled) checkpoints.

### Practical: scalars to watch first

1. `rolling50/task_reward_avg` — primary "how is it doing" curve.
2. `rolling50/episode_length_avg` and `rolling50/estop_episode_count` — episode-shape and safety, together.
3. `safety/estop_episodes` and `safety/controller_disconnect_episodes` — escalation signal; if these climb, intervene.
4. `artifacts/episodes_saved` vs `artifacts/episodes_removed_*` — confirms episodes are actually being kept.
5. `replay/success_buffer_size` — only useful if the learner is actually consuming (i.e., not in the rollout-only configuration).

For rollout-only collection (large `--min-replay-size-before-learning`), the `[collector_progress]` line is the single best terminal signal — it prints rolling50 on every episode without needing TensorBoard at all.

---

## Quick-reference: log directory layout

```
# td3_training.py
<log_parent_dir>/
  events.out.tfevents.*           # TB scalars (single writer)
  rollouts/                       # eval GIFs (n_gifs=1 per checkpoint)
  rollout_data_like_0.gif         # in-training snapshot
  *.pth                           # actor / critic / training_state checkpoints

# async_td3_real_modular.py — single unified run folder:
<data_root_dir>/<model_subdir>/data_<TS>/
  episode_hdf5/  reset_hdf5/      # per-step + reset-FSM trajectories
  episode_gifs/                   # side-by-side Box2D + camera GIFs
  episode_camera_videos/          # raw camera MP4s
  collector_tb/                   # rolling50, safety, artifacts, exploration, transitions
  learner_tb/                     # losses, charts/SPS, replay sizes (empty if learner idle)
  checkpoint_*/                   # periodic checkpoints (when --enable-periodic-checkpointing)
  latency_profiles/               # only with --enable-latency-profiling
  run_note.txt                    # optional human note from the startup prompt
```
