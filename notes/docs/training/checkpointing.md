# Checkpointing

Save/resume system for TD3 training state.

Simulation code: [`helper/td3_checkpointing.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/td3_checkpointing.py).
Real-world code: `_build_async_training_state` / `_save_async_checkpoint` in [`async_td3_real.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py) (shared library; entrypoint is [`async_td3_real_modular.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_modular.py)).

## Checkpoint schema

`build_training_state` (sim) and `_build_async_training_state` (real) produce a dict saved via `torch.save`. Key groups:

### Metadata

| Key | Type | Description |
|-----|------|-------------|
| `checkpoint_version` | int | Schema version (current: 2) |
| `global_step` | int | Total environment steps taken |
| `iteration` | int | Episode/iteration counter |
| `args` | dict | Full `Args` dataclass as a dict for reproducibility |

### Network state

| Key | Description |
|-----|-------------|
| `actor` | Actor state dict |
| `actor_target` | Target actor state dict |
| `qf1`, `qf2` | Twin critic state dicts |
| `qf1_target`, `qf2_target` | Target critic state dicts |
| `q_optimizer` | Critic optimizer state dict |
| `actor_optimizer` | Actor optimizer state dict |

### Replay buffers

| Key | Description |
|-----|-------------|
| `success_replay_buffer` | Success partition state dict (optional, controlled by `include_replay_buffer`) |
| `failure_replay_buffer` | Failure partition state dict |

Legacy checkpoints may have a single `replay_buffer` key instead -- loaded into the failure buffer for backward compatibility.

### Exploration and episode state

| Key | Description |
|-----|-------------|
| `primitive_selector` | `PrimitiveExplorationSelector` state dict (weights, active primitive, remaining steps) |
| `episode_trajectory` | In-progress episode transitions (list-of-tensors format) |
| `recent_episode_returns` | Deque of recent episode returns for success threshold |
| `episode_return_success_threshold` | Current quantile threshold |

### Kinematic tracking

| Key | Description |
|-----|-------------|
| `temporal_paddle_history` | Paddle position history tensor |
| `temporal_puck_history` | Puck position history tensor |
| `steps_since_done` | Steps since last episode reset per env |
| `current_velocity_mag` | Current velocity magnitude |
| `current_acceleration_mag` | Current acceleration magnitude |
| `current_jerk_mag` | Current jerk magnitude |
| `velocity_magnitudes`, `acceleration_magnitudes`, `jerk_magnitudes` | Rolling lists for logging |

### RNG state

| Key | Description |
|-----|-------------|
| `rng_states.python` | `random.getstate()` |
| `rng_states.numpy` | `np.random.get_state()` |
| `rng_states.torch_cpu` | `torch.get_rng_state()` |
| `rng_states.torch_cuda` | CUDA RNG states (saved but **not restored** on resume) |

CUDA RNG is intentionally skipped during restore to avoid GPU topology/device-count mismatch failures across runs (e.g., different `CUDA_VISIBLE_DEVICES` settings).

## Resume vs fine-tune

Two distinct load paths exist:

### Full resume (`load_resume_training_state`)

Restores everything: networks, optimizers, replay buffers, exploration state, RNG, kinematic history, episode staging, metrics. Intended for continuing an interrupted run with minimal state loss.

### Fine-tune (`load_fine_tune_optimizer_state`)

Restores **only** optimizer state dicts (not replay, not RNG, not exploration). Used when transferring a trained policy to a new task or reward configuration where the replay buffer contents would be stale.

## Legacy field migrations

The checkpoint loader handles several schema changes from older versions:

| Old field | New field | Migration |
|-----------|-----------|-----------|
| `temporal_done_history` + `temporal_position_count` | `steps_since_done` | Position count - 1, clamped to 0; any recent done in history resets to 0 |
| Single `replay_buffer` | `success_replay_buffer` + `failure_replay_buffer` | Old buffer loaded into failure partition |
| `episode_transition_staging` (list) | `episode_trajectory` (single) | First element of the list is used |
| `bootstrap_terminals` (in replay) | `dones` | `bootstrap_terminals` preferred when present for critic-correct semantics |
| Primitive weight vectors of length 3/4/5 | Length 6 | Padded with zeros for new primitive types |

## Resuming real-world async training

This is the canonical procedure for resuming an interrupted real-world TD3 run with [`async_td3_real_modular.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_modular.py). It picks up training where the previous run left off — model weights, optimizer momentum, replay buffer, learner step counters, run-elapsed timer, and rolling-window deques.

### TL;DR — resume command

```bash
python scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_modular.py \
    --train-args <previous-run>/args.yaml \
    --args-file scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml \
    --model-path <previous-run>/checkpoint_<TAG>/training_state.pth \
    --load-replay-from-checkpoint \
    --replay-source-priority checkpoint_only
```

`<TAG>` looks like `successeps_120_qupdates_4800` (periodic) or `final_qupdates_4800` (graceful exit). Pick the latest checkpoint inside the previous run's `data_<TIMESTAMP>/` folder.

### Pre-flight checklist (on the run you want to be able to resume)

The serializer is `_serialize_training_state_payload` ([`async_td3_real.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py) — `include_non_vital_training_state_fields` gates everything below).

- [ ] `enable_periodic_checkpointing: true` — without this no checkpoints are written.
- [ ] `include_non_vital_training_state_fields: true` — **required** for clean resume. Without this the checkpoint omits optimizer state, learner counters, `collector_total_steps`, `run_elapsed_total_s`, and the rolling-window deques. Resuming loses Adam momentum, restarts the TB step axis at 0, and starts rolling-50 stats cold. The orchestrator prints a loud `[main] WARNING` at startup if this combination drifts.
- [ ] `checkpoint_every_successful_online_episodes` — cadence. Default 10 (residual) / 20 (online).

Both [`td3_online.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml) and [`td3_residual.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual.yaml) ship with `include_non_vital_training_state_fields: true` already.

### Where checkpoints live

`_setup_run_data_dir` ([`async_td3_real.py:1917`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py)) creates a unified per-run folder:

```
<data_root_dir>/<model_subdir>/data_<TIMESTAMP>/
    episode_hdf5/                    ← per-episode trajectories (ground truth)
    reset_hdf5/                      ← reset-FSM trajectories
    episode_gifs/, episode_camera_videos/
    episode_summaries.jsonl          ← per-episode return + metadata log (one JSON per line)
    collector_tb/, learner_tb/       ← TensorBoard scalars
    checkpoint_successeps_<N>_qupdates_<M>/
        training_state.pth           ← THE file you point --model-path at
        model.pth, qf1.pth, qf2.pth, …, args.yaml, config.yaml
    checkpoint_final_qupdates_<M>/   ← graceful-exit checkpoint
    latency_profiles/                ← optional
    run_note.txt
```

Periodic checkpoints fire from `_run_sync_learner_iteration` ([`async_td3_real.py:1666`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py)) when the orchestrator increments `stats["checkpoint_save_request_id"]`. The graceful-exit checkpoint is written by `_finalize_sync_learner_state` ([`async_td3_real.py:1985`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py)) inside the modular orchestrator's `finally` block, so it covers normal exit AND `KeyboardInterrupt`.

### What gets restored on resume

Loaded by `_load_training_state_checkpoint` and `_init_sync_learner_state` ([`async_td3_real.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py)), then consumed by [`async_td3_real_modular.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_modular.py).

| Restored | Source field |
|---|---|
| Actor / actor_target / qf1 / qf2 / qf1_target / qf2_target weights | `actor`, `actor_target`, `qf1`, `qf2`, `qf1_target`, `qf2_target` (always) |
| RNG states (Python / NumPy / torch CPU; CUDA skipped) | `rng_states` (always) |
| Adam optimizer state for actor + critics | `actor_optimizer`, `q_optimizer` (non-vital) |
| Learner update counters (`total_updates`, `total_actor_updates`) | `learner_q_updates`, `learner_actor_updates` (non-vital) |
| `collector_total_steps` (TB x-axis continuity) | `collector_total_steps` (non-vital) |
| `run_elapsed_total_s` (run-elapsed clock) | `run_elapsed_total_s` (non-vital) |
| Rolling-window deques (`task`, `motion`, `length`, `estop`, `return`) | `rolling50_*_values` (non-vital) — smaller windows (5/10/25) derived on the fly |
| Replay buffers | `success_replay_buffer`, `failure_replay_buffer` (gated by `--load-replay-from-checkpoint` and `--replay-source-priority`) |
| Episode artifact id counter | recovered from the highest existing HDF5 in `episode_hdf5/` |

### What does NOT carry across resume

- **Primitive exploration selector** state (active primitive, remaining steps) — rebuilt fresh.
- **Transition-hold counters** — rebuilt fresh.
- **`recent_episode_returns` deque** used for the success-partition threshold — rebuilt empty (refills from new episodes).
- **A new run-data folder** is created — the old `data_<TIMESTAMP>/` is preserved untouched. New episodes/checkpoints/TB logs/episode_summaries.jsonl land in the new folder.

### Replay-source semantics on resume

Three priorities (`replay_source_priority`):

| Priority | Behavior |
|---|---|
| `checkpoint_only` | Load replay from checkpoint; ignore `warm_start_hdf5_dirs`. **Use this for resume.** |
| `checkpoint_then_append` | Load replay from checkpoint, then append warm-start HDF5s on top. |
| `warmstart_only` *(default in the YAMLs)* | Ignore the checkpoint replay (even if `--load-replay-from-checkpoint` is set), use only warm-start HDF5s. Right for cold-start; wrong for resume. |

For a resume you almost always want `checkpoint_only`. The CLI flag overrides the YAML value: `--replay-source-priority checkpoint_only --load-replay-from-checkpoint`.

### Stitching episode_summaries.jsonl across runs

Each run writes its own JSONL at `<run_data_dir>/episode_summaries.jsonl`. The old file is untouched — to mine the full multi-run history:

```python
import pandas as pd
df = pd.concat([
    pd.read_json(p, lines=True).assign(run=p.parent.name)
    for p in sorted(Path("real_runs").rglob("data_*/episode_summaries.jsonl"))
])
df.episode_return.plot()  # full return curve across resumes
```

`run_episode_index` is per-run (1-based); `total_steps`, `actor_version`, and `run_elapsed_total_s` are continuous across resumes (when `include_non_vital_training_state_fields=true`).

### Verifying a clean resume

After launching a resume, watch the first ~30s of stdout for:

- `[resume_replay] loaded from checkpoint success_rb=… failure_rb=…` — replay buffers actually loaded (only if `--load-replay-from-checkpoint` is set).
- `[init_sync_learner] resumed q_updates=… actor_updates=…` — non-zero counters confirm optimizer / counters were restored.
- TB scalar `runtime/elapsed_total_s` jumps to the previous run's elapsed at the very first periodic log — confirms `run_elapsed_total_s` carried across.
- `rolling50/episode_return_avg` is non-zero immediately (not warming up from empty) — confirms rolling deques carried across.
- No `[main] WARNING` line about `include_non_vital_training_state_fields` — config is right.

If any of these are missing, the previous run almost certainly was launched with the lean-checkpoint flag and you cannot fully resume that checkpoint — only the model weights and replay buffer can be transferred.

## Related docs

- [TD3 algorithm](td3-algorithm.md) -- what the checkpointed state is used for
- [Replay and episodes](replay-and-episodes.md) -- buffer state dict format
- [Async replay semantics](../environments/real-world/td3-async-replay.md) -- `dones` vs `bootstrap_terminals`
