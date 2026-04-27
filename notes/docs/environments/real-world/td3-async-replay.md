# Real-world async TD3: replay `dones` and legacy checkpoints

Training on hardware uses the async TD3 collector/learner path (shared-memory replay), not the synchronous vec-env script [`td3_training.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py).

| Piece | Location |
|-------|----------|
| Async real TD3 (collector + learner) | [`scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py) |
| Shared replay (success/failure partitions) | [`scripts/smooth_policy/amp_history/amp_training/td3/helper/shared_replay.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/shared_replay.py) |
| Sim TD3 reference (naming and bootstrap) | [`td3_training.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py) |

## High-level training flow (`async_td3_real.py`)

At a high level, the real-world async TD3 script runs one process that alternates between hardware data collection and learner updates against a shared replay:

1. **Startup and restore**
   - Parse args/config, build env/replay, optionally load replay from checkpoint, and optionally warm-start replay from saved HDF5 episodes.
2. **Initialize learner state**
   - Build actor + twin critics (+ target networks), set optimizers, and restore optimizer/rng state if available.
3. **Collect one policy episode on hardware**
   - Run policy inference with exploration (noise and primitive takeover logic).
   - Step the env, compute task + motion reward, classify stop/safety state, and append transitions to an in-memory episode buffer.
4. **End-of-episode processing**
   - Optionally truncate post-failure steps for readiness-fail safety handling.
   - Add the episode to success/failure replay partitions.
   - Run learner updates from shared replay; if actor changed, sync collector actor and apply transition hold.
   - Write artifacts (HDF5, optional GIF/video, latency profile) and update rolling metrics.
5. **Reset and safe handoff back to policy**
   - Run reset FSM or hard-reset path as needed, then soft-reset/prime state and re-enter policy with transition hold.
6. **Periodic runtime duties**
   - Emit TensorBoard + console stats, maintain rolling-50 summaries, and save periodic checkpoints.
7. **Shutdown/finalize**
   - On exit, write final checkpoint state and close env/writers cleanly.

After modularization, most heavy logic is grouped in helper modules:
- collector primitives/factories: `real_collector_factories.py`
- episode truncation buffers: `real_episode_buffers.py`
- reset and transition helpers: `real_collector_reset.py`
- rolling metrics + TB helpers: `real_collector_metrics.py`
- motion reward helpers: `real_motion_rewards.py`
- stop state classification helpers: `real_stop_state.py`
- warm-start HDF5 loading helpers: `real_warm_start.py`

## Naming (aligned with `td3_training.py`)

- **`terminations` / `truncations`**: flags from `env.step`, same idea as the vec-env arrays in `td3_training.py`.
- **`dones` (episode boundary):** `terminations | truncations | collector_stop` — ends the episode for resets, logging, and primitives.
- **`dones` (replay / critic):** stored in shared replay and used as **`sampled_dones`** in the learner Bellman update. Semantics match the synchronous buffer: **env termination (and collector stop), not time-limit truncation alone** — so truncated-but-not-terminated steps still bootstrap from \(Q(s')\), consistent with Gymnasium-style TD(0).

## Legacy replay checkpoints (two columns)

Older async builds wrote **two** float columns per partition:

| Legacy key | Role |
|------------|------|
| `dones` | Episode-end style mask (often `termination \| truncation \| stop`) |
| `bootstrap_terminals` | Mask actually used for Bellman and next-step prev-action zeroing (termination-like, excluding truncation-only ends) |

Current code stores **only** `dones`, with the **critic** semantics above (same as [`TD3ReplayBuffer`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/replay_buffer.py) in the sim trainer).

**Loading old snapshots:** `SharedReplayPartition.load_state_dict` prefers `bootstrap_terminals` when that key exists, and otherwise uses `dones`. That keeps critic training consistent when resuming from checkpoints that still carry the legacy two-field layout. If you resume from an old buffer where only the legacy `dones` column was saved (without `bootstrap_terminals`), interpret buffer compatibility with care — prefer checkpoints that include `bootstrap_terminals` or re-collect after a schema change.

## Launch commands

All three variants invoke the same entrypoint and require two YAML files:

- `--train-args <train_run>/args.yaml` — training-run args.yaml. Supplies architecture only (`agent_hidden_layer_size`, `agent_num_hidden_layers`, `q_hidden_layer_size`, `q_num_hidden_layers`, `action_scale`, `use_last_action_in_policy_state`). Not CLI-overridable.
- `--args-file` — online-behavior defaults (typically [`td3_online.yaml`](../../../../scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml)). CLI flags override. Only canonical field names are accepted (legacy aliases `agent_hidden_size`, `q_hidden_size`, `learning_starts`, `device` are no longer remapped). Architecture fields in this file are ignored.

Mirrored in the top-level [README](../../../../README.md) under "TD3 Real-World Commands".

### Eval only (run policy, no training, no checkpointing)

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config.yaml \
  --model-path ex_model/new_td3_model/checkpoint_325000/training_state.pth \
  --train-args ex_model/new_td3_model/checkpoint_325000/args.yaml \
  --args-file scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/online_run \
  --min-replay-size-before-learning 999999999 \
  --no-enable-periodic-checkpointing \
  --no-load-replay-from-checkpoint \
  --warm-start-hdf5-dirs
```

`--min-replay-size-before-learning 999999999` gates out learner updates (see the check at `async_td3_real.py:2626`). `--warm-start-hdf5-dirs` with no value disables replay warm-start from HDF5.

`--data-root-dir` is the single root for all collected per-episode artifacts (HDF5s, GIFs, camera videos). At startup, `_setup_run_data_dir` creates `<data_root_dir>/<model_path_parent_dir>/data_<YYYYMMDD-HHMMSS>/` and writes `episode_hdf5/`, `reset_hdf5/`, `episode_gifs/`, and `episode_camera_videos/` inside it. The `<model_path_parent_dir>` mirrors the directory portion of `--model-path` (e.g. `ex_model/new_td3_model/checkpoint_325000/`) so multiple runs against the same checkpoint share a parent.

### Online training from a pretrained checkpoint (collect + train)

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config.yaml \
  --model-path ex_model/td3_model/checkpoint_1515000/training_state.pth \
  --train-args ex_model/td3_model/checkpoint_1515000/args.yaml \
  --args-file scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/online_run
```

Learning behaviour comes from `td3_online.yaml`: `learning_starts: 0`, `q_updates: 20`, low LRs (`policy_lr: 5e-5`, `q_lr: 1e-4`), warm-start replay from `real_runs/warm_start_trajectories`, periodic checkpointing every 20 successful episodes. See [td3-real-world-configs](../../training/td3-real-world-configs.md).

### Resume training from a previous online run

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real \
  --config configs/real_configs/rollout_td3_config.yaml \
  --model-path real_runs/checkpoints/default/checkpoint_successeps_100_qupdates_1517000/training_state.pth \
  --train-args real_runs/checkpoints/default/checkpoint_successeps_100_qupdates_1517000/args.yaml \
  --args-file scripts/smooth_policy/amp_history/configs/td3_real_world/td3_online.yaml \
  --collector-device cpu \
  --learner-device cuda:0 \
  --data-root-dir real_runs/online_run \
  --load-replay-from-checkpoint \
  --include-non-vital-training-state-fields
```

`--load-replay-from-checkpoint` restores the shared replay from the checkpoint (instead of re-warm-starting from HDF5); `--include-non-vital-training-state-fields` also restores optimizer/rng state for a true resume.

## Staging scripts

Two wrapper scripts under `td3/extras/` launch `async_td3_real.py` with scheduled hyperparameter changes:

- [`run_td3_motion_weight_staged.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/run_td3_motion_weight_staged.py) -- schedules motion reward weight changes across training stages.
- [`run_td3_env_transfer_staged.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/run_td3_env_transfer_staged.py) -- schedules environment parameter changes for sim-to-real transfer curriculum.

Related reset-policy helper (single-process buffer, same `dones`-only convention): [`async_td3_real_reset_policy.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_reset_policy.py).
