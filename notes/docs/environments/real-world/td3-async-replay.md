# Real-world async TD3: replay `dones` and legacy checkpoints

Training on hardware uses the async TD3 collector/learner path (shared-memory replay), not the synchronous vec-env script [`td3_training.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py).

| Piece | Location |
|-------|----------|
| Async real TD3 (collector + learner) | [`scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py) |
| Shared replay (success/failure partitions) | [`scripts/smooth_policy/amp_history/amp_training/td3/helper/shared_replay.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/shared_replay.py) |
| Sim TD3 reference (naming and bootstrap) | [`td3_training.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py) |

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

Related reset-policy helper (single-process buffer, same `dones`-only convention): [`async_td3_real_reset_policy.py`](../../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_reset_policy.py).
