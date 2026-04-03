# Checkpointing

Save/resume system for TD3 training state.

Simulation code: [`helper/td3_checkpointing.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/helper/td3_checkpointing.py).
Real-world code: `_build_async_training_state` / `_save_async_checkpoint` in [`async_td3_real.py`](../../../scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py).

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

## Related docs

- [TD3 algorithm](td3-algorithm.md) -- what the checkpointed state is used for
- [Replay and episodes](replay-and-episodes.md) -- buffer state dict format
- [Async replay semantics](../environments/real-world/td3-async-replay.md) -- `dones` vs `bootstrap_terminals`
