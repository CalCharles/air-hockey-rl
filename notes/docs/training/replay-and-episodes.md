# Replay buffers and episode handling

How transitions are stored, partitioned, and sampled for TD3 training.

## Buffer types

### `TD3ReplayBuffer` (simulation)

**Code:** [`helper/replay_buffer.py`](../../../scripts/td3/helper/replay_buffer.py)

Fixed-capacity ring buffer on GPU. Stores per-transition:

| Field | Shape | Description |
|-------|-------|-------------|
| `observations` | `(buffer_size, *obs_shape)` | Current observation |
| `next_observations` | `(buffer_size, *obs_shape)` | Next observation |
| `actions` | `(buffer_size, *act_shape)` | Action taken |
| `prev_actions` | `(buffer_size, *act_shape)` | Previous action (for last-action augmentation) |
| `task_rewards` | `(buffer_size,)` | Task reward scalar |
| `motion_rewards` | `(buffer_size,)` | Motion reward scalar |
| `dones` | `(buffer_size,)` | Episode boundary mask (termination-style, not truncation-only) |

`add` handles wrap-around with a two-chunk copy when the batch crosses the ring boundary. `sample` draws uniform random indices.

### `TD3PrioritizedReplayBuffer` (simulation, PER)

**Code:** [`helper/prioritized_replay_buffer.py`](../../../scripts/td3/helper/prioritized_replay_buffer.py)

Same ring layout as `TD3ReplayBuffer`, plus a `priorities` vector. Implements proportional PER:

- **Sampling probability:** `p_i = priority_i^alpha / sum(priority_j^alpha)` with `alpha=0.6`
- **Importance-sampling weights:** `w_i = (N * p_i)^(-beta) / max(w)` with `beta` annealed during training
- **New transitions** enter with `max_priority` so they are sampled at least once
- `update_priorities` replaces priorities at given indices after a critic update
- `sample_uniform` provides unbiased samples for the actor update

### `SharedTD3Replay` (real-world)

**Code:** [`helper/shared_replay.py`](../../../scripts/td3/helper/shared_replay.py)

Two `SharedReplayPartition` instances (success and failure) backed by `torch.Tensor.share_memory_()` for cross-process access. Each partition is a ring buffer with the same field layout as the simulation buffers, protected by a `multiprocessing.Lock`.

Operations:
- `add_episode(partition, tensors)` -- add a complete episode's transitions to the named partition
- `sample(partition, batch_size, device)` -- uniform random sample, cloned and moved to device
- `state_dict` / `load_state_dict` -- checkpoint the full buffer contents

Legacy checkpoint compatibility: `load_state_dict` prefers `bootstrap_terminals` over `dones` when both keys exist (see [td3-async-replay.md](../environments/real-world/td3-async-replay.md)).

## Success / failure partitioning

Episodes are routed to the success or failure replay buffer based on a rolling quantile threshold:

1. After each episode, `episode_return` is appended to a `recent_episode_returns` deque (configurable window size).
2. The success threshold is the `(1 - success_top_fraction)` quantile of recent returns.
3. Episodes with `return >= threshold` go to the success buffer; others go to failure.

This adaptive threshold ensures roughly `success_top_fraction` of recent episodes are labeled "success" regardless of absolute reward scale.

**Code:** `finalize_episode_if_done` in [`helper/td3_episode_collection.py`](../../../scripts/td3/helper/td3_episode_collection.py).

## Episode trajectory staging

**Code:** `EpisodeTrajectory` in [`helper/td3_episode_collection.py`](../../../scripts/td3/helper/td3_episode_collection.py)

Transitions are accumulated in an `EpisodeTrajectory` dataclass during collection (list of tensors per field). At episode end:

1. `finalize_episode_if_done` computes the success threshold.
2. `flush_to_buffer` stacks the lists into batch tensors and calls `add` on the target replay buffer.
3. The trajectory is reset for the next episode.

`EpisodeTrajectory` tracks both `dones` (episode boundary) and `bootstrap_terminals` (critic mask) separately. For current TD3, these are typically identical, but the distinction exists for legacy checkpoint compatibility.

## Critic sampling strategy

**Code:** [`helper/td3_replay_sampling.py`](../../../scripts/td3/helper/td3_replay_sampling.py)

### Success/failure split

Each critic batch is split between success and failure buffers:

```
success_count = round(batch_size * critic_success_sample_fraction)
failure_count = batch_size - success_count
```

If one buffer is empty, the entire batch comes from the other.

### PER/uniform mix (within each source)

When PER is enabled, each source chunk is further split:

```
per_count = round(source_count * critic_per_fraction)
uniform_count = source_count - per_count
```

The PER and uniform sub-batches are concatenated. This mix lets the critic benefit from prioritized replay while retaining some coverage of the full distribution.

### Actor sampling

The actor always samples uniformly (via `sample_uniform` when PER is enabled) to avoid biased policy gradients from importance-weighted transitions.

## Related docs

- [TD3 algorithm](td3-algorithm.md) -- how sampled batches feed into critic/actor updates
- [Checkpointing](checkpointing.md) -- how replay buffer state is saved
- [Async replay semantics](../environments/real-world/td3-async-replay.md) -- `dones` vs `bootstrap_terminals` in real-world buffers
