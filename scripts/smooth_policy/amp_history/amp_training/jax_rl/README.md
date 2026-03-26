# jax_rl

JAX rewrite of the TD3 training pipeline. Algorithm updates (critic, actor, soft target) are JIT-compiled JAX. Environment collection and replay buffers are NumPy/CPU. The PyTorch `PrimitiveExplorationSelector` is kept as-is since it runs on CPU and is not in the hot path.

## Structure

```
jax_rl/
  networks.py        Flax modules: ResidualBlock, ResidualTrunk, DeterministicActor, DualHeadQNetwork
  replay_buffer.py   ReplayBuffer, PrioritizedReplayBuffer, EpisodeBuffer, finalize_episode_if_done
  utils.py           h_transform, h_inverse, soft_update, linear_anneal, reward helpers
  train_state.py     ActorCriticTrainState, make_train_state
  algorithm.py       RLAlgorithm protocol (generic interface for TD3, SAC, PPO, …)
  td3/
    td3_config.py    TD3Config dataclass (algorithm hyperparameters only)
    td3_algorithm.py TD3 class — init_train_state, update_critic, update_actor, select_action
    td3_training.py  Main training script — env loop, exploration, logging, checkpointing
```

## Known limitations / TODOs

### 1 — Single environment only
`td3_training.py` enforces `num_envs=1`, inherited from the original PyTorch script. This is fine because at 1 env the Box2D physics step (~5 ms) dominates and JAX/PyTorch conversion overhead (~0.1 ms) is invisible.

**To scale to N envs:**
- Remove the `num_envs=1` restriction.
- Move action selection into a single batched JAX call (already works — `select_action` accepts a batch).
- Replace `torch.as_tensor` / `.cpu().numpy()` wrappers around `PrimitiveExplorationSelector` with a JAX-native exploration primitive, or keep the PyTorch selector and accept the per-step CPU round-trip (still cheap at moderate N).
- `EpisodeBuffer` currently tracks a single episode (env 0). With N envs, maintain one `EpisodeBuffer` per env and route each to success/failure independently on `dones[i]`.
- Replay buffer writes currently batch the full episode on done; with N envs, batched writes still work since `ReplayBuffer.add` handles a batch dimension.

### 2 — Checkpointing is minimal
Checkpoints save raw parameter leaves with `np.savez`. There is no full-resume (optimizer state, replay buffer, global step) implemented yet.

**To add full resume:**
- Use `orbax.checkpoint` to save/restore the full `ActorCriticTrainState` pytree (params + opt_state).
- Separately pickle the NumPy replay buffer `state_dict()` and episode buffer.
- Save `global_step`, `recent_episode_returns`, rolling windows.
- On load, call `PrioritizedReplayBuffer.load_state_dict(...)` and restore the JAX train state via `orbax.checkpoint.PyTreeCheckpointer`.

### 3 — PyTorch remnants in the collection loop
`parse_motion_magnitudes_from_infos` returns PyTorch tensors which are immediately `.numpy()`'d. `PrimitiveExplorationSelector` takes and returns PyTorch tensors.

**At 1 env:** no measurable cost, leave as-is.

**When scaling:** port `parse_motion_magnitudes_from_infos` to return NumPy directly, and either port `PrimitiveExplorationSelector` to JAX or wrap it to accept/return NumPy.

### 4 — No model loading from existing PyTorch checkpoints
Old `.pth` files (PyTorch state dicts) cannot be loaded directly.

**To convert:** load with `torch.load`, extract the weight arrays with `.numpy()`, then manually assign them into the Flax param pytree using `jax.tree_util.tree_map`.

### 5 — Adding a new algorithm (SAC, PPO, …)
1. Add `<algo>/<algo>_config.py` with a config dataclass.
2. Add `<algo>/<algo>_algorithm.py` implementing the four methods of `RLAlgorithm` (see `algorithm.py`).
3. Write `<algo>/<algo>_training.py` importing `<algo>_algorithm.py` and reusing the shared `replay_buffer.py` infrastructure.
4. For on-policy algorithms (PPO), replace `ReplayBuffer` with a rollout buffer and remove the success/failure split.
