# Training throughput — where the time goes and what was done about it

*Written 2026-09-03 after profiling and re-engineering the sim training loop
(`scripts/td3/td3_training.py`, shared by the DR wrapper
`td3_training_dr.py`). The Box2D physics / env loop was not touched; the only env-side change is a reward-shape bug fix in three goal-task reward classes (see "Reach task fix" below).*

Experiment file with the raw numbers: [`notes/scratch/experiments/2026-09-03_04-20_training-throughput-optimization.md`](../../scratch/experiments/2026-09-03_04-20_training-throughput-optimization.md).

---

## TL;DR

| | Old trainer | New trainer | Gain |
|---|---|---|---|
| Env steps/s before `learning_starts` (random actions) | 249 | 1010 | 4.1× |
| Env steps/s while training (episodes ≈ 45 steps) | 52.6 | 530 | 10× |
| 60k-step DR run (`td3_paramrand_pm25`, 20k warmup, 2 checkpoints) | 841 s | 98 s | 8.6× |
| Projected 2M-step canonical DR run (training-phase rate) | ~10.5 h | ~1.1 h | |
| 100k-step sweep, 5 tasks × {DR, no-DR} (wall) | 758–1723 s | 136–226 s | 4.3–8.6× (8 cells; the old code crashed on `paddle_reach_position` until its reward-shape bug was fixed the same day) |

The environment itself steps at ~1700 steps/s in isolation (0.6 ms/step), so
the rollout side is now within ~2× of the physics floor; the update side is
GPU-bound at ~1.4 ms per critic update.

Learning is unchanged in kind: the update math is bit-identical to the
eager implementation on a fixed minibatch (verified), replay sampling draws
from the same distribution (verified against priority deciles), and the
short-budget eval returns of old and new runs fall in the same band. See the
experiment file for the per-task old-vs-new sweep.

---

## Where the time went (old trainer, 60k-step profile)

cProfile of the old loop on `td3_paramrand_pm25` (1 env, 64-wide 2-block
actor/critics, `q_updates=25`, `actor_updates_per_iteration=6`, batch 512):

| Cost centre | Share of wall | Why |
|---|---|---|
| Critic/actor forward+backward+Adam dispatch | ~45 % | One critic forward+backward is ~250 CUDA kernels but only 0.8 ms of GPU work; eager PyTorch spends ~7 ms of **CPU** time launching them. 25 critic + 6 actor updates per episode ⇒ ~560 ms of launch overhead per episode. CPU training was no faster (same op count). |
| `AsyncVectorEnv` pipe round-trips | ~15 % | 1 env in a subprocess: 680 steps/s through the pipe vs 1440 in-process. |
| Per-step GPU round-trips | ~12 % | Batch-1 actor inference on the GPU (0.83 ms vs 0.29 ms on CPU), 6 host→device tensor copies per step, and ~8 `.item()`-style syncs (exploration selector, trajectory staging, priority max tracking). 4.1 M `.item()` calls in 60k steps. |
| Replay sampling | ~5 % | 4 `sample()` calls + 8 `torch.cat` per critic minibatch, each `sample()` with a `.item()` sync. |
| GIF encoding | ~2–13 % | `imageio.mimsave` palette quantisation: 0.65 s per GIF every 50 episodes, on the training thread. And the `watch/` GIFs rendered a **separate env instance that was never stepped** (only the text overlay changed). |
| Per-checkpoint evaluation | ~20 s each | 5 envs × 4 episodes + one GIF; synchronous. Small at the old speed, ~30 % of wall at the new speed. |

## What changed

All throughput changes are in `scripts/td3/` (trainer + helpers). Env physics is untouched.

1. **CUDA-graph captured updates with in-graph sampling** —
   [`helper/td3_graphed_update.py`](../../../scripts/td3/helper/td3_graphed_update.py).
   Each critic update (PER + uniform sampling from both buffers → target →
   N-critic loss → backward → Adam → priority write-back → metrics) and each
   actor update is one `CUDAGraph.replay()`. Graphs are captured lazily per
   batch composition; warm-up side effects (weights, Adam state, priorities,
   RNG) are snapshotted and restored in place. The loss forward/backward is
   additionally `torch.compile`d (627 → 369 kernels per update) and Adam is
   `fused=True`. A training cycle went from ~560 ms to ~43 ms. The same class
   runs the identical math eagerly on CPU / when graphs are disabled.
2. **CPU rollout path.** A CPU replica of the actor (refreshed after every
   actor-update cycle with one flatten + one device→host copy) drives the env
   through a `torch.compile`d forward (273 → 140 µs per step). The
   exploration selector runs as a numpy backend
   (`NumpyPrimitiveExplorationSelector`, 125 → 53 µs per step; the torch
   class stays the reference and is still used for the simulator-space range
   mode and by the real-robot runtime) and per-episode trajectory staging
   lives on the CPU; the finished episode moves to the GPU replay buffer in
   one transfer.
3. **In-process single env.** `SingleEnvVector` replaces `AsyncVectorEnv`
   for `num_envs == 1` (still `AsyncVectorEnv` for more), skipping the pipe
   and gymnasium's per-key info bookkeeping (~60 info keys/step).
4. **Sync-free replay buffer.** `TD3PrioritizedReplayBuffer.sample()` uses a
   `torch.where` fallback instead of `.item()`, and the running max priority
   is tracked on-device and folded into the float lazily (`add` /
   `state_dict`). Checkpoint format unchanged.
5. **Async per-checkpoint eval.** `checkpoint_eval_async: true` (default)
   launches [`scripts/td3/checkpoint_eval.py`](../../../scripts/td3/checkpoint_eval.py)
   as a CPU-only subprocess (one at a time; the trainer waits if the previous
   one is still running). It reproduces the in-process behaviour exactly,
   including the DR wrapper's multi-env eval and its per-checkpoint eval-seed
   shift (`--eval-call-index`). Results land in the same files
   (`multi_env_eval.json`, `eval_0.gif`) plus `eval.log`; the summary line is
   printed when the trainer reaps the process. The final eval stays
   in-process.
6. **GIFs.** Encoding runs on a background thread; the `watch/` recorder now
   renders the *live* training env (it is disabled for `num_envs > 1`).
7. **Logging cut.** See [`monitoring.md`](monitoring.md). Training scalars
   are written every `train_metrics_log_interval` cycles (20) instead of a
   random 10 % of cycles, and only the 13 that get looked at; the rolling
   stats block is one console line every `stats_log_interval` steps (5000)
   instead of four lines every 500. Episode return/length are still logged
   per episode.

## Knobs (all in `Args`, see [`td3-args-reference.md`](td3-args-reference.md))

| Arg | Default | Purpose |
|---|---|---|
| `use_cuda_graphs` | `True` | CUDA-graph captured updates (GPU only). Auto-disabled for `target_critic_subset_size < num_critics`. |
| `compile_update` | `True` | `torch.compile` the loss inside the graphs. Falls back to uncompiled graphs on failure. One-time compile at the first training cycle: ~15 s for the 2-critic recipes, ~60 s for the residual recipe (5 critics + CQL, 3 critic forwards per critic). |
| `compile_rollout_actor` | `True` | `torch.compile` the CPU rollout actor (~10 s). Falls back to eager. |
| `rollout_device` | `"cpu"` | Device that drives the env. |
| `torch_num_threads` | `1` | Intra-op CPU threads (tiny tensors; a big OpenMP pool only spins). |
| `checkpoint_eval_async` | `True` | Background subprocess eval per checkpoint. |
| `train_metrics_log_interval` | `20` | Training cycles between loss-scalar writes. |
| `stats_log_interval` | `5000` | Env steps between rolling-stat writes / console line. |

## Reach task fix (env side, 2026-09-03)

`paddle_reach_position` crashed the old trainer: `paddle_reach_position_reward.py`
returned `-dist` as a 1-element array far from the goal and the scalar `bonus`
inside the goal radius, so `torch.stack` over an episode's rewards failed.
The single-sample path now returns a plain float (vectorised path unchanged,
`np.where`). The same pattern was fixed in
`paddle_reach_position_negative_regions_reward.py` and
`puck_goal_position_reward.py`. Rewards from every task are scalars again.

## Running batches of experiments

`scripts/td3/run_experiments.py` is the canonical batch entrypoint: give it
a set of TD3 args YAMLs (files, directories or globs), `--mode dr|nodr|auto`
(DR → `td3_training_dr`, plain → `td3_training`; auto decides by
`eval_param_seed`) and `--gpus`; jobs run in order with at most one per GPU,
stdout goes to `<out-root>/<name>.stdout.log`, and a `summary.md` with
wall-clock / phase SPS / final eval is written when the batch finishes.
Arguments after `--` are forwarded to every trainer call.

```bash
python -m scripts.td3.run_experiments --mode dr \
    --configs 'configs/td3/throughput_bench/full/*_dr.yaml' --gpus 0 2 3 \
    --out-root runs/td3/full_dr
```

`scripts/td3/extras/throughput_bench.py` is a thin wrapper around it that
runs the same configs from an old checkout too and prints speedups.

Set `TD3_PROFILE_SECTIONS=1` to get a per-section wall-clock breakdown
(`policy / env / bookkeeping / train_update / train_other`) printed with
every stats line.

## Compatibility notes

- Checkpoints are format-compatible in both directions. Resuming an old
  checkpoint into the new trainer works (`Adam(capturable=True)` moves the
  step counters to the GPU on load).
- `td3_training_dr.py` and `td3_training_gat.py` wrap the trainer unchanged
  (`evaluate_agent` / `make_env` are still module-level names).
- Do not use a parameter in an autograd-tracked op before the first update
  (e.g. `p.clone()` without `detach()`): its AccumulateGrad node would be
  bound to the legacy stream and graph capture fails. The trainer never
  does this; the updater runs `gc.collect()` before capture as a guard.

## What is left (measured, not done)

Per env step after the changes (~1.0 ms while training, ~0.85 ms before):
env physics 0.6 ms (floor), actor inference 0.14 ms, exploration selector
~0.05 ms, tensor bookkeeping ~0.15 ms. Per training cycle ~43 ms, GPU-bound
(≈1.4 ms × 25 critic updates + 6 × 0.6 ms actor updates).

- Batching the two critics into one stacked (`bmm`) network would roughly
  halve the critic kernel count again.
- Staging the episode trajectory as numpy rows instead of per-step tensor
  clones would save ~0.05 ms/step.
- Overlapping the update cycle with the next episode's rollout would hide
  the update entirely but changes which policy collects the next episode;
  not done on purpose.
