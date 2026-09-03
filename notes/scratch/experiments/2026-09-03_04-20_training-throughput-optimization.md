# Sim TD3 trainer throughput: profile, re-engineering, old-vs-new sweep over 5 tasks × {DR, no-DR}

- **Date**: 2026-09-03 04:20 UTC start
- **Status**: done (2026-09-03 ~09:00 UTC)
- **Run dirs**: `runs/td3/throughput_opt/_work/` (profiling + 60k A/B), `runs/td3/throughput_bench/{old,new}/<task>_<setting>` (sweep)
- **Configs**: `configs/td3/throughput_bench/*.yaml` (TD3 args, 100k budget), `configs/new_juggle/throughput_bench/sim_{dr,nodr}_<task>.yaml` (sim configs)
- **Code**: `scripts/td3/td3_training.py`, `scripts/td3/helper/td3_graphed_update.py` (new), `scripts/td3/checkpoint_eval.py` (new), `scripts/td3/helper/{prioritized_replay_buffer,td3_gif_recorder,td3_loop_logging,td3_metrics}.py`, `scripts/td3/extras/throughput_bench.py` (new runner). Canonical write-up: [`notes/docs/training/training-throughput.md`](../../docs/training/training-throughput.md).

## Question

Where does wall-clock go in `td3_training.py` / `td3_training_dr.py`, and how much faster can the loop run without touching the Box2D env (and with logging cut to what is actually read)?

## Setup

Profiling target: the canonical DR recipe `configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml` (1 env, 64-wide 2-block nets, batch 512, `q_updates=25`, `actor_updates_per_iteration=6`, PER on both buffers), shortened to 60k steps with `learning_starts=20000`. Machine: 64-core host, Quadro RTX 6000s (GPU 1 occupied by another user; all runs here on GPUs 0/2/3, one job per GPU).

Old code = git `bf9936e` (worktree at `runs/td3/throughput_opt/_work/baseline_wt`). Old and new use byte-identical YAMLs (the old `Args` rejects unknown keys, so no new knobs appear in the sweep configs).

Micro-benchmarks that drove the design (all measured in-session):

| Item | Measurement |
|---|---|
| Raw `AirHockeyEnv.step` (random actions) | 1746 steps/s (0.57 ms) |
| `SyncVectorEnv` (1 env) / `AsyncVectorEnv` (1 env) | 1440 / 679 steps/s |
| DR env `reset()` (rebuilds simulator) | 0.88 ms (0.52 ms without DR) |
| Actor batch-1 inference: GPU / CPU eager / CPU `torch.compile` | 829 / 292 / 140 µs |
| One critic fwd+bwd, eager GPU | 7.1 ms CPU dispatch, 0.78 ms GPU, 247 kernels |
| Training cycle (25 critic + 6 actor + Polyak), eager GPU / eager CPU (1 thread) | 560 / 502 ms |
| Same cycle, CUDA graphs (static inputs) | 51 ms |
| Same cycle, CUDA graphs incl. in-graph sampling + priority write-back | 52.7 ms |
| Same + `torch.compile`d loss + fused Adam | 42.7 ms (627 → 369 kernels/critic update) |
| Per-checkpoint DR eval (5 envs × 4 eps + 1 GIF) | 20.5 s (9.5 s of it the GIF episode) |
| One `watch/` GIF encode (`imageio.mimsave`) | ~0.65 s, every 50 episodes |

Correctness checks on `helper/td3_graphed_update.py`:
- With `policy_noise=0` (no in-graph randomness), 30 cycles of graphed vs eager updates from identical init on identical minibatches: max |Δparam| 7e-7 (critics), 4e-7 (actor), 2e-7 (targets, incl. fused Polyak).
- CQL path (`cql_alpha=20`) eager-class vs reference helper with shared RNG: bit-identical (0.0).
- Capture does not perturb state: params / Adam state / priorities / RNG restored in place; exactly the PER rows of one update change priority after one `critic_update` (249 rows for 358·0.7 draws with replacement, same as eager).
- In-graph PER sampling: decile histogram of 100k draws vs `p^0.6` expectation matches to <1 % per decile; empty buffer slots never drawn.
- Graphed vs eager loss trajectories over 30 cycles differ by less than two eager runs with different RNG.
- Age-decayed PER (`priority_age_decay=1e-3`, the residual recipes use `1e-4`): in-graph age-decile draw counts and mean IS weights match the eager buffer in both the filling and the wrapped regime (e.g. wrapped deciles 42097/31520/23042/… vs 42152/31373/22969/…). A first version applied the decay after the alpha exponent (decayed 1.6× too fast per decile); fixed before any run used it.
- numpy exploration selector vs torch selector over 60k steps at chance 0.15: takeover fraction 0.459 vs 0.456, stand-still share 0.062 vs 0.059, horizontal-dominant share 0.277 vs 0.272, mean magnitude 0.551 vs 0.550.
- Resuming an old-code checkpoint (`clean_baseline/checkpoint_50000/training_state.pth`, written by `bf9936e`) into the new trainer trains on with graphs + compile (needed a fix: the checkpoint's Adam param groups carry `capturable=False`).
- Residual + CQL recipe (`phaseC_actor2_1M.yaml`: 5 critics, α=20, age decay 1e-4) runs on graphs + compile; eval mode runs.

## Results

### 60k-step A/B on the canonical DR recipe (cuda:2 old, cuda:0 new)

| | Old (`bf9936e`) | New, graphs only | New, graphs + compile |
|---|---|---|---|
| 0–20k (random actions) SPS | 249 | 1009 | 1010 |
| 20k–60k (training, ep len ≈ 43) SPS | 52.6 | 418 | 530 |
| Wall for 60k steps incl. 2 checkpoint evals | 841 s | 115 s | 98 s |
| Multi-env eval mean return at ckpt 25k / 50k / final | 22.9 / 29.1 / 18.6 | 26.0 / 26.9 / 25.9 | 19.8 / 21.6 / 19.8 |

Eval returns at this budget are all "barely started learning" (20-30 on a task whose trained policies score 130+); the point is only that nothing is broken. The per-task 100k sweep below is the learning sanity check.

Where the old 60k run's time went (cProfile, 930 s profiled): network forward dispatch 222 s, backward 183 s, Adam 70 s, PER sampling 46 s, exploration selector 48 s, `AsyncVectorEnv` pipe reads 107 s, `torch.tensor` creation 16 s, GIF quantise 13 s, 4.1 M `.item()` calls 9 s.

Per-section breakdown of the new loop (`TD3_PROFILE_SECTIONS=1`, 8k-step smoke, ep len ≈ 45): env 28 %, policy (compiled actor + selector + tensor prep) 17 %, bookkeeping 7 %, update replay launches 7 %, GPU tail wait ~12 %.

### Non-DR canonical recipe (`td3_recommended_top50_hist2.yaml`, `sysid_best_params_hist2.yaml`)

Same code path (the DR wrapper only patches `evaluate_agent`); 8k-step smoke on cuda:3 ran clean at 1048 SPS pre-learning. Full numbers in the sweep table below.

### Sweep: 5 tasks × {DR, no-DR} × {old, new}, 100k steps each

Runner: `scripts/td3/extras/throughput_bench.py --gpus 0 2 3` (one job per GPU; GPU 1 was in use by another user). 100k steps, `learning_starts=20000`, 4 checkpoint evals. "Final eval return" = last checkpoint's `multi_env_eval.json` mean for DR runs; for no-DR runs it is the rolling training return over the last 10 % of steps (single-seed, tiny budget — a parity sanity check, not a learning result). Old = `bf9936e`.

| Job | Version | Wall (s) | Pre-learning SPS | Training SPS | Mean ep len | Final eval return |
|---|---|---:|---:|---:|---:|---:|
| juggle_dr | old | 1558 | 243 | 56 | 44 | 30.1 |
| juggle_dr | new | 181 | 1121 | 541 | 47 | 26.9 |
| juggle_dr | **speedup** | **8.6x** | 4.6x | 9.7x | | |
| juggle_nodr | old | 1553 | 246 | 56 | 42 | 21.7 |
| juggle_nodr | new | 196 | 1136 | 509 | 40 | 21.5 |
| juggle_nodr | **speedup** | **7.9x** | 4.6x | 9.1x | | |
| puck_vel_dr | old | 1442 | 246 | 61 | 50 | 234.0 |
| puck_vel_dr | new | 181 | 1103 | 555 | 51 | 270.7 |
| puck_vel_dr | **speedup** | **8.0x** | 4.5x | 9.1x | | |
| puck_vel_nodr | old | 1412 | 247 | 62 | 48 | 88.7 |
| puck_vel_nodr | new | 186 | 1123 | 557 | 48 | 102.4 |
| puck_vel_nodr | **speedup** | **7.6x** | 4.5x | 9.1x | | |
| reach_dr | old | crash | – | – | – | – |
| reach_dr | new | 141 | 1269 | 756 | 252 | -30.8 |
| reach_nodr | old | crash | – | – | – | – |
| reach_nodr | new | 136 | 1299 | 761 | 252 | -46.6 |
| reach_vel_dr | old | 779 | 259 | 122 | 252 | 134.2 |
| reach_vel_dr | new | 156 | 1215 | 734 | 252 | 140.9 |
| reach_vel_dr | **speedup** | **5.0x** | 4.7x | 6.0x | | |
| reach_vel_nodr | old | 758 | 254 | 129 | 252 | 180.3 |
| reach_vel_nodr | new | 176 | 1197 | 726 | 252 | 178.7 |
| reach_vel_nodr | **speedup** | **4.3x** | 4.7x | 5.6x | | |
| touch_dr | old | 1723 | 262 | 50 | 38 | 133.8 |
| touch_dr | new | 201 | 1071 | 491 | 39 | 161.7 |
| touch_dr | **speedup** | **8.6x** | 4.1x | 9.9x | | |
| touch_nodr | old | 1708 | 244 | 50 | 38 | 90.1 |
| touch_nodr | new | 226 | 1086 | 476 | 43 | 101.2 |
| touch_nodr | **speedup** | **7.6x** | 4.5x | 9.5x | | |

Notes:
- The old trainer could not run `paddle_reach_position`: `paddle_reach_position_reward.py` returned the reward as a 1-element array far from the goal (`-dist`) and a Python scalar inside the goal radius (`bonus`), and `EpisodeTrajectory.flush_to_buffer` died on `torch.stack` (`got [1] at entry 0 and [] at entry 11`). The new `SingleEnvVector` normalised rewards, which is why the new runs succeeded. Fixed at the source after the sweep (`np.where` + float on the single-sample path; same pattern fixed in `paddle_reach_position_negative_regions_reward.py` and `puck_goal_position_reward.py`), so old-code reach numbers do not exist in this table but the task now runs on both.
- Speedup scales with how much of the wall is update vs rollout: short-episode tasks (juggle / touch / puck_vel, ~45-step episodes → a 31-update cycle every 45 steps) get 9-10× in the training phase; `reach_vel` runs 252-step episodes (fewer cycles per step) and gets 5.6-6×. The pre-learning phase (no updates) is a uniform ~4.5×.
- New-code wall includes ~15 s of one-time `torch.compile` plus 4 background evals; old-code wall includes 4 synchronous evals.
- Final returns land in the same band old vs new on every task that both ran (differences are within single-seed noise at 100k steps: e.g. touch_dr 134 vs 162, reach_vel_nodr 180 vs 179, juggle_nodr 22 vs 22).

## Conclusion

The loop was CPU-dispatch-bound, not compute- or physics-bound: the tiny networks make per-op launch overhead the cost, and the fix is to launch once (CUDA graphs) rather than to use a bigger GPU. Across the 8 task×setting cells both versions ran, wall-clock for a 100k-step run dropped 4.3-8.6× (7.6-8.6× on the juggle-family tasks that matter), the training-phase rate 5.6-9.9×, with no visible change in what is learned at this budget. The env is now ~55 % of a rollout step and the update cycle is GPU-bound at ~43 ms. The projected canonical 2M-step DR run goes from ~10.5 h to ~1.1 h.

## Next

- Stack the two critics into one `bmm` network (halves critic kernels again).
- numpy rewrite of `PrimitiveExplorationSelector` + trajectory staging (~0.2 ms/step).
- Decide whether the 2M canonical DR run should be re-launched to confirm end-to-end parity at full budget (projected ~1.1 h).
