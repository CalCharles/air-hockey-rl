# Residual RL recipe — sim2sim and sim2real fine-tuning

The canonical recipe for fine-tuning a sim-pretrained TD3 policy onto a target with a meaningful sim-to-sim or sim-to-real gap. Three configs ship under [`configs/td3/sim2sim/warp075_p30_residual/`](../../../configs/td3/sim2sim/warp075_p30_residual/), each verified at 1M env steps. Pick by target severity.

## Canonical recipe

| Parameter | Value | Notes |
|-----------|-------|-------|
| Algorithm | TD3 residual (frozen base + residual head) | `full_checkpoint_load: residual` |
| `residual_scale` | **0.15** | Head can correct base by ±15% — wider than the 5% default; needed for non-trivial gaps |
| CQL penalty (`cql_alpha`) | **20** | Anti-overestimation regularizer; sweet zone 5–20 |
| BC anchor | **off** | Stacking BC anchoring on top of CQL hurts |
| Exploration primitives | **off** | Stacking exploration on top of CQL hurts |
| `num_critics` (Maxmin-N) | **5** | Min over all 5 target Q-heads |
| `q_updates` per env step | **1** | Stacking q_updates=4 on top of actor=2 backfires |
| `actor_updates_per_iteration` | **2** (warp ≤ 0.075) / **4** (warp 0.10) | Load-bearing knob found in the 2026-05-08 campaign |
| `target_network_frequency` | **2** | Polyak fix (2026-05-06) active |
| `q_weight_decay` | 1e-3 | 10× baseline |
| `q_lr`, `policy_lr` | 3e-4, default | |
| Budget | **1M** env steps | ≈ 2h45m on one RTX 6000 |
| Checkpoint interval | 50k | 20 checkpoints + final |

The recipe is recorded directly in each of the canonical YAMLs; no need to compose it manually.

## Recipe boundary

Verified on the canonical warp-y-sine target family. Gap severity is parameterized as `(warp amplitude, paddle scale)`.

| Target | Warp · paddle | zs baseline | Canonical config | actor_updates |
|--------|---------------|-------------|------------------|---------------|
| canonical big-gap | 0.075 · −30% | 48 | [`phaseC_actor2_1M.yaml`](../../../configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml) → [`configs/new_juggle/sim2sim_warp075_p30.yaml`](../../../configs/new_juggle/sim2sim_warp075_p30.yaml) | **2** |
| mild-paddle | 0.075 · −10% | 49 | [`phaseD_actor2_p10_1M.yaml`](../../../configs/td3/sim2sim/warp075_p30_residual/phaseD_actor2_p10_1M.yaml) → [`configs/new_juggle/sim2sim_warp075_p10.yaml`](../../../configs/new_juggle/sim2sim_warp075_p10.yaml) | **2** |
| harder-warp | 0.10 · −30% | (lower) | [`phaseD_actor4_w10_1M.yaml`](../../../configs/td3/sim2sim/warp075_p30_residual/phaseD_actor4_w10_1M.yaml) → [`configs/new_juggle/sim2sim_warp100_p30.yaml`](../../../configs/new_juggle/sim2sim_warp100_p30.yaml) | **4** |
| out of range | 0.125 · −30% | — | recipe **fails** — full-FT may help; consider a different source policy |

**1M single-seed best (back-half mean / peak):**
- `env_mild_p10` (warp 0.075, paddle −10%): mean **117** [94, 142], peak **177** (3.6× zs).
- `env_canonical_p30` (warp 0.075, paddle −30%): mean **97** [77, 121], peak **170**.

## Small-gap variant

For < 10% zero-shot drop, the simpler small-gap recipe applies — same residual scaffolding but with `recency_top50` data balance, no CQL. Config: [`configs/td3/sim2sim/td3_sim2sim_residual.yaml`](../../../configs/td3/sim2sim/td3_sim2sim_residual.yaml), sim target [`configs/new_juggle/sim2sim_combined.yaml`](../../../configs/new_juggle/sim2sim_combined.yaml). ≈ 30 min for the 100k budget.

## Real-world transfer

The real-robot residual fine-tune uses [`configs/td3_real_world/td3_residual.yaml`](../../../configs/td3_real_world/td3_residual.yaml). It points at [`configs/real_configs/rollout_config_residual.yaml`](../../../configs/real_configs/rollout_config_residual.yaml) (task `puck_juggle_upper_half_reward`) and otherwise mirrors the canonical big-gap residual recipe. Set `model_path:` to a `training_state.pth` from a sim TD3 run, then launch via `scripts/td3/extras/async_td3_real.py`. See [`environments/real-world/td3-async-replay.md`](../environments/real-world/td3-async-replay.md) for the async stack.

## Training and evaluation

### 1. Edit the config

In whichever recipe YAML you picked, set:

```yaml
config: <sim target YAML — already filled in the canonical files>
model_path: <path to source training_state.pth or model.pth — runs/td3/.../checkpoint_*/...>
log_parent_dir: runs/td3/sim2sim/<your_tag>/seed0
run_name: <your tag>
seed: 0      # change for each seed
device: cuda:0
```

### 2. Launch

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml \
  --num-envs 1
```

Run ≥ 3 seeds; copy the config, edit `seed:` and `log_parent_dir:` (must be unique per seed).

### 3. Per-checkpoint eval and best-checkpoint selection

The trainer writes a checkpoint every 50k steps and a `runs/.../args.yaml` snapshot. To eval a single checkpoint:

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml \
  --eval-mode \
  --model-path runs/td3/sim2sim/<tag>/seed0/checkpoint_<step>/model.pth \
  --total-timesteps 12500 \
  --num-envs 1
```

For per-checkpoint sweeps, iterate over `checkpoint_*` dirs manually and compare `eval_combined_ckpt_<step>/metrics.json["mean_return"]`. Ship the best-mean checkpoint.

## When to use this vs from-scratch vs full-FT

- **Use this recipe** for sim-to-sim or sim-to-real fine-tuning where the source policy is competent on the source environment. Residual is much cheaper than from-scratch (1M vs 1M but with a warm-started actor + frozen base = much faster rise) and more stable than full-FT on big gaps.
- **Use from-scratch** when you have the budget AND a from-scratch run can actually reach competitive performance on the target. For the warp075-family targets that's not the case (from-scratch peak ~112 at 400k on `warp075_p30`).
- **Use full-FT** when the source policy is poor on the target (so the residual head doesn't have a useful base) and a slow full-network fine-tune is cheaper than re-training.

## Background and history

- The single-knob fix that unlocked stable residual training was `success_top_fraction: 0.5` (median-split instead of 80th-percentile threshold) — see [`notes/scratch/residual_rl_drift_fix_log.md`](../../scratch/residual_rl_drift_fix_log.md) for the full investigation.
- The 2026-05-06 Polyak-averaging bug (silently using the actor instead of the actor-target during Q computation) affected all paddle50 v25–v30 variants and made their reported numbers unreliable.
- The 2026-05-08 hyperparameter campaign that identified `actor_updates_per_iteration` as the load-bearing knob is documented in [`notes/scratch/experiments/2026-05-08_02-17_cql-campaign.md`](../../scratch/experiments/2026-05-08_02-17_cql-campaign.md).
- The earlier paddle50 target (`sim2sim_combined.yaml` with paddle −50% mass-preserved) was deprecated because it is structurally untrainable from-scratch — 3.85M steps reached only peak 84, mean 47 — making "improvement over zero-shot" claims on it unfalsifiable.
- The deprecated paddle50 configs (v25/v27/v29/v30 family) were removed from the tree in the May 2026 cleanup. The experiment notes remain in `notes/scratch/`.
