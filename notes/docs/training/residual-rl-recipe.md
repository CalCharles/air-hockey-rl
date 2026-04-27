# Residual RL recipe — `recency_top50`

**Status:** validated 2026-04-26 across 3 seeds on `hist2_motion0 → sim2sim_combined`. This is the recommended recipe for residual fine-tuning a frozen-base TD3 policy on a perturbed-physics target environment.

**TL;DR:** the canonical config `scripts/smooth_policy/amp_history/configs/td3/sim2sim/td3_sim2sim_residual.yaml` already encodes this recipe. Just fill in `model_path`, `config`, `log_parent_dir`, `seed`, then run training + per-checkpoint eval.

---

## What changed (the single-knob fix)

`success_top_fraction: 0.2 → 0.5`. That's the entire fix.

Default config classified episodes as "successes" if their return was ≥ the 80th percentile of the recent 500 episodes — so the success threshold ratcheted up early in training and stayed high. Old peak transitions accumulated in `success_rb` ("museum of past peaks") and the actor's gradient kept seeing optimistic state-action pairs the current policy couldn't reproduce. Result: the policy degraded after an early peak, then catastrophically collapsed past step 100k.

Setting `success_top_fraction: 0.5` makes the threshold = MEDIAN of recent returns. ~50% of episodes go to `success_rb`, ~50% to `failure_rb` at all times. Threshold tracks current policy quality and can never lock in stale data.

Why not `top_fraction = 0.99` ("everything is a success")? Tested: it regresses because `failure_rb` starves (only the worst 1% of episodes go there) and the critic_failure_sample_fraction=0.7 then samples mostly an empty buffer. Median is the sweet spot.

---

## Full recipe

In `td3_sim2sim_residual.yaml`:

```yaml
# Data balance — the headline fix
success_top_fraction: 0.5            # MEDIAN split, was 0.2 (top-20%)
per_enabled: true                    # PER restored
critic_success_sample_fraction: 0.3  # default
critic_failure_sample_fraction: 0.7

# Residual head — give it room to learn
residual_scale: 0.15                 # was 0.05; head needs corrections > ±5%

# Critic — secondary regularisation
q_weight_decay: 0.001                # 10x baseline 1e-4; bounds Q magnitudes
q_updates: 4                         # `lower_qlr` setting
q_lr: 0.0003

# Budget — 100k is enough
total_timesteps: 100000              # peak window is 20-60k
checkpoint_interval: 10000           # saves 9 ckpts + final
```

The above is what `td3_sim2sim_residual.yaml` already contains.

---

## Training a residual policy

### 1. Edit the config to point at your source/target

In `scripts/smooth_policy/amp_history/configs/td3/sim2sim/td3_sim2sim_residual.yaml`, fill in:

```yaml
config: "<path to target sim YAML>"           # e.g. configs/new_juggle/sim2sim_combined.yaml
model_path: "<path to source checkpoint>"     # full path to source/<run>/checkpoint_<step>/model.pth
log_parent_dir: "runs/td3/sim2sim/<src>_to_<tgt>/residual/seed0"
run_name: "td3_sim2sim_residual_<your-tag>"
seed: 0                                       # change for each seed
device: "cuda:N"
```

`full_checkpoint_load: residual` should already be set (this loads the source as the frozen base, builds a fresh residual head and critic).

### 2. Launch training (run ≥3 seeds)

```bash
# Seed 0
.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
  --args-file scripts/smooth_policy/amp_history/configs/td3/sim2sim/td3_sim2sim_residual.yaml

# Seed 1, 2: copy the config, change `seed:` and `log_parent_dir:` (must be unique per seed)
```

A 100k run takes ~30 min on one Quadro RTX 6000.

### 3. Per-checkpoint deterministic eval

After each seed finishes, evaluate every saved checkpoint:

```bash
bash scripts/smooth_policy/eval_all_ckpts_residual.sh \
  <log_parent_dir> \
  <target_sim_config> \
  cuda:N
```

This writes `eval_combined_ckpt_<step>/metrics.json` for each checkpoint (n=50 episodes deterministic, seed=0).

### 4. Pick the best checkpoint to deploy

Aggregate results across seeds + steps:

```bash
.venv/bin/python notes/scratch/aggregate_driftfix_results.py <run_root>
```

Or for a quick single-run check:

```python
import json, glob, re
files = sorted(glob.glob(f"{run_dir}/eval_combined_ckpt_*/metrics.json"),
               key=lambda p: int(re.search(r"ckpt_([0-9]+)", p).group(1)))
best = max(files, key=lambda f: json.load(open(f))["mean_return"])
print(best, json.load(open(best))["mean_return"])
```

**Ship the best-mean checkpoint.** Final-step weights still vary across seeds — the per-checkpoint eval is the only reliable way to find peak.

---

## Reference numbers (sanity check)

On the canonical campaign (`hist2_motion0 → sim2sim_combined`, source `checkpoint_975000`):

| metric | zero-shot | from-scratch 400k | from-scratch 1M | residual recency_top50 (3-seed) |
|---|---:|---:|---:|---:|
| peak mean | 95.78 | 82.86 | 130.28 | 100.7 |
| mean across all ckpts | — | 43.0 | 73.9 | **93.9** |
| tail (last 3-5 ckpts) | — | 72.1 | 121.0 | **94.8** |
| budget | 0 | 400k | 1M | 100k |
| catastrophic collapse? | — | n/a | no | **no** (0/3 seeds) |

Per-seed top50 detail at 100k:

| seed | peak | mean(9) | last3 |
|---|---:|---:|---:|
| 0 | 110.7 | 103.7 | 104.1 |
| 1 | 92.4 | 88.8 | 91.4 |
| 2 | 98.9 | 89.2 | 88.9 |

If you see numbers significantly worse than this, something is off — check that `success_top_fraction: 0.5` is actually loaded (the prior recipe with `per_enabled: false` would give different and worse results).

---

## When to use this vs from-scratch

- **Use this recipe when**: budget is constrained (100k), you have a working source policy, target perturbations are moderate (<35% zero-shot drop). Hits ceiling around 100-110 mean.
- **Use from-scratch (1M+ budget) when**: you have the budget and want maximum performance. From-scratch can reach ~130 mean on this target with no drift, but at 10x the env-step cost.

---

## Mechanism diagnosis (why this fix matters)

The drift-fix campaign of 21 single-seed and 6 multi-seed runs (2026-04-26) tested:
- Actor-side regularization (residual head WD, output L2, scale anneal) → all rejected
- Critic-side regularization (`q_weight_decay`) → helps secondarily
- Disabling PER + success bias → unstable across seeds
- EMA actor → operational tool, doesn't fix drift
- **Data-balance variants (`success_top_fraction`, `success_buffer_size`, `recent_episode_window_size`)** → `top_fraction: 0.5` is the unique winner

The post-peak collapse was traced to two mechanisms in the original drift study:
1. **Museum of past peaks** in `success_rb` — fixed by `top_fraction: 0.5` (this recipe)
2. **Q runaway** in the critic — secondarily addressed by `q_weight_decay: 0.001`

Setting `top_fraction: 0.5` largely fixes (1), and the critic L2 keeps (2) in check.

Full chronological log: [`notes/scratch/residual_rl_drift_fix_log.md`](../../scratch/residual_rl_drift_fix_log.md). Single-knob ablation table: §5 of that log.

---

## Related code knobs (in `td3_training.py` Args)

These were added during the campaign for ablations. The default values keep them inactive — you don't need to touch them for this recipe, but they're available if you want to revisit:

| Args field | Purpose | Default | Tested? |
|---|---|---|---|
| `residual_weight_decay: float` | Adam weight_decay on residual head | 0.0 | Rejected (any value) |
| `residual_scale_end: float \| None` | Linear anneal of residual_scale | None | Rejected |
| `residual_ema_decay: float \| None` | EMA copy of residual head, saves `model_ema.pth` | None | Operational tool |
| `residual_action_l2: float` | L2 penalty on residual *output* | 0.0 | Rejected |

If you set `residual_ema_decay: 0.9999`, also use `bash scripts/smooth_policy/eval_all_ckpts_residual_ema.sh` to evaluate the EMA actor copy (saved as `model_ema.pth` per ckpt).

---

## Real-world residual

The same recipe runs on the real-world async pipeline via
`async_td3_real_modular.py` with `full_checkpoint_load: "residual"` in the
config. Canonical config:
[`scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual.yaml).

```bash
.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real_modular \
  --train-args <source_ckpt>/args.yaml \
  --args-file scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual.yaml
```

Wiring is identical to sim2sim:
- The same `ResidualActor` wrapper from `scripts/smooth_policy/residual_agent.py`.
- The same data-balance recipe (`success_top_fraction: 0.5`, `q_weight_decay: 0.001`,
  `residual_scale: 0.15`, `q_lr: 0.0003`, `q_updates: 4`, PER on).
- The same per-checkpoint-eval requirement: train with
  `enable_periodic_checkpointing: true` and ship the best checkpoint, NOT
  the final one.

The one functional delta vs sim2sim:

- **Replay seed**: real residual MUST use `replay_source_priority: "warmstart_only"`
  with HDF5 dirs in `warm_start_hdf5_dirs`. Loading a checkpoint replay (which
  was collected under the source's dynamics) would teach the new critic to
  value the obsolete dynamics — the canonical config keeps
  `load_replay_from_checkpoint: false` for this reason.

`model.pth` from a residual run contains the wrapped `ResidualActor` state_dict
(base + residual + clamp buffers); rollout / eval scripts need to rebuild the
same `ResidualActor` shell to load it. Standard sim2sim eval drivers already
do this; verify your real-world rollout target supports it before deploying.

## Open follow-ups

- **5-seed re-run** of `recency_top50` — current 3-seed sample tightens variance but more seeds would help.
- **200k budget extension** — does stability hold past 100k under this recipe?
- **`top50 + smaller_buf` combo** — both target the museum from different angles; might compound.
- **Generalisation** — test on other sim2sim pairs (or on sim2real).
