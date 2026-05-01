# Residual RL recipe — by gap size

**Status (2026-05-01):**

Two recipes, picked by the size of the source→target gap:

| Gap size | Recipe | Canonical config | Verified |
|---|---|---|---|
| **Small (<10% zs drop)** — e.g. paddle full-size variants | `recency_top50` (`success_top_fraction: 0.5`) | [`td3_sim2sim_residual.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3/sim2sim/td3_sim2sim_residual.yaml) | 3-seed @ 100k (2026-04-26) |
| **Big (~30% zs drop)** — paddle -50% mass-preserved | **v27 = `q_updates: 1` + Maxmin-5 critics** (`num_critics: 5`) | [`paddle50/td3_residual_v27_ensemble5.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v27_ensemble5.yaml) | 5-seed @ 300k + 1M extension (2026-04-30/05-01) |

Both recipes need per-checkpoint deterministic eval — final-step weights are unsafe for either. See `eval_all_ckpts_residual.sh`.

**Big-gap headline result (v27, 1M):** peak **98.3** (highest single-seed peak across all 30+ variants), 84% of ckpts above zs (67.54), mean 75.9. v27 is the only recipe with positive mean gain over zs at 1M; full_ft and full_ft+ensemble both drift to 0% above zs past 300k.

**Caveat:** even v27's mean is +8 over zs vs peak +31 — a structural ceiling. Diagnosed mechanism: residual-specific Q-overestimation, mitigated but not eliminated by Maxmin-5. Open follow-up campaign in [`notes/scratch/residual_exploration_plan.md`](../../scratch/residual_exploration_plan.md) (adaptation-phase exploration, NOT YET RUN — GPU contention 2026-05-01).

Full chronological logs:
- [`notes/scratch/residual_rl_drift_fix_log.md`](../../scratch/residual_rl_drift_fix_log.md) — small-gap (OLD env) campaign, 2026-04-26
- [`notes/scratch/residual_rl_paddle50_log.md`](../../scratch/residual_rl_paddle50_log.md) — big-gap (paddle50) campaign, 2026-04-29 → 2026-05-01

**TL;DR:**
- Small gap → `td3_sim2sim_residual.yaml` (recency_top50, sf=0.5).
- Big gap → `paddle50/td3_residual_v27_ensemble5.yaml` (q_updates=1 + Maxmin-5).
- Both decisively beat full_ft on stability metrics; v27 ties full_ft on peak.

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

## When to use this vs from-scratch vs full-FT

- **Use `recency_top50` (this recipe, small-gap) when**: budget is constrained (100k), target perturbations are SMALL (<10% zero-shot drop). Hits ceiling around 100-110 mean on the OLD `sim2sim_combined`.
- **Use v27 (Maxmin-5 ensemble, big-gap) when**: target has a BIG gap (>20% zs drop) and you have a working source policy. 5-seed verified peak 87.9 ± 4.8, last5 66.1, 70% above zs at 300k. 1M extension single-seed reaches peak 98.3 (the highest single-seed peak ever observed). See [§ Big-gap recipe — v27](#how-to-use-the-v27-recipe-the-best-big-gap-residual-recipe-2026-04-30-pm--maxmin-5-critics).
- **Use full-FT when**: target has a big gap AND you don't trust per-checkpoint eval. On paddle-50 at 300k, full-FT 3-seed peak = 89.6 ties v27 (87.9), but full-FT drifts to 0% above zs past 300k while v27 stays positive at 1M.
- **Use from-scratch (1M+ budget) when**: you have the budget AND a from-scratch run can actually reach competitive performance. Not the case for paddle-50: 1M from-scratch (q_updates=4, no primitives) plateaued at peak 30 — well below zero-shot 67.54. A "true from-scratch" run with `td3_recommended.yaml`'s full exploration is queued (see `notes/scratch/residual_exploration_plan.md` Part A) but not yet run.

## Big-gap envs (2026-04-29 — early iterations, superseded by v27 below)

> The v1/v2/v3 + 3-seed `no_per+qwd` results in this section are kept for
> historical context. They've been **decisively superseded by v27 (Maxmin-5)**
> documented in the next section. Skip to [§ How to use the v27 recipe](#how-to-use-the-v27-recipe-the-best-big-gap-residual-recipe-2026-04-30-pm--maxmin-5-critics)
> if you only want the current canonical big-gap recipe.

We re-ran the residual RL iteration loop on the new `sim2sim_combined.yaml`
(paddle -50% mass-preserved; **zero-shot 67.54** vs old env's 95.78 — ~30% gap).
Tested 3 recipe variants at 300k each, then 3-seed verified the winner.

### Iteration summary (single-seed, n=50 deterministic eval per ckpt)

| variant | recipe diff vs canonical | peak | mean(29) | last5 | >zs |
|---|---|---:|---:|---:|---:|
| v1 | (canonical: recency_top50, rs=0.15) | 82.5 | 49.4 | 43.3 | 2/29 |
| v2 | rs=0.15 → 0.25 | 81.9 | 43.8 | 26.3 | 4/29 |
| **v3** | per_enabled false, sf=0.0 (no_per+qwd) | **81.4** | **55.9** | 37.7 | 4/29 |
| full_ft | (full-model FT, lr 10x lower) | 91.7 | 68.2 | 64.0 | 13/29 |

All three residual variants peak at ~82 — **the peak is a structural ceiling
for residual on this base+target combo (30% gap), not a recipe issue**.
v3 (no_per+qwd) wins on `mean(29)` because it avoids the museum-of-past-peaks
collapse the canonical recipe still suffers from on big gaps.

### 3-seed verification — v3 vs full_ft (300k each)

| metric | residual v3 (3-seed mean) | full_ft (3-seed mean) | gap |
|---|---:|---:|---:|
| peak | **80.7 ± 1.6** | **89.6 ± 2.3** | +8.9 (full_ft) |
| mean(29) | 55.0 | 62.9 | +7.9 |
| last5_mean | 38.2 | 50.2 | +12.0 |
| % ckpts >zs | 27.6% | 34.5% | +6.9pp |
| peak step (median) | 70k | 40k | full_ft converges 2x faster |
| catastrophic collapse | 0/3 | 0/3 | both stable in 30-65 last5 range |

### Recommendation by gap size

| zs drop | recipe | peak (vs zs) | last5 (vs zs) | std on peak | budget |
|---|---|---:|---:|---:|---:|
| ~5% (OLD env) | residual + recency_top50 | +5 (100.7 vs 95.78) | +0 (94.8) | tight (3/3 last3 ≥ 88) | 100k |
| **~30% (NEW env, peak)** | **residual v27 (q_updates=1 + Maxmin-5)** | **+20 (87.9 vs 67.54)** | **-1 (66.1)** | 4.8 | 300k |
| **~30% (NEW env, tail)** | **residual v29 (q_updates=1 + REDQ-10-2)** | +18 (85.2) | **+3 (70.8)** | **4.4** | 300k (~2× compute of v27) |
| ~30% (NEW env, prior) | residual v25 (q_updates=1) | +18 (85.5) | -10 (57.6) | 9.0 | 300k |
| ~30% (NEW env) | full_ft | +21 (88.6) | -24 (43.3) | tight (2.0) | 300k |

**For the new env (big gap)**: pick v27 or v29 based on deployment style:
- **v27 (Maxmin-5)** — higher peak (87.9, ties full_ft); best when you can do
  per-ckpt eval and ship the best ckpt. Best single-seed peak ever observed (95.7).
- **v29 (REDQ-10-2)** — flatter trajectory (last5 70.8, **77% of ckpts above zs**);
  best for "fire-and-forget" deployment of the final-step weights. The only recipe
  where the policy *improves past 100k* (cross-seed mean peaks at 200k, not 100k).
- Both supersede v25 and v21. Both dominate full_ft on stability metrics.

### How to use the v27 recipe (the BEST big-gap residual recipe, 2026-04-30 PM — Maxmin-5 critics)

**v27 = v25 + `num_critics: 5`** (Maxmin-5: ensemble of 5 critics, min over all
5 target Qs). Builds on v25's drift fix by further tightening the Q estimate
through critic ensembling. Beats every prior recipe on every metric and
matches full_ft's peak while dominating its tail.

Canonical paper: **REDQ** (Chen et al. ICLR 2021). v27 uses the simpler
**Maxmin** variant (Lan 2020) — min over all critics, vs REDQ's random subset.

Copy `td3_sim2sim_residual.yaml` and apply these edits:

```yaml
success_top_fraction: 0.15           # was 0.5; OLD env median-split does NOT transfer to big-gap targets
priority_age_decay: 0.0001           # NEW arg (2026-04-29) — age-weighted PER
q_updates: 1                         # was 4 — drift fix (2026-04-30 AM)
num_critics: 5                       # was 2 — Maxmin-5 ensemble (2026-04-30 PM)
# target_critic_subset_size: None    # default = Maxmin (use all 5). Set to e.g. 2 for REDQ-5-2 variant.
# Everything else (q_weight_decay 1e-3, residual_scale 0.15, q_lr 3e-4) stays the same.
```

Verified config (5-seed): `scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v27_ensemble5.yaml`.

Expected performance (5-seed mean):
- Peak: **87.9 ± 4.8** (range 82.8–95.7 across seeds — best mean and tighter than v25)
- Mean(29): **71.2** — best of any recipe (+4.5 vs v25, +10.9 vs full_ft)
- Last5: **66.1 ± 13.6** — best of any recipe (+8.5 vs v25, +22.8 vs full_ft)
- **70% of ckpts above zs** across training — best ever (+22pp vs v25, +41pp vs full_ft)

Decay shape (cross-seed): peaks at ~80 around 100k, holds 63-65 through end of
300k. Two of five seeds end with last5 ≥ 76 — strictly above every full_ft seed.

vs full_ft (3-seed): v27 effectively ties on peak (87.9 vs 88.6) and dominates
on every stability metric. **For big-gap residual sim2sim/sim2real, v27 is the
new default.**

Per-checkpoint eval still recommended for picking the best ckpt, but the
final-step weights are now competitive with peak-eval for "fire-and-forget"
deployment.

### Why ensemble critics fix drift (mechanism)

Drift root cause: in residual RL the critic over-extrapolates Q-values to OOD
actions (small action subspace × frozen base → critic has limited grounding).
Q1 grows 2.6–4× during training; actor exploits this by pushing residual head
norm 5–10× higher; real returns degrade.

The standard TD3 twin-critic min reduces overestimation but with only N=2 the
bound is loose. With N=5 critics each independently initialized:
- Variance of `min(Q_1, ..., Q_N)` decreases with N
- Each critic makes different OOD extrapolation errors; min cancels them out
- More critic diversity → tighter target → less Q runaway → less actor exploit

We tested 4 ensemble configs (Phase 15 single-seed, then Phase 16 5-seed):
- Maxmin-3: too small, loose bound (84/61/52)
- **Maxmin-5: sweet spot** (88/71/66 — winning recipe)
- REDQ-5-2: random subset added variance, peak 86 single-seed but unstable
- REDQ-10-2: most pessimistic — flat tail (last5=74) but suppressed peak (80)

REDQ's random subset sampling is canonical but adds gradient variance during
the early phase; Maxmin (min-over-all) is more deterministic and won on peak.

### v29 alternative (REDQ-10-2): the tail-stability variant

Followup 5-seed verification of v29 (`num_critics: 10, target_critic_subset_size: 2`):

| metric | v29 (5s) | v27 (5s) |
|---|---:|---:|
| peak | 85.2 ± 4.4 | **87.9 ± 4.8** |
| **mean(29)** | **73.4** | 71.1 |
| **last5** | **70.8 ± 9.1** | 65.7 ± 13.2 |
| **%>zs** | **77%** | 70% |

**v29 has the most stable trajectory of any recipe ever**: cross-seed mean
trajectory is 75→72→73→72→**78**→74→66 across steps 10k→290k. Peaks at step
*200k*, not 100k. This is qualitatively different from every other recipe's
"early-peak then decay" pattern.

When to choose v29 over v27:
- Deploying final-step weights (no per-ckpt eval available)
- Need consistent above-zs performance (77% vs 70%)
- Tighter cross-seed last5 std (9.1 vs 13.2)

When to choose v27 over v29:
- Picking peak ckpt after eval (v27 peak 87.9 > v29 85.2)
- Compute-constrained (v27 N=5 vs v29 N=10 — about 2× faster)
- Want highest single-seed peak chance (v27 best 95.7 vs v29 89.6)

Verified config: `scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v29_redq10.yaml`.

```yaml
# v29 = v25 + REDQ-10-2 ensemble:
num_critics: 10
target_critic_subset_size: 2
# All other v25 knobs unchanged.
```

### 1M extension findings (2026-05-01)

Single-seed 1M extensions of v27 and v29 confirmed v27 scales well but v29 has
a delayed cliff:

| variant | peak (step) | mean(99) | last5 | %>zs |
|---|---:|---:|---:|---:|
| v27 1M (residual+Maxmin-5) | **98.3** @ 90k | 75.9 | 69.8 | **84%** |
| v29 1M (residual+REDQ-10-2) | 86.8 @ 200k | 62.2 | 58.2 | 34% |
| v32fix 1M (full_ft+Maxmin-5) | 91.0 @ 30k | 52.9 | 47.4 | 21% |
| v33fix 1M (full_ft+REDQ-10-2) | 91.5 @ 20k | 59.7 | 54.3 | 32% |
| full_ft 1M baseline (no ensemble) | 88.7 @ 100k | 53.8 | 42.1 | 23% |

(v32fix/v33fix/baseline numbers through step 560k — ran slower than residual variants;
see paddle50 log §8.16 for details.)

**Key 1M findings:**
- **v27 scales to 1M cleanly**: 84% above zs across all 99 ckpts; peak 98.3 is the
  highest single-seed peak ever observed across all 30+ residual variants.
- **v29 has a delayed cliff past step 300k**: 0-300k = 83% above zs, 300k-1M = 13% above zs.
  v29's 5-seed-300k results were misleading; do NOT trust v29 past 300k.
- **Residual + ensemble decisively beats full_ft + ensemble at 1M**: v27 is the
  ONLY recipe with positive mean gain over zs. Every other 1M run — residual or
  full_ft, with or without ensemble — averages BELOW zero-shot across the run.
- **Ensemble does NOT fix full_ft drift**: full_ft + Maxmin-5 / REDQ-10-2 give
  small peak boost (~+2 over baseline 88.7 → 91) but don't change the post-200k
  cliff. Drift mechanism is residual-specific; ensemble fix only applies there.

### Caveat on v27's "minor mean gains"

Even v27's 1M mean is only +8 over zs (75.9 vs 67.54), while peak is +31. Most
training-time ckpts hover 5-15 above zs. Three diagnoses:
1. Q-overestimation is mitigated by Maxmin-5 but not eliminated (Q1 still grows ~3×).
2. Residual head capacity (64×2, scale 0.15) may be insufficient for big-gap targets.
3. The base policy itself may be wrong in subtle ways for paddle -50% — fundamentally
   limiting what residual fine-tuning can achieve.

Open follow-ups for further improvement: TD7-style layer-norm critic, bigger
network, EMA actor, multi-paddle base policy retraining. See `residual_rl_paddle50_log.md`
§8.16 for full analysis.

### v25, v21 (deprecated; superseded by v27 and v29, kept for context)

**v25** = `q_updates: 1` (no ensemble). 5-seed peak 85.5 ± 9.0, last5 57.6, 48%>zs.
The first recipe to break the 200k drift cliff. Replaced by v27 which adds
critic ensembling on top.

**v21** = `success_top_fraction: 0.15` + `priority_age_decay: 0.0001` with default
q_updates=4 and N=2. 5-seed peak 83.5, last5 36.2 (drift cliff at 200k). Use v27
or v25 instead.

### Why 0.15 not 0.5 (mechanism)

On the OLD env (5% gap), the actor needed help with subtle perturbations.
The success buffer's MEDIAN was meaningful — episodes above the median
were genuinely better than typical. sf=0.5 worked because the median
captured useful peak signal.

On the NEW env (30% gap), peak performance (~80-90 return) is far above
sustainable performance (~40-60 return). The median is essentially noise.
sf=0.5 puts noise into the success buffer, the actor doesn't differentiate
peaks from average, and the museum effect amplifies. sf=0.15 (top 15%) puts
ONLY the rare peak transitions into success_rb. The actor learns to chase
those peaks specifically.

Adding `priority_age_decay: 0.0001` further down-weights stale peak
transitions in sampling, so the success buffer effectively holds a
"rolling top 15% of recent transitions". This combination produced the
v21 result above.

### Earlier variant: no_per+qwd (still works, but v21 is better)

Pre-v21 finding (2026-04-29 morning):

```yaml
per_enabled: false                       # was true
critic_success_sample_fraction: 0.0      # was 0.3
critic_failure_sample_fraction: 1.0      # was 0.7
```

Config: `…/paddle50/td3_residual_v3_no_per_qwd.yaml`. 3-seed peak 80.7 ± 1.6.
Less seed-sensitive than v21 (tighter cross-seed std) but lower peak ceiling
and lower mean(29). Only choose this over v21 if seed reproducibility matters
more than absolute performance.

### Scope: this fix is residual-specific (verified 2026-04-27)

We tested whether `success_top_fraction: 0.5` helps the full-model from-scratch case too, by re-running `td3_recommended.yaml` with only that single change on `sysid_best_params_hist2.yaml` for 1M steps. Per-checkpoint deterministic eval (39 ckpts + final, n=50 seed=0):

| metric | baseline (top_fraction=0.2, `hist2_motion0`) | top50 (top_fraction=0.5) |
|---|---:|---:|
| peak | **170.2** @ 750k | 157.6 @ 625k |
| mean(39) | 108.6 | 109.9 |
| last5_mean | 143.6 | 144.5 |
| final | **169.8** | 113.7 |

**Tied on average performance (mean/last5 within 1 SE), baseline slightly better on peak and final.** The data-balance fix doesn't help full-model from-scratch — likely because the museum-of-past-peaks effect compounds specifically when the actor is constrained to small residual corrections around a fixed base; in full-model training, the actor adapts more freely and the optimistic critic Q has less leverage to corrupt the policy.

**Recommendation**: keep `success_top_fraction: 0.2` as the default for `td3_recommended.yaml` (full-model, from-scratch). Only switch to `0.5` for residual fine-tuning configs.

---

## Mechanism diagnosis (why this fix matters)

### Small-gap (OLD env) drift fix
The drift-fix campaign of 21 single-seed and 6 multi-seed runs (2026-04-26) tested:
- Actor-side regularization (residual head WD, output L2, scale anneal) → all rejected
- Critic-side regularization (`q_weight_decay`) → helps secondarily
- Disabling PER + success bias → unstable across seeds
- EMA actor → operational tool, doesn't fix drift
- **Data-balance variants (`success_top_fraction`, `success_buffer_size`, `recent_episode_window_size`)** → `top_fraction: 0.5` is the unique winner

The post-peak collapse on the OLD env was traced to:
1. **Museum of past peaks** in `success_rb` — fixed by `top_fraction: 0.5`
2. **Q runaway** in the critic — secondarily addressed by `q_weight_decay: 0.001`

### Big-gap (NEW env) drift fix (2026-04-30, root cause identified)

On the new env (paddle -50%, 30% gap), the OLD env recipe failed. After 22
variants and 12 phases, we instrumented Q-value and residual-head trajectories
across all completed runs and found a universal pattern:

| metric | start of training | end of training | growth |
|---|---:|---:|---:|
| critic Q1 task mean | 0.4 - 0.5 | 1.0 - 1.6 | **2.6 - 4.0×** |
| residual head output norm | 0.015 - 0.025 | 0.07 - 0.11 | **5 - 10×** |
| return | peak 80-90 @ ~50-150k | drift to 30-45 by 290k | -50% |

**The drift mechanism is residual-specific Q-overestimation:**
1. Critic Q-values grow unboundedly because the residual setting limits action-space
   coverage (small `residual_scale` × frozen base) — the critic over-extrapolates.
2. Actor follows the gradient `-Q(s, a_base + residual)`, pushing residual head
   norm up to chase phantom Q values.
3. Real returns degrade because the residual is overshooting; critic Q keeps
   growing (no grounding signal).

Cross-check: full_ft critic Q1 *declines* over training (3.7 → 2.7 across
270k) — full FT has full action-space coverage and the critic stays grounded.

### The fix: reduce critic update frequency

We tried 4 mitigations targeting this mechanism:

| variant | knob | result |
|---|---|---|
| BC anchor on residual output | `residual_action_l2: 1.0 / 10.0` | suppresses peak below zs (peak 65 / 63) |
| Stronger critic L2 | `q_weight_decay: 0.01` (10× default) | peak ok (77) but drift not fixed |
| **Reduced critic capacity** | **`q_updates: 1` (1/4 default)** | **same peak (80.8), drift dramatically reduced (last5 65)** |

**Reducing `q_updates: 4 → 1` is the fix.** With only one critic gradient
update per env step, the critic doesn't have enough capacity to drift far
from the data, so Q stays better calibrated, residual head doesn't grow
unboundedly, and the policy stays near peak across the full 300k window.

5-seed verification: peak 85.5 ± 9.0, mean(29) 66.7 (best of any recipe),
last5 57.6 (vs v21 36.2, full_ft 43.3), 48% of ckpts above zs.

Full chronological logs:
- [`notes/scratch/residual_rl_drift_fix_log.md`](../../scratch/residual_rl_drift_fix_log.md) — OLD env (small-gap) campaign
- [`notes/scratch/residual_rl_paddle50_log.md`](../../scratch/residual_rl_paddle50_log.md) — NEW env (big-gap) campaign, ending in §8.13 Phase 13/14 fix

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

Active plan (queued, blocked on GPU contention 2026-05-01):
[`notes/scratch/residual_exploration_plan.md`](../../scratch/residual_exploration_plan.md) —
adaptation-phase exploration (Part A: true from-scratch on paddle50 with
`td3_recommended.yaml`'s full primitive exploration; Part B: v30 family =
v27 + primitives). Hypothesis: v27's structural ceiling (mean +8 vs peak +31)
is partly because the residual head only sees data near the frozen base
trajectory — re-introducing primitive exploration during adaptation should
broaden the rollout distribution.

Other ideas (lower priority — see paddle50 log §8.16 for analysis):
- TD7-style layer-norm critic (architectural — should reduce Q-overestimation
  more than ensembling alone)
- Bigger residual head capacity (64×2 may be insufficient for 30%-gap targets)
- Multi-paddle base policy retraining (the base policy itself may be
  fundamentally wrong for paddle -50%)
- Best-of-eval-checkpoint tracker in `td3_training.py` (eliminates the
  per-checkpoint eval requirement for deployment)
