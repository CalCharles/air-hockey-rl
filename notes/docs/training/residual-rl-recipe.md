# Residual RL recipe — by gap size

> ## ✅ Canonical big-gap recipe (2026-05-08): `redesign_cql` on `warp075_p30`.
>
> **Target**: `configs/new_juggle/sim2sim_warp075_p30.yaml` (paddle −30% mass-preserved
> + sine-y puck-obs warp 0.075; zs=48; from-scratch peak 112 at 400k).
>
> **Recipe**: CQL α=20, no BC anchor, no primitive exploration, num_critics=5,
> residual_scale=0.15, q_updates=1, target_network_frequency=2 (Polyak fix from
> 2026-05-06 active). Config: [`scripts/smooth_policy/amp_history/configs/td3/sim2sim/warp075_p30_residual/redesign_cql_1M.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3/sim2sim/warp075_p30_residual/redesign_cql_1M.yaml).
>
> **1M result, single seed**: trajectory rises through 600k, plateaus at mean **95**
> with band [74, 116] in 600k–1M. Peak 154.8 @ 492k (3.2× zs, 1.4× from-scratch peak).
> Acceptance: band lower edge sustained at zs+26, mean +47 above zs. See
> [`notes/scratch/experiments/2026-05-07_21-30_residual-on-warp075-p30.md`](../../scratch/experiments/2026-05-07_21-30_residual-on-warp075-p30.md).
>
> **The previous big-gap target (`sim2sim_combined.yaml`, paddle50) is deprecated** —
> structurally untrainable from-scratch (3.85M peak 84, mean 47), making any
> "improvement over zs" claim on it unfalsifiable. All v25–v30 paddle50 recipes
> (`v27_ensemble5`, `v29_redq10`, `v30_explore_lite`, …) were also confounded by
> the silent Polyak-averaging bug discovered 2026-05-06. Sections below covering
> v27 / v30_explore_lite are historical record only; **do not use them as
> recipe-level guidance**. Small-gap recipe (`td3_sim2sim_residual.yaml`,
> `recency_top50`) is unaffected and still applies for <10% zs drops.
>
> **Round-2 ablations on the new target (all hurt vs CQL alone at 300k):**
> BC λ=0.5 (−10), BC λ=1.0 (−13), N=10 (−11 in back-half), CQL+exploration (−19 to −31).
> Lesson: CQL alone is sufficient; stacking additional anchoring or exploration suppresses learning.

**Status (2026-05-04, superseded as of 2026-05-07):**

Two recipes for big-gap; pick by deployment style.

| Gap size | Recipes (5-seed verified) | Canonical configs |
|---|---|---|
| **Small (<10% zs drop)** — e.g. paddle full-size variants | `recency_top50` (`success_top_fraction: 0.5`) | [`td3_sim2sim_residual.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3/sim2sim/td3_sim2sim_residual.yaml) (3-seed @ 100k) |
| **Big (~30% zs drop)** — paddle -50% mass-preserved | **v27** (Maxmin-5, 1M-verified) for highest peak; **v30_explore_lite** (v27 + lite adaptation exploration) for tighter cross-seed last5 | [`paddle50/td3_residual_v27_ensemble5.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v27_ensemble5.yaml) ⋅ [`paddle50/td3_residual_v30_explore_lite.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v30_explore_lite.yaml) |

These are the structural lessons from the residual-RL drift-fix campaign
(both small-gap OLD env and big-gap paddle50). They should govern recipe
choices in any future residual / sim2real fine-tuning work in this repo.

### Big-gap recipe choice (5-seed cross-seed @ 300k)

| metric | v27 5-seed | v30_explore_lite 5-seed |
|---|---:|---:|
| peak | **87.94 ± 4.82** | 83.43 ± 8.08 |
| mean(29) | 71.05 ± 8.62 | 70.90 ± 6.31 |
| **last5** | 65.67 ± 13.20 | **68.40 ± 3.25** ← ~4× tighter std |
| %>zs | 69.66 ± 22.80 | 67.59 ± 27.65 |

The two recipes **tie within seed noise on peak / mean / %>zs**. The
real difference is **cross-seed last5 variance**: v30_explore_lite's
late-stage policies are dramatically more consistent across seeds (std
3.25 vs v27's 13.20). When deploying *without* per-checkpoint eval —
i.e. shipping the final-step or last-5 average policy — v30_explore_lite
is the safer choice. When deploying *with* per-checkpoint eval — picking
the highest-eval ckpt — v27's slightly higher peak makes it the better
choice.

**v27 is also 1M-verified** (peak 98.3 single-seed at 1M, highest of any
recipe across 30+ variants tested). v30_explore_lite is only verified
through 300k.

### Big-gap pick rule

| Deployment style | Use |
|---|---|
| Per-checkpoint eval, deploy peak ckpt | **v27** (peak 87.94 vs v30_lite 83.43 mean over 5 seeds) |
| Final-step or last-5 deployment, no per-ckpt eval | **v30_explore_lite** (last5 std 3.25 vs v27 13.20) |
| 1M+ training budget | **v27** (only recipe with 1M verification; peak 98.3 / 84% above zs at 1M) |

### Time-to-peak (paddle50, RTX 6000, residual mode q_updates=1)

≈ 8 sec per 1k env steps.

| milestone | env steps | wall clock |
|---|---:|---:|
| First ckpt above zs (both recipes, most seeds) | 10k | ≈80 sec |
| Highest single-ckpt peak (seed-dependent) | 30k–250k | 4–35 min |
| Full 300k run | 300k | ≈41 min |
| Full 1M extension (v27 only) | 1M | ≈2h45m |

See [§ Time to peak](#time-to-peak-and-budget-guidance) for budget guidance.

### From-scratch ceiling on paddle50 (2026-05-04)

We tested whether from-scratch can match transfer learning on paddle50:

| variant | budget | peak | %>zs |
|---|---:|---:|---:|
| Original-sim from-scratch (`td3_recommended.yaml`, hist4) | 300k | **>100** | n/a (different env) |
| Paddle50 from-scratch (`td3_recommended.yaml` exact) | 300k | ~30 | 0% |
| Paddle50 from-scratch + bigger network (128×3) | 300k | **36.3** | 0% |
| Paddle50 from-scratch + Maxmin-5 ensemble | 300k | 25.6 | 0% (HURTS) |
| Paddle50 from-scratch (Part A, full exploration) | **1M** | 63.3 | 0% |
| Paddle50 zero-shot (transfer, no training) | 0 | 67.54 | n/a |
| Paddle50 v27 residual | 300k | 87.94 | 70% |

**Paddle50 is structurally much harder than the original sim.** No
300k from-scratch recipe reaches zero-shot. Even 1M from-scratch with
full exploration peaks at 63 — below zero-shot. **Use a transfer
recipe (residual or full_ft); from-scratch isn't viable on paddle50.**

Maxmin-5 helps residual (where Q-overestimation is the dominant failure
mode) but actively hurts from-scratch (the over-pessimism slows initial
Q-learning).

Full chronological logs:
- [`notes/scratch/residual_rl_drift_fix_log.md`](../../scratch/residual_rl_drift_fix_log.md) — small-gap (OLD env) campaign, 2026-04-26
- [`notes/scratch/residual_rl_paddle50_log.md`](../../scratch/residual_rl_paddle50_log.md) — big-gap (paddle50) campaign, 2026-04-29 → 2026-05-04 (§8.17–§8.19)

**TL;DR:**
- Small gap → `td3_sim2sim_residual.yaml` (recency_top50, sf=0.5).
- Big gap, peak deployment → **`paddle50/td3_residual_v27_ensemble5.yaml`** (5-seed peak 87.94 ± 4.82, 1M-verified).
- Big gap, fire-and-forget deployment → **`paddle50/td3_residual_v30_explore_lite.yaml`** (5-seed last5 68.40 ± 3.25, ~4× tighter cross-seed tail std).
- Both decisively beat full_ft on stability and tie on peak.
- From-scratch on paddle50 doesn't work — even 1M with bigger net plateaus below zero-shot. Use residual.

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

### How to use the v27 recipe (the canonical big-gap residual recipe, 2026-04-30 PM — Maxmin-5 critics)

**v27 = v25 + `num_critics: 5`** (Maxmin-5: ensemble of 5 critics, min over all
5 target Qs). Builds on v25's drift fix by further tightening the Q estimate
through critic ensembling. Beats every prior recipe on every metric and
matches full_ft's peak while dominating its tail.

**This is the canonical recipe. Build off this. Future sim2sim and
sim2real residual work should treat v27 as the standard baseline.**
Reasons:
- Highest peak (5-seed 87.94 ± 4.82, single-seed 1M peak 98.3 — best of
  30+ variants tested).
- Fastest rise (most seeds above zero-shot by step 10k ≈ 80 sec).
- Most stable 1M trajectory (84% of all 99 ckpts above zs across 1M; the
  only recipe with positive mean gain over zs at 1M).
- Conservative posture (zero adaptation-phase exploration, small
  `residual_scale: 0.15`, `q_updates: 1`, Maxmin-5) — every other recipe
  that loosens any of these knobs does worse.
- **Ensemble size matters a lot.** Maxmin-3 (peak 84, mean 61, last5 52)
  is dramatically worse than Maxmin-5 (peak 88, mean 71, last5 66). N=5
  is the sweet spot — pessimistic enough to stop Q-runaway, not so
  pessimistic that learning slows.

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
fallback default** (preferred is v30_explore_lite, see next section).

Per-checkpoint eval still recommended for picking the best ckpt.

### How to use the v30_explore_lite recipe (alternative — fire-and-forget only)

> **This is an alternative, not the default. Pick v27 unless you have a
> specific deployment reason to ship final-step weights without per-ckpt eval.**
> v30_explore_lite trades v27's higher peak (87.94 → 83.43) for a tighter
> cross-seed last5 std (13.20 → 3.25). It is verified at 300k only — the
> 1M-step behavior is unknown. At 1M, only v27 has been verified.

**Note on exploration findings:** Across all the v30 family experiments, the
clear takeaway was that **adaptation-phase exploration only ever ranges
between "neutral" and "actively harmful"** for residual fine-tuning. Conservative,
low or zero exploration is what works:

- **v27 (zero adaptation-phase exploration)** — best peak.
- **v30_explore_lite (~3% chance, half base strength)** — ties v27 on
  peak/mean within seed noise; only wins on cross-seed last5 std.
- **v30_explore_full (matches base-policy strength, ~5–15% chance)** —
  collapses policies. `%>zs` drops from 70% → 19%, peak drops by ~10.
- v30_explore_directional_only (lite chance, only directional primitives)
  — last5 cliff to 41 (vs lite's 71).

The pattern is: **more exploration → faster collapse and lower peaks**.
Future sim2sim/sim2real work should default to v27 (no exploration) and
treat any reintroduction of adaptation-phase exploration as a deliberate
last-resort experiment, not a starting point.

**v30_explore_lite = v27 (Maxmin-5) + moderate adaptation-phase primitive exploration.**
v27 zeros all primitive exploration during residual fine-tuning, so the rollout
distribution stays narrow around the frozen base policy and the critic only sees
near-base data. v30_explore_lite re-enables the same primitive exploration the
base policy was trained with, but at **half** the strength (chance 0.10→0.03
over 50k steps, vs. base training's 0.15→0.05). This broadens the data
distribution the residual head learns from without disrupting it.

Copy `td3_residual_v27_ensemble5.yaml` and apply these edits:

```yaml
# Adaptation-phase exploration — the v30_explore_lite addition
exploration_noise: 0.1                 # was 0.05; matches base-policy training
exploration_primitive_chance: 0.03                       # steady-state primitive override rate (lite)
exploration_primitive_chance_start: 0.10                 # start higher; anneal in
exploration_primitive_chance_pre_learning_starts: 0.10
exploration_primitive_chance_anneal_steps: 50000

# Full primitive weight set — same as base-policy training
exploration_primitive_weight_stand_still: 0.2
exploration_primitive_weight_same_direction: 1.0
exploration_primitive_weight_y_aligned: 1.0
exploration_primitive_weight_target_position_directional: 1.0
# (anneal-phase weights identical)

# Everything else stays at v27 settings (num_critics=5, q_updates=1, sf=0.15,
# priority_age_decay=1e-4, residual_scale=0.15, q_weight_decay=1e-3, q_lr=3e-4).
```

Verified config (5-seed): `scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v30_explore_lite.yaml`
(seed1-4 variants: `…_seed{1,2,3,4}.yaml`).

Expected performance (5-seed @ 300k, n=50 deterministic eval per ckpt, zs=67.54):
- Peak: **83.43 ± 8.08** — slightly below v27's 87.94 ± 4.82 (-4.5 mean)
- Mean(29): **70.90 ± 6.31** — ties v27's 71.05 ± 8.62
- **Last5: 68.40 ± 3.25** — beats v27's 65.67 ± 13.20; **cross-seed std ~4× tighter**
- %>zs: 67.59 ± 27.65 — ties v27's 69.66 ± 22.80

Per-seed:

| seed | peak | mean(29) | last5 | %>zs |
|---|---:|---:|---:|---:|
| 0 | 80.72 | 73.63 | 72.58 | 83% |
| 1 | 94.44 | 79.02 | 68.53 | 90% |
| 2 | 72.30 | 61.73 | 63.74 | 21% |
| 3 | 86.34 | 70.39 | 69.82 | 79% |
| 4 | 83.36 | 70.09 | 68.35 | 67% |

The 2-seed signal in the original §8.17 writeup (which suggested v30_lite
strictly dominated v27) was an optimistic sample — both early seeds
landed in the high quartile. The full 5-seed picture is more honest:
**v30_lite trades v27's slightly higher peak for dramatically more
consistent late-stage policies.**

#### When to choose v30_explore_lite over v27

- **Fire-and-forget deployment** (no per-ckpt eval at deployment time):
  v30_lite's last5 std 3.25 vs v27's 13.20 means you're getting a much
  more predictable late-stage policy. With per-ckpt eval, v27's higher
  peak is more valuable.
- **Predictable expected returns** matter more than max returns: v30_lite
  has tighter mean std (6.31 vs 8.62) and last5 std (3.25 vs 13.20).
- **You only have 300k budget**: v27 has been verified at 1M (peak 98.3,
  84% > zs); v30_lite has not. If you can spend 1M, v27 has the edge.

#### When to choose v27 over v30_explore_lite

- **Per-ckpt eval is part of the deployment pipeline**: v27's higher peak
  (87.94 vs 83.43 mean) wins; per-ckpt eval recovers the gap.
- **Highest single-seed peak chance**: v27 max-seed-peak 95.74 vs
  v30_lite's 94.44. Marginal but real.
- **1M training budget**: only v27 is 1M-verified.

#### Tested but inferior variants (2026-05-03)

| variant | exploration knob | peak (2-seed) | mean | last5 | %>zs |
|---|---|---:|---:|---:|---:|
| **v30_explore_lite** | chance 0.10→0.03, full primitive set | 87.58 | 76.33 | 70.55 | 86.5% |
| v30_explore_directional_only | chance 0.10→0.03, only same_direction + target_position_directional | 84.22 | 57.90 | 41.06 | 46.5% |
| v30_explore_full | chance 0.15→0.05, full primitive set (=base-training strength) | 76.41 | 56.98 | 51.20 | 19% |

**Two clear "wrong" directions:**
1. **Too much exploration (v30_explore_full) HURTS.** Matching base-policy
   exploration strength (chance 0.15→0.05) drops %>zs from 70% to 19%. The
   primitive override drowns out the residual head's learning signal.
2. **Stripping out diversity primitives HURTS.** v30_explore_directional_only
   keeps only `same_direction` and `target_position_directional`. Cliff
   returns (last5 41 vs lite's 71).

The diversity primitives (`stand_still`, `y_aligned`) matter even though the
base already handles vertical alignment — they widen the off-base data
distribution the critic sees.

#### Mechanism (why lite exploration helps the residual but full exploration hurts)

In residual RL, the critic over-extrapolates Q-values to OOD actions because
the action subspace it explores is small (residual head produces tiny
corrections × frozen base → narrow rollout distribution). Maxmin-5 ensembling
(v27) tightens the target-Q bound but doesn't change the data distribution.

Lite primitive exploration replaces ~3-10% of env steps with raw primitive
actions — broadening rollout coverage AWAY from the base policy's narrow
behavior basin. The residual head's critic now grounds its Q-estimates against
genuinely diverse data, reducing OOD extrapolation drift.

Full primitive exploration goes too far: when ~5-15% of env steps are primitive
overrides, the residual head's learning signal is dominated by primitive
trajectories rather than its own (base + residual) trajectories. The residual
head can no longer "find" its corrective signal because the data doesn't
reflect the residual's actual contribution.

#### Caveat: 300k only (1M behavior unknown)

v30_explore_lite is verified at 5 seeds × 300k. v27 is verified at 5 seeds ×
300k + 1 seed × 1M. v30_lite's 1M behavior is unknown — until tested,
prefer v27 for 1M-scale runs.

### Time to peak and budget guidance

Both recipes are extremely fast on a Quadro RTX 6000 (paddle50 sim, residual
mode with q_updates=1):
- **300k env steps**: ≈41 min wall clock (≈8.2 sec per 1k steps)
- **1M env steps**: ≈2h45m wall clock (v27 1M extension reference)

Time-to-checkpoint-above-zs (n=50 deterministic eval, per-ckpt):
- v30_explore_lite: 4/5 seeds above zs at step 10k = **≈80 sec wall clock**
  (first checkpoint logged with `checkpoint_interval: 10000`)
- v27: about 4/5 seeds above zs by step 10k

Time-to-absolute-peak (highest single ckpt across the run; *which* ckpt is
peak is seed-dependent):
- v30_explore_lite 5-seed peaks: 20k, 40k, 50k, 220k, 250k (mean ~116k) ≈ 3–35 min
- v27 5-seed peaks: 30k, 90k, 100k, 120k, 250k (mean ~118k) ≈ 7–35 min

Both recipes give roughly the same time-to-peak distribution. Some seeds
peak as early as 20-40k (≈3-7 min), others as late as 220-250k (≈30-35 min).
You don't know which seed you got until you eval all checkpoints.

**Practical implication**: don't budget the full 300k just to "wait for the
peak". With per-checkpoint eval, a useful checkpoint arrives within minutes.
The reason to run 300k+ is to collect more candidate ckpts for eval (more
ckpts = better best-of-eval). Diminishing returns past 200k for residual.

For a sim2real fine-tune where eval is expensive, run 100k (≈14 min) and eval
all 9 checkpoints. For a research comparison run 300k; for a 1M definitive
result, run 1M.

### How to use the v30_explore_lite recipe (alternative big-gap recipe, 2026-05-04 5-seed verified)

**v30_explore_lite = v27 (Maxmin-5) + moderate adaptation-phase primitive exploration.**
v27 zeros all primitive exploration during residual fine-tuning, so the rollout
distribution stays narrow around the frozen base policy and the critic only sees
near-base data. v30_explore_lite re-enables the same primitive exploration the
base policy was trained with, but at **half** the strength (chance 0.10→0.03
over 50k steps, vs. base training's 0.15→0.05). This broadens the data
distribution the residual head learns from without disrupting it.

Copy `td3_residual_v27_ensemble5.yaml` and apply these edits:

```yaml
# Adaptation-phase exploration — the v30_explore_lite addition
exploration_noise: 0.1                 # was 0.05; matches base-policy training
exploration_primitive_chance: 0.03                       # steady-state primitive override rate (lite)
exploration_primitive_chance_start: 0.10                 # start higher; anneal in
exploration_primitive_chance_pre_learning_starts: 0.10
exploration_primitive_chance_anneal_steps: 50000

# Full primitive weight set — same as base-policy training
exploration_primitive_weight_stand_still: 0.2
exploration_primitive_weight_same_direction: 1.0
exploration_primitive_weight_y_aligned: 1.0
exploration_primitive_weight_target_position_directional: 1.0
# (anneal-phase weights identical)

# Everything else stays at v27 settings (num_critics=5, q_updates=1, sf=0.15,
# priority_age_decay=1e-4, residual_scale=0.15, q_weight_decay=1e-3, q_lr=3e-4).
```

Verified config (5-seed): `scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v30_explore_lite.yaml`
(seed1-4 variants: `…_seed{1,2,3,4}.yaml`).

Expected performance (5-seed @ 300k, n=50 deterministic eval per ckpt, zs=67.54):
- Peak: **83.43 ± 8.08** — slightly below v27's 87.94 ± 4.82 (-4.5 mean)
- Mean(29): **70.90 ± 6.31** — ties v27's 71.05 ± 8.62
- **Last5: 68.40 ± 3.25** — beats v27's 65.67 ± 13.20; **cross-seed std ~4× tighter**
- %>zs: 67.59 ± 27.65 — ties v27's 69.66 ± 22.80

Per-seed:

| seed | peak | mean(29) | last5 | %>zs |
|---|---:|---:|---:|---:|
| 0 | 80.72 | 73.63 | 72.58 | 83% |
| 1 | 94.44 | 79.02 | 68.53 | 90% |
| 2 | 72.30 | 61.73 | 63.74 | 21% |
| 3 | 86.34 | 70.39 | 69.82 | 79% |
| 4 | 83.36 | 70.09 | 68.35 | 67% |

The 2-seed signal in the original §8.17 writeup (which suggested v30_lite
strictly dominated v27) was an optimistic sample — both early seeds
landed in the high quartile. The full 5-seed picture is more honest:
**v30_lite trades v27's slightly higher peak for dramatically more
consistent late-stage policies.**

#### When to choose v30_explore_lite over v27

- **Fire-and-forget deployment** (no per-ckpt eval at deployment time):
  v30_lite's last5 std 3.25 vs v27's 13.20 means you're getting a much
  more predictable late-stage policy. With per-ckpt eval, v27's higher
  peak is more valuable.
- **Predictable expected returns** matter more than max returns: v30_lite
  has tighter mean std (6.31 vs 8.62) and last5 std (3.25 vs 13.20).
- **You only have 300k budget**: v27 has been verified at 1M (peak 98.3,
  84% > zs); v30_lite has not. If you can spend 1M, v27 has the edge.

#### When to choose v27 over v30_explore_lite

- **Per-ckpt eval is part of the deployment pipeline**: v27's higher peak
  (87.94 vs 83.43 mean) wins; per-ckpt eval recovers the gap.
- **Highest single-seed peak chance**: v27 max-seed-peak 95.74 vs
  v30_lite's 94.44. Marginal but real.
- **1M training budget**: only v27 is 1M-verified.

#### Tested but inferior variants (2026-05-03)

| variant | exploration knob | peak (2-seed) | mean | last5 | %>zs |
|---|---|---:|---:|---:|---:|
| **v30_explore_lite** | chance 0.10→0.03, full primitive set | 87.58 | 76.33 | 70.55 | 86.5% |
| v30_explore_directional_only | chance 0.10→0.03, only same_direction + target_position_directional | 84.22 | 57.90 | 41.06 | 46.5% |
| v30_explore_full | chance 0.15→0.05, full primitive set (=base-training strength) | 76.41 | 56.98 | 51.20 | 19% |

**Two clear "wrong" directions:**
1. **Too much exploration (v30_explore_full) HURTS.** Matching base-policy
   exploration strength (chance 0.15→0.05) drops %>zs from 70% to 19%. The
   primitive override drowns out the residual head's learning signal.
2. **Stripping out diversity primitives HURTS.** v30_explore_directional_only
   keeps only `same_direction` and `target_position_directional`. Cliff
   returns (last5 41 vs lite's 71).

The diversity primitives (`stand_still`, `y_aligned`) matter even though the
base already handles vertical alignment — they widen the off-base data
distribution the critic sees.

#### Mechanism (why lite exploration helps the residual but full exploration hurts)

In residual RL, the critic over-extrapolates Q-values to OOD actions because
the action subspace it explores is small (residual head produces tiny
corrections × frozen base → narrow rollout distribution). Maxmin-5 ensembling
(v27) tightens the target-Q bound but doesn't change the data distribution.

Lite primitive exploration replaces ~3-10% of env steps with raw primitive
actions — broadening rollout coverage AWAY from the base policy's narrow
behavior basin. The residual head's critic now grounds its Q-estimates against
genuinely diverse data, reducing OOD extrapolation drift.

Full primitive exploration goes too far: when ~5-15% of env steps are primitive
overrides, the residual head's learning signal is dominated by primitive
trajectories rather than its own (base + residual) trajectories. The residual
head can no longer "find" its corrective signal because the data doesn't
reflect the residual's actual contribution.

#### Caveat: 300k only (1M behavior unknown)

v30_explore_lite is verified at 5 seeds × 300k. v27 is verified at 5 seeds ×
300k + 1 seed × 1M. v30_lite's 1M behavior is unknown — until tested,
prefer v27 for 1M-scale runs.

### Time to peak and budget guidance

Both recipes are extremely fast on a Quadro RTX 6000 (paddle50 sim, residual
mode with q_updates=1):
- **300k env steps**: ≈41 min wall clock (≈8.2 sec per 1k steps)
- **1M env steps**: ≈2h45m wall clock (v27 1M extension reference)

Time-to-checkpoint-above-zs (n=50 deterministic eval, per-ckpt):
- v30_explore_lite: 4/5 seeds above zs at step 10k = **≈80 sec wall clock**
  (first checkpoint logged with `checkpoint_interval: 10000`)
- v27: about 4/5 seeds above zs by step 10k

Time-to-absolute-peak (highest single ckpt across the run; *which* ckpt is
peak is seed-dependent):
- v30_explore_lite 5-seed peaks: 20k, 40k, 50k, 220k, 250k (mean ~116k) ≈ 3–35 min
- v27 5-seed peaks: 30k, 90k, 100k, 120k, 250k (mean ~118k) ≈ 7–35 min

Both recipes give roughly the same time-to-peak distribution. Some seeds
peak as early as 20-40k (≈3-7 min), others as late as 220-250k (≈30-35 min).
You don't know which seed you got until you eval all checkpoints.

**Practical implication**: don't budget the full 300k just to "wait for the
peak". With per-checkpoint eval, a useful checkpoint arrives within minutes.
The reason to run 300k+ is to collect more candidate ckpts for eval (more
ckpts = better best-of-eval). Diminishing returns past 200k for residual.

For a sim2real fine-tune where eval is expensive, run 100k (≈14 min) and eval
all 9 checkpoints. For a research comparison run 300k; for a 1M definitive
result, run 1M.

### Why ensemble critics fix drift (mechanism)

Drift root cause: in residual RL the critic over-extrapolates Q-values to OOD
actions (small action subspace × frozen base → critic has limited grounding).
Q1 grows 2.6–4× during training; actor exploits this by pushing residual head
norm 5–10× higher; real returns degrade.

**Without an explicit Q-overestimation control, the residual policy
collapses.** Every recipe in this doc that lacks one of {Maxmin-N (N≥5),
REDQ, low `q_updates`, or strong `q_weight_decay`} ends in catastrophic
post-200k drift. With any of these mechanisms in place, the policy stays
near peak. Bounding Q is the single load-bearing knob.

The standard TD3 twin-critic min reduces overestimation but with only N=2 the
bound is loose. With N=5 critics each independently initialized:
- Variance of `min(Q_1, ..., Q_N)` decreases with N
- Each critic makes different OOD extrapolation errors; min cancels them out
- More critic diversity → tighter target → less Q runaway → less actor exploit

#### Ensemble size matters a lot — N=5 is the sweet spot, not N=3

We tested 4 ensemble configs (Phase 15 single-seed, then Phase 16 5-seed):
- Maxmin-3: too small, loose bound (84/61/52)
- **Maxmin-5: sweet spot** (88/71/66 — winning recipe)
- REDQ-5-2: random subset added variance, peak 86 single-seed but unstable
- REDQ-10-2: most pessimistic — flat tail (last5=74) but suppressed peak (80)

The jump from N=3 → N=5 is the most consequential single decision in
the entire campaign:

| ensemble | peak | mean | last5 |
|---|---:|---:|---:|
| Maxmin-3 (single-seed) | 84 | 61 | 52 |
| **Maxmin-5 (single-seed)** | **88** | **71** | **66** |
| Δ (N=3 → N=5) | +4 | +10 | **+14** |

That is not an incremental improvement — it is the difference between
"recipe drifts in the tail" and "recipe stays near peak through 1M steps".
**Do not drop below N=5.** The 1M-step v27 extension (single-seed peak
98.3, 84% of all 99 ckpts above zs) shows N=5 hits a regime where the
bound is tight enough to suppress runaway essentially permanently —
something Maxmin-3 demonstrably cannot do, and pushing further toward
N=10 / REDQ-10-2 sacrifices peak (and v29 develops a delayed cliff
past 300k).

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
- **v27 scales to 1M cleanly and is exceptionally stable across 1M steps**:
  84% above zs across all 99 ckpts; peak 98.3 is the highest single-seed
  peak ever observed across all 30+ residual variants. v27 hit a sweet
  spot — Maxmin-5 is tight enough to suppress Q-runaway permanently
  (not just through 300k), so the policy stays near peak across the full
  1M-step run rather than developing a delayed cliff. This is the
  primary reason v27 is the canonical recipe.
- **v29 has a delayed cliff past step 300k**: 0-300k = 83% above zs, 300k-1M = 13% above zs.
  v29's 5-seed-300k results were misleading; do NOT trust v29 past 300k.
  This is also why N=10 / REDQ-10-2 is not preferable to N=5 / Maxmin-5 —
  v27's calibration is the one that holds up under extended training.
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

## Real-world residual — v27 (canonical)

The async real-world pipeline (`async_td3_real_modular.py`) supports the full
v27 recipe as of 2026-05-04. The Maxmin-N / REDQ-N-M code paths in
`async_td3_real.py` (`_init_sync_learner_state`, `_run_sync_learner_iteration`,
`_save_async_checkpoint`) were generalised from the original twin-TD3 pair to
an N-critic ensemble; everything else (replay, exploration, checkpointing) was
already shared.

Canonical configs:
- args-file (online behaviour): [`td3_real_world/td3_residual.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual.yaml)
- train-args (architecture + ensemble): [`td3_real_world/td3_residual_train_args.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual_train_args.yaml)

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real_modular \
  --train-args scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual_train_args.yaml \
  --args-file  scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual.yaml \
  --model-path <path-to-source-checkpoint>/training_state.pth
```

`--model-path` must point at a `training_state.pth`, NOT a bare `model.pth`.
`_load_training_state_checkpoint` validates that the loaded dict has
`actor`/`qf1..qfN`/`rng_states`/replay-buffer keys (i.e., it's the full
state-dict file `td3_training.py` writes alongside `model.pth` in every
checkpoint dir). Residual mode then extracts just `training_state["actor"]`
as the frozen base; the source's critics, replay, and optimizer state are
discarded — the new run starts with a fresh 5-critic Maxmin ensemble and
fresh Adam momentum, as a residual fine-tune should.

### Resuming a residual run

To continue an in-progress residual training run from a previous checkpoint
(e.g. picking up after the 2000-step fresh-fill phase has already been
crossed, or just continuing where the last session stopped), use the
`residual_resume` mode added 2026-05-04:

```bash
python -m scripts.smooth_policy.amp_history.amp_training.td3.extras.async_td3_real_modular \
  --train-args scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual_train_args.yaml \
  --args-file  scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual.yaml \
  --full-checkpoint-load residual_resume \
  --learning-starts-fresh-steps 0 \
  --load-replay-from-checkpoint \
  --model-path <prev-run>/checkpoint_<tag>/training_state.pth
```

The four overrides do specific work:

- **`--full-checkpoint-load residual_resume`** — new mode added 2026-05-04.
  The default `residual` mode passes the saved actor through
  `extract_deterministic_state_dict`, which strips all the keys a wrapped
  `ResidualActor` state_dict has (`base.*`, `residual.*`, `action_low`,
  `action_high`) — the load is silent and the base actor stays at fresh
  init. `residual_resume` skips that filter and loads the wrapped state_dict
  directly, restoring both the frozen base and the trained residual head.
  Critics + targets are restored from `qf1..qfN` keys; optimizer state and
  RNG are restored too.
- **`--learning-starts-fresh-steps 0`** — disables the fresh-buffer-fill
  gate. The prior run already crossed it; re-engaging would waste 2000 steps
  of robot time before the critic moves. Set to a non-zero value if you
  *want* to re-engage (e.g. resuming from very stale data).
- **`--load-replay-from-checkpoint`** — restores the success/failure
  replay buffers from the saved `training_state.pth`. Without this, the
  replay starts empty and (combined with `learning-starts-fresh-steps 0`)
  the critic samples from a near-empty buffer for the first few episodes.
  This flag is the resume-time inverse of the `load_replay_from_checkpoint:
  false` default in the args-file (which is correct for fresh starts to
  block stale source-dynamics replay, but wrong for continuing your own run).
- **`--model-path <prev>/training_state.pth`** — point at the **full
  training_state.pth** from the prior checkpoint, NOT `model.pth`.
  `residual_resume` requires the dict format with `actor`, `actor_target`,
  `qf1..qfN`, `qf1_target..qfN_target`, `rng_states`, and (optionally)
  `q_optimizer`, `actor_optimizer`, replay buffers, learner counters,
  rolling stats. `model.pth` is just the actor's state_dict and would
  fail the validation in `_load_training_state_checkpoint`.

Robust resume requires the prior run to have been saved with
`include_non_vital_training_state_fields: true` (the default in
`td3_residual.yaml`). Without it, the training_state.pth lacks optimizer
state, learner counters, `collector_total_steps`, `run_elapsed_total_s`,
and the rolling-window deques — resume falls back to a "weights only"
residual_resume: Adam momentum resets, TB step axis restarts at 0, and
rolling-50 stats start cold. Functional but lossy. See
[`notes/docs/training/checkpointing.md#resuming-real-world-async-training`](checkpointing.md#resuming-real-world-async-training)
for the full resume contract.

Architecture matching: the `--train-args` YAML's `agent_num_hidden_layers`,
`q_num_hidden_layers`, `num_critics`, and `use_last_action_in_policy_state`
must match the prior run's. Easiest source of truth: the prior run wrote
its own `args.yaml` next to `training_state.pth` — pass that as
`--train-args` if you're not sure.

`num_critics` and `target_critic_subset_size` live in the train-args YAML
(not the args-file) because they are architecture fields — the same fields
`td3_training.py` writes to its `args.yaml`. A source actor's saved
`args.yaml` from a `td3_training.py` run is itself a valid `--train-args`
input (older args.yamls predate the ensemble keys and resolve to N=2 via
the safe defaults in `_load_train_args`); the dedicated
`td3_residual_train_args.yaml` adds `num_critics: 5` so the new run spins
up the Maxmin-5 ensemble even when the source was twin-TD3.

How v27 maps onto async-real:
- `success_top_fraction: 0.15`, smaller buffers (6000/14000), 500-episode
  recency window — already wired into `_add_episode_to_shared_replay`.
- `q_updates: 1`, `q_lr: 3e-4`, `q_weight_decay: 1e-3`, `target_network_frequency: 2`,
  `actor_updates_per_iteration: 1` — already wired.
- `num_critics: 5`, `target_critic_subset_size: null` — generalised in the
  learner step (Maxmin or REDQ-N-M, branchless on N=2 to stay bit-identical
  with the legacy twin-critic kernel).
- `residual_scale: 0.15`, `residual_weight_decay: 0.0`, `residual_action_l2: 0.0` — already wired.

Difference vs sim2sim v27 (only one, intentional):
- **No PER / `priority_age_decay`** in async-real. The real-world replay
  uses a uniform success/failure mix (`SharedTD3Replay.sample`); PER and
  age decay aren't ported. v27's primary lever (Maxmin-5) carries the
  load-bearing Q-control regardless.

Everything else mirrors sim2sim v27 verbatim, including:
- **Fresh-buffer-fill phase**: `learning_starts_fresh_steps: 2000` mirrors
  sim2sim v27's `learning_starts: 2000`. For the first 2000 FRESH
  post-launch collector steps the agent rolls out (frozen base + zero-init
  residual + `exploration_noise: 0.05`) and pushes transitions into replay,
  but the critic does NOT update — its first gradient step lands on a
  buffer that has seen pure on-policy data. Run-relative semantics (counts
  only post-launch steps, ignores warm-start replay size and prior-run
  totals); a resume re-engages the gate until 2000 *new* steps land. Set
  to 0 to disable.
- **Reward weights**: `task_reward_weight: 1.0`, `motion_reward_weight: 0.0`
  (motion-Q head is still trained, but doesn't contribute to the actor
  objective — preserves v27's task-reward-only residual posture).
- **Exploration mirrored 1:1**: `exploration_primitive_chance: 0.0`,
`exploration_noise: 0.05`, weights set so `same_direction` is the only
non-zero one (moot at runtime when chance=0; mirrored verbatim for
clarity). The recipe is explicit that "v27 with **zero adaptation-phase
exploration** is the right default" and that any non-zero chance is
"unhelpful at best, actively harmful at worst" — so the real-world config
does the same. The earlier real-world residual config (predating v27) used
a `0.025` hedge; the v27 update removed it.

If you specifically need adaptation-phase exploration on the real robot,
copy v30_explore_lite (chance 0.10→0.03 anneal) wholesale — do NOT leave
the chance half-tuned between 0 and 0.03, which is uncharted territory.

Operational requirements (same as sim2sim v27):
1. `enable_periodic_checkpointing: true` — every N successful online
   episodes, the learner snapshots `qf1.pth..qf5.pth`, `qf1_target.pth..qf5_target.pth`,
   and `model.pth` into `<checkpoint_root_dir>/checkpoint_<tag>/`.
2. Per-checkpoint deterministic eval (real-world: replay the saved
   trajectories; sim2sim: rerun in sim).
3. **Ship the best checkpoint, not the final one.** v27's 1M trace shows
   84% of ckpts above zs but the per-ckpt return is still seed-dependent
   in a ±10 band.

Replay-source rule:
- **Default (canonical v27 — mirrors sim2sim)**: `warm_start_hdf5_dirs: []`
  + `load_replay_from_checkpoint: false`. Replay starts empty, exactly like
  sim2sim v27. The `learning_starts_fresh_steps: 2000` gate then fills the
  buffer with 2000 fresh on-policy transitions before the critic's first
  gradient step. Same buffer-content semantics as sim2sim. `replay_source_priority: "warmstart_only"`
  is kept as a defensive guard — if someone later flips `load_replay_from_checkpoint` on,
  this priority forces the (empty) warm-start to win and keeps the buffer
  clean of stale source-dynamics data.
- **Optional warm-start mode**: if you want to seed the buffer with prior
  HDF5 trajectories (for fire-and-forget runs without a fresh-fill phase),
  point `warm_start_hdf5_dirs` at directories of HDF5 episodes collected
  with the *same base policy on the same robot*, and set `learning_starts_fresh_steps: 0`.
  NEVER load a checkpoint replay from a source policy run
  (`load_replay_from_checkpoint: false` must stay false) — it was collected
  under the source's dynamics and would teach the new critic to value the
  obsolete dynamics.

`model.pth` from a residual run is a wrapped `ResidualActor` state_dict
(base + residual + clamp buffers). Rollout / eval scripts must rebuild the
same `ResidualActor` shell to load it; standard sim2sim eval drivers
already do this — verify your real-world rollout target supports it before
deploying.

### Data layout across resumes

`_setup_run_data_dir` creates a **new** timestamped directory for **every
launch** at `<data_root_dir>/<model_subdir>/data_<TIMESTAMP>/`. So a single
training session that goes through `launch → checkpoint → kill → resume`
produces two sibling directories:

```
runs/async_td3/data/
└── runs/td3_training/.../checkpoint_975000/
    └── data_20260504-100000/                  ← initial launch
        ├── episode_hdf5/                      ← episode IDs 0..N
        ├── reset_hdf5/
        ├── episode_summaries.jsonl
        ├── run_events.jsonl
        ├── collector_tb/  learner_tb/
        └── checkpoint_successeps_50_qupdates_3000/
            ├── training_state.pth             ← <- you resume from this
            ├── model.pth   args.yaml   config.yaml
            └── qf1.pth ... qf5.pth
└── runs/async_td3/data/<...>/data_20260504-100000/checkpoint_successeps_50_qupdates_3000/
    └── data_20260504-150000/                  ← resume launch (nested under prev ckpt)
        ├── episode_hdf5/                      ← episode IDs 0..M (RESTART)
        ├── episode_summaries.jsonl            ← per-launch
        ├── collector_tb/  learner_tb/         ← per-launch
        └── checkpoint_*/
```

What CONTINUES across launches (load-bearing for training):

- **Network weights** (actor, residual head, all 5 critics, all 5 targets) —
  via `training_state.pth`.
- **Optimizer state** (Adam momentum for both q_optimizer and actor_optimizer)
  — gated on `include_non_vital_training_state_fields: true` in the saving
  run's args-file (default in our v27 config) AND `--full-checkpoint-load
  residual_resume`.
- **Replay buffer** — gated on `--load-replay-from-checkpoint` flag.
- **RNG state** — always restored.
- **Step / time counters** (`collector_total_steps`, `run_elapsed_total_s`,
  `learner_q_updates`, `learner_actor_updates`) — restored. TB step axis
  continues from the resumed step count, not from 0.
- **Rolling-50 episode statistics** (task/motion/length/return/estop/juggles
  /contacts deques) — restored if the saving run had `include_non_vital_training_state_fields: true`.

What FRAGMENTS across launches (per-launch artifacts):

- **HDF5 trajectories** in `episode_hdf5/` — per-launch directory. Episode
  IDs restart at 0 within each launch directory. `_next_available_episode_id`
  scans the new (initially empty) folder.
- **GIFs / camera videos** in `episode_gifs/` and `episode_camera_videos/`
  — same per-launch story.
- **`episode_summaries.jsonl`** and **`run_events.jsonl`** — append-only
  files, one per launch. Each row in `episode_summaries.jsonl` records
  `episode_id` (per-launch) plus `total_steps` (global) and `actor_version`
  (global) — so episodes are uniquely identifiable across launches via
  `(launch_dir, episode_id)` or via `total_steps`.
- **TensorBoard logs** in `collector_tb/` and `learner_tb/` — per-launch
  directories. Step axis continues from the resumed counter (so curves
  visually continue if you point TB at the parent dir).
- **Latency profiles** in `latency_profiles/` — per-launch.

To view aggregated TB across all launches of a multi-launch session:

```bash
tensorboard --logdir runs/async_td3/data/<source_subdir>/
```

TB will discover `collector_tb/` and `learner_tb/` directories at any depth,
and merge runs into a single chart per scalar (each launch becomes a
separate run-line, but they overlap on the global step axis).

To aggregate the per-launch JSONLs:

```bash
find runs/async_td3/data/<source_subdir>/ -name 'episode_summaries.jsonl' \
  -exec cat {} \; | sort -k <by-total_steps>  # rough — use jq for real work
```

This means **your data is preserved across resumes** — no episode HDF5
ever gets overwritten or deleted. The fragmentation is in the directory
structure only, and the global step counter + `total_steps` field in each
JSONL row gives you a unified timeline.

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
