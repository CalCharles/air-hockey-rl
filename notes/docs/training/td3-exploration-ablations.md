# TD3 Exploration Ablations — Warm-start and Bootstrap

> **2026-05-04 path note:** the `td3_recommended.yaml` referenced throughout this
> doc is the original hist4 variant, which is now at
> `scripts/smooth_policy/amp_history/configs/td3/legacy/td3_recommended.yaml`.
> Likewise `sysid_best_params_hist4.yaml` lives in `new_juggle/legacy/`. The
> sweeps were run against those exact files; the active default for **new** runs
> is `td3_recommended_top50_hist2.yaml` ([`td3-configs.md`](td3-configs.md)). All
> ablation conclusions below still apply (only `config:` and
> `success_top_fraction` differ between active and legacy).

Effect of exploration knobs on juggle-task learning. All Phase-1 runs use
`td3_recommended.yaml` as the args file (2-layer, q=25/a=6,
`sysid_best_params.yaml` sim) and vary a single exploration knob. Runs
launched 2026-04-17, 1 seed each, 500k timesteps. Anchor is the existing
`upd_sweep` run (P1a), truncated to 500k.

See [`exploration/td3-primitives.md`](../exploration/td3-primitives.md) for
what each primitive does and
[`training/td3-ablations-updates-and-depth.md`](td3-ablations-updates-and-depth.md)
for the preceding update/depth study that produced the anchor config.

## What was varied

The recommended config bundles four exploration mechanisms that all fire
simultaneously: constant Gaussian noise, annealed primitive takeover,
a forced 100% primitive bootstrap over the first 20k steps
(`exploration_primitive_chance_pre_learning_starts=1.0`), and a warm-start
policy loaded as one of the primitives (weight 0.2 in both annealing and
post-anneal phases). None of these had been ablated.

All run directories are under `runs/td3/sysid_params/` (relative to repo root,
absolute: `/home/air-hockey/daliu/air-hockey-rl/runs/td3/sysid_params/`).

| Run | Dir | Change vs anchor | GPU |
|---|---|---|---|
| E0 anchor | `runs/td3/sysid_params/upd_sweep/` | — | cuda:1 (historical) |
| E2 no-warmstart | `runs/td3/sysid_params/expl_no_warmstart/` | `policy_takeover` weight 0 (anneal + post-anneal) | cuda:1 |
| E5 warmstart-heavy | `runs/td3/sysid_params/expl_warmstart_heavy/` | `policy_takeover` weight 1.0 (anneal + post-anneal) | cuda:2 |
| E4 no-bootstrap | `runs/td3/sysid_params/expl_no_bootstrap/` | `chance_pre_learning_starts 0` (falls back to `chance_start=0.15` during first 20k) | cuda:3 |

### Upstream sweep run locations

The layer-size and update-ratio sweeps that produced the anchor config
(`td3_recommended.yaml`, 2-layer, q=25/a=6) also live under
`runs/td3/sysid_params/`. Full write-up in
[`td3-ablations-updates-and-depth.md`](td3-ablations-updates-and-depth.md).

| Sweep | Run | Dir | Config |
|---|---|---|---|
| Depth (Part 1) | baseline | `runs/td3/sysid_params/delayr1/` | q=200, a=50, 5-layer (killed @ 400k) |
| Depth (Part 1) | A | `runs/td3/sysid_params/ablater1/` | q=50, a=12, 5-layer |
| Depth (Part 1) | B | `runs/td3/sysid_params/ablate_l2/` | q=50, a=12, 2-layer |
| Depth (Part 1) | C | `runs/td3/sysid_params/ablate_l3/` | q=50, a=12, 3-layer |
| Update volume (Phase 1) | P1a (anchor) | `runs/td3/sysid_params/upd_sweep/` | q=25, a=6, 2-layer |
| Update volume (Phase 1) | P1b | `runs/td3/sysid_params/upd_sweepr1/` | q=12, a=3, 2-layer |
| Update volume (Phase 1) | P1c | `runs/td3/sysid_params/upd_sweepr2/` | q=6, a=2, 2-layer (killed, under-training) |
| Actor:Q ratio (Phase 2) | P2a | `runs/td3/sysid_params/ratio_sweep/` | q=29, a=2 (ratio 0.07) |
| Actor:Q ratio (Phase 2) | P2b | `runs/td3/sysid_params/ratio_sweepr1/` | q=21, a=10 (ratio 0.48) |
| Actor:Q ratio (Phase 2) | P2c | `runs/td3/sysid_params/ratio_sweepr2/` | q=10, a=21 (ratio 2.10) |

## Results at 500k

| Run | ret@250k | ret@500k | tail10 | tail50 | max_ret | pos_frac | step→≥100 | wall |
|---|---|---|---|---|---|---|---|---|
| E0 anchor | **75.2** | 94.3 | 96.6 | 87.6 | 206 | 0.60 | 9.9k | 2.28h* |
| E2 no-warmstart | 67.5 | 79.0 | 101.7 | 79.1 | 207 | 0.61 | 65.9k | 2.74h |
| E5 warmstart-heavy | 63.8 | 78.2 | **48.7** | 80.9 | 209 | 0.62 | 6.6k | 2.80h |
| **E4 no-bootstrap** | 64.7 | **109.2** | **117.3** | **110.1** | 207 | 0.60 | **592** | 2.73h |

`ret@N k` = rolling-2k-episode return. `tail10`/`tail50` = mean of last 10/50
episodic returns at 500k. `max_ret` = `charts/max_episodic_return` over
[0, 500k]. `step→≥100` = first env-step at which `charts/episodic_return`
crossed 100.

\*Anchor wall time is the first 500k segment of the original 1M run, which
ran on a dedicated GPU. E2/E4/E5 ran concurrently with other training on
cuda:0, so throughput numbers are not apples-to-apples for wall clock. The
step-indexed return metrics are unaffected.

## Main observations

### Bootstrap forcing is actively harmful (E4)

Removing the `exploration_primitive_chance_pre_learning_starts=1.0` override
gave the largest improvement of the sweep. At 500k, E4's rolling-2k return
is **+15 points over anchor (109 vs 94)** and its tail-50 is **+22 points
(110 vs 88)**. E4 also reached episodic return 100 at step 592 vs anchor's
9.9k.

Interpretation: forcing primitive activation 100% of the time during the
first 20k random-action steps produces less useful bootstrap data than the
default `exploration_pre_learning_action_source="random"` behavior
(uniform-random actions with a 15% primitive activation rate from the
annealing schedule). The over-opinionated bootstrap appears to restrict
early replay diversity in a way that slows later learning.

The single-seed caveat applies — 15-point gaps are within plausible seed
noise. But the signal is coherent across every late-window metric (ret@500k,
tail10, tail50, step→≥100), which is harder to attribute to noise alone.

### Warm-start weight is not monotonic (E2 vs anchor vs E5)

Both directions of adjusting `policy_takeover` weight underperform the
anchor at ret@500k:

- **E2 (weight 0, anchor is 0.2)**: ret@500k 79 vs 94. Learns slower through
  250k, but `tail10=101.7` is *higher* than anchor — suggests E2 catches
  up late.
- **E5 (weight 1.0)**: ret@500k 78 with **tail10=48.7**. Late-training is
  *worse*, not just slower. The policy appears to be destabilizing —
  possibly because 5× warm-start pulls the learned policy back toward the
  frozen demo distribution rather than letting it explore refinements.

So anchor weight 0.2 sits near a reasonable operating point. Zero-weight
sacrifices early-learning speed. Heavy-weight actively hurts steady-state
behavior.

### Peak ceiling is invariant

Max episodic return in [0, 500k] is 206–209 across all four runs. Whatever
exploration does, it doesn't change the policy's maximum capability — it
changes how often the policy achieves that capability (steady-state return)
and how fast it gets there (step→≥100).

### pos_frac is flat

`rewards/sampled_task_reward_positive_fraction` = 0.60–0.62 across all runs.
The Q-function's view of "good samples" in replay is stable regardless of
exploration choices in this sweep.

## Recommended config (committed 2026-04-19)

`td3_recommended.yaml` now bakes in the following defaults based on this
sweep plus the depth/update-volume study:

- `exploration_primitive_chance_pre_learning_starts: null` (E4 — no bootstrap forcing)
- `exploration_primitive_weight_policy_takeover: 0.0` (E2 — no external warmstart)
- `exploration_primitive_weight_anneal_policy_takeover: 0.0`
- `exploration_policy_takeover_enabled: false`
- `config: sysid_best_params_hist4.yaml` (hist_len=4 PID target filter)
- 2-layer nets, `q_updates=25`, `actor_updates_per_iteration=6`
  (from the depth/update sweep)

The +15 pt E4 gap was single-seed and strong but not conclusive; the
Phase-2 seed re-run is still tracked in
[`notes/scratch/exploration_optimization_plan.md`](../../scratch/exploration_optimization_plan.md).
Until that runs, treat the committed default as the best current guess
rather than the proven optimum.

## Continuing this work

The concrete next-experiment plan (seeds, overrides, GPU assignments,
decision rules) lives at
[`notes/scratch/exploration_optimization_plan.md`](../../scratch/exploration_optimization_plan.md).
Start there for "continue optimizing exploration" prompts.

## Commands used

```bash
# All Phase-1 runs share:
BASE="python scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py \
  --args-file scripts/smooth_policy/amp_history/configs/td3/td3_recommended.yaml \
  --total-timesteps 500000"

# E2: no warm-start (cuda:1)
$BASE --device cuda:1 \
  --log-parent-dir runs/td3/sysid_params/expl_no_warmstart \
  --run-name expl_no_warmstart \
  --exploration-primitive-weight-policy-takeover 0 \
  --exploration-primitive-weight-anneal-policy-takeover 0

# E5: warm-start heavy (cuda:2)
$BASE --device cuda:2 \
  --log-parent-dir runs/td3/sysid_params/expl_warmstart_heavy \
  --run-name expl_warmstart_heavy \
  --exploration-primitive-weight-policy-takeover 1.0 \
  --exploration-primitive-weight-anneal-policy-takeover 1.0

# E4: no bootstrap forcing (cuda:3)
$BASE --device cuda:3 \
  --log-parent-dir runs/td3/sysid_params/expl_no_bootstrap \
  --run-name expl_no_bootstrap \
  --exploration-primitive-chance-pre-learning-starts 0
```

## Metric extraction

Script: `notes/scratch/extract_expl_metrics.py` (reads tensorboard events,
emits the table above).
