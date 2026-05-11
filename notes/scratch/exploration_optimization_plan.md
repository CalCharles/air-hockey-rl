# Exploration optimization — continuation plan

Handoff doc. If prompted with "continue optimizing exploration" (or similar),
start here.

## Status

**Phase 1 done** — see full writeup at
[`notes/docs/training/td3-exploration-ablations.md`](../docs/training/td3-exploration-ablations.md).

Three single-seed 500k ablations were run against the `td3_recommended.yaml`
anchor (P1a, 2-layer, q=25/a=6, `sysid_best_params.yaml` sim):

| Run | Change | ret@500k | tail10 | tail50 | Verdict |
|---|---|---|---|---|---|
| E0 anchor | — | 94.3 | 96.6 | 87.6 | baseline |
| E2 no-warmstart | `policy_takeover` weight 0 | 79.0 | 101.7 | 79.1 | slower; tail catches up |
| E5 warmstart-heavy | `policy_takeover` weight 1.0 | 78.2 | **48.7** | 80.9 | tail regresses — 5× warm-start destabilizes |
| **E4 no-bootstrap** | `chance_pre_learning_starts 0` | **109.2** | **117.3** | **110.1** | clear win |

Max episodic return is flat at 206–209 across all four — exploration shifts
sample efficiency, not ceiling.

Run dirs live under `runs/td3/sysid_params/expl_*`. Anchor is `upd_sweep/`.

## What to do next, in priority order

### 1. Confirm E4 on seed=1 (must do before committing config change)

Single-seed +15 ret@500k is strong but not conclusive. One re-run on a new
seed is enough to decide whether to commit
`exploration_primitive_chance_pre_learning_starts=None` (or equivalently,
remove the line) to `td3_recommended.yaml`.

**If E4-seed1 reproduces the win** (ret@500k ≳ 100, tail50 ≳ 105): commit
the config change. Drop the `exploration_primitive_chance_pre_learning_starts`
line from `configs/td3/td3_recommended.yaml`.
Also update the "Recommended config change" section in
`td3-exploration-ablations.md` to "confirmed".

**If it doesn't reproduce**: run a third seed. If signal is still ambiguous,
the override isn't actively harmful and isn't worth removing.

### 2. Stack winners — E4 + reduced-warm-start (interesting, not required)

E2 (no warm-start) was slow early but caught up on tail10. The slow early
phase might be caused by the 100% bootstrap cliff (what E4 removes), not by
warm-start being missing. Run **E4 ∩ E2**: both overrides removed
(`chance_pre_learning_starts=0` AND `policy_takeover=0`). If it matches
E4-alone performance, warm-start is strictly redundant and can be removed.

### 3. Fine warm-start sweep (optional, low priority)

E2 (weight 0) and E5 (weight 1.0) both lost to anchor (0.2). That suggests
anchor is near-optimal but the true minimum could be slightly off. Sweep
`policy_takeover` weight ∈ {0.1, 0.4} around anchor. Only pursue if Phase-1
confirmation leaves time — the effect size is likely <5 points.

### 4. Gaussian noise magnitude (deferred)

`exploration_noise=0.1` was held constant. Could try 0.05 / 0.2. Low-priority
— typical TD3 default works; no signal yet that noise magnitude matters here.

### Explicitly out of scope (per user, 2026-04-17)

- **Do not** ablate `exploration_pre_contact_hit_variant`. Keep chance at 0.
- **Do not** disable *all* primitives at once (won't train).

## How to run an experiment

All experiments use the recommended config as a base and override only the
exploration knob under test. One seed per GPU, one run per GPU.

**GPU policy**: cuda:1, cuda:2, cuda:3. Never two processes on the same GPU.
Check with `nvidia-smi --query-gpu=index,memory.used --format=csv` first.

Template (fill in `RUN_NAME`, `GPU`, `OVERRIDES`, `SEED`):

```bash
cd /home/air-hockey/daliu/air-hockey-rl
source .venv/bin/activate
nohup python scripts/td3/td3_training.py \
  --args-file configs/td3/td3_recommended.yaml \
  --total-timesteps 500000 \
  --device cuda:${GPU} \
  --log-parent-dir runs/td3/sysid_params/${RUN_NAME} \
  --run-name ${RUN_NAME} \
  --seed ${SEED} \
  ${OVERRIDES} \
  > runs/td3/sysid_params/logs/${RUN_NAME}.stdout.log 2>&1 &
disown
```

**Critical**: `--log-parent-dir` must be the per-run directory, not a
grouping parent. The training script treats `log_parent_dir` as the actual
log dir; if it already exists, it appends `r1`/`r2`/... to the path,
creating diverging siblings like `sysid_paramsr1`. This tripped the first
launch attempt on 2026-04-17; correct usage is the full per-run path above.

### Concrete overrides for the next experiments

Priority 1 (confirm E4 on seed=1):
```
RUN_NAME=expl_no_bootstrap_seed1
GPU=1
SEED=1
OVERRIDES="--exploration-primitive-chance-pre-learning-starts 0"
```

Priority 2 (stack winners, seed=0):
```
RUN_NAME=expl_no_bootstrap_no_warmstart
GPU=2
SEED=0
OVERRIDES="--exploration-primitive-chance-pre-learning-starts 0 \
           --exploration-primitive-weight-policy-takeover 0 \
           --exploration-primitive-weight-anneal-policy-takeover 0"
```

Priority 3 (warm-start fine sweep, seed=0, two runs):
```
RUN_NAME=expl_warmstart_0p1         GPU=1   SEED=0
OVERRIDES="--exploration-primitive-weight-policy-takeover 0.1 \
           --exploration-primitive-weight-anneal-policy-takeover 0.1"

RUN_NAME=expl_warmstart_0p4         GPU=2   SEED=0
OVERRIDES="--exploration-primitive-weight-policy-takeover 0.4 \
           --exploration-primitive-weight-anneal-policy-takeover 0.4"
```

Each run takes ~2.75h on shared GPU, ~2.3h on dedicated.

## How to analyze results

1. Add the new run name+path to the `RUNS` dict in
   [`notes/scratch/extract_expl_metrics.py`](extract_expl_metrics.py).
2. `source .venv/bin/activate && python notes/scratch/extract_expl_metrics.py`
3. Compare headline metrics (ret@500k, tail10, tail50, max_ret, pos_frac)
   against anchor E0 and relevant prior ablation.
4. Append a row to the results table in `td3-exploration-ablations.md` and
   a short `## Phase N — ...` section summarizing interpretation.

Key metric definitions:
- `ret@Nk (rolling2k)` = last `charts/rolling2k_avg_episode_return` value at
  step ≤ N. Smoother than raw `episodic_return`.
- `tail10`/`tail50` = mean of last 10/50 `charts/episodic_return` events
  before 500k. `tail10` is noisy; `tail50` is the main late-training signal.
- `max_ret` = `charts/max_episodic_return` over [0, 500k].
- `pos_frac` = last `rewards/sampled_task_reward_positive_fraction` — proxy
  for Q-quality / replay health.

Wall times in the results table are not apples-to-apples across runs
(GPU contention varies). Don't draw throughput conclusions from them.

## Decision rules

- **Commit a config change** to `td3_recommended.yaml` only when a result
  is reproduced on ≥2 seeds or the effect size is large (≳20 points on
  tail50).
- **Stop iterating** on a knob when the effect size dips below ~5 points
  on tail50 across a run pair.
- **One seed is for screening, not shipping.** Phase-1 results are
  actionable but every headline config change needs a seed-1 re-run.

## Files / locations (quick reference)

- Results writeup: `notes/docs/training/td3-exploration-ablations.md`
- This plan: `notes/scratch/exploration_optimization_plan.md`
- Metric extractor: `notes/scratch/extract_expl_metrics.py`
- Recommended config: `configs/td3/td3_recommended.yaml`
- Exploration primitives doc: `notes/docs/exploration/td3-primitives.md`
- Exploration selector code: `scripts/td3/helper/exploration_selector.py`
- Update/depth ablations (preceding study): `notes/docs/training/td3-ablations-updates-and-depth.md`
- Run dirs for Phase 1: `runs/td3/sysid_params/{upd_sweep,expl_no_warmstart,expl_warmstart_heavy,expl_no_bootstrap}/`
