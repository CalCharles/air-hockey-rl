# Post-Polyak-fix rerun: which v25–v30 paddle50 residual findings survive a working bootstrap?

- **Date**: 2026-05-06 18:29 UTC start
- **Status**: 300k pass done; 1M extension done for 4/5 (`fix_v30_lite_1M` still running ~67k of 1M). All four finished cells **fail** every acceptance criterion — back-half collapses to 30–45 vs zs ≈ 67.5. Follow-up TD3+BC sweep launched 2026-05-07 03:48 UTC: see [`2026-05-07_03-48_td3bc-residual-sweep.md`](2026-05-07_03-48_td3bc-residual-sweep.md).
- **Run dirs**: `runs/td3/sim2sim/post_polyak_fix/<name>/seed0/`
- **Configs**: `configs/td3/sim2sim/paddle50/post_polyak_fix/`
- **Launcher**: `scripts/smooth_policy/run_post_polyak_fix.sh <gpu_id>`
- **Logs**: `notes/scratch/post_polyak_fix_logs/`

## Question

Earlier today we discovered that all paddle50 residual configs from `v25_q_updates1`
through `v30_explore_lite` (i.e. the entire late-cycle big-gap residual sweep that the
recipe doc names as canonical) had `q_updates: 1` paired with `target_network_frequency: 2`.
The Polyak-averaging gate was `(q_update_idx + 1) % target_network_frequency == 0` against
an inner-loop index that resets every cycle, so for these configs the modulo never hits
zero and **target networks never updated** for the full training run. `actor_target` stayed
at frozen-base + zero-init residual head ≈ source policy; `qfs_target` stayed at random init.
TD targets were essentially `r + γ·Q_random_init(s', π_source(s'))` — the bootstrap was noise.

The gate is now fixed (`td3_training.py` :: `total_critic_updates % target_network_frequency`,
2026-05-06). With a working bootstrap, do the headline findings from those broken runs still
hold? Three claims need re-verification before they can be promoted to recipe-level guidance:

1. **Maxmin-N=5 is the ensemble sweet spot** (v27 ensemble5 vs v26 ensemble3 vs v29 redq10 — all broken).
2. **q_updates=1 (UTD-1) is fine for residual** (the v25 finding that *triggered* the bug).
3. **Adaptation-phase exploration is neutral-to-harmful** (the v30 series — all broken).

## Setup

5 runs, 1 seed each at 300k steps, on the paddle50 (big-gap) sim2sim transfer. Each run
is the v27 baseline with **one** knob changed, so each delta isolates one of the questions
above. Source model and downstream env match the original v27/v30 setup (`hist2_motion0`
checkpoint_975000 → `sim2sim_combined.yaml`), so results are directly comparable to the
broken-target numbers in `notes/scratch/residual_rl_paddle50_log.md`.

| Run | num_critics | q_updates | exploration | GPU | Tests claim |
|---|---:|---:|---|---:|---|
| `fix_v27_baseline` | 5 | 1 | none | 0 | control — post-fix replication of v27 |
| `fix_twin` | 2 | 1 | none | 2 | (1) does Maxmin-5 still beat plain twin? |
| `fix_redq10` | 10 | 1 | none | 3 | (1) does N=10 still underperform N=5? |
| `fix_v27_q4` | 5 | 4 | none | 1 | (2) does q=1 actually match q=4? |
| `fix_v30_lite` | 5 | 1 | lite primitives + ε=0.10 | 0 (queued after baseline) | (3) does explore_lite still beat / match v27? |

Everything else (buffer, success/failure split with `recency_top50` / `success_top_fraction=0.15`,
PER, age-decay, residual_scale=0.15, q_lr/policy_lr=3e-4, q_weight_decay=1e-3, network shapes,
checkpoint cadence) is held at the v27 settings. The only knob that differs from v27 in each
non-control row is the one labeled in the table.

Coarse design — 1 seed per cell. If a result inverts a previous claim, follow up with a 3-seed
verification in a separate dated file before touching the recipe doc.

### Launching

```bash
# from repo root, in 4 separate shells (or tmux panes)
bash scripts/smooth_policy/run_post_polyak_fix.sh 0   # baseline -> v30_lite
bash scripts/smooth_policy/run_post_polyak_fix.sh 1   # v27_q4
bash scripts/smooth_policy/run_post_polyak_fix.sh 2   # twin
bash scripts/smooth_policy/run_post_polyak_fix.sh 3   # redq10
```

Each run writes scalars to TensorBoard under its `log_parent_dir` and dumps stdout to
`notes/scratch/post_polyak_fix_logs/<name>.log`. Per-GPU pipeline status lands in
`pipeline_gpu{N}.log` in the same directory.

### Reading results

Per the existing convention (`notes/docs/training/monitoring.md`):

- **Per-checkpoint eval** is implicit: each 10k-step checkpoint runs `evaluate_agent` with
  4 episodes. Use `eval_all_ckpts_residual.sh` to get the longer (16-episode) eval signal
  if a run looks promising.
- **Headline metric**: best-of-eval-checkpoint mean episodic return on the target sim,
  plus fraction of eval checkpoints above the zero-shot baseline (~80 on paddle50).
- **Trajectory shape matters more than peak-vs-final** — see the auto-memory note from
  2026-04-26. For each run, report the trajectory of evaluation means across all
  checkpoints, not just the peak number.

## Results

**Status (2026-05-06 ~19:21 UTC)**: 3 of 5 finished — the entire Maxmin-N axis. `fix_v27_q4`
still running on cuda:1 (~280k); `fix_v30_lite` started on cuda:0 after baseline finished.

**Metric**: `charts/rolling2k_avg_episode_return` — smoothed training-rollout return (ε=0.05
exploration noise). The deterministic per-checkpoint `evaluate_agent` only renders GIFs and
doesn't log a scalar (cleanup follow-up: write its 4-ep mean into TB so we get a less-noisy
signal). Numbers below are from this single training-rollout signal at 300k, single seed.

| Run | N | Peak | AUC mean | Last-20 mean ± std | %>zs(80) | Notes |
|---|---:|---:|---:|---:|---:|---|
| `fix_twin` | 2 | 128.8 @ 22k | **77.4** | 75.6 ± 13.7 | **43%** | wins all three smoothed metrics |
| `fix_v27_baseline` | 5 | 116.5 @ 22k | 66.1 | 63.4 ± 8.5 | 17% | the canonical v27 recipe |
| `fix_redq10` | 10 | 114.8 @ 192k | 72.8 | 70.5 ± 8.0 | 27% | between twin and v27 |
| `fix_v27_q4` | 5 (q=4) | 108.2 @ 140k | 56.0 | 38.4 ± 6.4 | 17% | back-half collapse to 30–41 |
| `fix_v30_lite` | 5 + lite expl | 101.4 @ 162k | 57.5 | 54.0 ± 8.2 | 4% | uniformly ≤ v27 across run |

Sampled trajectories (every ~21k steps), Maxmin-N axis:

```
   step   twin(N=2)    v27(N=5)   redq(N=10)
   21500       118.6       114.3         77.6
   43000        69.3        72.3         71.1
   64500        71.5        80.7         74.2
   85500        66.7        62.0         71.9
  107000        59.6        76.8         58.9
  128500        90.4        87.5         69.5
  150000        72.6        74.1         68.7
  171000        85.7        48.0         70.6
  192500        85.2        53.3         92.7
  214000        77.9        60.9         63.6
  235000        62.4        55.7         62.0
  256500        65.8        60.3         84.2
  278000        66.7        57.5         78.1
  299500        65.4        83.9         62.1
```

### Provisional read on Axis 1 (Maxmin-N)

The pre-fix finding was **N=5 ≫ N=3, N=10 underperforms**. With the bootstrap fix, the
single-seed ordering on these data is **N=2 ≥ N=10 ≥ N=5** — i.e., the headline result
is **inverted**. N=5 (the recipe pointer in memory) is the worst by AUC, last-20 mean,
and %>zs.

**Caveats with equal weight to the result:**
1. **Single seed.** Last-20 std is 8–14; the AUC gap between N=2 and N=5 (≈14) is ~1σ
   of within-run noise. **This is not yet a 3-seed-verified claim.**
2. Comparison budget (300k) matches the original v27/v26/v29 sweep, but the canonical
   v27 headline of "peak 98.3 at 1M" is a different (longer) budget.
3. Metric is noisy training rollout; the deterministic per-ckpt eval would tighten it
   but isn't currently logged.

If this ordering survives 3-seed verification it would mean the "Maxmin-N=5 is essential"
claim in the recipe doc was a frozen-target artifact, and the recipe should default back
to twin TD3 (with whatever else from v22/v23/v24 remains validated). **Do not change the
recipe doc until the 3-seed follow-up lands.**

### Provisional read on Axis 2 (UTD)

```
   step     q=1       q=4
   21500    114.3      84.7
  107000     76.8      89.2     <- q=4 holds
  150000     74.1      59.6     <- q=4 starts to fall
  192500     53.3      33.7
  235000     55.7      33.0
  299500     83.9      36.0     <- q=4 collapsed
```

q=4 tracks q=1 through ~150k, then collapses to 30–41 for the back half. Final-20-step
mean: q=1 = 63.4 ± 8.5 vs q=4 = 38.4 ± 6.4 — a >2σ gap in the metric's noise band. The
v25 headline ("q=1 is fine for residual") **survives** the bootstrap fix — the conclusion
was correct, even if the originally-stated mechanism ("lower UTD prevents drift") was
confounded with the frozen-target bug. With working Polyak, q=4 looks *worse*, plausibly
because a now-moving actor target chases a higher-UTD (more unstable) critic.

Same single-seed caveat — the back-half collapse could be one bad seed rather than a
trend. Recommend a 3-seed verification before publishing as a stable claim, but the
direction is clear and consistent with the original v25 result.

### Provisional read on Axis 3 (Exploration)

`v30_lite` (chance 0.10→0.03 over 50k, ε=0.10, lite primitive weights) is uniformly
≤ no-exploration `v27_baseline` across the 300k run by every aggregate metric and at
every sampled step except a small bump at 43k. The v30 finding "exploration ranges
from neutral to harmful" **survives** the bootstrap fix. The narrative that "the base
policy already produces sensible behavior; primitives over-disturb it" looks correct
even with working targets.

## Conclusion

Across all three axes, single-seed at 300k:

| Axis | Pre-fix claim | Post-fix result | Verdict |
|---|---|---|---|
| Maxmin-N | N=5 ≫ N=3, N=10 underperforms | N=2 ≥ N=10 ≥ N=5 | **INVERTED** |
| UTD (q_updates) | q=1 fine for residual | q=1 > q=4 (q=4 collapses) | Survives |
| Exploration | Neutral-to-harmful | v30_lite < no-expl on every metric | Survives |

The single load-bearing claim from the v25–v30 era that **does not survive** is the
"Maxmin-N=5 is the sweet spot" canonical-recipe finding. With proper Polyak averaging,
plain twin TD3 (N=2) wins by AUC, last-20 mean, and %>zs on this seed. The other two
findings (UTD-1 is fine, exploration is harmful) reproduce.

This means the recipe pointer in CLAUDE.md ("v27 ensemble5 is canonical for big-gap
sim2sim/sim2real") may be wrong — but the surrounding choices (UTD-1, no exploration
primitives, residual_scale=0.15, recency-top-50 / age-decay buffer logic from v22-v24)
all still seem fine. **Do not edit the recipe doc on a single seed.**

## 1M extension (launched 2026-05-07)

**User decision (2026-05-07)**: skip the 3-seed verification. Instead, extend each
of the 5 configs to a full 1M trajectory and look **holistically at curve shape**,
not single-step peak/last-N numbers. The headline question is "does each variant
*hold* performance over the full budget, or collapse?" — exactly the question that
made the broken-target v27 famous in the first place ("the only recipe with a
stable 1M trajectory").

**Why this is the right test even on 1 seed**: trajectory shape over 700k extra
steps is a much richer signal than 1-seed numbers at 300k. A run that holds 60–90
in a band for 700k tells us much more than three seeds at 300k showing 65 ± 10.

**Mechanism**: fresh runs from source at `total_timesteps: 1000000`, same seed=0,
same source policy, fresh log dirs under `runs/td3/sim2sim/post_polyak_fix_1M/`.
Same approach as the original `td3_residual_v27_ensemble5_1M.yaml` precedent
(no in-place residual-resume code path exists). Same seed = the 0–300k segment
should reproduce the existing post-fix 300k trajectory closely; 300k–1M is new.

**Configs**: `<name>_1M.yaml` siblings of the 300k configs in the same dir.
**Launcher**: `bash scripts/smooth_policy/run_post_polyak_fix.sh <gpu_id> _1M`.
**Queue files**: `_queue_gpu{0..3}_1M.txt`.

| GPU | queue | est. wall (h) |
|---:|---|---:|
| 0 | `fix_v27_baseline_1M` | ~5 |
| 1 | `fix_v27_q4_1M` | ~5 |
| 2 | `fix_twin_1M` → `fix_v30_lite_1M` | ~4 + ~5 = 9 |
| 3 | `fix_redq10_1M` | ~6 |

Bottleneck wall ≈ 9h on GPU 2.

### 1M results (4/5 finished; v30_lite still running)

zs (paddle50) ≈ 67.5. Acceptance bar: 500k–1M band lower edge ≥ ~77.5.

| Run | 0-200k mean[band] | 200-500k mean[band] | 500-700k | 700-900k | 900k-1M | Verdict |
|---|---|---|---|---|---|---|
| `fix_v27_baseline_1M` (N=5) | 68 [51,90] | 51 [35,68] | 46 [34,62] | 39 [30,49] | **37 [30,46]** | **fails** |
| `fix_twin_1M` (N=2) | 72 [55,92] | 42 [31,65] | 36 [28,45] | 38 [29,48] | **38 [30,48]** | **fails** |
| `fix_redq10_1M` (N=10) | 75 [58,96] | 74 [57,92] | 54 [39,72] | 44 [34,55] | **38 [29,52]** | **fails** (latest collapser) |
| `fix_v27_q4_1M` (N=5, q=4) | 69 [39,94] | 37 [29,47] | 40 [30,50] | 42 [32,53] | (still running) | **fails** (early collapse, no recovery) |
| `fix_v30_lite_1M` | (~67k of 1M, will land in ~9h) | | | | | TBD |

**Holistic shape across all 4 finished cells**:
- Each peaks in 0–200k clearly above zs (band lower edge 51–58, top 90–96).
- Each then drifts down, with a sharp drop in 200–500k for twin and v27_q4, a softer
  drift for v27_baseline, and the latest collapse for redq10 (held 200–500k before
  cliffing).
- All converge to a 30–45 hold band in 700k–1M — well below zs.

This is exactly the §8.13 drift mechanism (Q overestimation → residual head exploits)
that the broken-target era was chasing. The Polyak fix correctly enabled bootstrap
learning, but in doing so removed the implicit damping that frozen targets had been
providing — so the underlying drift now expresses fully. The "v27 holds 1M" finding
from the broken-target era was almost certainly a frozen-target-as-regularizer
artifact.

**Verdict on the parent question** ("which axes' findings survive?"): all 4 finished
post-fix cells fail acceptance. The 300k Maxmin-N "inversion" was real *at 300k*, but
with a longer trajectory all variants collapse, so the cross-axis comparisons aren't
load-bearing — what matters now is fixing the drift itself.

**Next iteration**: TD3+BC sweep on N=5 baseline, λ ∈ {0.5, 1.0, 2.0}, 1M each.
Launched 03:48 UTC, results in ~5h. See `2026-05-07_03-48_td3bc-residual-sweep.md`.

### How to read the curves when they land

For each run, look at `charts/rolling2k_avg_episode_return` over 0–1M:

1. **Hold band**: where does the trajectory live in the back half (500k–1M)?
   A 60–90 band for 700k is the signature of a stable recipe. A monotonically
   decreasing trajectory or a "cliff" mid-run is the signature of collapse.
2. **Cliff timing**: if there's a collapse, when does it happen? The pre-fix
   v27 had no cliff at 1M (its claim to fame). The pre-fix v29 (REDQ-10) had
   a "delayed cliff past 300k" that was invisible at the 300k cutoff. The
   `fix_v27_q4` 300k run already showed a back-half collapse at 150k+; will
   it stay collapsed through 1M, or recover?
3. **Width of band**: a 30-point band (e.g. 60–90) is healthier than a 60-point
   band that swings 30↔90. Wider bands suggest bootstrap instability that may
   amplify with more data.
4. **Don't reduce to a single peak.** Per the auto-memory note from 2026-04-26,
   peak number alone is misleading on noisy 1-seed RL. Show the curve, or at
   minimum a hold-band + cliff-timing summary.

The cross-axis questions stay the same as the 300k pass:
- **Maxmin-N**: do twin / N=5 / N=10 still rank twin ≥ N=10 ≥ N=5, or does the
  ordering change as the bootstrap accumulates effect over 1M? (The pre-fix
  story was specifically "N=5 wins at 1M because it's the only one that holds";
  the post-fix 300k inversion may or may not survive the longer budget.)
- **UTD**: does q=4 stay collapsed, or claw back? Does q=1 hold?
- **Exploration**: does v30_lite stay below baseline through 1M, or does the
  primitive-driven diversity start helping at later steps?

## Acceptance criteria (set 2026-05-07)

User goal is **improved performance that is maintained**. Concretely, a "satisfactory"
1M cell must satisfy ALL of:

- **Above zs floor**: back-half (500k–1M) hold band's *lower edge* stays clearly above
  the paddle50 zero-shot baseline (zs ≈ 67.5; per `project_residual_drift_fix_in_flight`
  memory). "Clearly" = ≥ 10 points above zs sustained, not a one-checkpoint blip.
- **No cliff**: no monotonic drop or step-down in the back half. A single dip that
  recovers is OK; sustained collapse to zs or below is not.
- **Reasonable band width**: 30 points or less (e.g. 78–108) is healthy; 60-point
  swings (40↔100) suggest bootstrap instability that may not survive a re-seed.
- **Mean above zs**: the simplest sanity — last-100k mean ≥ zs + 10.

The 300k single-seed pass had only one cell above zs (`fix_twin` at AUC 77 / last-20 75,
~7 above zs). All others were at or below zs. So **odds are non-trivial that the 1M
extension lands unsatisfactory across the board**, in which case the next iteration
draws from the known-method menu below.

## If 1M is unsatisfactory: known sim2real / off-policy-fine-tune methods to try

Try these **before** novel ideas. Order is rough "well-validated for this regime first,"
not "best for our setup" — the latter requires an experiment.

1. **TD3+BC** (Fujimoto & Gu 2021, "A Minimalist Approach to Offline RL").
   Actor loss: `−Q(s, π(s)) + λ · ‖π(s) − π_source(s)‖²`. λ in {0.1, 1, 10} sweep.
   Smooth knob between "stay at zs" and "free residual" — very natural for our
   residual + frozen-base setup, and addresses exactly the failure mode
   (residual head exploits Q overestimation and drifts off the source).
2. **CQL** (Kumar et al. 2020). Adds a conservative penalty `α · log Σ_a exp Q(s,a) − Q(s,a*)`
   to critic loss to suppress OOD actions. Principled fix for the v25–v30 drift mechanism
   (critic Q1 grows 2.6–4× per the §8.13 instrumentation), without needing an explicit
   anchor policy.
3. **Layer-norm critic / TD7** (Fujimoto et al. 2023). LN before each ReLU in the critic
   stabilizes Q under distribution shift. Already on the project's deferred list per
   `paddle50_log.md` §8.16. Cheapest of the three — pure architecture change, no new
   loss term or hyperparameter.
4. **Stronger source-side domain randomization**. The current source policy
   (`hist2_motion0_v2`, promoted 2026-05-05) was retrained with collision randomization
   but paddle radius isn't randomized at source-time. Wider source DR makes the source
   policy's basin of competence larger, which makes residual fine-tuning easier on the
   target — this is the highest-leverage fix if it works, but requires retraining the
   source (~1M+ steps). Defer unless 1–3 all fail.
5. **Replay seeding from source-env data**. The `full_checkpoint_load: "fine_tune"` mode
   already supports `fine_tune_replay_keep_total`; we just don't use it in residual mode.
   Worth combining with TD3+BC.
6. **Online distillation from source on target-env states**. Periodically minimize
   `‖π(s_target) − π_source(s_target)‖` while still doing residual RL. Lighter-touch
   than TD3+BC, but redundant with it; pick one.

**Iteration protocol**: pick ONE method (probably TD3+BC first, since it's the canonical
fine-tune-with-anchor recipe and the cleanest fit for residual mode). Build one config
at λ ≈ 1.0 on top of the most-promising 300k cell (likely `fix_twin` if it holds, or
`fix_v27_baseline` if not). Run 1 seed × 1M. Look at the curve. Don't stack methods.

## Next

1. **Wait for the 1M curves and analyze holistically.** Update this file with
   the trajectory shapes (sampled every ~70k or so) and a written read of each
   axis through the lens of the four shape criteria above. **Apply the acceptance
   criteria explicitly** — call out which (if any) cells meet them.
2. **If unsatisfactory**: pick one method from the menu above (TD3+BC unless there's
   a specific reason to try a different one), draft the config, run 1 seed × 1M,
   write up in a NEW dated experiment file (per the convention) that supersedes this
   one for the chosen direction.
3. **If satisfactory**: update `notes/docs/training/residual-rl-recipe.md`
   and the `project_residual_drift_fix_in_flight.md` memory entry to point at the
   winning recipe and demote v27.

### Original next-steps (deferred per user 2026-05-07)

1. **3-seed verification of the Maxmin-N inversion** (highest priority). Re-run
   `fix_twin`, `fix_v27_baseline`, `fix_redq10` on seeds 1–3. If twin TD3 still wins
   (or ties N=5 within seed-noise), update `notes/docs/training/residual-rl-recipe.md`
   to default to N=2 for big-gap residual and demote the v27 ensemble5 pointer in
   CLAUDE.md / `project_residual_drift_fix_in_flight.md`.
2. **3-seed verification of the q=4 collapse** (lower priority — direction matches
   the original v25 result, but the back-half collapse magnitude is suspicious for a
   single seed). Re-run `fix_v27_q4` on seeds 1–3. If the collapse persists, update
   the recipe doc to flag UTD>1 as actively harmful (not just "unnecessary").
3. **No re-test of v30 exploration** for now — same direction as the original finding
   on this single seed; promote to next priority only if (1) and (2) both hold.
4. **Logging follow-up**: wire `evaluate_agent`'s 4-episode mean into TensorBoard so
   future runs have a deterministic per-checkpoint signal alongside the noisy
   training rollout. (Smaller change in `td3_training.py` :: the `evaluate_agent`
   call site.)
