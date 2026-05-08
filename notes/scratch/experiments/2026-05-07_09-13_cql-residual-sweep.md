# CQL sweep on residual fine-tune (post-Polyak-fix collapse rescue, attempt #2)

- **Date**: 2026-05-07 09:13 UTC start
- **Status**: done. CQL is the post-fix winning mechanism. `α=20, N=5, residual_scale=0.15` is the best single-seed 1M recipe so far (900k-1M mean 68; 700-900k band [60, 92]). 4-cell stacking sweep (CQL+BC, CQL+twin, CQL+N=10) was killed at 0-300k by user request — partial signal: BC hurts, N=10 converges fastest, twin matches α=20 alone. See "Cross-method holistic comparison" section at the bottom for the full picture.
- **Supersedes (in priority for the next iteration)**:
  [`2026-05-07_03-48_td3bc-residual-sweep.md`](2026-05-07_03-48_td3bc-residual-sweep.md)
  — the BC sweep concluded that no λ in {0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0} gives
  improved + maintained performance.
- **Run dirs**: `runs/td3/sim2sim/post_polyak_fix_1M/fix_cql_alpha{01,1,5}/seed0/`
- **Configs**: `scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/post_polyak_fix/fix_cql_alpha{01,1,5}.yaml`
- **Launcher**: `bash scripts/smooth_policy/run_post_polyak_fix.sh <gpu> _cql`

## Question

The BC sweep showed BC at any λ either drifts (λ ≤ 0.1) or pins residual to zero
(λ ≥ 0.5). Mechanism diagnosis: BC penalizes the actor, but the underlying problem
is **Q-overestimation** (`§8.13`-instrumented: Q1_task_mean grows 2.6–4× over training).
BC is a fixed-magnitude penalty, while policy-gradient grows with Q — so BC's relative
pressure decays, and any λ light enough to allow learning is light enough to allow
drift.

CQL (Kumar et al. 2020) attacks Q-overestimation directly by adding a conservative
penalty to the critic loss:

```
L_critic += α · (logsumexp_a Q(s,a) − Q(s, π(s)))
```

logsumexp samples N random uniform actions per state; the term pushes Q *down* for
OOD actions (`logsumexp_a` ≈ max) while keeping Q up for the policy's action. This
prevents the Q-bloat that drives the residual head's drift.

## Setup

α ∈ {0.1, 1.0, 5.0} sweep on the canonical `fix_v27_baseline_1M` config (N=5, q=1,
no exploration, no BC, residual_scale=0.15). 1 seed each at 1M.

| Run | α | GPU | Base config |
|---|---:|---:|---|
| `fix_cql_alpha01` | 0.1 | 0 | v27 baseline (N=5, q=1, no expl) |
| `fix_cql_alpha1`  | 1.0 | 1 | same |
| `fix_cql_alpha5`  | 5.0 | 3 | same |

GPU 2 held idle for a follow-up at α=2 or α=10 if the {0.1, 1, 5} reveals a sweet
spot direction.

### Implementation

Two new args added to `td3_training.py` (`Args` dataclass):
- `cql_alpha: float = 0.0` — penalty strength; 0 disables (default).
- `cql_n_random: int = 10` — number of uniform random actions sampled per state
  for the logsumexp approximation.

Critic-loss site (`td3_training.py` ~line 1885) extended: when `cql_alpha > 0`,
samples `cql_n_random` random actions in [-1,1]^act_dim per state, computes
`logsumexp(Q_random) − Q(s, actor(s))` per critic on the task head only (motion
head is a small auxiliary; not the source of overestimation drift), and adds
`α · penalty` to that critic's task loss before the optimizer step. The actor
forward through `deterministic_actor_action` is wrapped in `torch.no_grad()` so
the actor isn't backprop'd through the critic loss.

First launch (commit ~03:48–09:13) failed with NameError because I used
`sampled_policy_observations` (only computed in the actor block at line 2087).
Fixed by computing it locally from `sampled_observations` + `sampled_prev_actions`,
both already in scope at the critic-loss site.

## Acceptance criteria (carried over)

A cell is satisfactory if back-half (500k–1M):
1. band lower edge ≥ ~77.5 (zs + 10)
2. no cliff
3. band width ≤ 30
4. mean ≥ ~77.5

If CQL also fails to clear the bar, fall back to source-side stronger DR
(retrain `hist2_motion0_v2` with paddle-radius randomization) — that's the next
known method in the parent doc's menu.

## Results

| Run | α | residual_scale | Peak (step) | 0-200k | 500-700k | **900k-1M** | Cliff? | Above zs? | Verdict |
|---|---:|---:|---:|---|---|---|---|---|---|
| `fix_v27_baseline_1M` (control) | 0 | 0.15 | 110 @ 131k | 68 [51,90] | 46 [34,62] | 37 [30,46] | yes | no | **drift / collapse** |
| `fix_cql_alpha01` | 0.1 | 0.15 | 132 @ **455k** | 64 [46,86] | 66 [51,84] | **72 [57,90]** | no | **yes (+4.5)** | maintained, mild improvement |
| `fix_cql_alpha1`  | 1.0 | 0.15 | 134 @ **988k** | 62 [48,77] | 62 [47,75] | **72 [55,90]** | no | **yes (+4.5)** | maintained, mild improvement |
| `fix_cql_alpha5`  | 5.0 | 0.15 | 126 @ **952k** | 68 [53,85] | 69 [53,87] | **72 [55,91]** | no | **yes (+4.5)** | maintained, mild improvement |

### Headline finding (2026-05-07 ~12:00 UTC)

CQL is **the first method on the menu that maintains performance ≥ zs across 1M**.
All three α strengths produce essentially identical end-of-1M numbers (mean 72,
band [55, 90]) and have **peaks at the END of training** (455k, 988k, 952k) —
which is the "policy keeps improving" signature, not the "drift past peak"
shape we've seen everywhere else.

Comparison to control: +35 mean in 900k-1M window.
Comparison to BC pinned: +10 mean.
Strict criterion-1 (band lower edge ≥ 77.5) still fails (lower edge 55-57).
Mean is +4.5 above zs (deterministic eval would be higher — training rollout
includes ε=0.05 noise dilution).

**CQL is robust over 50× α range** — choice of α matters less than presence-vs-absence.

### Extension sweep launched 2026-05-07 ~12:00 UTC

Two simple extensions, both within the CQL framework, to push further toward
strict criterion-1 (band lower edge ≥ 77.5):

| Run | α | residual_scale | GPU | Hypothesis |
|---|---:|---:|---:|---|
| `fix_cql_alpha10` | 10.0 | 0.15 | 0 | stronger Q-anchor → higher floor? |
| `fix_cql_alpha20` | 20.0 | 0.15 | 1 | even stronger; if same as α=5/10, CQL is saturated |
| `fix_cql_alpha5_rs030` | 5.0 | 0.30 | 2 | 2× action range; CQL keeps Q honest with bigger residual |
| `fix_cql_alpha5_rs050` | 5.0 | 0.50 | 3 | 3.3× action range; aggressive; tests residual ceiling |

## Conclusion

(Pending.) Decision tree:

- **Sweet spot found**: declare CQL the post-fix recipe. Update
  `notes/docs/training/residual-rl-recipe.md` (big-gap section) and the
  `project_residual_drift_fix_in_flight.md` memory entry.
- **All α drift like control**: CQL too weak in this range — sweep α ∈ {10, 20}.
- **All α kill the peak**: CQL too aggressive — sweep α ∈ {0.01, 0.05}.
- **All α partial wins (some criteria pass, some fail)**: write up, decide
  next mechanism (source-side DR) or combination strategy.

## Cross-method holistic comparison (final, single seed, 1M each unless noted)

zs (paddle50) ≈ 67.5; bar for "improved + maintained" = back-half mean ≥ 77.5.

```
recipe                          0-200k  200-500  500-700  700-900  900k-1M     verdict
control (no fine-tune, source)    67       67       67       67       67     baseline
from-scratch 1M default           19       30       37       43       47     well below zs
from-scratch 3.85M (big net)      29       43       45       46       47     plateau, structural ceiling
post-fix raw (no anchor)          68       51       46       39       37     drift / collapse
TD3+BC any λ (0.5–2.0)            62       62       63       62       61     pinned at zs
TD3+BC low λ (0.05–0.2)           60       60       60       58       57     mild drift
CQL α=0.1                         64       68       66       65       72     maintained, mild rise
CQL α=1                           62       64       62       63       72     maintained, late peak
CQL α=5                           68       69       69       66       72     maintained, late peak
CQL α=10                          69       74       74       64       69     best 200-700k
CQL α=20  ★                       67       69       75       74       68     best mid-late, lower edge 60
CQL α=5, rs=0.30                  62       64       56       59       62     larger rs hurts
CQL α=5, rs=0.50                  43       39       39       41       41     larger rs collapses

partial CQL+stack (killed at 276k-300k of 1M; only 0-300k shown):
                                0-100k  100-200  200-300
α=20 alone (reference)            66       69       66
α=20 + BC λ=0.1                   59       56       54     ← BC hurts
α=20 + BC λ=1.0                   60       60       66     ← still hurts
α=20 + N=2 (twin)                 68       70       71     ← matches α=20 alone
α=20 + N=10                       69       73       75     ← converges fastest
```

## Findings (in order of confidence)

1. **From-scratch is structurally dead on paddle50** — plateaus at mean 47 at
   3.85M (5× our budget). Source policy + transfer is mandatory.
2. **Polyak fix exposed real Q-overestimation drift** that frozen targets had
   been incidentally damping. The "v27 holds 1M" finding from the broken-target
   era was a frozen-target artifact.
3. **TD3+BC alone is binary** (drift or pin, no sweet spot across 200× λ).
   Mechanism: BC is fixed magnitude while PG grows with Q.
4. **CQL is the right mechanism.** First method that maintains performance ≥ zs
   across 1M, with peaks landing at end of training (455k–988k) — the "still
   improving" signature.
5. **CQL is robust over 50× α range** (0.1 to 20 all qualitatively similar).
   α=10–20 best for sustained mid-run.
6. **residual_scale=0.15 is the right ceiling.** rs=0.30 hurts, 0.50 collapses.
7. **BC stacking on top of CQL HURTS** (early-300k signal). CQL provides
   sufficient anchoring; adding BC's drag-to-zero trades off learning signal.
8. **N=10 + CQL converges fastest** (early-300k partial). Whether it sustains
   through 1M or collapses sooner is unknown — sweep was killed.
9. **Twin + CQL works fine.** Maxmin-5 not essential once CQL handles Q.

## Best recipe so far

**`fix_cql_alpha20.yaml`** (CQL α=20, N=5, residual_scale=0.15, all other knobs
inherited from v27 baseline). 1M single seed:
- 900k-1M mean **68** (≈ zs); 700-900k band **[60, 92]** (best sustained window)
- Peak **124 @ 620k** (1.85× zs)
- Improvement: +21 vs from-scratch 3.85M, +31 vs no-CQL drift, ~0–5 vs zero-shot
  in noisy training rollout (deterministic eval likely +5–10)

## What's not achieved

- Strict criterion 1 (back-half band lower edge ≥ 77.5) — best is 60.
- Mean +10 sustained — best mid-run mean is 75; not sustained the full back half.
- Single seed only.
- Training-rollout metric (with ε=0.05 noise dilution); deterministic eval not
  logged as a TB scalar.

## Next moves (in priority order)

1. **Source-side stronger DR** (highest leverage, biggest commitment): retrain
   `hist2_motion0_v2` with paddle-radius randomization. Wider source basin →
   easier residual fine-tune. ~1M+ source retrain — flagged for human decision.
2. **CQL + N=10, full 1M** (cheap; partial signal promising): re-launch the
   killed run. If it sustains the early lead, new best recipe.
3. **CQL α=20 + longer training (2M)**: see if α=20 keeps improving past 1M.
4. **Wire deterministic-eval mean into TB** (small code change): cleaner signal;
   may push us above the strict bar without algorithm changes.
