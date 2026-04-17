# TD3 Ablations — Update Count and Network Depth

Wall-clock vs sample efficiency trade-offs for TD3 on the juggle task. All runs use `sysid_best_params.yaml` sim config and `td3_no_alignment.yaml` as the args file. Run dates: 2026-04-16 (Part 1) and 2026-04-17 (Parts 2–3).

The study has three parts:
1. **Part 1** — cut a very expensive baseline (q=200/a=50, 5-layer) down by 4× updates, and sweep network depth.
2. **Part 2 (Phase 1)** — on the 2-layer net, keep cutting update volume at the current a:q ratio to find the efficiency sweet spot N*.
3. **Part 3 (Phase 2)** — at N*, vary the actor:Q ratio.

## Setup

| Run | Dir | q_updates | actor_updates | Layers | GPU | Status |
|---|---|---|---|---|---|---|
| baseline | `runs/td3/sysid_params/delayr1/` | 200 | 50 | 5 | cuda:1 | killed @ 400k |
| A | `runs/td3/sysid_params/ablater1/` | 50 | 12 | 5 | cuda:3 | 1M done |
| B | `runs/td3/sysid_params/ablate_l2/` | 50 | 12 | 2 | cuda:1 | 1M done |
| C | `runs/td3/sysid_params/ablate_l3/` | 50 | 12 | 3 | cuda:2 | 1M done |

Baseline was killed because throughput was 10 SPS — 1M would have taken ~28h.

## Results at 400k (apples-to-apples with killed baseline)

| Run | Wall | ret@400k | max_ret ≤400k | pos_frac |
|---|---|---|---|---|
| baseline (q200, a50, l5) | **9.49h** | **112.0** | 131.1 | **0.635** |
| A (q50, a12, l5) | 3.41h | 76.7 | 120.2 | 0.596 |
| B (q50, a12, l2) | 2.21h | 71.9 | 128.3 | 0.574 |
| C (q50, a12, l3) | 2.63h | 75.4 | 123.8 | 0.584 |

## Results at 1M (reduced-update runs only)

| Run | Wall | mean SPS | final ret | max ret | pos_frac | Step to ret>50 |
|---|---|---|---|---|---|---|
| A (l=5) | 7.29h | 32 | 83.1 | 142.9 | 0.559 | 110.5k |
| B (l=2) | **4.82h** | **50** | 102.2 | **157.0** | 0.568 | 115.5k |
| **C (l=3)** | 5.67h | 42 | **114.1** | 154.1 | **0.602** | **99.5k** |

## Main observations

### Update count (baseline vs reduced)
- **Baseline was gradient-bound, not env-bound.** 10 SPS means the update step ate ~90% of wall-clock. Cutting q→50, a→12 gave A (same 5-layer arch) **3.2× higher SPS**.
- **Per-step quality drops modestly; per-wall-clock quality improves sharply.** At step 400k, baseline ret=112 beats reduced 72–77. But at the same 3.4h wall-clock, reduced-update A had reached step 400k while baseline was only at ~150k.
- **No ceiling advantage for the baseline.** Baseline peak 131 @394k vs reduced peaks 143–157 @~550k. The extra updates don't reach higher — they just reach a given quality sooner per step. The reduced runs keep climbing past where baseline ran out of budget.
- **Sample efficiency per update is similar.** Baseline's pos_frac=0.635 is the highest, but reduced tails converge to ~0.60. Gap is small.

### Network depth (A vs B vs C, all reduced)
- **3-layer is the sweet spot.** C leads on final return (114), pos_frac (0.602), and time-to-ret>50 (99.5k steps). Only 18% slower wall-clock than the shallowest (B).
- **5-layer is strictly dominated.** A is slowest (32 SPS, 7.3h) and ends lowest (83). Extra depth costs ~50% more wall time with no quality payoff on this task.
- **2-layer peaks nearly as high as 3-layer** (157 vs 154) but finishes lower (102). Fast but less consistent late-training.
- **Peak returns cluster at 143–157 across depths.** Depth affects throughput and early learning more than ceiling.

## Part 2 — Phase 1: lower update volume on 2-layer (2026-04-16 / -17)

Given 2-layer nets (B) give up only ~3 pts max vs 3-layer with 20% more throughput, we doubled down on 2-layer and kept cutting update volume at the same a:q ratio (~0.24).

| Run | Dir | q/a | Total/ep | Result |
|---|---|---|---|---|
| B (anchor, Part 1) | `.../ablate_l2/` | 50/12 | 62 | max=157, final=102, 4.82h |
| P1a | `.../upd_sweep/` | 25/6 | 31 | max=156, final=94, 6.17h* |
| P1b | `.../upd_sweepr1/` | 12/3 | 15 | max=137, final=96, 5.50h* |
| P1c (killed) | `.../upd_sweepr2/` | 6/2 | 8 | max stuck at 33 @ 590k — under-training |

\*Wall times for P1a/P1b were inflated by heavy GPU contention; mean-SPS is the cleaner throughput signal.

### Observations
- **Halving updates (N=62→31) preserves the peak.** P1a max 156 ≈ B max 157. Mean SPS +13% (57 vs 50 clean). The 1M ceiling survives.
- **Quartering (N=62→15) gives up ~13% peak.** P1b max 137, never crosses 100 until step 512k (vs 199k for B). Still learns, just slower per step.
- **N=8 is below the learning threshold.** P1c plateaued at max=33 and was killed. Somewhere between 8 and 15, learning becomes gradient-starved.
- **Optimum total per episode: N* ≈ 31** at a:q ≈ 0.24. Preserves ceiling with ~15% wall-clock improvement vs B in clean-GPU conditions. Gain is **modest, not transformative** — halving updates again (to 15) would save more time but costs ceiling.

## Part 3 — Phase 2: actor:Q ratio sweep at N*=31 (2026-04-17)

Fixed total updates at N*=31 per episode, varied the actor:Q ratio.

| Run | Dir | q/a | ratio | max | final | tail10 | pos | t>100 | meanSPS |
|---|---|---|---|---|---|---|---|---|---|
| P2a | `.../ratio_sweep/`   | 29/2  | **0.07** | 129.0 | 75.4 | 92.8 | **0.629** | 564k | 66 |
| **P1a (anchor)** | `.../upd_sweep/`  | 25/6  | **0.24** | **156.1** | 93.7 | 90.8 | 0.543 | 295k | 57 |
| P2b | `.../ratio_sweepr1/` | 21/10 | **0.48** | 136.4 | 96.4 | 89.4 | 0.602 | **272k** | 42 |
| P2c | `.../ratio_sweepr2/` | 10/21 | **2.10** | 140.9 | 97.8 | 89.2 | 0.592 | 342k | 75 |

### Observations
- **a:q ≈ 0.24 is clearly the best ratio on peak return.** P1a's max 156 stands ~15 pts above every other ratio (129–141). This matches TD3's theoretical delayed-policy-update prescription (fewer actor than Q updates).
- **Tail-10 and final returns are flat across ratios (~89–98).** Peak quality and steady-state quality decouple — P1a spikes higher but converges to similar final policy as actor-rich/starved variants. One seed, so noise could explain most of this.
- **Actor-starved (0.07) is the worst per-step learner** (t>100 at 564k vs 272k for moderate ratio) but has the highest pos_frac (0.629). The Q function stays accurate; the policy just moves slowly toward it.
- **Actor-rich (2.10) learns fast and throughputs well** (75 SPS) but caps ~16 pts below P1a. Too many actor updates per Q update destabilizes the critic-anchored target.
- **Throughput is not monotonic in ratio.** P2c (2.10) runs at 75 SPS, P2b (0.48) at 42 SPS, despite same total updates. Actor and Q update cost differently per their loop structure (target-network sync, PER updates, etc.).

### Recommended config (2-layer)

**`q_updates=25, actor_updates_per_iteration=6`** (P1a, N=31, ratio 0.24). Preserves B's peak (~157), modest throughput gain (~15% clean), and ratio 0.24 clearly beats other ratios on peak quality. If throughput matters more than peak, q=10/a=21 also reaches ~141 max at 75 SPS (fastest) — acceptable for sweeps, not for final runs.

## Overall recommendation

- **Network:** 2-layer (fastest, gives up only ~3 pts peak vs 3-layer).
- **Updates:** `q_updates=25, actor_updates_per_iteration=6` (P1a config).
- **Fallback for hyperparameter sweeps:** `q_updates=10, actor_updates_per_iteration=21` or `q_updates=12, actor_updates_per_iteration=3` if throughput is the priority and ~15% peak loss is acceptable.

## Commands used

```bash
# Part 1: Baseline (killed at 400k)
python scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py \
  --args-file scripts/smooth_policy/amp_history/configs/td3/td3_no_alignment.yaml \
  --config   scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params.yaml \
  --total-timesteps 1000000 --q-updates 200 --actor-updates-per-iteration 50 \
  --agent-num-hidden-layers 5 --q-num-hidden-layers 5 --enable-puck-delay-interpolation

# Part 1: Reduced-update + depth sweep (A/B/C differ only in --agent/q-num-hidden-layers = 5/2/3)
python .../td3_training.py \
  --args-file .../td3_no_alignment.yaml --config .../sysid_best_params.yaml \
  --total-timesteps 1000000 --q-updates 50 --actor-updates-per-iteration 12 \
  --agent-num-hidden-layers {2,3,5} --q-num-hidden-layers {2,3,5} \
  --enable-puck-delay-interpolation

# Part 2 (P1a/b/c): 2-layer, same a:q, varying total
#   P1a: --q-updates 25 --actor-updates-per-iteration 6
#   P1b: --q-updates 12 --actor-updates-per-iteration 3
#   P1c: --q-updates 6  --actor-updates-per-iteration 2  (killed, under-training)

# Part 3 (P2a/b/c): 2-layer, total=31, varying a:q
#   P2a: --q-updates 29 --actor-updates-per-iteration 2   (ratio 0.07)
#   P2b: --q-updates 21 --actor-updates-per-iteration 10  (ratio 0.48)
#   P2c: --q-updates 10 --actor-updates-per-iteration 21  (ratio 2.10)
```
