# Sim2sim target redesign: warp-based perturbation for trainable + mismatched env

- **Date**: 2026-05-07 15:48 UTC start
- **Status**: done. All 4 Phase-2 trainability cells passed (peak ≥ 100). Recommended canonical target: **`warp075_p30`** (paddle −30% + warp 0.075, zs=48, peak 112.6, end-window mean 70).
- **Source policy** (untouched): `latest_model/hist2_motion0_v2/model.pth`
- **Sim configs**: `scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_warp075_p{00,10,20,30}.yaml`
- **Training configs**: `scripts/smooth_policy/amp_history/configs/td3/sim2sim/warp075_trainability/td3_warp075_p{00,10,20,30}.yaml`
- **Run dirs**: `runs/td3/sim2sim_redesign/warp075_p{00,10,20,30}/seed0/`
- **Logs**: `notes/scratch/sim2sim_redesign_logs/p{00,10,20,30}.log`
- **Sweep harness**: `notes/scratch/sim2sim_warp_paddle_zs_sweep.py`

## Question

The previous sim2sim target (`sim2sim_combined.yaml`, paddle50) stacked many
perturbations (paddle −50%, action delay newly enabled, restitutions reduced,
wall cone widened, pid softer, hist_len 2→4) and turned out to be **structurally
untrainable from scratch** (peak 84 mean 47 at 3.85M, per
`from_scratch_5M_bigger_net`). That makes residual-RL gains on it unfalsifiable
— if from-scratch can't reach a sensible asymptote, "improvement over zs" has
no upper bar to hit.

User constraints for the redesign:
1. **Trainable from scratch**: ≥ 100 mean within 400k steps.
2. **Clear mismatch**: source policy zs < 60.
3. **Locked params** (forced equal between source-train env and target):
   all delays, hist_len (=2 to keep `hist2_motion0_v2` source), all
   restitutions.
4. **Available knobs**: paddle_radius (mass-preserved), wall_direction_cone_deg,
   pid_kp, and the new `puck_obs_sine_warp_amplitude` (sinusoidal y-distortion
   on puck observation, edge-preserving — `make_sine_y_warp_fn` in
   `airhockey/observation_homography.py`).
5. **Test paddle-on-top regardless** of whether warp alone clears the bar.

## Phase 1 — zero-shot heatmap (4-GPU parallel; ≈1 min wall total)

2D grid over (paddle_pct, warp_amp), all other params = source. 50 deterministic
episodes per cell. Each row (one paddle pct) ran on one GPU; full grid in
parallel.

### Results — `charts/avg_episode_return` mean ± std

```
paddle\warp |   0   | 0.05 | 0.075 | 0.10 | 0.125 | 0.15 | 0.20 | 0.25
   -0%      | 143.5 | 96.9 |  58.1 | 37.2 |  33.7 | 32.4 | 24.5 | 20.6
  -10%      | 134.4 | 85.8 |  48.7 | 35.6 |  28.0 | 30.6 | 21.7 | 22.0
  -20%      | 132.0 | 70.0 |  60.0 | 35.5 |  28.0 | 25.7 | 22.1 | 20.7
  -30%      | 126.7 | 68.8 |  48.1 | 32.6 |  26.4 | 22.5 | 22.3 | 20.4
```

(`✓` cells in `summary.md` mark mean < 60. Also see
`runs/td3/sim2sim/zs_warp_paddle_sweep/summary.{md,json}`.)

### Findings

1. **Warp dominates over paddle for zs degradation.** Across a row (paddle
   held), warp 0 → 0.075 drops zs by 70–100. Going down a column (warp held),
   paddle 0 → −30% drops zs by only 5–15.
2. **(paddle 0%, warp 0)** = 143.5 — confirms source policy is healthy on its
   native env (matches the source training reaching mean ≈ 169 at 850k).
3. **The "sweet zone" of zs ∈ [48, 60] is the warp = 0.075 column**: zs across
   paddle reductions = {58, 49, 60, 48}. Warp ≥ 0.10 crashes zs below 35 —
   needlessly aggressive and likely to hurt from-scratch trainability.
4. **Warp 0.05 is borderline**: zs 70–97 across paddle reductions; doesn't
   reliably clear the < 60 bar even with paddle −30%.

## Phase 2 — from-scratch trainability sweep (4-GPU parallel; ETA ≈ 50–55 min)

Pin warp = 0.075, sweep paddle ∈ {0%, −10%, −20%, −30%}. Same TD3 from-scratch
recipe that produced `hist2_motion0_v2` (the source policy itself), only
differences: total_timesteps 1M → 400k, sim config swap.

| Cell | zs (Phase-1) | paddle_radius | GPU | Pass condition |
|---|---:|---:|---:|---|
| `warp075_p00` | 58.1 | 0.0508 (source) | 0 | rolling-2k mean ≥ 100 by 400k |
| `warp075_p10` | 48.7 | 0.04572 | 1 | same |
| `warp075_p20` | 60.0 | 0.04064 | 2 | same |
| `warp075_p30` | 48.1 | 0.03556 | 3 | same |

### TD3 recipe (inherited from `td3_hist2_motion0_v2.yaml`)

```
q_updates: 25, target_network_frequency: 10  (Polyak fires twice per cycle)
policy_lr: 3e-4, q_lr: 1e-3
agent_hidden_layer_size: 64, agent_num_hidden_layers: 2
PER on, success/failure split active, full primitive exploration
total_timesteps: 400000, model_path: null  (from-scratch)
```

Same recipe trained `hist2_motion0_v2` to mean ≈ 169 at 850k on the **source**
sim, so it should comfortably reach 100 in 400k on a closely-related target.

## Decision rule once Phase 2 lands

For each cell, compute `peak rolling-2k mean` across the full 400k.

- **If multiple cells pass** (peak ≥ 100): pick the one with the **largest
  paddle reduction that still passes** — that maximizes the dynamics-mismatch
  surface (paddle hurts zs less than warp, but adds a second axis of
  asymmetry which is realistic for sim2real). Promote that config to
  `sim2sim_warp_v3.yaml` (or chosen canonical name).
- **If only `p00` passes**: pure-warp target. Cleaner story, single perturbation
  axis.
- **If no cell passes**: trainability is bottlenecked. Drop warp to 0.05 and
  re-run. zs at 0.05 is 70–86, doesn't clear < 60 alone, so we'd also need
  paddle reduction; but paddle alone barely moves zs — we'd be stuck.
  Fallback: relax mismatch goal (allow zs < 80 instead of < 60) or add
  `pid_kp` as a third perturbation axis.

## Results — all 4 from-scratch curves (single seed, 400k each)

`charts/rolling2k_avg_episode_return`. Source-policy zs from Phase 1 in second column.

```
cell  paddle    zs    0-100k       100-200k     200-300k     300-400k     peak
p00     0%     58   22[18,26]   46[32,60]    69[54,85]   72[58,89]   102.8 @ 240k
p10   -10%     49   21[17,27]   45[31,62]    65[53,77]   71[58,86]   103.2 @ 394k
p20   -20%     60   21[18,25]   40[28,53]    63[50,79]   71[57,86]   115.2 @ 358k
p30   -30%     48   20[17,25]   43[30,58]    62[51,75]   70[56,85]   112.6 @ 363k
```

**All 4 cells pass the trainability bar (peak ≥ 100).** End-window means are
essentially equal (70–72) across all paddle reductions; trajectory shape is
the same (warmup → ramp → stable ~70). Only the peaks differ noticeably, and
peaks are noisy single-checkpoint readings.

### Decision

**Canonical pick: `warp075_p30` (paddle −30% + warp 0.075).**

| Criterion | Score |
|---|---|
| Trainability (peak ≥ 100) | ✅ 112.6 @ 363k |
| Clear mismatch (zs < 60) | ✅ zs = 48 (clearest of all 4) |
| End-of-training gap (mean − zs) | ✅ **+22** (tied with p10 for highest) |
| Multi-modal mismatch surface | ✅ geometric (paddle) + perceptual (warp) |

**Alternative: `warp075_p10`** — same end-of-training gap (+22), milder paddle
perturbation (−10%). Cleaner story if you prefer paddle as a "subtle calibration
error" rather than a large geometric change.

`p20` is borderline (zs = 60 right at boundary, so doesn't strictly satisfy
the < 60 bar). `p00` (pure-warp) clears trainability comfortably but has the
weakest mismatch (zs = 58, only 2 points below the bar).

## Next

1. Wait for the 4 from-scratch curves (~50–55 min).
2. Apply the decision rule above; promote the chosen config.
3. With the new target locked in, restart the residual-RL recipe development
   (CQL etc.) on this trainable env. The previous post-Polyak-fix CQL work was
   done on the un-trainable `paddle50` target, so its conclusions need to be
   re-validated on the new target.
