# Zero-shot sim2real ablation extension (500k → 700k) + new `no_obs_delay_randomization`

- **Date**: 2026-05-09 13:36 UTC start, 18:49 UTC end
- **Status**: 12/12 trained at 700k (exit 0). Ready for sim2real handoff.
- **Run dirs**: `runs/td3/zeroshot_ablations_700k/<name>/seed0/` (500k originals at `runs/td3/zeroshot_ablations/...` left untouched)
- **Configs**:
  - sim YAMLs: `configs/new_juggle/zeroshot_ablations/sim_<name>.yaml` (existing) + new `sim_no_obs_delay_randomization.yaml`
  - TD3 args YAMLs: `configs/td3/zeroshot_ablations_700k/td3_zeroshot_<name>{_extend}.yaml`
  - Generators: `_generate.py` in each of the two dirs above (sim generator was extended; td3 700k generator is new)
  - Launcher: `scripts/smooth_policy/run_zeroshot_ablations_700k.sh <gpu_id>`

## Question

Two things in one sweep, both motivated by the 500k results
([2026-05-05_17-38_zero-shot-sim2real-ablations.md](2026-05-05_17-38_zero-shot-sim2real-ablations.md)):

1. **Are the 500k policies converged?** The base run `hist2_motion0_v2` peaked
   at 850k @ 169.72 mean (1M total), so 500k is mid-rise. The user wanted
   another +200k of training to make sure the ablation comparison isn't
   distorted by under-training.
2. **What does the obs-delay ablation actually look like once we decouple
   it from the puck_history density bug?** The original `no_obs_delay`
   flatlined at mean 17 because `enable_observation_delay: false` collapses
   the breakpoints sub-step loop and 3×-shrinks puck-history temporal density
   (see the 500k diagnosis). Replace it with `no_obs_delay_randomization`,
   which keeps `enable_observation_delay: true` (sub-stepping unchanged) but
   flips `randomize_delay: true → false` so the ±25% per-step jitter is
   removed cleanly.

## Setup

**Recipe** = each ablation's existing 500k td3 args YAML, with these
overrides:
- `total_timesteps: 700000` (was 500000)
- For continuations: `model_path:
  runs/td3/zeroshot_ablations/<name>/seed0/training_state.pth` (full_resume
  default — restores actor/target/qfs/optimizers/replay/global_step/RNG)
- `log_parent_dir: runs/td3/zeroshot_ablations_700k/<name>/seed0` (new dir
  so the 500k final state isn't overwritten)

For `no_obs_delay_randomization`: fresh 700k run from scratch (no
warmstart) on a brand-new sim YAML.

**Single seed per ablation** (seed 0; same as the 500k sweep).

### GPU split

5h total wall clock — both pipelines started 13:36 UTC.

| GPU | runs | wall clock | finished |
|-----|------|---|---|
| cuda:0 | 5 (1 fresh 700k + 4 continuations) | 5h 14m | 18:49 UTC |
| cuda:1 | 7 continuations | 4h 28m | 18:04 UTC |

Per-run: continuations 31–41 min (mean 39 min); the fresh 700k took 151 min.
No failures.

### Pre-launch smoke test

Before launching the full sweep I ran the `baseline_extend` config
manually for ~60s and confirmed:
- `Loading model/checkpoint from runs/td3/zeroshot_ablations/baseline/seed0/training_state.pth`
- `Resuming training from global_step=500000, iteration=500000`
- Replay buffers re-loaded (success_capacity=30k, failure_capacity=70k)
- First step's rolling avg return = 79.4, matching the baseline's 500k
  end-of-training mean of 81 — the resumed policy is the same one we
  trained.

## Results

### Final mean rolling return at 700k

Mean over the last 20 reports of `Rolling(2k) Avg Return` (≈ last 10k env
steps; same convention as the 500k writeup). All numbers single-seed —
SE not estimable from one seed; treat ±10 as the noise floor based on
the 500k sweep's spread.

| ablation | 500k mean | 700k mean | Δ vs 500k | 700k success |
|---|---:|---:|---:|---:|
| baseline | 81 | **97** | **+16** | 0.85 |
| sysid_off | 116 | 110 | -6 | 0.80 |
| no_paddle_puck_strength | 90 | 100 | +10 | 0.77 |
| no_paddle_puck_direction | 80 | 98 | +18 | 0.89 |
| no_wall_direction | 95 | 88 | -7 | 0.80 |
| no_action_attenuation | 82 | 91 | +9 | 0.81 |
| start_100_near_top | 83 | 97 | +14 | 0.85 |
| start_100_near_paddle | 73 | 102 | +29 | 0.84 |
| no_puck_noise | 105 | 122 | +17 | 0.85 |
| no_occlusions | 98 | 90 | -8 | 0.82 |
| **no_obs_delay_randomization** | — *(new — replaces broken `no_obs_delay`)* | **90** | new | 0.83 |
| all_sysid_no_rand | 77 | 100 | +23 | 0.71 |

Headline: 8/11 continuations went up (mostly +10 to +29), 3 went down
slightly (sysid_off, no_wall_direction, no_occlusions; all -6 to -8 —
within the ±10 single-seed noise band). The recipe is robust enough that
the +200k extension didn't unlock a dramatic separation between healthy
ablations — they all sit in 88–122 mean now, vs 73–116 at 500k.

### Trajectory shape (avoid one-number-overclaim)

Per the long-running guidance to not collapse single-seed runs to one
number, here are means at 500k (the resume baseline), 600k, and 700k for
each run. Several runs dipped between 500k and 600k before recovering at
700k — typical PER + Polyak noise on a single seed. No collapses.

| ablation | 500k | 600k | 700k |
|---|---:|---:|---:|
| baseline | 79 | 97 | 97 |
| sysid_off | 113 | 125 | 110 |
| no_paddle_puck_strength | 75 | 69 | 100 |
| no_paddle_puck_direction | 81 | 72 | 98 |
| no_wall_direction | 81 | 73 | 88 |
| no_action_attenuation | 89 | 104 | 91 |
| start_100_near_top | 78 | 106 | 97 |
| start_100_near_paddle | 55 | 101 | 102 |
| no_puck_noise | 122 | 118 | 122 |
| no_occlusions | 73 | 68 | 90 |
| no_obs_delay_randomization | 78 | 88 | 90 |
| all_sysid_no_rand | 83 | 112 | 100 |

### `no_obs_delay_randomization` — the key new datapoint

The new ablation **trained successfully**: 90 mean / 0.83 success at
700k (sits in the middle of the 88–122 band of all the other healthy
ablations). This isolates the actual ablation question — *does
randomizing the observation delay help* — from the env-level
puck_history density coupling that broke the original `no_obs_delay`
ablation at 500k. Specifically:

- `enable_observation_delay: true` is **kept on**, so the breakpoints
  sub-step loop in `airhockey_box2d.py:1680` runs over `[0, t_obs,
  time_per_step]` (3 sub-steps) and the puck_history is appended at
  the same density the recipe was tuned for.
- `randomize_delay: true → false` is the only diff from canonical. The
  jitter code path (`airhockey_box2d.py:1668`) is gated on
  `randomize_delay AND delay_relative_range > 0`, so the per-step jitter
  is fully removed without touching the density.

Implication for the paper: the `no_obs_delay_randomization` row is the
correct ablation to report for "delay randomization off" — the original
`no_obs_delay` 500k row should be marked deferred (or omitted) until the
puck_history-append-inside-breakpoints coupling in the simulator is
fixed.

## Conclusion

All 12 ablations at 700k. The `randomize_delay` ablation isolates the
intended knob (delay-jitter) without the env coupling and trains to a
healthy 90 mean — i.e., removing delay-jitter alone is **not** a
training failure, ruling out "delay randomization is essential" as a
strong claim. The original 500k `no_obs_delay` flatline was an
artifact of the sub-step coupling, not a domain-randomization signal.

The 11 healthy continuations + 1 new fresh run are ready for the
sim2real transfer comparison on the user's other machine. The original
500k checkpoints and `runs/td3/zeroshot_ablations/...` paths are still
intact for reproducing the earlier comparison; the 700k extensions live
under `runs/td3/zeroshot_ablations_700k/<name>/seed0/`.

## Caveats

- **Single seed**. The 88–122 spread across ablations at 700k is
  single-seed noise plus randomization-knob effect mixed together. To
  separate signal from noise, would need ≥3 seeds per ablation. Treat
  cross-ablation deltas under ~10 mean as not statistically meaningful.
- **`Rolling(2k)` is over only ~10k env steps**, not the full 50k+
  needed for a tight estimate. The "700k mean" column above is a
  smoothed end-of-training summary, not a settled long-horizon eval.
  Recommend running `eval_all_ckpts_residual.sh` per ablation on the
  canonical sim before the sim2real handoff if the transfer comparison
  is sensitive.
- **No off-by-one fix**. The trainer still doesn't write a
  `checkpoint_700000` (only every-25k intervals up to 675000), but the
  final `model.pth` and `training_state.pth` at the run root *do* hold
  the exact 700k state — those are the right files to hand off.
- **`all_sysid_no_rand` continuation kept the original
  `enable_observation_delay: false` semantics** (rather than switching
  to `randomize_delay: false`) so we could full_resume from the 500k
  checkpoint without an env shift. This means the "all randomization
  off" bucket still has the puck_history density change baked in. If
  the paper wants strict consistency between "all randomization off"
  and the new `no_obs_delay_randomization` convention, that ablation
  needs a fresh 700k from scratch with the updated sim YAML.

## Next

- ✅ Both GPU pipelines completed cleanly (12/12 exit 0). Ready for
  sim2real handoff.
- Hand off `model.pth` (or training_state.pth for further fine-tuning)
  of each of the 12 ablations to the real-robot machine. Recommended
  pre-handoff: run `eval_all_ckpts_residual.sh` per ablation on the
  canonical sim to pick the best ckpt rather than just using the final
  one.
- Open follow-up (still): fix the
  puck_history-append-inside-breakpoints coupling in
  `airhockey/sims/airhockey_box2d.py` so a true `no_obs_delay`
  ablation (delay fully off, sub-steps preserved) is also runnable.
- Open follow-up (still): implement empirical-start-state init in the
  env so the real-world-starts ablation can be run.
