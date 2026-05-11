# Retrain hist2_motion0 on the latest collision-randomization sim

- **Date**: 2026-05-05 02:55 UTC start, 06:55 UTC done
- **Status**: done
- **Supersedes**: prior `latest_model/hist2_motion0/` (trained on a sim that did NOT have paddle-puck strength/direction or wall-direction randomization). `runs/td3/hist_motion_collision/hist2_motion0/checkpoint_975000/` was a closer match (same sim) but weaker peak (148.08 vs new 169.72).
- **Run dir**: `runs/td3/hist_motion_collision/hist2_motion0_v2/seed0/`
- **Config**: `configs/td3/td3_hist2_motion0_v2.yaml`
- **Promoted to**: `latest_model/hist2_motion0_v2/`

## Question

The model at `latest_model/hist2_motion0/` was the source policy for
all sim2sim and sim2real residual-RL work, but it was trained on a sim
that lacked the latest collision randomization knobs (paddle-puck
strength, paddle-puck direction, wall direction). Going forward we
want a base policy trained on the up-to-date source sim
(`sysid_best_params_hist2.yaml`, which has all those randomizations
enabled).

## Setup

Recipe = exactly `td3_recommended.yaml` with one override:
- `config:` swapped to `sysid_best_params_hist2.yaml` (the canonical
  hist=2 source sim, with the latest collision randomization).

All other knobs unchanged: `q_updates: 25`, `actor_updates_per_iteration: 6`,
`exploration_noise: 0.1`, primitive chance 0.15→0.05 over 200k, full
primitive set, 64×2 actor + critic, 1M total steps, checkpoint every
25k.

Single seed, single GPU (RTX 6000), 1M env steps. Training started
2026-05-05 02:55 UTC and finished 06:26 UTC (~3h 33m wall clock).

## Results

39 checkpoints + final model.pth, all evaluated on the same source sim
(`sysid_best_params_hist2.yaml`), n=50 deterministic eval per
checkpoint.

| metric | value |
|---|---:|
| **Best ckpt** | **850k @ mean 169.72** |
| Runner-up | 925k @ mean 165.66 |
| Final-step model.pth | mean 152.50 |
| Trajectory range | 17 (early) → 170 (peak) |
| Mean across all 39 ckpts | 145 |

Top 10 ckpts by mean return:

| step | mean |
|---:|---:|
| 850000 | 169.72 |
| 925000 | 165.66 |
| 625000 | 154.90 |
| 700000 | 154.36 |
| 675000 | 153.80 |
| 350000 | 153.38 |
| 550000 | 150.68 |
| 575000 | 150.42 |
| 600000 | 147.64 |
| 875000 | 146.38 |

Last 10 ckpts (post-peak settle):

| step | mean |
|---:|---:|
| 750000 | 135.02 |
| 775000 | 144.44 |
| 800000 | 140.90 |
| 825000 | 136.08 |
| **850000** | **169.72** ← peak |
| 875000 | 146.38 |
| 900000 | 134.04 |
| 925000 | 165.66 |
| 950000 | 133.36 |
| 975000 | 135.66 |

Comparison vs prior `runs/td3/hist_motion_collision/hist2_motion0/`
(same sim, prior single-seed run):

| metric | prior hist2_motion0 | **hist2_motion0_v2 (this run)** |
|---|---:|---:|
| best ckpt mean | 148.08 (@ 975k) | **169.72 (@ 850k)** |
| final-step mean | 148.08 | 152.50 |

## Conclusion

`hist2_motion0_v2` is the new sim-to-real ground truth source policy.
Promoted to `latest_model/hist2_motion0_v2/` (peak ckpt 850k).
Significantly stronger than the prior version (+21 peak), which is
typical seed variance — same recipe, same 1M budget, just a fresh seed
landed a higher ckpt.

This is the model future sim2sim and sim2real residual-RL recipes
should reference (in `model_path:`).

## Next

- Future configs that currently reference
  `runs/td3/hist_motion_collision/hist2_motion0/checkpoint_975000/model.pth`
  (e.g., `td3_residual_v27_ensemble5.yaml`) can stay as-is for
  reproducibility of past experiments; new configs going forward should
  use `latest_model/hist2_motion0_v2/model.pth`.
- Re-run zero-shot eval of the new base on `sim2sim_combined.yaml`
  (paddle50) to establish the new zero-shot reference for residual
  experiments — the prior `zs=67.54` was for the OLD base policy. New
  zero-shot may differ, which would shift all "% above zs" metrics.
  → File when run: `experiments/<date>_<time>_hist2-motion0-v2-paddle50-zero-shot.md`.
