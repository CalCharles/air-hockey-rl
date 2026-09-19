# Four-timestep PID-target smoothing (`hist_len: 4`) on the five canonical tasks — worse than `hist_len: 2` on every non-saturated task

- **Date**: 2026-09-04 22:19 UTC (writeup); runs launched 2026-09-04 ~14:00 UTC
- **Status**: done
- **Run dirs**: `runs/td3/tasks_hist4_20260904/` (hist4, this experiment) · `runs/td3/tasks_20260904/{sysid,dr}/` (hist2 baseline, from [2026-09-04_04-50_five-tasks-new-recipe.md](2026-09-04_04-50_five-tasks-new-recipe.md))
- **Configs**: `configs/td3/tasks/*.yaml` (TD3 args, **unchanged** from the hist2 baseline) · `configs/new_juggle/tasks/sim_{sysid,dr}_<task>.yaml` (sim configs, flipped to hist4)

## Question

The five canonical tasks were validated under `hist_len: 2` (2-timestep moving-average low-pass on
the PID target). Does moving to `hist_len: 4` — the smoothing setting we want for the real robot —
hold up, so it can be adopted as the canonical recipe?

## Setup

Identical TD3 recipe to the hist2 baseline — **no hyperparameter changed**. Only the sim configs moved:

| Key | hist2 baseline | hist4 (this run) |
|---|---|---|
| `hist_len` | 2 | **4** |
| `pid_kp` | 9000 | **7500** |
| `paddle_density` | 3000 | **3500** |
| `_dr` `paddle_density` range | 2250–3750 | **2625–4375** |

`pid_kp` / `paddle_density` move with `hist_len` because the filter changes the paddle's effective
command dynamics: each `hist_len` has its own paddle sysid fit. The hist4 values come from a
windowed-10 3D grid search over teleop data **re-recorded under `hist_len: 4`** (2026-05-21), carried
in `configs/new_juggle/sysid_best_params_hist4.yaml`. `puck_damping` / `gravity` are puck-side and
unchanged. So this compares two internally-calibrated configurations, **not** smoothing in isolation
(see *Next*).

Budget as before: `_sysid` 1M steps, `_dr` 2M steps, single seed per cell, 10 jobs on 4 GPUs.

## Results

All 10 runs exited rc=0.

**DR runs** — per-checkpoint `multi_env_eval.json` (5 envs × 4 eps). "back-half" = mean over the
last 40 of 79 checkpoints; ± is SE **across those checkpoints** (within-run stability), not across seeds.

| Task | hist2 back-half | hist4 back-half | Δ | hist2 max | hist4 max |
|---|---:|---:|---:|---:|---:|
| juggle | **116.8 ± 1.6** | 94.5 ± 1.9 | −22.3 | 137.2 | 113.3 |
| puck_vel | **57.7 ± 1.3** | 25.0 ± 0.5 | −32.7 | 79.8 | 31.1 |
| reach | **10.0 ± 0.0** | 6.1 ± 0.7 | −3.9 | 10.0 | 10.0 |
| reach_vel | **10.0 ± 0.0** | 9.6 ± 0.2 | −0.4 | 10.0 | 10.0 |
| touch | 1.0 ± 0.0 | 1.0 ± 0.0 | 0.0 | 1.0 | 1.0 |

**sysid runs** — `charts/avg_episodic_return`, back half of training:

| Task | hist2 back-half | hist4 back-half | Δ |
|---|---:|---:|---:|
| juggle | **83.4 ± 1.2** | 64.3 ± 1.0 | −19.1 |
| puck_vel | **20.6 ± 0.3** | 17.5 ± 0.2 | −3.1 |
| reach | 10.0 ± 0.0 | 10.0 ± 0.0 | 0.0 |
| reach_vel | 10.0 ± 0.0 | 10.0 ± 0.0 | 0.0 |
| touch | 0.9 ± 0.0 | 0.9 ± 0.0 | 0.0 |

**Trajectory shape matters more than the endpoints** (DR, eval mean every 200k steps):

```
juggle    hist2:   23   65   91   95  108  113  119  132  119  127
juggle    hist4:   21   51   68   82   60   95   95   86  108   81

puck_vel  hist2:    2   21   33   37   53   60   46   44   64   48
puck_vel  hist4:    1   15   20   22   22   26   26   27   28   24

reach     hist2:    0   10   10   10    9   10   10   10   10   10
reach     hist4:    0    2    0    0    0    1    2    7   10   10
```

- **juggle**: hist4 tracks below hist2 the whole way and is visibly noisier; it is not a slower rise
  to the same plateau.
- **puck_vel**: hist4 plateaus at ~25 and never exceeds 31 — hist2's *max* is 79.8. This is a lower
  asymptote, not a lag.
- **reach**: the largest effect. hist2 saturates by 200k and holds. hist4 sits at 0–2 for ~1.4M steps
  and only reaches 10 near the very end. The back-half mean of 6.1 reflects a late, incomplete rescue,
  not steady mid-range performance. Under sysid (no DR) reach is unaffected — the interaction is
  specifically hist4 × domain randomization.

## Conclusion

**`hist_len: 4` as specified is worse than `hist_len: 2` on every task that is not already saturated,
in both sysid and DR, and it is not close.** The three sparse/saturated tasks (touch, reach_vel, and
reach under sysid) are indifferent because they sit at their reward ceiling; they carry no signal
either way. On the two tasks with headroom — juggle and puck_vel — hist4 loses ~19–23 % (juggle) and
~15–57 % (puck_vel). Under DR, hist4 additionally breaks reach's learning speed almost entirely.

Confidence: the *direction* is solid — single seed per cell, but the gap is consistent across 4
independent comparisons (2 tasks × 2 physics settings), holds across the whole trajectory rather than
at one endpoint, and is far larger than within-run checkpoint SE. The *magnitude* is single-seed and
should not be quoted precisely.

**On this evidence hist4 should not be adopted as the canonical recipe.** The configs in
`configs/new_juggle/tasks/` are currently flipped to hist4 (that was the instruction that prompted
this run); reverting them to hist2 is a one-line-per-file change.

## What is untested / confounded

The comparison changes smoothing **and** the paddle gains together, because each `hist_len` carries
its own sysid fit. So we cannot yet say whether the regression comes from the 4-step filter itself or
from the hist4 gains (`kp` 9000→7500, `density` 3000→3500). Both configurations are individually
calibrated, so this is a fair test of *the two deployable configurations* — but not of smoothing alone.

Nothing here says anything about **real-robot** transfer, which is the reason hist4 was wanted. A sim
regression of this size is a strong argument against it, but sim return is not the transfer metric.

## Next

Each gets its own file in this directory:

1. **Disentangle the confound** — run `hist_len: 4` with the *hist2* gains (kp 9000 / density 3000),
   juggle + puck_vel only, sysid + DR. Cheap (~4 cells) and decides whether the filter or the refit
   is responsible.
2. **Seed the headline cells** — if hist4 is still wanted, 3 seeds on juggle_dr and puck_vel_dr for
   both hist2 and hist4, so the magnitude can be quoted.
3. **hist4 × DR reach interaction** — reach fails only under DR at hist4. Worth a look; it may be the
   recentred `paddle_density` range (2625–4375) rather than the filter.
