# Five canonical tasks × {sysid, DR} under the 2026-09-04 recipe (q_weight_decay 0, single flat replay buffer, ×10 sparse rewards)

- **Date**: 2026-09-04 04:50 UTC start
- **Status**: done (`touch_dr` was at 1.61M / 2M steps, stable, when this was written; it finishes unattended and the runner writes `summary.md`)
- **Run dirs**: `runs/td3/tasks_20260904/sysid/<task>_sysid`, `runs/td3/tasks_20260904/dr/<task>_dr`
- **Configs**: `configs/td3/tasks/<task>_{sysid,dr}.yaml`, `configs/new_juggle/tasks/sim_{sysid,dr}_<task>.yaml`
- **Supersedes** (as the reference 5-task batch): the `full_nodr_20260903_2252` runs of `2026-09-04_01-05_sparse-task-collapse-diagnosis.md`
- **Related**: `2026-09-04_01-05_sparse-task-collapse-diagnosis.md` (why the recipe changed)

## Question

Does the recipe that fixed the three collapsing sparse tasks — `q_weight_decay: 0`, `single_replay_buffer: true` (one 1M-transition buffer), and ×10 baked into the reach / reach_vel / puck_vel reward classes — train all five canonical tasks, with sysid physics and with ±25 % physics randomization, without hurting juggle and touch?

## Setup

`scripts/td3/run_experiments.py --mode auto --configs configs/td3/tasks/*_{sysid,dr}.yaml`, seed 0, two to three jobs per GPU. sysid = 1M steps (`td3_training.py`); dr = 2M steps (`td3_training_dr.py`, 5-env × 4-episode checkpoint eval). Everything else is the canonical `td3_recommended_top50_hist2.yaml` / `td3_paramrand_pm25.yaml`. Rewards: juggle unchanged, touch +1, reach +10, reach_vel +10, puck_vel 10 × upward displacement.

## Results

Training curves (rolling 5k-step bins). "min succ last 50 %" is the lowest bin in the second half of training — the "stays there" check.

| Run | steps | success, last 20 % | min success, last 50 % | return, last 20 % | ep len, last 20 % |
|---|---:|---:|---:|---:|---:|
| juggle_sysid | 1M | 0.88 | 0.65 | **83.9** (old recipe, same day: 55–75) | 133 |
| touch_sysid | 1M | 0.89 | 0.80 | 0.89 (old: 0.85) | 6.4 |
| reach_sysid | 1M | **1.00** | 0.95 | 10.0 (old: 0.08) | 7.4 |
| reach_vel_sysid | 1M | **1.00** | 0.99 | 10.0 (old: 0.01) | 9.6 |
| puck_vel_sysid | 1M | 0.86 | 0.68 | **21.2** = 2.1 m of upward puck travel / episode (old: 0.06; random 0.14; scripted intercept 0.59) | 84 |
| juggle_dr | 2M | 0.90 | 0.67 | 93.6 | 144 |
| touch_dr | 1.61M | 0.89 | 0.83 | 0.89 | 6.5 |
| reach_dr | 2M | **1.00** | 0.99 | 10.0 | 5.9 |
| reach_vel_dr | 2M | **1.00** | 0.98 | 10.0 | 8.3 |
| puck_vel_dr | 2M | 0.89 | 0.68 | **23.7** | 86 |

Final DR checkpoint multi-env eval (5 envs × 4 eps, `multi_env_eval.json`): juggle return 106.6 / success 1.0; touch 1.0 / 1.0; reach 10.0 / 1.0; reach_vel 10.0 / 1.0; puck_vel 54.1 / 1.0. Caveat: `td3_training_dr` hard-codes `env.max_timesteps = 200` for that eval and counts "ran the full budget without terminating" as success (juggle semantics), so for puck_vel the eval return (5.4 unscaled over 200 steps) and success are not comparable to the training curve; the training curve is the like-for-like number.

## Conclusion

The recipe works on all five tasks in both physics settings. reach and reach_vel are at 100 % from ~150–300k and never drop below 0.95 afterwards; puck_vel holds 0.86–0.89 with ~2 m of upward puck travel per episode; touch is unchanged (0.89 vs 0.85) and juggle is better than under the old recipe (84 vs 55–75 at 1M sysid), though that is one seed and juggle's curve is noisy (min bin 0.65). No task regressed. Single seed throughout.

## Follow-up (same day, after these runs)

- The DR checkpoint eval (`td3_training_dr._rollout_metrics`) and the GIF evals (`evaluate.py`, `scripts/utils.py`) no longer force a 200-step budget or the survive-to-the-end success rule; they use the task's `max_timesteps` and `info["success"]`. Re-evaluating `puck_vel_dr` @975k under the corrected eval: mean episode length 99.7 (budget 100), success 1.0, return 31.9 across the 5 DR envs.
- `spawn_paddle` / `spawn_puck` now seed the 5-frame histories with the actual spawn pose (flag 0) instead of the (−0.8, 0) placeholder flagged 1. **The 10 runs above were trained with the old first-frame placeholder**; future runs see the corrected first observation.

## Next

- Second seed on juggle / touch to confirm the recipe is neutral-to-positive there.
- Re-run the 10-task batch once with the corrected first observation (expected to be neutral: one frame per episode).
