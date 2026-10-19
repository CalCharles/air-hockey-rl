# 10-episode online fine-tuning of the hist4 juggle policies on the puck-lag sim

- **Date**: 2026-10-18 08:13 UTC
- **Status**: done
- **Root**: `runs/td3/sysid_v2_real_workspace/puck_lagcam_online_finetune/`
  - `envs.json` is a hand-written manifest with a single target env
  - per-seed runs are in `online/env1/hist4_seed<S>/` (`result.json`, `online_progress.jsonl`, `online_tb/`)
  - the mean ± SE TensorBoard run is `online_agg/`
- **Target**: `configs/new_juggle/sysid_v2_hist4_puck_lagcam.yaml`. This is the 120 ms puck-only lag on a 30 fps frame grid (120–153 ms) from [`2026-10-18_08-03_puck-camera-lag-sim2sim.md`](2026-10-18_08-03_puck-camera-lag-sim2sim.md).
- **Source policies**: `runs/td3/sysid_v2_real_workspace/hist4_seed{0..4}/juggle_sysid_v2_hist4/checkpoint_1975000/training_state.pth`

## Setup

The workers of `scripts/td3/sim2sim_online_finetune.py --worker` run the real-robot recipe (`configs/td3/td3_online_real_finetune/juggle_hist4.yaml`):
- actor / critics / targets / Adam states loaded in place
- actor LR 1e-5, critic LR 1e-3
- one online buffer with no sim data
- K = 5·T critic updates after each episode
- one actor update per 20 critic updates
- exploration noise 0.1

Settings for this run:
- no warm start
- 10 online episodes per seed, i.e. 10 training rounds
- after every round, a deterministic eval of 10 episodes on fixed seeds (`EVAL_SEED_BASE`), the same for every round and policy

Command, per seed S on GPU S % 4:

```
.venv/bin/python -u -m scripts.td3.sim2sim_online_finetune --worker --root runs/td3/sysid_v2_real_workspace/puck_lagcam_online_finetune \
  --env 1 --seed S --learner-device cuda:G --num-online-episodes 10 --eval-episodes 10 --online-eval-every 1 \
  --online-eval-episodes 10 --min-episode-steps 1 --warm-start-episodes 0 \
  --source-checkpoint 'runs/td3/sysid_v2_real_workspace/hist4_seed{seed}/juggle_sysid_v2_hist4/checkpoint_1975000/training_state.pth'
.venv/bin/python scripts/analysis/aggregate_online_tb.py runs/td3/sysid_v2_real_workspace/puck_lagcam_online_finetune/online
```

## Results

Deterministic 10-episode eval; the mean row is mean ± SE across the 5 seeds.

| Policy | Zero-shot (round 0) | Online episodes 1–10 (with exploration noise) | After 10 rounds |
|---|---|---|---|
| hist4_seed0 | 45.9 ± 6.9 | 30.7 | 35.9 ± 6.9 |
| hist4_seed1 | 58.0 ± 11.4 | 35.9 | 50.2 ± 5.0 |
| hist4_seed2 | 44.4 ± 4.4 | 51.7 | 28.6 ± 6.9 |
| hist4_seed3 | 59.9 ± 7.8 | 54.7 | 35.8 ± 9.2 |
| hist4_seed4 | 66.1 ± 12.8 | 45.3 | 43.0 ± 11.3 |
| **Mean ± SE** | **54.9 ± 4.2** | 43.7 ± 4.6 | **38.7 ± 3.7** |

- Paired change after 10 rounds: −16.2 ± 3.3. All 5 seeds are worse.
- Deterministic eval by round, 0 to 10 (mean): 54.9, 46.9, 45.8, 37.9, 42.9, 41.9, 40.8, 46.3, 34.3, 36.1, 38.7.
- For scale: the lag-free source sim scores ~109.
- The zero-shot value here (54.9) is on different eval seeds than the 49.6 of the lag eval.

## Conclusion

- **Ten episodes of the online recipe do not recover the lag loss. They make it worse:** 54.9 → 38.7, every seed. The decline starts in the first rounds.
- **The update rate is 5 updates per transition from a buffer of a few hundred transitions, with no sim data.** That is enough to move a policy trained without lag away from its sim behaviour before the critic has learned anything useful about the lagged dynamics.
- **Confidence.** This is a single 10-episode budget, matching what is feasible on the robot. A longer budget, a warm start, or a lower actor LR / UTD might behave differently.
