# Real-robot juggle eval of the five new-sim strike-shaping policies (SWA from 1.2M), 10,000-step episode cap

- **Date**: 2026-11-01 21:53 UTC start (last episode 2026-11-02 00:10 UTC)
- **Status**: done (3–5 episodes per seed; more episodes would tighten the error bars)
- **Run dirs**: `real_runs/eval_new_sim_T10000/{s1,s11,s21,s31,s41}/data_*/`
- **Configs**: `configs/real_configs/rollout_td3_config_hist4_T10000.yaml` (the hist4 juggle real config, with `max_timesteps` 10000 instead of 1000)
- **Policies**: `data/new_sim_best_policies/<night>__ft_term03_nopretanh_s<seed>__swa1200k/`

## Question

How well do the five new sim-trained juggling policies transfer to the real robot
when episodes are allowed to run long (up to 10,000 steps instead of 1000)?

## Setup

- **Policies.** Fine-tuned `ft_term03_nopretanh` runs (training sim config
  `configs/ceiling/generated/<night>/ft_term03_nopretanh_sim.yaml`). They were
  initialized from `runs/ceiling/shape_seeds_term03/term03_weak3_pout2_s1/checkpoint_2700000`,
  which is the edge-termination strike-shaping recipe from
  [`2026-10-30_12-47`](2026-10-30_12-47_strike-shaping-edge-termination-5seeds.md).
  Strike-shaping args: `shape_edge_coef` 0.5 and `shape_pout_coef` 2.0. The edge
  velocity fix is off. Each `model.pth` is an SWA (weight average) of checkpoints
  1.2M–1.9M plus the final model; the full list is in `sources.txt` in each policy
  folder.
- **Architecture.** History agent (`rma_mode: history`, 50-step window), obs 30,
  act 2, hist_len 4.
- **Training horizon.** The sim training config uses `max_timesteps: 2000`, so the
  real episodes longer than 2000 steps go past the training horizon.
- **Command:**
  ```
  python -m scripts.td3.extras.async_td3_real_eval --agent history \
    --config configs/real_configs/rollout_td3_config_hist4_T10000.yaml \
    --model-path data/new_sim_best_policies/<policy> \
    --collector-device cpu --eval-episodes <n> --episode-gif-max-frames 2000 --append \
    --first-reset-side right \
    --data-root-dir real_runs/eval_new_sim_T10000/s<seed>
  ```
  Runs were resumed with `--append`, which is why episode timestamps have gaps.
  Episodes between resets use the `ResetPolicyFSM` puck sweep.
- **Task.** `puck_juggle_upper_half_reward`, which terminates on `puck_hit_bottom`.
- **Data source.** Numbers are from `eval_per_episode.jsonl`. s11's
  `eval_summary.json` is stale: it lists 1 episode, but the jsonl has 3 after the
  resume.

All values are **mean ± SE** (ddof=1, /√n), per episode.

## Results

### Per policy

| Seed | Policy | n | Return | Juggles | Contacts | Length (steps) | Best ep (steps / juggles) |
|---|---|---|---|---|---|---|---|
| s1 | night1 | 5 | 315.0 ± 104.3 | 8.0 ± 2.9 | 8.8 ± 3.0 | 543 ± 182 | 1247 / 19 |
| s11 | night7 | 3 | 613.0 ± 337.8 | 12.0 ± 6.7 | 12.7 ± 6.8 | 859 ± 459 | 1757 / 25 |
| s21 | night8 | 4 | 481.8 ± 208.3 | 11.0 ± 5.3 | 15.0 ± 7.9 | 838 ± 388 | 1711 / 21 |
| s31 | night8 | 4 | 486.8 ± 178.3 | 10.5 ± 4.6 | 12.2 ± 4.3 | 788 ± 278 | 1589 / 24 |
| **s41** | night8 | 5 | **1161.0 ± 274.1** | **25.6 ± 5.9** | **26.8 ± 6.3** | **1722 ± 398** | 2666 / 38 |

### Aggregate

| Grouping | n | Return | Juggles | Length (steps) |
|---|---|---|---|---|
| All episodes pooled | 21 | 623.5 ± 113.2 | 13.8 ± 2.5 | 972 ± 168 |
| Seed means (n = seeds) | 5 | 611.5 ± 145.3 | 13.4 ± 3.1 | 950 ± 201 |
| Seed means without s41 | 4 | 474.1 ± 61.1 | 10.4 ± 0.9 | 757 ± 73 |
| night8 seeds pooled (s21/s31/s41) | 13 | 744.5 ± 156.4 | 16.5 ± 3.6 | 1163 ± 233 |

Episode lengths: 12 of 21 episodes lasted ≥ 500 steps, 9 lasted ≥ 1000, and 2
lasted ≥ 2000 (both s41).

### Per episode

| Seed | Ep | Steps | Return | Juggles | Contacts | End |
|---|---|---|---|---|---|---|
| s1 | 0 | 248 | 153 | 3 | 3 | puck_hit_bottom |
| s1 | 1 | 340 | 184 | 5 | 6 | puck_hit_bottom |
| s1 | 2 | 530 | 308 | 8 | 9 | puck_hit_bottom |
| s1 | 3 | 1247 | 719 | 19 | 20 | puck_hit_bottom |
| s1 | 4 | 351 | 211 | 5 | 6 | puck_hit_bottom |
| s11 | 0 | 1757 | 1277 | 25 | 26 | puck_hit_bottom |
| s11 | 1 | 249 | 173 | 3 | 4 | puck_hit_bottom |
| s11 | 2 | 570 | 389 | 8 | 8 | puck_hit_bottom |
| s21 | 0 | 1258 | 785 | 19 | 20 | puck_hit_bottom |
| s21 | 1 | 318 | 209 | 4 | 5 | puck_hit_bottom |
| s21 | 2 | 1711 | 887 | 21 | 35 | puck_hit_bottom |
| s21 | 3 | 64 | 46 | 0 | 0 | puck_hit_bottom |
| s31 | 0 | 1589 | 988 | 24 | 25 | puck_hit_bottom |
| s31 | 1 | 737 | 487 | 9 | 10 | puck_hit_bottom |
| s31 | 2 | 457 | 283 | 5 | 7 | puck_hit_bottom |
| s31 | 3 | 369 | 189 | 4 | 7 | puck_hit_bottom |
| s41 | 0 | 479 | 293 | 7 | 7 | puck_hit_bottom |
| s41 | 1 | 2437 | 1705 | 38 | 38 | puck_hit_bottom |
| s41 | 2 | 1261 | 819 | 19 | 20 | puck_hit_bottom |
| s41 | 3 | 2666 | 1733 | 38 | 42 | **e-stop** (collector_readiness_fail_5steps) |
| s41 | 5 | 1766 | 1255 | 26 | 27 | puck_hit_bottom |

### Safety and data-quality notes

- **s41 ep 3 ended in an e-stop** at step 2666, while the puck was still being
  juggled. It is kept in the stats, so s41's numbers are a lower bound for that
  episode. That is also the episode with the largest drift between the actual and
  commanded paddle position (see below).
- **s41 ep 4 was discarded** (`short_episode`, 16 steps, protective stop at the
  start).
- **No other protective stops or controller disconnects.** Every other episode
  ended by `puck_hit_bottom`.
- **The paddle touched the robot-end wall in long episodes (> ~2000 steps).** This
  is not a workspace-clip bug: the commanded target never passes x = −0.45. The
  actual TCP drifts up to ~4.5 cm away from the command over a long episode.
  Details: [`tcp-drift-long-episodes.md`](../../docs/environments/real-world/tcp-drift-long-episodes.md).
  The drift grows with episode length, so it mostly affects s41 and is a confound
  for any episode over ~1500 steps.

## Conclusion

- **s41 is clearly the best policy on the real robot:** 1161 ± 274 return,
  25.6 ± 5.9 juggles, and the only episodes past 2000 steps. The other four seeds
  are close to each other (seed means 474 ± 61 return, about 10 juggles per
  episode).
- **Small samples.** With 3–5 episodes per seed and heavy-tailed episode lengths,
  the per-seed SEs are large. The ranking of s1 / s11 / s21 / s31 is not resolved;
  only s41 standing out is.
- **No policy got anywhere near the 10,000-step cap.** The longest episode was
  2666 steps.
- **Long episodes are confounded by the TCP drift.** Past ~2000 steps the paddle
  operates up to ~4 cm off from where the policy thinks it is commanding it. Fix
  the drift before reading too much into long-episode performance.

## Next

- Fix or bound the actual-TCP drift (forceMode x/y deviation limits), then rerun
  s41 for long episodes.
- More episodes per seed (≥ 10), especially s41 and one of the mid seeds, to
  tighten the SEs.
- Compare with the sim survival numbers for these exact SWA policies (not
  documented here) and with the previous hist4 real baseline.
