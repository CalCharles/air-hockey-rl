# State distribution of the real hist4 juggle rollouts vs the offset-sim replay buffers, and whether high-return real episodes are more sim-like

- **Date**: 2026-10-18 08:01 UTC
- **Status**: done (analysis only)
- **Run dirs**: real `results/eval_4_tasks_of_policies_hist4_2026-09-19/juggle_sysid/data_20261014-201743` (10 eps, 1,020 steps); sim buffers `runs/td3/sanity_hist4_retrain/sanity_hist4_seed{0-4}/training_state.pth` (`success_replay_buffer`, 1M each = last 1M of 3M steps)
- **Configs**: `configs/new_juggle/sysid_v2_hist4.yaml` (with the real x/y offsets + paddle workspace, commit af73fbe)
- **Outputs**: `results/eval_4_tasks_of_policies_hist4_2026-09-19/state_dist_sim_vs_real/` (`compare_states.py`, `fig1`–`fig6`, `summary.json`, `report.html`)

## Question

How do the states seen by the policy on the real robot compare with the states in the sim replay buffer? And are the high-return real episodes closer to the sim distribution than the low-return ones?

## Setup

- Real 30-dim history obs rebuilt from the logged `pose[:2]` / `puck[:3]` (already in the obs frame), last 5 rows, padded with row 0.
- Per-feature Wasserstein distances against a null from 100 random 1,020-step sim blocks (W/null95).
- Gradient-boosted real-vs-sim classifier, grouped 5-fold cross-validation, out-of-fold AUC.
- Nearest-neighbour distance in an 8-d z-scored space (paddle/puck position + velocity), sim seeds 0–3 as reference, seed 4 as the in-distribution baseline.
- Per-episode Spearman correlations against return with permutation p-values, over the whole episode and the first 40 steps.
- **Confound:** the eval policy (`data/policies_hist4_2026-09-19/juggle/sysid`, pre-offset) is not on this machine. The buffer comes from a different hist4 policy with exploration noise 0.1.

## Results

- Coverage: 2.8 % of real states beyond the sim held-out p99 NN distance (1 % expected); 6.7 % beyond p95. Out-of-distribution states are driven mainly by puck vx (39 %) and puck x (30 %).
- Classifier AUC: full obs + last action 0.94; last action 0.88 (mostly policy differences + the 8-step hold); paddle history 0.82; puck history 0.79; current xy 0.76; occlusion flags 0.54.
- Largest marginal gaps (W/null95): puck occluded 3.7 (13.5 % vs 8.8 % of frames; run mean 2.5 vs 1.2, max 17 vs 7), puck x 3.3, puck − paddle x 3.1 (the puck sits behind the paddle in > 5 % of real steps, clustered at x 0.8–0.9, y −0.2 to −0.35), puck vx 2.0 (p5 −0.80 vs −1.74 m/s), paddle vy 2.0, paddle x 1.5. Real paddle start pose y = ±0.376 lies past the sim workspace edge (0.352).
- Paddle response at |a| > 0.9: real 0.22 / 0.27 m/s vs sim 0.38 / 0.49 m/s (x / y).
- Per episode: return ~ length (ρ 0.77). Whole-episode in-distribution scores vs return: NN ρ +0.02 (p 0.97), fraction beyond p99 −0.12 (0.74), classifier −0.39 (0.26). First 40 steps: paddle speed ρ +0.75 (p 0.016, 1 of 14 tests), puck speed −0.64 (0.05), classifier-xy +0.60 (0.08, the *opposite* of the hypothesis).
- Top-3 vs bottom-4: equal distance to sim on paddle features; the bottom group is further on puck x / relative x / speed, driven by the frames where the puck passes the paddle as the episode ends, plus detector jumps in ep0.
- Start side: paddle-at-y=+0.376 starts R 61.6 vs 34.8 (Mann–Whitney p 0.15).

## Conclusion

The real states lie mostly inside the sim buffer's coverage. The gap is in motion texture (slow paddle, weaker hits, long occlusions, quantised and unevenly spaced puck frames, puck behind the paddle), consistent with `2026-10-15_19-43`. High-return episodes are not detectably more in distribution (n = 10, so only strong effects are ruled out). The best lead is early paddle speed and start side.

## Next

- Evaluate a `sanity_hist4` seed on the robot (its buffer is available) to remove the policy confound, and roll the same policy in sim from the real start states.
- Collect 30–40 real episodes to test moderate correlations.
