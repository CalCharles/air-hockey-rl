# Overnight search for ~99 % sustained juggling on the new sim (training-side changes only)

- **Date**: 2026-10-31 07:21 UTC start (02:21 CDT), autonomous overnight run on GPUs 0–1
- **Status**: done (no recipe reached the 99 % target; one new best single policy at 94 % reach 10k; 5-seed recipe 79.4 ± 4.5 %; the outcome follows the source checkpoint)
- **Follows**: [`2026-10-31_02-00_new-sim-retraining-and-finetuning.md`](2026-10-31_02-00_new-sim-retraining-and-finetuning.md)
- **New sim** (unchanged, never edited): `configs/new_juggle/sysid_v2_hist4_realplant.yaml` + 2-step action latency. Training cap 2000 steps (`max_timesteps: 2000` in every generated sim yaml; the inert `random_variables` / ranges are only there because the RMA trainer requires them, `domain_random` stays off). No edge-velocity fix.
- **Run dirs**: `runs/ceiling/{night1,night1b,night2,night3,night3b,night4,night5}/`; weight-averaged policies `runs/analysis/swa/`; evals `runs/analysis/new_sim_eval/<label>/results.json` (labels `base_ft_select`, `night1_select`, `night1_rerun_select`, `swa1`, `night2_ab`, `night2_ab_swa`, `night3b`, `night3b_swa`, `night3_select`, `night4_select`, `night5_select`, `final_test_candidates`)
- **Configs**: `configs/ceiling/{night1,night1b,night2,night3,night3b,night4,night5}.yaml` (each header says what the round tests), generated jobs in `configs/ceiling/generated/<round>/`
- **Code**: `scripts/analysis/new_sim_eval.py` (new-sim eval with survival + hit-style metrics and a checkpoint-screen mode); training-only `side` shaping term in `scripts/rma/motion_shaping.py` (`shape_side_coef / _min / _ref / _target` in `scripts/rma/train_base_policy.py`, default off). Scratch tools (not committed): `swa.py` (actor weight averaging over checkpoints), `fail_diag.py` (how episodes end), `ensemble_eval.py` (action-averaging ensemble).

## Question

Can a training-side change (reward shaping, hyperparameters, algorithm tricks) get a policy to ~99 % of evaluation episodes reaching 5k and 10k steps on the new sim, with natural hits (no contact at the top edge of the workspace)?

## Protocol

All numbers are mean ± std err over episodes. Selection: every checkpoint ≥ `--min-step` plus the final model screened on 20 episodes at the 2000 cap (seeds 800000+), the best 1–2 per run re-evaluated at the 10k cap on fresh seeds (900000+), 20 or 40 episodes. Final test: 100 episodes, 10k cap, seeds 1000000+. Hit metrics per contact: edge hit = paddle within 2 cm of `x_min_lim`; dtop = median distance from that edge; near-edge = fraction of steps within 2 cm. **Every policy in this note has 0 % edge hits** (dtop 9–18 cm); near-edge time is ≤ 2 % for all new runs (the old `ft_edgefix` seeds: 5–34 %).

Bar to beat (`base_ft_select`, 20 episodes, 10k cap): `realplant_screen2/ft_edgefix_s1` 100 ± 0 % reach 5k, 90 ± 7 % reach 10k; the other nine fine-tuned seeds 45–90 % / 20–65 %.

## Rounds

| Round | What changed (from the best warm starts unless noted) | Result (best checkpoint per run, 10k cap) |
|---|---|---|
| night1 (15 runs, 2M steps) | continue fine-tuning ft_term03 / ft_edgefix with term03 shaping; + gamma 0.998 & actor lr 1e-4; fine-tune strike_shaped sources; strike_term03 sources without the pre-tanh penalty | no run beats the bar. Best: `ft_term03_nopretanh_s1` final 90 ± 7 / 80 ± 9 %; c_* runs 55–85 / 30–70 %; gamma 0.998 + lr 1e-4 worst (20–65 / 0–50 %) |
| night1b (3 runs) | as c_edgefix_t03 with exploration noise 0.05, target noise 0.1 | `noise05_s1@1.6M` 100 ± 0 / 85 ± 8 %; s0 80 / 50 %, s2 90 / 80 % |
| SWA (offline) | average actor weights over a run's checkpoints ≥ 1.0–1.2M | ≈ best selected checkpoint, without the selection: `nopretanh_s1` SWA 95 ± 5 / 90 ± 7 %; ft_edgefix seeds 45–80 / 15–55 % (no better than their best checkpoints) |
| night2 (2 runs, 1M) | "polish": actor lr 5e-5, critic 3e-4, noise 0.05 / 0.1, batch 1024 | **worse than the start**: 42–72 / 10–55 % (40 episodes), SWA 55–80 / 38–62 %; seeds 2–3 cancelled |
| night3 (6 runs, 1.2M) | term03 + one-sided per-step penalty on paddle y > 0.27 (coef 0.5, 1.5) | **worse, dose-dependent**: coef 0.5 25–55 / 0–25 %, coef 1.5 0–5 / 0 % |
| night3b (2 runs) | coef 1.0 + actor frozen for the first 100k steps | 2–55 / 0–38 % |
| night4 (4 runs, 1.2M) | term03 + penalty on the *puck* at y > 0.25, full at 0.38 (coef 0.3, 1.0), seeds 0 and 2 inits | **worse, dose-dependent** (40 episodes): coef 0.3 40–48 % reach 10k (SWA 50–68 %), coef 1.0 8–15 % (SWA 5–20 %) |
| night5 (3 runs, 1.2M) | term03 + actor frozen for the first 100k steps (no side term) | **worse than the start** (40 episodes): 30–55 % reach 10k, SWA 35–62 % |
| ensemble (offline) | mean action of nopretanh_s1 SWA + ft_edgefix_s1 + noise05_s1@1.6M | 92 ± 4 / 88 ± 5 % (40 episodes) vs members 80–85 % reach 10k on the same 40 seeds, but only 87 ± 3 / 77 ± 4 % in the 100-episode final test (below its best member) |
| night6 (2 runs, 2M) | seeds 3–4 of `ft_term03_nopretanh` (identical recipe), to give the best recipe 5 seeds | SWA ≥ 1.2M final test: s3 85 ± 4 / 76 ± 4 %, s4 88 ± 3 / 79 ± 4 % |
| night7 (3 runs, 2M) | `ft_term03_nopretanh` again from the seed 0–2 sources with new RNG seeds 10–12 (is seed 1's 94 % the recipe or luck?) | SWA ≥ 1.2M final test, reach 5k / 10k: from the s0 source 71 ± 5 / 49 ± 5 %, **s1 source 94 ± 2 / 93 ± 3 %**, s2 source 89 ± 3 / 83 ± 4 % |

The 6 OOM failures of night1's first launch (8 jobs per GPU; each job holds ~3.9 GB of replay on the GPU, so at most 5 fit on a 24 GB card) were rerun at 4 per GPU; their numbers are in the table.

## Why the remaining episodes fail (`fail_diag.py`, 4 best policies × 40 episodes × 10k cap, 38k contacts, 27 lost pucks)

| paddle y at the contact (Box2D, table centre 0) | contacts | lost within that hit | per-hit loss |
|---|---|---|---|
| < −0.25 | 7119 | 3 | 0.42 / 1000 |
| −0.25 … −0.10 | 8908 | 6 | 0.67 / 1000 |
| −0.10 … +0.10 | 6480 | 0 | 0 |
| +0.10 … +0.25 | 8075 | 1 | 0.12 / 1000 |
| **> +0.25** | **7372** | **17** | **2.3 / 1000** |

Lost pucks are almost all the hit right before the end (puck passes the paddle 2–5 steps later): weak glancing contacts (exit speed 0.3–1.0 m/s vs 1.45 m/s typical) with the puck sent back down. 63 % of them happen with the paddle near the +y wall, a zone with 19 % of the contacts; hits just as close to the −y wall are ~5× safer. Removing that zone's excess hazard alone would take reach-10k from ~85 % to ~94 %.

Penalizing the paddle for being there (night3 / 3b) is the wrong fix: the policy stops following the puck to the +y side and the failure mode flips to "puck hits the bottom while the paddle waits at y ≈ +0.1" (night3b s0: 20/20 lost, most 50–80 steps after the last hit, puck at y +0.37–0.39). night4 instead penalizes the puck's position, which the policy can only reduce by aiming.

## Final test (100 episodes, 10k cap, seeds 1000000+)

| Policy | Return | Length | Reach 5k | Reach 10k | dtop | near-edge |
|---|---|---|---|---|---|---|
| **`night1/ft_term03_nopretanh_s1`, SWA of checkpoints ≥ 1.2M** (`runs/analysis/swa/night1__ft_term03_nopretanh_s1__swa1200k`) | 7056 ± 163 | 9501 ± 218 | **95 ± 2 %** | **94 ± 2 %** | 10.6 cm | 0 % |
| `realplant_screen2/ft_edgefix_s1` (previous best) | 6378 ± 190 | 9080 ± 269 | 91 ± 3 % | 85 ± 4 % | 10.4 cm | 8 % |
| `night1b/c_edgefix_t03_noise05_s1@1.6M` | 5784 ± 225 | 8308 ± 322 | 84 ± 4 % | 72 ± 5 % | 9.9 cm | 0 % |
| ensemble of the three above | 6463 ± 215 | 8740 ± 289 | 87 ± 3 % | 77 ± 4 % | 11.8 cm | 0 % |
| `night1/ft_term03_nopretanh_s0`, SWA ≥ 1.2M | 5472 ± 232 | 7928 ± 335 | 78 ± 4 % | 66 ± 5 % | 14.8 cm | 0 % |
| `night1/ft_term03_nopretanh_s2`, SWA ≥ 1.2M | 6329 ± 201 | 8853 ± 281 | 88 ± 3 % | 82 ± 4 % | 13.4 cm | 0 % |
| `night6/ft_term03_nopretanh_s3`, SWA ≥ 1.2M | 6198 ± 226 | 8519 ± 308 | 85 ± 4 % | 76 ± 4 % | 10.3 cm | 0 % |
| `night6/ft_term03_nopretanh_s4`, SWA ≥ 1.2M | 6423 ± 213 | 8771 ± 289 | 88 ± 3 % | 79 ± 4 % | 11.4 cm | 0 % |

All 0 % edge hits. The best single policy (94 % reach 10k, natural hits, no time at the edge) is new tonight, but the recipe behind it is not reliable: its five seeds give 66 / 94 / 82 / 76 / 79 % reach 10k (**79.4 ± 4.5 % across seeds**; reach 5k 86.8 ± 2.7 %), about the same as the strike_term03 recipe on the original sim (79.6 ± 4.8 %).

**The source checkpoint, not the fine-tuning RNG, decides the outcome.** Rerunning seeds 0–2 with new RNG seeds (night7) reproduces the ranking: from the `strike_term03` seed-1 source (`shape_seeds_term03/term03_weak3_pout2_s1/checkpoint_2700000`) 94 → 93 % reach 10k, seed-2 source 82 → 83 %, seed-0 source 66 → 49 %. So a second policy at 93 ± 3 % exists (`runs/analysis/swa/night7__ft_term03_nopretanh_s11__swa1200k`), and the way to a better policy is a better old-sim source, not more fine-tuning seeds.

## Conclusion

- **No recipe reaches the 99 % target.** The best single policy is new tonight: `ft_term03_nopretanh_s1` with its actor averaged over checkpoints ≥ 1.2M, 95 ± 2 % reach 5k and 94 ± 2 % reach 10k over 100 fresh episodes, with natural hits (0 % edge contacts, 10.6 cm median below the edge, 0 % of the time near the edge). The previous best `ft_edgefix_s1` scores 91 ± 3 / 85 ± 4 % on the same seeds and spends 8 % of the time near the edge. Across the recipe's five seeds, reach-10k is 79.4 ± 4.5 %, but the seed-1 result replicates (93 ± 3 % with a new RNG seed): what matters is which old-sim checkpoint you start from.
- **Further fine-tuning of a good new-sim policy hurts, whatever the knob.** Continued fine-tuning (night1), lower learning rates and noise with a bigger batch (night2), and freezing the actor while the critic warms up (night5) all end below their starting checkpoints, and survival swings by 20–40 points between checkpoints of one run. Every warm start begins with an empty replay buffer (`training_state.pth` does not save it).
- **Weight averaging (SWA) is a cheap, consistent help:** about equal to the best selected checkpoint, without the selection noise. Action ensembles of different seeds do not help (87 / 77 % in the final test).
- **The remaining failures are specific:** weak glancing contacts with the paddle near the +y wall carry ~5× the per-hit loss rate of other contacts. Neither shaping term aimed at them works: penalizing the paddle there stops it chasing the puck, and penalizing the puck there degrades the policy just as much, worse as the coefficient grows.

## Next

- Source selection: fine-tune from several checkpoints of the good source run (`term03_weak3_pout2_s1`, e.g. 2.3–3.0M) and of the other strike runs, pick the source by new-sim survival after fine-tuning.
- Keep the replay buffer across warm starts (`save_replay_buffer` + a resume path that loads it but resets the step budget), to test whether the fresh-buffer shock is what degrades fine-tuning.
- Attack the +y-wall failures without restricting the paddle: e.g. oversample transitions near those contacts in replay, or reward exit speed only on contacts with |y| > 0.25.
