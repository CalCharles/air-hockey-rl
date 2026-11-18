# New sim: 5-seed check of the best fine-tuning recipe from the best source (2026-10-31)

Follow-up to [2026-10-31_07-21](2026-10-31_07-21_new-sim-overnight-sustained-juggling-search.md). Question: is the 94 % reach-10k
policy (night1 `ft_term03_nopretanh_s1`, actor SWA ≥ 1.2M) a reliable recipe from its source, or a lucky seed?

**Setup.** Same recipe as night1 s1 / night7 s11, 3 new RNG seeds (21 / 31 / 41), `configs/ceiling/night8.yaml`
(generated args differ from night7 s11 only in seed, run name and log dir; sim config identical). Source for all 5:
`runs/ceiling/shape_seeds_term03/term03_weak3_pout2_s1/checkpoint_2700000` (old-sim strike_term03, 3M from scratch).
Fine-tune 2M steps on the new sim (`sysid_v2_hist4_realplant.yaml` + 2-step action latency, cap 2000), term03 shaping,
`actor_pre_tanh_penalty: 0`; then actor SWA over checkpoints ≥ 1.2M (9 checkpoints). Final test
`scripts.analysis.new_sim_eval`, 100 episodes, 10k cap, seeds 1000000+ (never used for selection).

| Seed | Policy | Reach 5k | Reach 10k | Edge hits | Near-edge |
|---|---|---|---|---|---|
| 1 | `runs/analysis/swa/night1__ft_term03_nopretanh_s1__swa1200k` | 95 ± 2 % | 94 ± 2 % | 0 % | 0.1 % |
| 11 | `runs/analysis/swa/night7__ft_term03_nopretanh_s11__swa1200k` | 94 ± 2 % | 93 ± 3 % | 0 % | 0.2 % |
| 21 | `runs/analysis/swa/night8__ft_term03_nopretanh_s21__swa1200k` | 80 ± 4 % | 70 ± 5 % | 0 % | 0.1 % |
| 31 | `runs/analysis/swa/night8__ft_term03_nopretanh_s31__swa1200k` | 94 ± 2 % | 93 ± 3 % | 0 % | 0.1 % |
| 41 | `runs/analysis/swa/night8__ft_term03_nopretanh_s41__swa1200k` | 90 ± 3 % | 87 ± 3 % | 0 % | 0.2 % |
| **5-seed mean** | | **90.6 ± 2.8 %** | **87.4 ± 4.5 %** | | |

Per-seed entries are ± std err over 100 episodes; the mean row is ± std err over the 5 seeds.

**Conclusion.** The recipe from this source is good but not fully reliable: 3 of 5 seeds reach 10k in 93–94 % of episodes,
one 87 %, one 70 %. The 5-seed mean (87.4 ± 4.5 %) is above the 5-source mean (79.4 ± 4.5 %), but far from the 99 % goal.
All seeds hit naturally (0 % top-edge hits). Results files: `runs/analysis/new_sim_eval/{final_test_candidates,
final_test_nopretanh_swa_rep,final_test_nopretanh_swa_s1src}/results.json`.
