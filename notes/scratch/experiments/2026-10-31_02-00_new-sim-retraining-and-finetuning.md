# Training sustained juggling on the new sim (real-plant proxy + 2-step action latency)

- **Date**: 2026-10-31 02:00 CDT (work from 2026-10-30 18:23 CDT on)
- **Status**: done for screening, fine-tuning seeds 0–4 and the hist4 sim-vs-real check. The 100-episode final test of the selected policies is not done yet.
- **Follows**: [`2026-10-30_17-40_ceiling-sim2real-paddle-plant-gap.md`](2026-10-30_17-40_ceiling-sim2real-paddle-plant-gap.md) (why the ceiling policies fail on the robot; how the new sim was built)
- **"New sim"** = `configs/new_juggle/sysid_v2_hist4_realplant.yaml` + 2 control steps of action latency (`action_latency_steps: 2` in `scripts/rma/train_base_policy.py`, `--action-latency 2` in `scripts/analysis/proxy_sim_eval.py`). `airhockey/` and `sysid_v2_hist4.yaml` (the "original sim") are unchanged.
- **Run dirs**: `runs/ceiling/realplant_screen1/`, `runs/ceiling/realplant_screen2/`, `runs/ceiling/realplant_seeds34/`; evals `runs/analysis/proxy_sim_eval/{rp_screen1_early_final,rp1_term03_s1,rp2_screen,rp_cap10k,rp_seeds34_sel,h4_orig,h4_new_lat0,h4_new_lat2}/results.json`
- **Configs**: `configs/ceiling/realplant_screen2.yaml` (seeds 0–2 of the three recipes that worked) and `configs/ceiling/realplant_seeds34.yaml` (seeds 3–4 of the two fine-tune recipes); generated jobs in `configs/ceiling/generated/{realplant_screen2,realplant_seeds34}/`. Screen 1 (`configs/ceiling/realplant_screen1.yaml`) is not committed; its spec is recorded below.
- **Code**: `init_from: {seed: checkpoint_dir}` in `scripts/ceiling/sweep.py` variants warm-starts a seed from that checkpoint's `training_state.pth` (actor, critics and targets; with `full_checkpoint_load: weights_only` the optimiser, replay and step counter start fresh).
- GPUs 0–1 only. All numbers are mean ± std err.

## New sim vs original sim

| # | Setting | Original sim | New sim | Why |
|---|---|---|---|---|
| 1 | `enable_observation_delay` | `true` | `false` | With it on, each 50 ms control step is two 25 ms Box2D steps. Box2D damps the paddle by `v *= 1 − h·c` per step, so the paddle runs ~2× too fast. The PID sysid was fit with it off. |
| 2 | `puck_obs_lag_seconds` | 0 | 0.025 | Keeps the newest puck observation 25 ms old (measured 24.9 ± 0.1 ms in both sims), observation-only, no physics change. |
| 3 | `pid_kp` | 7531.5 | 9038 (×1.2) | Open-loop replay of logged real actions then matches the real paddle (speed ratio 1.00–1.08, x error 2–3 cm, r² 0.94–0.96). |
| 4 | action latency (trainer / eval, not a sim key) | 0 | 2 control steps | The real paddle responds 3–4 steps after the command; the sim already produces 1–2. |

Side effects of (1): the paddle observation is no longer 25 ms stale (real: fresh telemetry); history frames are 50 ms apart instead of 25 ms (real: 50 ms); the `hist_len` 4 filter spans 4 control steps instead of 2.

The fix in commit 4a64790 (`history_sample_once_per_step`) only changes how often the observation history is sampled; it is opt-in, off in `sysid_v2_hist4.yaml`, and does not change the 2 × 25 ms physics split.

### Original sim vs new sim vs real robot

Paddle speed (hist4 seeds 0–3, per-step speed; real = the 2026-10-20 eval logs, sim = 10 episodes × 4 seeds each):

| | Original sim | New sim | Real robot |
|---|---|---|---|
| real ÷ sim paddle speed, same action sequence (open-loop replay) | 0.46–0.52 | 1.00–1.08 | 1 |
| median / p90 / p99 / max (m/s) | 1.00 / 1.88 / 2.26 / 2.40 | 0.38 / 0.69 / 0.87 / 1.04 | 0.39 / 0.79 / 0.98 / 1.21 |
| extra real lag vs sim (replay) | 3–4 steps (150–200 ms) | 0 | — |
| replay x / y error | ~14 / 24–27 cm | 1.5–3 / 5–8 cm (one episode 29 cm y) | — |
| obs history frame spacing | 25 ms | 50 ms | 50 ms |

The new sim's fastest moves are 10–15 % slower than the robot's; the y drift in replay is not explained.

hist4 return and length. Protocol of the 2026-10-18 eval (`scripts/td3/extras/eval_occlusion_map_sim2sim.run_episodes`, one env reused, reset seeds 1000–1049, 50 episodes per seed). The original sim keeps its config cap of 250 steps; the new sim runs to 10,000 (no episode exceeded 824). Real = 10 kept episodes per seed, cap 1000, episodes < 50 steps and stopped episodes discarded by the real eval.

| hist4 return | seed 0 | seed 1 | seed 2 | seed 3 | mean ± SE, seeds 0–3 |
|---|---:|---:|---:|---:|---:|
| original sim, cap 250 | 114 ± 7 | 120 ± 6 | 111 ± 6 | 87 ± 8 | 108.0 ± 7.2 |
| **new sim, cap 10k** | **127 ± 12** | **84 ± 10** | **94 ± 13** | **72 ± 8** | **94.0 ± 11.8** |
| **real robot** | **135 ± 34** | **73 ± 14** | **98 ± 22** | **82 ± 10** | **97.2 ± 13.7** |

| hist4 episode length | seed 0 | seed 1 | seed 2 | seed 3 | mean ± SE, seeds 0–3 |
|---|---:|---:|---:|---:|---:|
| original sim, cap 250 | 183 ± 9 | 197 ± 8 | 182 ± 10 | 145 ± 11 | 176.7 ± 11.1 |
| **new sim, cap 10k** | **225 ± 18** | **151 ± 15** | **165 ± 20** | **147 ± 14** | **171.9 ± 18.2** |
| **real robot** | **223 ± 51** | **134 ± 27** | **185 ± 38** | **135 ± 18** | **169.4 ± 21.4** |

- The new sim tracks the robot seed by seed (seed 0 best, seed 1 near the bottom); the original sim ranks seed 1 best.
- With the real eval's ≥ 50-step filter applied to the sim, the new sim gives 109 return (seeds 0–3), ~12 % above the robot. Stopped episodes cannot be filtered in sim, so the fair number lies between 94 and 109.
- Without latency the new sim gives 131 (cap 10k): the 2-step latency is needed.
- The reproduction of the earlier 108.7 ± 8.6 (5 seeds × reset seeds 1000–1009, cap 250) gives 108.9 ± 8.6. My first `proxy_sim_eval` comparison (158) used a 2000-step cap by mistake.
- Temporary scripts (not committed; they import the repo code unchanged): `hist4_cap250_eval.py` (the protocol above, `CAP` env var overrides the cap, optional latency wrapper), `speed_dist.py`, `full_action_speed.py`, `puck_staleness.py`, in the session scratchpad `/tmp/claude-1023/-data-rohanp-air-hockey-rl/e74ea17c-778d-4316-a824-903129f8452f/scratchpad/`.

## Screen 1: from scratch on the new sim (mostly failed; spec not committed)

`configs/ceiling/realplant_screen1.yaml` → `runs/ceiling/realplant_screen1/`, 3M steps, seeds 0–2, training cap 2000. Common args: the `final_privcritic_pretanh` recipe (TD3, 50-step history actor, true-state critic, `actor_pre_tanh_penalty: 0.001`, `gamma: 0.995`) + `action_latency_steps: 2`.

| variant | extra args | training return over 3M steps | eval (new sim, 20 eps, cap 2000) |
|---|---|---|---|
| `dribble` | none | flat at 17 (never hits the puck) | — |
| `dribble_dr` | `domain_random: true`, ±25 % paddle_density, puck_damping, gravity, pid_kp | flat at 17 | 21 ± 3, paddle nearly still |
| `term03` | `shape_edge_coef 0.5`, `shape_edge_term_margin 0.03`, `shape_weak_coef 3.0`, `shape_pout_coef 2.0` | seed 1 rises to 203; seeds 0 / 2 start rising only at the end (35 / 27) | seed 1: 1066 ± 129, 60 % reach 2000 (10k cap: 1258 ± 282, 10 % reach 5k, 0 % 10k); seed 2: 66 ± 10 |

On the slower paddle with latency, random exploration with the dribble recipe never finds a hit. The strike shaping gives enough signal for one seed in three.

## Screen 2 + seeds 3–4: fine-tuning and a from-scratch variant (worked)

`configs/ceiling/realplant_screen2.yaml` (seeds 0–2) and `configs/ceiling/realplant_seeds34.yaml` (seeds 3–4), GPUs 0–1.

| variant | start | args beyond the screen-1 base | steps |
|---|---|---|---|
| `ft_term03` | `latest_models/ceiling/strike_term03/seed<N>` source checkpoints (`SOURCE` files) | `term03` shaping; `full_checkpoint_load: weights_only`, `exploration_pre_learning_action_source: policy`, `exploration_primitive_chance_start: 0.05` | 1.5M |
| `ft_edgefix` | `latest_models/ceiling/edgefix_privcritic_pretanh/seed<N>` source checkpoints | as above without shaping, plus `edge_velocity_fix: true` (as the source was trained) | 1.5M |
| `term03_nopretanh` | from scratch (seeds 0–2 only) | `term03` shaping, `actor_pre_tanh_penalty: 0.0` | 3M |

Training return (`avg_episodic_return`, 250k-step bins): every `ft_*` seed starts at 35–65 (the old policies collapse in the new sim) and climbs to 220–280 by 1.5M. `term03_nopretanh`: seed 0 rises from 1.25M to 193, seed 2 from 0.5M to 316, seed 1 never learns.

Checkpoint screen (new sim, 20 episodes, cap 2000, seed base 800000): every fine-tuned seed has a checkpoint at 85–100 % reach 2000.

| run | 1.2M | 1.4M | final (1.5M / 3M) |
|---|---|---|---|
| ft_term03_s0 | 1064 ± 100, 55 % | 1435 ± 18, 90 % | 1196 ± 99, 75 % |
| ft_term03_s1 | 1307 ± 84, 90 % | 1446 ± 12, 90 % | 1351 ± 71, 85 % |
| ft_term03_s2 | 1044 ± 103, 50 % | 1308 ± 41, 85 % | 1175 ± 94, 70 % |
| ft_edgefix_s0 | 1082 ± 121, 65 % | 1324 ± 63, 90 % | 1208 ± 71, 70 % |
| ft_edgefix_s1 | 1167 ± 93, 75 % | 1123 ± 99, 55 % | 1416 ± 7, 100 % |
| ft_edgefix_s2 | 1138 ± 111, 75 % | 1250 ± 58, 90 % | 1030 ± 101, 65 % |
| term03_nopretanh_s0 | 2.6M 692 ± 102, 15 % | 2.8M 748 ± 115, 25 % | 258 ± 52, 0 % |
| term03_nopretanh_s2 | 2.6M 1370 ± 16, 95 % | 2.8M 1051 ± 111, 65 % | 1305 ± 17, 95 % |

Seeds 3–4 of `ft_term03` / `ft_edgefix`: see "Seeds 3–4" below.

Long-horizon eval of the best checkpoint per seed (new sim, 20 episodes, cap 10,000, seed base 900000; checkpoints picked on the 2000-cap screen above, so slightly optimistic):

| policy | return | length | reach 5k | reach 10k | paddle speed |
|---|---:|---:|---:|---:|---:|
| ft_edgefix_s0 (1.4M) | 5916 ± 422 | 8495 ± 605 | 90 ± 7 % | 65 ± 11 % | 0.28 |
| ft_edgefix_s1 (final) | **6865 ± 130** | **9750 ± 180** | **100 %** | **90 ± 7 %** | 0.25 |
| ft_edgefix_s2 (1.4M) | 4814 ± 534 | 7247 ± 800 | 75 ± 10 % | 55 ± 11 % | 0.22 |
| ft_term03_s0 (1.4M) | 3064 ± 617 | 4229 ± 852 | 45 ± 11 % | 20 ± 9 % | 0.26 |
| ft_term03_s1 (1.4M) | 5098 ± 631 | 7006 ± 864 | 70 ± 11 % | 55 ± 11 % | 0.28 |
| ft_term03_s2 (1.4M) | 4492 ± 625 | 6519 ± 903 | 65 ± 11 % | 40 ± 11 % | 0.26 |
| term03_nopretanh_s2 (final) | 5173 ± 523 | 7778 ± 788 | 75 ± 10 % | 55 ± 11 % | 0.37 |
| term03_s1, screen 1 (final) | 1258 ± 282 | 1752 ± 382 | 10 ± 7 % | 0 % | 0.27 |
| hist4 seeds 0–3 | 89–106 | 169–193 | 0 % | 0 % | 0.39–0.42 |
| old ceiling policies, seed 0 (4 recipes, not fine-tuned) | 24–49 | 60–104 | 0 % | 0 % | 0.26–0.28 |

### Hit style in the new sim (seed 0, `scripts/ceiling/hit_motion` metrics, 5 episodes × 1000 steps)

| policy | edge hits | median hit distance from top edge | paddle speed into puck | swing | puck exit | time near edge |
|---|---:|---:|---:|---:|---:|---:|
| hist4 seed 0 | 0 % | 19.9 cm | 0.33 m/s | 12.7 cm | 1.74 m/s | 8 % |
| ft_edgefix_s0 | 0 % | 11.3 cm | 0.18 m/s | 18.0 cm | 1.48 m/s | 19 % |
| ft_term03_s0 | 0 % | 11.3 cm | 0.21 m/s | 22.3 cm | 1.58 m/s | 0 % |
| term03_nopretanh_s0 | 0 % | 14.7 cm | 0.28 m/s | 19.9 cm | 1.78 m/s | 0 % |

- No policy dribbles at the top edge, including `ft_edgefix`, whose source hit at the edge 51 % of the time in the original sim.
- All contact the puck about every 7 steps, like hist4, with softer contacts (0.18–0.28 vs 0.33 m/s).
- `term03_nopretanh` (no anti-saturation penalty) is the closest to hist4's hit style.

## Seeds 3–4

`runs/ceiling/realplant_seeds34/` (finished 00:47 CDT). Same checkpoint screen (new sim, 20 episodes, cap 2000, seed base 800000; `runs/analysis/proxy_sim_eval/rp_seeds34_sel/`):

| run | 1.2M | 1.4M | final (1.5M) |
|---|---|---|---|
| ft_term03_s3 | 922 ± 117, 30 % | 1277 ± 58, 75 % | 976 ± 123, 50 % |
| ft_term03_s4 | 1188 ± 80, 75 % | 1291 ± 50, 80 % | 1348 ± 57, 95 % |
| ft_edgefix_s3 | 926 ± 113, 40 % | 1271 ± 34, 90 % | 1201 ± 71, 85 % |
| ft_edgefix_s4 | 1351 ± 53, 85 % | 1203 ± 61, 85 % | 1302 ± 63, 90 % |

Both recipes reproduce on seeds 3–4: every seed has a checkpoint at 80–95 % reach 2000. Not yet run at the 10k cap.

## Conclusions

- **Fine-tuning the old ceiling policies is what works on the new sim.** All 10 fine-tuned seeds recover within 1.5M steps; from scratch, the dribble recipe never learns, and the strike-shaped recipe learns in 1–2 seeds of 3.
- **`ft_edgefix` is the strongest recipe so far** (seeds 0–2: 88 % reach 5k, 70 % reach 10k on single checkpoints), then `term03_nopretanh` (best seed 55 % at 10k), then `ft_term03`.
- **The new sim is credible for transfer.** It reproduces the hist4 real-robot return (94.0 ± 11.8 vs 97.2 ± 13.7), length and seed ranking, and the real paddle speed distribution, without having been tuned on any policy return.
- `strike_shaped` was not fine-tuned (`strike_term03` was chosen as the striking source because it was better in the original sim). Before fine-tuning, `strike_shaped` seed 0 was the best old policy in the new sim (49 ± 14 vs 29 ± 7).

## Next

- 100-episode final test at a 10k cap of the selected checkpoint per seed, both fine-tune recipes, then save the winners to `latest_models/ceiling/<name>/`.
- Real-robot test of `runs/ceiling/realplant_screen2/ft_edgefix_s1` (final model), the best single policy.
- Optionally: fine-tune `strike_shaped`; `term03_nopretanh` on more seeds.
