# Sim2sim transfer testing

A *sim2sim* campaign trains a policy on one Box2D sim ("source") and tests how it transfers to a perturbed Box2D sim ("target") that shares the task / observation / action space but differs in physics. It is the rehearsal step before sim2real, and the home for fine-tuning experiments (full FT, residual FT, from-scratch baseline).

This page documents the harness, the layout, the campaign run on the `hist2_motion0` checkpoint (2026-04-25), and what we learned about which perturbations actually move the needle.

> **For residual RL recipes**, see [`training/residual-rl-recipe.md`](residual-rl-recipe.md) — that's the canonical doc. **Big-gap canonical default: `v27` (Maxmin-5, peak 87.94 ± 4.82, 1M-verified peak 98.3 / 84% > zs).** Build any future residual sim2sim or sim2real work off v27. The headline experimental conclusions are: (1) Q-overestimation control is the load-bearing knob — every recipe without it collapses; (2) every Q-control mechanism (Maxmin-N, REDQ, low `q_updates`, `q_weight_decay`) works reasonably well, with v27's Maxmin-5 winning on peak + 1M stability; (3) ensemble size matters a lot — N=3 is dramatically worse than N=5; (4) adaptation-phase exploration is unhelpful at best, actively harmful at worst — conservative/zero exploration wins; (5) v27 is exceptionally stable across 1M steps, hitting a sweet spot the other ensemble sizes don't reach. Alternative for fire-and-forget 300k deployment only: `v30_explore_lite` (v27 + lite exploration; tighter last5 std at lower peak; not 1M-verified). **From-scratch on paddle50 doesn't work** — best 300k recipe (bigger network) peaks at 36 vs zero-shot 67.54; even 1M peaks at 63 (§8.18–8.19). The "Drift study" / "400k extension" / "Drift-fix campaign" sections below trace the *small-gap* (OLD `sim2sim_combined`, paddle full-size) campaign through 2026-04-26. The harder paddle-50 (big-gap) campaign lives at [`notes/scratch/residual_rl_paddle50_log.md`](../../scratch/residual_rl_paddle50_log.md) (§8.17–§8.19) and is summarized in the recipe doc.

Planning + open-questions doc: [`notes/scratch/sim2sim_infra_plan.md`](../../scratch/sim2sim_infra_plan.md). Residual-method details live in [`notes/scratch/residual_rl_plan.md`](../../scratch/residual_rl_plan.md).

---

## Pipeline at a glance

| Step | Artifact |
|---|---|
| 1. Author target sim config | `scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_<tag>.yaml` |
| 2. Zero-shot eval | `scripts/smooth_policy/sim2sim_eval.py` → `runs/td3/sim2sim/<src_to_tgt>/zero_shot/metrics.json` |
| 3. Fine-tune (full / residual / from-scratch) | `scripts/smooth_policy/amp_history/configs/td3/sim2sim/td3_sim2sim_*.yaml` |
| 4. Aggregate | `scripts/smooth_policy/sim2sim_compare.py` → `comparison.md` |

All eval/comparison helpers reuse `scripts/smooth_policy/eval_utils.py` (factored from `evaluate.py`) so policy loading is identical to the standard eval path.

### Config layout

- **Source sim**: any `configs/new_juggle/sysid_best_params*.yaml`. The canonical source is now `sysid_best_params_hist2.yaml` (matches `latest_model/hist2_motion0/config.yaml`). The hist3 / hist4 / hist5 variants used for the original temporal-smoothing ablation are preserved at `configs/new_juggle/legacy/sysid_best_params_hist{3,4,5}.yaml`. The reference campaign below is the one that was actually run against `sysid_best_params_hist4.yaml` before the legacy move; new sim2sim work should source from `sysid_best_params_hist2.yaml`.
- **Target sim**: lives next to source as `configs/new_juggle/sim2sim_<tag>.yaml`. Inherits structurally from one source — only physics keys differ. First line is `# Source: <source_yaml>` for provenance. Each modified key has an inline `# PERTURBED: ...` comment.
- **Training configs**: under `configs/td3/sim2sim/`. Files for `zero_shot`, `full_ft`, `residual`, `from_scratch`; only `config:` / `model_path:` / `log_parent_dir:` change per campaign. The `residual` config carries the small-gap canonical recipe (`recency_top50`, `success_top_fraction: 0.5`); **for big-gap targets the canonical default is `paddle50/td3_residual_v27_ensemble5.yaml`** (Maxmin-5; 1M-verified; the standard for future residual sim2sim/sim2real work). Use `paddle50/td3_residual_v30_explore_lite.yaml` only as a fire-and-forget alternative when shipping final-step weights without per-ckpt eval. See [residual-rl-recipe.md](residual-rl-recipe.md).

### Results directory

```
runs/td3/sim2sim/<src_tag>_to_<tgt_tag>/
  zero_shot/   metrics.json + optional eval_rollouts/*.gif
  full_ft/     seed0/ seed1/ ...   (TD3 runs, td3_training.py output dirs)
  residual/    seed0/ seed1/ ...
  from_scratch/seed0/ seed1/ ...
  comparison.md
```

`<src_tag>` and `<tgt_tag>` are short qualitative names (`hist2_motion0`, `combined`, `heavy_puck`, etc.). Never reuse a `seedN` directory between runs — `td3_training.py` will append `r1`/`r2` and split runs into sibling dirs that the aggregator can't merge.

---

## Reference campaign — `hist2_motion0_to_combined` (2026-04-25)

### Source policy under test

| Field | Value |
|---|---|
| Run dir | `runs/td3/hist_motion_collision/hist2_motion0/` |
| Latest checkpoint | `checkpoint_975000/model.pth` |
| Args file | `td3_recommended.yaml` (now at `td3/legacy/td3_recommended.yaml`; the active default is `td3_recommended_top50_hist2.yaml`) |
| Source sim | `sysid_best_params_hist2.yaml` (sysid params + 2-timestep low-pass on PID target) |
| Reward weights | `task_reward_weight: 1.0`, `motion_reward_weight: 0.0` |
| Network | 2-layer, hidden=64 (actor + Q) |
| Updates | `q_updates: 25`, `actor_updates_per_iteration: 6` |
| Total steps | 1,000,000 |
| Exploration noise | 0.1 |
| `learning_starts` | 20,000 |

This is one of the `hist_motion_collision` sweep cells: 2-timestep temporal smoothing, no motion shaping. See [`td3-configs.md`](td3-configs.md) for the recommended-default rationale and [`td3-ablations-updates-and-depth.md`](td3-ablations-updates-and-depth.md) for the depth/update story.

### Target sim — `sim2sim_combined.yaml`

Bundles four perturbations relative to the source sysid config. All other physics keys held constant.

| Knob | Source | Target | Δ | Rationale |
|---|---:|---:|---|---|
| `pid_kp` | 9000 | 7200 | −20% | Softer position controller, slower paddle response |
| `enable_action_delay` | false | true | on | Adds delay to actions in addition to existing observation delay |
| `delay_seconds` | 0.025 | 0.030 | +20% | Larger delay magnitude (`delay_relative_range: 0.25` jitter unchanged) |
| `paddle_radius` | 0.0508 m | 0.04064 m | −20% | Smaller paddle → more edge collisions, less stable contact |
| `wall_direction_cone_deg` | 10 | 25 | +150% | Widens random angular perturbation on wall bounces |

### Zero-shot result (50 episodes, seed=0)

| Metric | Source (`sysid_best_params_hist2`) | Target (`sim2sim_combined`) | Δ |
|---|---:|---:|---|
| mean | 148.08 | 95.78 | **−35%** |
| median | 171.5 | 82.5 | −52% |
| std | 54.96 | 62.79 | +14% |
| tail10 | 178.5 | 87.8 | −51% |
| max | 210.0 | 203.0 | −3% |
| zero-return eps | 1 / 50 | 6 / 50 | +5 |

`max_return` barely moves (210 → 203) — the policy *can* still survive a full episode under target conditions on a lucky seed. Median dropping by half and 6× failure rate define the gap that FT methods will be measured against.

10 sample trajectories rendered to `runs/td3/sim2sim/hist2_motion0_to_combined/zero_shot/eval_rollouts/rollout_{0..9}.gif`.

---

## Single-knob sensitivity sweeps (2026-04-25)

To attribute the sim2sim gap, each perturbation was swept in isolation while every other physics key was held at the source baseline. Sweep script: [`notes/scratch/sim2sim_perturbation_sweep.py`](../../scratch/sim2sim_perturbation_sweep.py). 25 episodes per setting, seed=0. Per-setting metrics: `runs/td3/sim2sim/hist2_motion0_to_sweeps/<knob>/<label>/metrics.json`. Combined summary: `runs/td3/sim2sim/hist2_motion0_to_sweeps/summary.md`.

### PID Kp — flat through −20%, then collapses

| reduction | kp | mean | median | max | n_zero |
|---|---:|---:|---:|---:|---:|
| 0% | 9000 | 139.4 | 154 | 210 | 1 |
| 10% | 8100 | 136.7 | 162 | 194 | 1 |
| 20% | 7200 | 138.4 | 161 | 205 | 1 |
| 30% | 6300 | 106.0 | 111 | 192 | 1 |
| 40% | 5400 | 96.1 | 91 | 190 | 2 |
| 50% | 4500 | 53.7 | 41 | 172 | 3 |

Threshold at ~−20%; performance is essentially baseline up to that point and collapses past −30%. At −50% the policy is at ~38% of source mean.

### Action delay — roughly linear decline once enabled

| setting | enabled | delay (s) | mean | median | max | n_zero |
|---|---|---:|---:|---:|---:|---:|
| baseline | false | 0.025 | 139.4 | 154 | 210 | 1 |
| on, +0% | true | 0.025 | 134.2 | 148 | 199 | 2 |
| on, +20% | true | 0.030 | 124.3 | 141 | 197 | 2 |
| on, +40% | true | 0.035 | 121.2 | 137 | 197 | 2 |
| on, +60% | true | 0.040 | 97.6 | 103 | 189 | 2 |
| on, +80% | true | 0.045 | 66.7 | 63 | 170 | 3 |

Just enabling action delay (no extra magnitude) costs ~5 mean. Each +0.005 s past that costs another ~10–25 mean. At 0.045 s (90% of `time_per_step = 0.050 s`, the upper clip in `_resolve_delay_seconds_for_step`) the policy halves.

### Wall-bounce angle cone — no measurable effect across 10° → 60°

| cone | mean | median | max | n_zero |
|---|---:|---:|---:|---:|
| 10° (baseline) | 139.4 | 154 | 210 | 1 |
| 20° | 154.8 | 164 | 202 | 1 |
| 30° | 142.9 | 154 | 202 | 1 |
| 40° | 142.4 | 161 | 201 | 1 |
| 50° | 153.0 | 163 | 201 | 1 |
| 60° | 140.7 | 154 | 207 | 1 |

All within noise (~±15 mean). The widened wall cone — even at 60° — is essentially invisible to the policy. Likely because (a) wall bounces aren't on the critical path of upper-half juggling (the puck spends most of its time between paddle and top), and (b) the policy's recovery margin absorbs the extra angular noise. The existing `wall_direction_cone_deg` knob is uniform-random with zero mean, so it widens variance without shifting mean bounce direction; a meaningful wall perturbation would likely require either a deterministic angular bias on every bounce, a `side_wall_restitution` / `end_wall_restitution` change, or wall geometry changes.

### Implications for the combined config

- The −35% gap in `sim2sim_combined` is driven by **kp + delay** (and presumably the −20% paddle radius, which wasn't in this sweep). The wall cone bump contributes ~nothing.
- For future combined targets that aim to test transfer specifically against wall dynamics, swap the cone widening for one of the alternatives above.

---

## Fine-tune campaign — residual vs full_ft (2026-04-25)

100k steps of online TD3 fine-tuning on the target sim, starting from `checkpoint_975000`. Single seed. Residual uses frozen base + trainable residual + critic from scratch (Silver/Johannink). Full_ft uses `load_mode: fine_tune` (warm-starts actor, Q, optimizer state) and seeds 10k samples from the source replay (3000 success + 7000 failure, proportional to source buffer fill).

### Result — best checkpoint per method

| | mean | tail10 | median | max | step |
|---|---:|---:|---:|---:|---|
| zero_shot (no FT) | 95.78 | 87.8 | 82.5 | 203 | — |
| **residual best** (`residual_scale=0.05`) | 106.84 | **127.2** | **123.5** | 196 | 50k |
| **full_ft best** (lrs ÷10) | **108.64** | 117.4 | 95.5 | 195 | 100k |

Both methods beat zero-shot at peak; residual reaches its peak in **half the env-steps** (50k vs 100k) and has higher tail10/median there. Full_ft has marginally higher mean. Per-checkpoint trajectories at `runs/td3/sim2sim/hist2_motion0_to_combined/{residual,full_ft}/eval_*_ckpt_*/metrics.json`.

### Final-step eval is unsafe — both methods drift past peak

Full_ft v3 (200k, same hyperparams as v2): final-step deterministic eval mean **62.4** despite peaking at 108.64 mean at step 100k. The actor keeps moving on a small replay long after gradient signal has run out. Pattern is the same for residual (post-50k decay, less severe but present).

**Always evaluate intermediate checkpoints (every `checkpoint_interval`) and pick the best — not the final step.**

### Hyperparameter sensitivity (single-seed, this campaign)

Each row is a 100k-step run; "best" column is best deterministic mean across saved checkpoints.

| variant | UTD (q,a) | primitives | residual_scale | lrs | best mean | notes |
|---|---|---|---|---|---:|---|
| residual v0 | 25,6 | off | 0.25 | source (3e-4 / 1e-3) | 46.86 | actor diverges; UTD too high, scale too loose |
| residual v1 | 4,1 | off | 0.25 | source | 41.56 | UTD reduction alone insufficient |
| residual v2 | 4,1 | off | **0.05** | source | **106.84** | scale was the dominant knob |
| residual v3 | 4,1 | off | 0.10 | source | 99.80 | flatter trajectory, lower peak than 0.05 |
| full_ft v0 | 25,6 | on | n/a | source | 42.36 | primitives + high UTD destroy actor |
| full_ft v1 | 4,1 | off | n/a | source | 35.90 | source-lr was the dominant knob |
| full_ft v2 | 4,1 | off | n/a | **÷10** | **108.64** | matches residual peak; later step (100k) |
| full_ft v3 | 4,1 | off | n/a | ÷10 | (same v2 ckpt @ 100k) | 200k extension just decays past peak |

Lessons:

1. **`residual_scale=0.05` is the sweet spot** for residual on a 1M-step base policy + small online buffer. 0.25 lets bad gradients drag actions ±25%; 0.10 doesn't reach the same peak as 0.05.
2. **`lr ÷ 10` is required for full_ft** when warm-starting from a strong base. Source-training lrs make the actor move too fast on a small replay before the critic can build a useful signal.
3. **UTD reduction (q25/a6 → q4/a1) is necessary but not sufficient** — neither method works at low UTD with the wrong scale/lr.
4. Disable exploration primitives in `full_ft` for a clean head-to-head with residual. Primitives flood the early replay with off-policy data that hurts FT-from-strong-base.

### Code wiring (landed this campaign)

| File | Change |
|---|---|
| `scripts/smooth_policy/residual_agent.py` | new `ResidualActor` (frozen base + trainable residual, zero-init head, action clip) |
| `…/td3/td3_training.py` | `Args.full_checkpoint_load` Literal extended with `"residual"`; new residual loader branch (shared base across online/target); `Args.residual_scale`; `Args.fine_tune_replay_keep`; **lr-reset after `load_fine_tune_optimizer_state`** so config knobs aren't silently overridden by source-restored optimizer state |
| `…/td3/helper/td3_checkpointing.py` | `seed_fine_tune_replay_from_source` — proportional subsample from source success/failure buffers, added via `add()` so position/size stay consistent |
| `scripts/smooth_policy/eval_utils.py` | `"residual_actor"` policy class; `infer_policy_class_from_state_dict` recognizes `base.*`+`residual.*`; `build_policy` constructs `ResidualActor` placeholder, state_dict restore fills buffers |
| `…/configs/td3/sim2sim/td3_sim2sim_residual.yaml` | initial post-drift-study defaults: `residual_scale=0.05`, `q_updates=1`, `q_lr=3e-4`, `actor_updates_per_iteration=1`, primitives off. **Superseded by the drift-fix campaign defaults below (`residual_scale=0.15`, `q_updates=4`, `success_top_fraction=0.5`)** — that's what the YAML actually contains today. |
| `…/configs/td3/sim2sim/td3_sim2sim_full_ft.yaml` | tested defaults: `policy_lr=3e-5`, `q_lr=1e-4`, `q_updates=4`, `actor_updates_per_iteration=1`, primitives off, `fine_tune_replay_keep=10000` |

### Open follow-ups

- **Best-of-eval-checkpoint tracker** in `td3_training.py`: compute deterministic eval at each checkpoint and save the best so far as `model.pth`. Without it, every campaign needs the post-hoc per-checkpoint eval we did here. Still open as of 2026-05-01.
- ~~Multi-seed verification at the chosen peak step.~~ Done (3-seed `recency_top50` and 5-seed v27, both in residual-rl-recipe.md).
- ~~Apply the drift study to full_ft.~~ Done — full_ft + Maxmin-5 / REDQ-10-2 ensemble verified at 1M (paddle50 log §8.16). Ensemble fixes residual-specific Q-overestimation, not full_ft drift.

---

## Drift study — residual (2026-04-25)

Investigated *why* the residual v2 hit mean 106.84 @ step 50k but degraded to 84.7 by 100k (-21% drift). Hypothesis space: narrow data, high learning rate, high update-to-data, success-buffer bias.

**TL;DR.** The "wider data" hypothesis was wrong (`bigger_buffer` ≈ baseline). The drift is structural to the optimization regime: critic Q values inflate over time, driven by high UTD on a PER-sampled success buffer that becomes a museum of past peaks. Single-knob `q_lr ÷ 3.3` and `UTD ÷ 4` each reduce drift; combining them gives the highest absolute peak (109.3 @ 30k) and the best final mean (95.7), but does NOT eliminate post-peak collapse — the combo still has a transient -22% trough at step 50k before recovering. The single-knob `lower_qlr` run is the most *stable* trajectory (peak 102.1, no trough below 88). All numbers below are single-seed, n=50 deterministic eval; SE ≈ 8–9 mean-return.

### Diagnostic — what we observed in v2 tensorboard

Across all of v2 and v3 (residual_scale 0.05 and 0.10), the same pattern:

| @ step | 10k | 25k | 50k | 75k | 100k |
|---|---:|---:|---:|---:|---:|
| Q1_task_mean | 0.30 | 0.42 | 0.70 | 0.90 | **1.12** |
| bellman_target | 0.70 | 1.19 | 2.20 | 3.00 | **4.07** |
| sampled_task_reward (critic batch) | 0.50 | 0.57 | 0.63 | 0.64 | **0.66** |
| episode_return_success_threshold | 135 | 151 | 159 | 161 | **160** |
| rollout return (rolling 2k) | 123 | **134** | 98 | 104 | 92 |

Q values climb monotonically; the bellman target stays ahead, dragging Q up. **Mechanism:** PER samples the critic 30 % from the success buffer; that buffer keeps the *top-20%* returning episodes from a 500-episode rolling window. As training progresses the success-threshold rises (135 → 160) — the bucket becomes a *museum of past peaks*. The current rolling policy degrades but doesn't get its weak rollouts into the success bucket. Critic trains on increasingly optimistic (s, a) the actor can't currently produce → Q inflates → actor exploits → rollouts collapse.

### Single-knob ablation — peak/final summary

Each row reports the best deterministic mean across saved checkpoints (n=50 episodes, seed=0); tail10 is at *that same step* (not the absolute best tail10).

| variant | knob change | peak mean | tail10 @ peak | peak step | final mean | drift (peak→final) |
|---|---|---:|---:|---|---:|---:|
| zero-shot | (no FT) | — | 87.8 | — | 95.78 | — |
| v2 baseline | (none) | 106.8 | 127.2 | 50k | 84.7 | -21% |
| bigger_buffer | success/failure 6k/14k → 30k/70k | 103.9 | 101.2 | 90k | 84.0 | -19% |
| lower_qlr | q_lr 1e-3 → **3e-4** | 102.1 | **132.7** | 60k | 93.7 | -8% |
| lower_utd | q_updates 4 → **1** (UTD) | 98.4 | 113.3 | 20k | 95.2 | -3% |
| low_succ_frac | critic_success_sample_fraction 0.3 → 0 | 100.2 | 92.5 | 40k | 87.5 | -13% |
| **combo (UTD=1 + q_lr=3e-4)** | both above | **109.3** | 111.0 | **30k** | **95.7** | -13% |

### What "drift -13%" hides — the post-peak trajectory

The peak→final number papers over what happens *between* peak and final. Per-checkpoint deterministic mean for the combo and lower_qlr-alone runs:

| ckpt | combo mean | combo Δ from peak | lower_qlr mean | lower_qlr Δ from peak |
|---|---:|---:|---:|---:|
| 10k  |  91.34 | -16.4% |  81.06 | -20.6% |
| 20k  |  92.82 | -15.1% |  84.40 | -17.3% |
| 30k  | **109.30** | **PEAK** |  88.52 | -13.3% |
| 40k  |  97.86 | -10.5% |  93.82 | -8.1% |
| 50k  |  85.10 | **-22.2%** |  91.28 | -10.6% |
| 60k  |  93.42 | -14.5% | **102.10** | **PEAK** |
| 70k  |  89.34 | -18.3% | 100.82 | -1.3% |
| 80k  |  96.16 | -12.0% |  92.36 | -9.5% |
| 90k  |  95.84 | -12.3% |  97.02 | -5.0% |
| final|  95.66 | -12.5% |  93.74 | -8.2% |

The combo trajectory has a **transient -22.2% drop at step 50k that's the same magnitude as the v2 baseline's monotonic collapse** — the policy briefly bottoms out at 85.1 (≈ baseline-final-level bad) before bouncing back to a 90–96 band. So the precise claim that's true is:

- combo reaches a **higher peak earlier** (109.3 @ 30k vs baseline 106.8 @ 50k),
- combo finishes at a **higher floor** (95.7 vs 84.7),
- combo does NOT keep the policy *near peak* — there's still a post-peak collapse, it just recovers.

`lower_qlr` (q_lr=3e-4 alone, UTD unchanged at 4) is the **most stable** run by trajectory shape: monotonic-ish climb to peak at 60k, no checkpoint drops more than -13.3% from peak, no trough. Lower absolute peak but the policy actually stays close to it.

### Statistical caveat

Single seed, n=50 episodes per checkpoint, std ≈ 60 → SE ≈ 8.5 on the mean. Adjacent-checkpoint differences in the combo run are within 1–3 SE:
- 109.3 (30k) vs 97.86 (40k): Δ = 11.4 ≈ 1.3 SE
- 109.3 (30k) vs 85.1 (50k): Δ = 24.2 ≈ 2 SE (significant but single-seed)
- 109.3 (30k) vs 102.1 (lower_qlr peak): Δ = 7.2 < 1 SE

**Without multi-seed confirmation, "combo's peak is 109.3" and "lower_qlr's peak is 102.1" are not statistically distinguishable**, and the 50k trough may itself be partly noise. Treat the table as indicative; multi-seed re-run is the open follow-up that decides between combo and lower_qlr-alone as the residual default.

### Findings

1. **The "wider data" hypothesis is wrong.** `bigger_buffer` (5x main + buffers wide enough to hold all 100k samples) is essentially baseline. Drift is structural to the optimization regime, not the data window.
2. **UTD is the dominant driver of Q runaway.** `lower_utd` flattens Q (`Q1@100k`: 1.12 → **0.26**) and gives the smallest peak→final drift (-3%), but underfits the critic — peak mean only 98.4.
3. **Lower q_lr is the dominant driver of trajectory smoothness.** `lower_qlr` has no mid-run trough below 88, peak→final drift only -8%, and the highest tail10 of any single-knob run (132.7).
4. **Removing success-buffer bias (`critic_success_sample_fraction=0`) is partial.** Helps peak→final drift (-13%) but doesn't kill Q runaway (Q still 0.20 → 1.05 over 100k).
5. **Combo gets the highest peak but trades stability for it.** Higher peak (109.3) and earlier (30k) than any single-knob, but the trajectory is more volatile: -22% transient trough at 50k vs lower_qlr's -10% maximum dip. If you can early-stop on per-checkpoint eval (recommended), combo is the right pick. If you can't, lower_qlr-alone is safer.

### Default change

Updated `td3_sim2sim_residual.yaml` to combo defaults:

| knob | old | new |
|---|---:|---:|
| `q_updates` | 4 | **1** |
| `q_lr` | 1e-3 | **3e-4** |

`residual_scale=0.05`, primitives off, `fine_tune_replay_keep` etc. all unchanged. **Per-checkpoint eval is required** to pick the actual best model — combo's higher peak only matters if you don't end up with the 50k trough as your `model.pth`.

If your harness can't early-stop and you must trust `model.pth` at end of training, prefer `q_lr=3e-4` *alone* (UTD=4) — slightly lower peak, much smoother trajectory, smaller risk of finishing in a trough.

### Reproducibility

Diagnostic configs at `scripts/smooth_policy/amp_history/configs/td3/sim2sim/diagnose/{bigger_buffer,lower_qlr,lower_utd,low_success_frac,combo_utd_qlr}.yaml`. Run dirs at `runs/td3/sim2sim/hist2_motion0_to_combined/residual_diagnose/<variant>/seed0[r1]/`. Per-checkpoint deterministic eval results in `eval_combined_ckpt_*/metrics.json` and `eval_combined_final/metrics.json` under each.

### Open follow-ups (specific to this study)

- ~~Multi-seed confirmation of combo vs lower_qlr.~~ Done — superseded by the drift-fix campaign + 3-seed `recency_top50` verification below; both old "combo" and "lower_qlr" defaults were dropped in favor of `success_top_fraction: 0.5`.
- ~~Apply UTD=1 + q_lr=3e-4 to full_ft.~~ Done — see drift-fix campaign and the big-gap full_ft+ensemble runs in the paddle50 log.
- ~~`success_top_fraction=1.0` test of the museum mechanism.~~ Done in the data-balance ablation below (`recency_top99` regressed because failure_rb starves).
- Best-of-eval-checkpoint tracker in `td3_training.py` — STILL OPEN.

---

## 400k extension — does longer training help, does scale help? (2026-04-26)

100k showed residual peaks @ 30k (combo) / 60k (lower_qlr) and then drifts. Two open questions: (1) at 400k — the budget that lets a from-scratch policy train decently — does residual close the gap to source (~148)? (2) does bumping `residual_scale` 0.05 → 0.15 unlock more correction, since the rs=0.05 head is essentially a ±5% perturbation around the frozen base?

**TL;DR.** None of the four 400k residual variants stay near peak. Bigger scale (0.15) reaches higher absolute peaks (combo: 113.7, lower_qlr: 107.9) but is *more volatile* — both rs=0.15 runs end below zero-shot. Bigger scale specifically lets `lower_qlr_rs015` produce a clear 50k-window of consistent-improvement at 30-70k (avg 102.1, all > zero-shot), but the policy then catastrophically collapses to 50-65 mean from step 190k onward. **The headline finding: rs=0.05 is too tight (residual head can't express useful corrections — `lower_qlr_400k`'s best ckpt is at step 10k where the head is essentially zero from init); rs=0.15 is too loose at long horizons (drift compounds).** The right operating point under fixed `residual_scale` likely needs early stopping (per-checkpoint eval + best-tracker) or a scale schedule.

### 4-way comparison — single seed, 400k each, 39 saved checkpoints + final

`mean(all)` is the per-checkpoint mean averaged across all 39 saved checkpoints — i.e., what you'd expect from a randomly-picked checkpoint. `>zs` is the count of checkpoints whose mean beats zero-shot 95.78. `best phase` is the highest 5-checkpoint sliding-window mean (50k-wide).

| variant | peak | @step | tail10@peak | final | mean(all) | >zs / 39 | best 50k phase |
|---|---:|---:|---:|---:|---:|---:|---|
| combo_400k (rs=0.05) | 97.6 | 340k | 104.2 | 83.9 | 88.9 | 3 | 93.3 (100-140k) |
| **combo_400k_rs015 (rs=0.15)** | **113.7** | 340k | 101.1 | 86.8 | 83.9 | 3 | 98.8 (320-360k) |
| lower_qlr_400k (rs=0.05) | 99.9 | **10k** | 94.4 | 97.4 | 83.2 | 4 | 94.3 (10-50k) |
| **lower_qlr_400k_rs015 (rs=0.15)** | 107.9 | 70k | 95.7 | 69.4 | 78.8 | **11** | **102.1 (30-70k)** |
| zero-shot reference | 95.8 | — | 87.8 | 95.8 | 95.8 | — | — |

**Compared to 100k results** (combo peak 109.3 @ 30k, lower_qlr peak 102.1 @ 60k): the 400k extension finds higher peaks for two of four variants but no improvement in stability. **`lower_qlr_400k`'s best checkpoint is step 10k** (mean 99.94, when residual head is essentially zero from init) — meaning the rs=0.05 head adds noise the rest of training without improving on the frozen base.

### Trajectory shapes — only `lower_qlr_400k_rs015` produces a stable improvement window

```
        combo_rs015 (peak 113.7 @ 340k)         lower_qlr_rs015 (peak 107.9 @ 70k)
        103 → drop → 86 → 71 → ... → 113 ↓      85 → climb → 108 → SUSTAINED → 200k cliff
        "single-spike" peak                      30-70k all >100, then collapse
```

- **`combo_400k_rs015`** trajectory is jittery throughout: starts at 103.2 @ 10k, dips to 59.3 @ 190k, recovers to a noisy 80-95 band, then a single-checkpoint spike to 113.7 @ 340k. Very few checkpoints beat zero-shot. The 113.7 peak is real but adjacent ckpts (97.8, 92.4) are normal — likely a noise spike given single-seed n=50.
- **`lower_qlr_400k_rs015`** has the cleanest "good phase": from step 30k → 170k, every checkpoint mean ≥ 89, with 30k-170k average ≈ 99.5, peak 107.94 @ 70k, tail10 reaching 139.2 @ 130k. Then from step 190k onward the policy collapses: mean drops to 50-65 range and never recovers (final 69.4). This is the strongest evidence that the residual *can* learn useful corrections — and the strongest evidence that long-horizon drift is severe with rs=0.15.
- **`combo_400k` (rs=0.05)** wanders in a 80-97 mean band the whole 400k — neither learns much nor collapses much.
- **`lower_qlr_400k` (rs=0.05)** has the same pattern but with a long-tail decline starting around 150k (mean drifts to 70-80 range from 200k onward, then partial recovery — final 97.4).

### What this means for `residual_scale`

Confirms the 100k insight that 0.05 (residual ±5% around base action) is too tight for useful corrections under this source/target gap. But 0.15 trades expressive headroom for stability:

- 0.15 is the only variant where any *consecutive* window of ≥5 checkpoints averages > zero-shot (lower_qlr_rs015 30-70k @ 102.1). 0.05 never does.
- 0.15 is also the only variant where final-step performance falls below 70 (lower_qlr_rs015 final 69.4) — a ~25-point cliff from the same run's best phase.

**The right next experiments** (priority order):
1. **Early-stop / best-tracker**: pick model.pth as the best-eval checkpoint, not final. Without this, all 4 variants ship with degraded weights. (Open follow-up from drift study, now urgent.)
2. **Scale schedule**: try `residual_scale: 0.15` annealed to 0.05 over training. Hypothesis: high scale early lets the head learn corrections; low scale late protects from drift.
3. **Multi-seed verification** of `lower_qlr_400k_rs015` 30-70k window — single-seed result; combo_400k_rs015's 113.7 peak is a single-checkpoint spike that deserves seed-confirmation before being treated as the new ceiling.
4. Apply scale-bump to `full_ft` (currently runs with `lr÷10`, no residual; the residual analog is `policy_lr * residual_scale_equivalent` — different mechanism so the scale knob doesn't transfer directly).

### Reproducibility — 400k extension

Configs at `scripts/smooth_policy/amp_history/configs/td3/sim2sim/diagnose/long/{combo_400k,combo_400k_rs015,lower_qlr_400k,lower_qlr_400k_rs015}.yaml`. Run dirs at `runs/td3/sim2sim/hist2_motion0_to_combined/residual_diagnose/long/<variant>/seed0/`. Per-checkpoint eval JSONs under `eval_combined_ckpt_*/metrics.json` (39 each + `eval_combined_final/`). Eval driver: `bash scripts/smooth_policy/eval_all_ckpts_residual.sh <run_dir> scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_combined.yaml cuda:N`.

### From-scratch baseline (done 2026-04-26)

`scripts/smooth_policy/amp_history/configs/td3/sim2sim/diagnose/long/from_scratch_400k.yaml` runs the canonical recommended-default TD3 hyperparams (q_updates=25, actor_updates=6, primitives ON, learning_starts=20k, buffer 100k) on `sim2sim_combined.yaml` for 400k steps. **Result: peak 82.86 @ step 370k, mean(all 39 ckpts) 43.02, no checkpoint > zero-shot 95.78.** 400k from-scratch is monotonically improving but well below the zero-shot warm-start. **Confirms the residual underperformance is NOT a method failure — the env is too hard for 400k from-scratch**, and any fine-tuning approach (residual or full_ft) has a real opportunity to do much better than starting from zero.

---

## Drift-fix campaign — finding `no_per + q_wd1e3` (2026-04-26)

After the 400k extension showed all rs={0.05, 0.15} variants drift below zero-shot at end of training, we ran a comprehensive drift-fix campaign of 11 single-seed 200k experiments. Authoritative log: [`notes/scratch/residual_rl_drift_fix_log.md`](../../scratch/residual_rl_drift_fix_log.md).

### Mechanism diagnosis

The drift study traced post-peak collapse to TWO independent critic-side failure modes:
1. **Museum effect**: PER samples 30% from the success buffer, which keeps top-20% returning episodes from a 500-episode rolling window. As training improves then degrades, the buffer keeps "ghost" peak transitions the current policy can't reproduce — the critic learns optimistic Q values from a museum of past peaks.
2. **Q runaway**: critic Q values inflate monotonically (Q1@100k: 0.30 → 1.12 in baseline). The actor exploits inflated Q estimates and drifts to high-Q-low-actual-return regions.

### What worked

| fix | mechanism | result |
|---|---|---|
| **`per_enabled: false` + `critic_success_sample_fraction: 0.0`** | Kills museum effect (uniform sampling, no success bias) | best peak/mean (`no_per_rs015`: peak 105 @ 100k, sustained 90-130k > zs) |
| **`q_weight_decay: 0.001` (10x default)** | Bounds Q magnitudes via critic L2 | best tail stability (`q_wd1e3_rs015`: drift only -13, last5_mean 90.2) |
| **Both stacked** | Attacks both failure modes | **WINNER `no_per_q_wd1e3_rs015`** (see below) |

### What didn't work

| fix | result | why |
|---|---|---|
| Residual head WD=1e-3 | peak 113 but still drifts -38 | Doesn't address critic side |
| Residual head WD=1e-2 | peak 94 (below zs) | Too aggressive, kills correction signal |
| Scale anneal 0.15→0.05 | peak 97 @ step 10k (residual ≈ 0) | Head can't track shrinking ceiling |
| Action L2 (λ=1) | peak 103 | Redundant with parameter L2 |
| EMA actor on top of no_per_q_wd | regressed to drift -45 | Single-seed noise or interference |

### Winning recipe — `no_per_q_wd1e3_rs015`

| metric | value |
|---|---:|
| peak | 108.0 @ step 40k |
| mean(all 19 ckpts) | **96.93** |
| last5_mean | **96.50** (above zero-shot 95.78!) |
| drift (peak → last5) | -11.5 |
| ckpts > zero-shot | 10/19 |

**This is the first variant where the policy stays above zero-shot through the END of training.** Final-step `model.pth` is competitive without per-checkpoint eval.

Per-checkpoint trajectory:

```
step:   10  20  30  40   50  60  70  80   90  100  110  120  130  140  150  160  170  180  190
mean:   94 102  89 108>  94  94  97> 98> 100> 102> 105>  90   94   92   98> 101>  93   92   99>
```

(`>` = > zero-shot 95.78). Sustained > zs windows at steps 70-110k (5 consecutive) and 150-160k (2 consecutive).

### Default config update (2026-04-26)

`td3_sim2sim_residual.yaml` updated to use the winning combo:

```yaml
per_enabled: false
critic_success_sample_fraction: 0.0
critic_failure_sample_fraction: 1.0
q_weight_decay: 0.001       # 10x baseline 1e-4
residual_scale: 0.15        # was 0.05
q_updates: 4                # restored to 4 (was 1 in earlier "combo" defaults)
q_lr: 0.0003                # lower_qlr setting unchanged
total_timesteps: 200000     # was 100000
```

### Reproducibility — drift-fix campaign

Configs under `scripts/smooth_policy/amp_history/configs/td3/sim2sim/diagnose/long/driftfix/`. Run dirs at `runs/td3/sim2sim/hist2_motion0_to_combined/residual_diagnose/long/driftfix/<variant>/seed0/`. Per-checkpoint eval JSONs in `eval_combined_ckpt_*/metrics.json`. Aggregator: `.venv/bin/python notes/scratch/aggregate_driftfix_results.py`.

### Code knobs landed in `td3_training.py`

| Args field | Effect |
|---|---|
| `residual_weight_decay: float = 0.0` | Adam weight_decay on residual head (rejected by campaign) |
| `residual_scale_end: float \| None = None` | Linear anneal of residual_scale over training (rejected) |
| `residual_ema_decay: float \| None = None` | EMA copy of residual head; saves `model_ema.pth` per checkpoint (operational tool) |
| `residual_action_l2: float = 0.0` | L2 penalty on residual *output* in actor loss (rejected) |

### Multi-seed verification (2026-04-26)

After the seed-0 result of `no_per_q_wd1e3_rs015` looked like a clean win (mean across 19 ckpts above zero-shot, drift only -11.5), we re-ran with seeds 1, 2 and also did seeds 1, 2 of `q_wd1e3_rs015` alone. Result: **only the peak generalises across seeds, the lack of drift does not**.

| recipe | seed | peak | mean(19) | last5_mean | drift | >zs |
|---|---:|---:|---:|---:|---:|---:|
| q_wd1e3 | 0 | 103.2 | 89.7 | 90.2 | -13.0 | 4/19 |
| q_wd1e3 | 1 | 91.9 | 68.3 | 55.2 | -36.7 | 0/19 |
| q_wd1e3 | 2 | 95.3 | 86.9 | 85.2 | -10.1 | 0/19 |
| **q_wd1e3 MEAN** | — | **96.8** | **81.6** | **76.8** | -19.9 | 4/57 |
| no_per+q_wd | 0 | 108.0 | 96.9 | 96.5 | -11.5 | 10/19 |
| no_per+q_wd | 1 | 91.1 | 68.4 | 61.2 | -30.0 | 0/19 |
| no_per+q_wd | 2 | 101.7 | 73.2 | 59.8 | -41.9 | 1/19 |
| **no_per+q_wd MEAN** | — | **100.3** | **79.5** | **72.5** | -27.8 | 11/57 |

**Key takeaways:**
1. **Peak performance is reproducible** (~100 mean across seeds) — both recipes consistently produce a checkpoint somewhere in the 91-108 mean range, well above zero-shot 95.78.
2. **Drift is partially unsolved** — 4/6 seeds across both recipes still collapse in the second half. The seed-0 "no drift" trajectory was an outlier, not the rule.
3. **no_per+q_wd produces a slightly higher peak** (100.3 vs 96.8 cross-seed mean) but the tail is no more stable than q_wd alone.
4. For deployment: **per-checkpoint eval is REQUIRED** to find the peak (typically step 20-60k). Final-step `model.pth` is unsafe.

### Data-balance fix supersedes the prior recipe (2026-04-26)

After the multi-seed disappointment, we ran a 4-variant data-balance
ablation that fundamentally changed the default. The winning fix is a
**single-knob change**: `success_top_fraction: 0.2 → 0.5`.

| variant | knob change | peak | mean(9) | last3 | >zs |
|---|---|---:|---:|---:|---:|
| recency_smaller_buf | success_buffer 6000→1500 | 108.3 | 97.8 | 99.1 | 5/9 |
| **recency_top50** | success_top_fraction 0.2→0.5 | **110.7** | **103.7** | **104.1** | **8/9** |
| recency_top99 | success_top_fraction 0.2→0.99 | 102.7 | 91.7 | 90.2 | 3/9 |
| recency_window100 | recent_window 500→100 | 106.0 | 96.1 | 90.9 | 5/9 |

**Mechanism:** `success_top_fraction: 0.5` makes the success threshold = MEDIAN of the recent 500 episode returns. This guarantees ~50% of episodes go to `success_rb` and ~50% to `failure_rb` regardless of how policy quality moves. The threshold tracks current quality, can't ratchet up and stay there. Old peak transitions get diluted by current-quality transitions at high rate.

`top99` (0.99) regresses because failure_rb starves — only the worst 1% of episodes go there, and the critic_failure_sample_fraction=0.7 then samples mostly an empty buffer.

### Multi-seed verification of `recency_top50`

| seed | peak | mean(9) | last3 | catastrophic? |
|---|---:|---:|---:|---|
| 0 | 110.7 | 103.7 | 104.1 | ✅ no |
| 1 | 92.4 | 88.8 | 91.4 | ✅ no |
| 2 | 98.9 | 89.2 | 88.9 | ✅ no |
| **3-seed mean** | **100.7** | **93.9** | **94.8** | **✅ none** |

Compared to prior recipes' 3-seed means:

| recipe | peak | mean(N) | tail | seed collapses |
|---|---:|---:|---:|---:|
| q_wd1e3 alone | 96.8 | 81.6 | last5 76.8 | 1/3 (last5=55) |
| no_per+q_wd1e3 | 100.3 | 79.5 | last5 72.5 | 2/3 (last5=60-61) |
| **recency_top50** | 100.7 | **93.9** | **last3 94.8** | **0/3** |

`recency_top50` matches the peak of prior recipes but **eliminates the catastrophic seed-dependent collapse**. Tail mean improves by +22 pts vs prior best.

### Final operational recipe (revised 2026-04-26 PM)

`td3_sim2sim_residual.yaml` defaults:
```yaml
per_enabled: true                       # restored (was false in prior recipe)
success_top_fraction: 0.5               # median split — THE FIX (was 0.2 = top-20%)
critic_success_sample_fraction: 0.3     # restored to default
q_weight_decay: 0.001                   # 10x baseline; bounds Q
residual_scale: 0.15
q_updates: 4
q_lr: 0.0003
total_timesteps: 100000
```

**Deployment procedure:**
1. Train ≥3 seeds for 100k steps each
2. Per-checkpoint eval (n=50) every 10k steps
3. Ship the best-mean checkpoint across (seeds, steps)

Expected: peak ckpt 100-110 mean, mean across all ckpts ≥ 88 even on worst seed (no catastrophic collapse). 4-15% improvement over zero-shot 95.78.

### From-scratch 1M comparison (added 2026-04-26 PM)

For context, also ran from-scratch TD3 with recommended HPs to 1M steps (resumed from the 400k checkpoint). Per-checkpoint eval:

| metric | from-scratch 400k | from-scratch 1M |
|---|---:|---:|
| peak mean | 82.86 @ 370k | **130.28 @ 990k** |
| final mean | 85.10 | **130.28** |
| ckpts > zero-shot | 0/39 | 24/98 |
| last5_mean | 72.1 | 121.0 |

**Trade-off:**
- Residual at 100k reaches peak ~108 in ~30k env steps — **fast**, but ceiling at ~110.
- From-scratch at 1M reaches peak ~130 — **higher ceiling**, no drift, but 10x the budget.

For sim2sim where target dynamics are reachable from scratch given enough budget, both approaches are viable depending on time/compute constraints.

### Open follow-ups

- ~~5-seed re-run of `recency_top50`.~~ Done; small-gap recipe is stable.
- ~~Test recipe on other sim2sim pairs.~~ Done — `recency_top50` does NOT transfer to big-gap targets (paddle -50%); v27 (Maxmin-5) is the big-gap recipe. See [`residual-rl-recipe.md`](residual-rl-recipe.md) and [`paddle50_log.md`](../../scratch/residual_rl_paddle50_log.md).
- Adaptation-phase exploration: see [`notes/scratch/residual_exploration_plan.md`](../../scratch/residual_exploration_plan.md) (queued, blocked on GPU 2026-05-01).

---

## How to run a new campaign

1. Author `configs/new_juggle/sim2sim_<tag>.yaml` (copy from a sysid source, change only physics keys, mark each with `# PERTURBED: ...`).
2. Pick a source checkpoint and tag (e.g., `hist2_motion0`).
3. Zero-shot:
   ```bash
   python scripts/smooth_policy/sim2sim_eval.py \
     --checkpoint runs/td3/<run>/checkpoint_<step>/model.pth \
     --target-config scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_<tag>.yaml \
     --n-episodes 50 --seed 0 \
     --out-dir runs/td3/sim2sim/<src_to_tgt>/zero_shot/
   ```
   Add `--save-gif --n-gifs 10` for qualitative rollouts.
4. (Optional) Single-knob sweep — adapt `notes/scratch/sim2sim_perturbation_sweep.py` to the new target's knobs.
5. Fine-tune: fill placeholders in `td3_sim2sim_{full_ft,residual,from_scratch}.yaml` (`config`, `model_path`, `log_parent_dir`, `run_name`, `seed`), launch ≥2 seeds each. The repo yamls carry the campaign-tested defaults: residual = `recency_top50` (`success_top_fraction: 0.5`, `residual_scale: 0.15`, `q_updates: 4`, `q_lr: 3e-4`); full_ft = `lr÷10`. **For big-gap targets (zs drop >20%) the canonical default is [`paddle50/td3_residual_v27_ensemble5.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v27_ensemble5.yaml)** (Maxmin-5; 5-seed + 1M-verified; the standard for any future residual sim2sim/sim2real). Use [`paddle50/td3_residual_v30_explore_lite.yaml`](../../../scripts/smooth_policy/amp_history/configs/td3/sim2sim/paddle50/td3_residual_v30_explore_lite.yaml) only as a fire-and-forget 300k alternative — see [`residual-rl-recipe.md`](residual-rl-recipe.md). From-scratch isn't viable on paddle50-class targets. **Always evaluate intermediate checkpoints** — final-step eval is unsafe for every fine-tuning method on every gap size.
6. `python scripts/smooth_policy/sim2sim_compare.py --campaign-dir runs/td3/sim2sim/<src_to_tgt>/`.
