# Sim2sim transfer testing

A *sim2sim* campaign trains a policy on one Box2D sim ("source") and tests how it transfers to a perturbed Box2D sim ("target") that shares the task / observation / action space but differs in physics. It is the rehearsal step before sim2real, and the home for fine-tuning experiments (full FT, residual FT, from-scratch baseline).

This page documents the harness, the layout, the campaign run on the `hist2_motion0` checkpoint (2026-04-25), and what we learned about which perturbations actually move the needle.

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

- **Source sim**: any `configs/new_juggle/sysid_best_params*.yaml`. Currently the canonical source is `sysid_best_params_hist4.yaml`; ablations use the matching `_hist2.yaml` / `_hist3.yaml` / `_hist5.yaml`.
- **Target sim**: lives next to source as `configs/new_juggle/sim2sim_<tag>.yaml`. Inherits structurally from one source — only physics keys differ. First line is `# Source: <source_yaml>` for provenance. Each modified key has an inline `# PERTURBED: ...` comment.
- **Training configs**: under `configs/td3/sim2sim/`. Four files (`zero_shot`, `full_ft`, `residual`, `from_scratch`); only `config:` / `model_path:` / `log_parent_dir:` change per campaign. The `residual` config is a stub until `ResidualActor` lands (see residual_rl_plan).

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
| Args file | `td3_recommended.yaml` |
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
| `…/configs/td3/sim2sim/td3_sim2sim_residual.yaml` | tested defaults: `residual_scale=0.05`, `q_updates=1`, `q_lr=3e-4`, `actor_updates_per_iteration=1`, primitives off (kept one nominal weight non-zero so the selector accepts the distribution at init). `q_updates`/`q_lr` reflect the 2026-04-25 drift-study update — see "Drift study" section below. |
| `…/configs/td3/sim2sim/td3_sim2sim_full_ft.yaml` | tested defaults: `policy_lr=3e-5`, `q_lr=1e-4`, `q_updates=4`, `actor_updates_per_iteration=1`, primitives off, `fine_tune_replay_keep=10000` |

### Open follow-ups

- **Best-of-eval-checkpoint tracker** in `td3_training.py`: compute deterministic eval at each checkpoint and save the best so far as `model.pth`. Without it, every campaign needs the post-hoc per-checkpoint eval we did here.
- **Multi-seed verification** at the chosen peak step (now 30k for residual w/ updated defaults, 100k for full_ft). Single-seed numbers above are indicative, not statistically tight.
- **Apply the same drift study to full_ft.** The drift study below was residual-only. full_ft warm-starts the source critic, so its dynamics are different — the same UTD/q_lr knobs may or may not move the needle.

---

## Drift study — residual (2026-04-25)

Investigated *why* the residual v2 hit mean 106.84 @ step 50k but degraded to 84.7 by 100k (-21% drift). Hypothesis space: narrow data, high learning rate, high update-to-data, success-buffer bias.

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

### Single-knob ablation (each row vs. v2 baseline; same seed=0; same target)

Each row reports the best deterministic mean across saved checkpoints (n=50 episodes, seed=0); tail10 is at *that same step* (not the absolute best tail10).

| variant | knob change | peak mean | tail10 @ peak | peak step | final mean | drift |
|---|---|---:|---:|---|---:|---:|
| zero-shot | (no FT) | — | 87.8 | — | 95.78 | — |
| v2 baseline | (none) | 106.8 | 127.2 | 50k | 84.7 | -21% |
| bigger_buffer | success/failure 6k/14k → 30k/70k | 103.9 | 101.2 | 90k | 84.0 | -19% |
| lower_qlr | q_lr 1e-3 → **3e-4** | 102.1 | **132.7** | 60k | 93.7 | -8% |
| lower_utd | q_updates 4 → **1** (UTD) | 98.4 | 113.3 | 20k | 95.2 | **-3%** |
| low_succ_frac | critic_success_sample_fraction 0.3 → 0 | 100.2 | 92.5 | 40k | 87.5 | -13% |
| **combo (UTD=1 + q_lr=3e-4)** | both above | **109.3** | 111.0 | **30k** | **95.7** | -13% |

Findings:

1. **The "wider data" hypothesis is wrong.** `bigger_buffer` (5x main + buffers wide enough to hold all 100k samples) is essentially baseline. The drift is structural to the optimization regime, not the data window.
2. **UTD is the dominant driver.** `lower_utd` flattens Q (`Q1@100k`: 1.12 → **0.26**) and reduces drift to -3 %, but underfits the critic — peak mean only 98.4.
3. **Lower q_lr halves drift on its own.** q_lr ÷ 3.3 also gives the highest tail10 of any single-knob run (**132.7**, beating baseline's 127.2). Q still climbs but slower.
4. **Removing success-buffer bias (`critic_success_sample_fraction=0`) is partial.** Helps drift (-13 %) but doesn't kill Q runaway (Q still 0.20 → 1.05 over 100k); current critic still chases bellman.
5. **Combining UTD=1 + q_lr=3e-4 wins overall.** Highest peak mean (109.3 > baseline 106.8), reaches peak in **40 % less compute** (30k vs 50k), final mean stays at 95.7 (above zero-shot 95.78). Drift is still -13 % so per-checkpoint eval remains required, but the trough is much shallower.

### Default change

Updated `td3_sim2sim_residual.yaml`:

| knob | old | new |
|---|---:|---:|
| `q_updates` | 4 | **1** |
| `q_lr` | 1e-3 | **3e-4** |

`residual_scale=0.05`, primitives off, `fine_tune_replay_keep` etc. all unchanged.

### Reproducibility

Diagnostic configs at `scripts/smooth_policy/amp_history/configs/td3/sim2sim/diagnose/{bigger_buffer,lower_qlr,lower_utd,low_success_frac,combo_utd_qlr}.yaml`. Run dirs at `runs/td3/sim2sim/hist2_motion0_to_combined/residual_diagnose/<variant>/seed0[r1]/`. Per-checkpoint deterministic eval results in `eval_combined_ckpt_*/metrics.json` and `eval_combined_final/metrics.json` under each.

### Open follow-ups (specific to this study)

- Apply the same UTD=1 + q_lr=3e-4 to `full_ft` and check whether it also gets a higher peak / less drift.
- **Multi-seed (3+)** confirmation that combo's peak (109.3) and final (95.7) replicate. Single-seed numbers in the table are indicative.
- A `success_top_fraction=1.0` (FIFO success bucket — no top-20% retention) variant would directly test the "museum" mechanism. Skipped to keep this study to single-knob + one combo.

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
5. Fine-tune: fill placeholders in `td3_sim2sim_{full_ft,residual,from_scratch}.yaml` (`config`, `model_path`, `log_parent_dir`, `run_name`, `seed`), launch ≥2 seeds each. The residual and full_ft yamls in repo carry the campaign-tested defaults (residual `scale=0.05` + `q_updates=1` + `q_lr=3e-4`, and full_ft `lr÷10`); revisit those if your source policy is much weaker / target gap is much wider. **Always evaluate intermediate checkpoints** — final-step eval is unsafe (see "Drift study" and "Fine-tune campaign" sections above).
6. `python scripts/smooth_policy/sim2sim_compare.py --campaign-dir runs/td3/sim2sim/<src_to_tgt>/`.
