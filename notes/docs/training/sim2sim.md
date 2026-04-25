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
5. Fine-tune: fill placeholders in `td3_sim2sim_full_ft.yaml` / `td3_sim2sim_from_scratch.yaml` (`config`, `model_path`, `log_parent_dir`, `run_name`, `seed`), launch ≥2 seeds each. `residual` is blocked on `ResidualActor` landing.
6. `python scripts/smooth_policy/sim2sim_compare.py --campaign-dir runs/td3/sim2sim/<src_to_tgt>/`.
