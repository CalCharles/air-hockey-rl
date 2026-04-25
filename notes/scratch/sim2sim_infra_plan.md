# Sim2sim infrastructure — planning doc

Handoff doc. If prompted with "continue on sim2sim" or "work on sim2sim
infrastructure" (or similar), start here.

## Status

**Infrastructure implemented 2026-04-20.** Committed pieces:

- `scripts/smooth_policy/eval_utils.py` — shared eval helpers factored out
  of `evaluate.py` (checkpoint unwrap, policy-class inference, policy
  construction, env-view builder). `evaluate.py` now imports from it.
- `scripts/smooth_policy/sim2sim_eval.py` — zero-shot eval harness.
  Writes `metrics.json` with per-episode returns; optional GIFs.
  Validated: eval of `runs/td3/sysid_params/upd_sweep/checkpoint_500000/model.pth`
  on its own source config gave mean_return 97.9 / max 181 over 10
  episodes, matching training tail metrics (94.3 / 96.6 tail10).
- `scripts/smooth_policy/sim2sim_compare.py` — aggregator. Walks
  `runs/td3/sim2sim/<src_to_tgt>/`, pulls zero-shot JSON + per-seed TB
  scalars, writes `comparison.md`.
- `scripts/smooth_policy/amp_history/configs/td3/sim2sim/` — four YAML
  stubs (`zero_shot`, `full_ft`, `from_scratch`, `residual`). The
  `residual` stub is **not runnable** until `ResidualActor` and the
  `load_mode: residual` branch land (see `residual_rl_plan.md`).

**Still TODO (left for next sessions):**
- Run an FT campaign (`full_ft` + `from_scratch` ≥2 seeds each) on the
  `hist2_motion0_to_combined` target so the comparison aggregator can be
  exercised end-to-end. `residual` is still blocked on `residual_rl_plan.md`.

**Done since 2026-04-20:**
- Authored `configs/new_juggle/sim2sim_combined.yaml` (combined kp/delay/
  paddle-radius/wall-cone perturbations) — 2026-04-25.
- Zero-shot evaluated `hist2_motion0/checkpoint_975000` on source vs.
  target (mean 148 → 96, −35%) — 2026-04-25.
- Single-knob sensitivity sweep (kp / action delay / wall cone) at
  `runs/td3/sim2sim/hist2_motion0_to_sweeps/` — found wall cone is
  ineffective as a perturbation lever — 2026-04-25.
- Promoted stable sections to formal doc:
  [`notes/docs/training/sim2sim.md`](../docs/training/sim2sim.md). Pointer
  added to `notes/docs/index.md` and `CLAUDE.md`.

## Scope

**In scope:** directory/naming conventions, zero-shot transfer eval harness,
wiring for regular FT on a target env, wiring for residual FT on a target
env (infra only — residual method design lives in
[`residual_rl_plan.md`](residual_rl_plan.md)), results directory layout,
metrics comparison workflow, doc placement.

**Out of scope:** the specific target sim parameters (gravity, damping,
density perturbations) — these are a later config-authoring step. The
residual RL method itself is covered by `residual_rl_plan.md`.

## Existing state — what gets removed

Survey on 2026-04-20 found **no active sim2sim code** in the repo:

- The phrase "sim2sim" appears only in `notes/scratch/residual_rl_plan.md`
  (design doc, no code).
- `scripts/domain_adaptation/` is legacy per CLAUDE.md and not imported by
  the active TD3 path. **Not part of sim2sim.** Leave it alone unless the
  user explicitly asks for cleanup.
- No "run checkpoint on a different config" eval harness exists beyond the
  general `scripts/smooth_policy/evaluate.py` (single-env rollouts + GIF).

**Conclusion:** clean slate. No removal step required. This plan is
additive.

## Terminology

- **Source sim**: env the base policy is trained on. Canonical choice is
  `sysid_best_params_hist4.yaml` (the real-world sysid ground truth, which
  is what `td3_recommended.yaml` already points to).
- **Target sim**: perturbed env used to test transfer and run online FT.
  Shares task / obs / action space with source; only physics parameters
  differ.
- **Zero-shot**: run base policy on target sim with no fine-tuning.
- **Regular FT** / **full FT**: resume TD3 training on target sim using the
  existing `load_mode: fine_tune` path. Actor + critics + optimizer state
  warm-start from the source checkpoint.
- **Residual FT**: frozen source actor + trainable residual actor, critic
  from scratch. See [`residual_rl_plan.md`](residual_rl_plan.md).
- **From-scratch**: TD3 on target with no warm-start. Lower baseline.

## 1. Config layout and naming

### 1.1 Sim configs

Target sim configs live alongside source configs under
`scripts/smooth_policy/amp_history/configs/new_juggle/`, with a `sim2sim_`
prefix so they group together in `ls`:

```
configs/new_juggle/
  sysid_best_params.yaml             # source / ground truth
  sysid_best_params_hist{3,4,5}.yaml # source w/ different history lens
  sim2sim_<tag>.yaml                 # target sim(s) — authored later
```

`<tag>` names the perturbation qualitatively (e.g.,
`sim2sim_heavy_puck.yaml`, `sim2sim_soft_paddle.yaml`,
`sim2sim_combined.yaml`). Each target inherits structurally from the
source it was derived from — only physics keys differ. When a target is
authored, add a `# Source: sysid_best_params_hist4.yaml` comment at the top
so provenance is explicit.

### 1.2 Training configs

Four TD3 config files live under
`scripts/smooth_policy/amp_history/configs/td3/sim2sim/`:

```
configs/td3/sim2sim/
  td3_sim2sim_zero_shot.yaml   # eval-only; not a training config (see §2)
  td3_sim2sim_full_ft.yaml     # load_mode: fine_tune
  td3_sim2sim_residual.yaml    # load_mode: residual  (see residual_rl_plan.md)
  td3_sim2sim_from_scratch.yaml # no warm-start
```

Each of these has:
- `config: <path to target sim config>`  ← the target env
- `model_path: <path to source-trained checkpoint>` (omitted for from-scratch)
- `load_mode:` as above
- `total_timesteps: 100_000` (FT budget, not full training)
- Otherwise forks from `td3_recommended.yaml`.

The source-trained checkpoint path is fixed per experiment campaign. When
the source checkpoint changes, update it in one place (these four files).

### 1.3 Why a separate `sim2sim/` subdir

Keeps the main `configs/td3/` directory readable. Sim2sim configs are
*experimental* (one campaign per source/target pair), while the files at
the top level are the canonical recommended/standard training configs.

## 2. Zero-shot evaluation harness

**New script:** `scripts/smooth_policy/sim2sim_eval.py`.

Responsibilities:
- Load a base checkpoint.
- Instantiate `AirHockeyEnv` from an arbitrary sim config (the target).
- Run N episodes deterministically, collect per-episode returns.
- Log to a results dir: `runs/td3/sim2sim/<source_tag>_to_<target_tag>/zero_shot/`.
- Write a `metrics.json` with: `n_episodes`, `mean_return`, `std_return`,
  `median_return`, `tail10`, `max_return`, `source_checkpoint_path`,
  `target_config_path`, `seed`.

CLI sketch:

```
python scripts/smooth_policy/sim2sim_eval.py \
  --checkpoint runs/td3/.../checkpoint.pt \
  --target-config .../new_juggle/sim2sim_<tag>.yaml \
  --n-episodes 50 --seed 0 \
  --out-dir runs/td3/sim2sim/<src>_to_<tgt>/zero_shot/
```

Reuse `evaluate.py`'s `_unwrap_eval_state_dict` and `_infer_policy_class_from_state_dict`
helpers (factor them to a shared module, e.g.
`scripts/smooth_policy/eval_utils.py`) rather than copy-paste.

Should **not** emit GIFs by default — pure metric collection. Add a
`--save-gif` flag for the qualitative-debug case.

## 3. Regular fine-tuning — reuse existing path

No training-code changes required. `td3_training.py` already supports
`load_mode: fine_tune` (line ~798-829, loads actor + qf1 + qf2 + optimizer
state, skips replay/runtime). For sim2sim:

- Point `config:` at the target sim config.
- Point `model_path:` at the source checkpoint.
- Reduce `total_timesteps` to the FT budget (100k default).
- Reduce `buffer_size` (default 20k — online FT is short).
- Reduce `learning_starts` (~2k — the base already explores the task).
- Reduce `exploration_noise` (~0.05 — preserve base behavior early).

Those values live in `td3_sim2sim_full_ft.yaml`. Nothing else is new.

## 4. Residual fine-tuning — hook only

Per [`residual_rl_plan.md`](residual_rl_plan.md), residual RL requires:
1. A new `ResidualActor` wrapper (`scripts/smooth_policy/residual_agent.py`).
2. A new `load_mode: residual` branch in `td3_training.py`.
3. A `td3_sim2sim_residual.yaml` config.

This infra plan only records **where** each piece lives and ensures naming
and run-dir conventions stay consistent with §1/§5. The method design and
implementation steps are in `residual_rl_plan.md` — do not duplicate them
here.

## 5. Results directory layout

One parent per (source, target) pair, one subdir per method:

```
runs/td3/sim2sim/
  <src_tag>_to_<tgt_tag>/
    zero_shot/                  # eval only, no training
      metrics.json
      eval_rollouts/            # optional GIFs
    full_ft/
      seed0/  seed1/  ...       # one per seed
    residual/
      seed0/  seed1/  ...
    from_scratch/
      seed0/  seed1/  ...
    comparison.md               # written by the metrics script (§6)
```

`<src_tag>` = short name of the source config (e.g. `sysid_hist4`).
`<tgt_tag>` = short name of the target config (e.g. `heavy_puck`).

**Rule (inherited from exploration plan):** never reuse a
`<...>/<method>/seedN` directory for two runs — if it exists,
`td3_training.py` will append `r1`/`r2`, which splits runs into sibling
dirs that are hard to aggregate.

## 6. Metrics aggregation

**New script:** `scripts/smooth_policy/sim2sim_compare.py` (or similar),
modeled after `notes/scratch/extract_expl_metrics.py`.

Input: a `<src_to_tgt>/` directory.
Output: a table (stdout + written to `comparison.md`) with rows per
method × seed and columns `ret@100k`, `tail10`, `tail50`, `max_ret`,
`pos_frac`, plus a summary row per method (mean ± std over seeds). Also
pulls the `zero_shot/metrics.json` single-row result.

Keep the metric definitions consistent with `extract_expl_metrics.py` so
numbers are comparable across campaigns.

## 7. Documentation placement

While this is in scratch, the handoff doc is this file. Once the infra is
implemented and at least one sim2sim campaign has been run end-to-end,
promote the stable sections (§1 config layout, §2 eval harness API, §5
results layout, §6 metrics) to a formal doc:

```
notes/docs/training/sim2sim.md
```

That doc should:
- Describe the source/target config convention.
- Document the eval harness CLI.
- Document the run-dir layout and the aggregator script.
- Link to `residual_rl_plan.md` (or its promoted form) for the residual
  method details.

Leave this scratch file as the campaign log + next-steps tracker after
promotion, same pattern as the exploration optimization plan.

Also update `CLAUDE.md` with a one-line pointer under "Active code paths"
or "Key docs" once the formal doc exists — not before.

## 8. Open questions (decide during implementation, not now)

- **How many seeds per method for a sim2sim campaign?** Tentative: 2
  (screening) per method, bumping to 3 for the chosen method before any
  real-robot attempt. Same rule as the exploration plan.
- **Should `sim2sim_eval.py` share env-construction code with the trainer?**
  Probably yes — factor a tiny `build_env_from_config(cfg_path)` helper
  rather than duplicating `AirHockeyEnv` kwarg assembly. Decide at
  implementation time based on how gnarly the existing construction is.
- **Where do we store the "canonical source checkpoint" for a campaign?**
  Option A: a fixed path written into the four `td3_sim2sim_*.yaml`
  files. Option B: a symlink `runs/td3/sim2sim/<campaign>/source_ckpt`.
  Option A is simpler; revisit if we run many campaigns against different
  base checkpoints.
- **Deterministic seeds for the zero-shot eval.** The env has stochastic
  resets. Decide whether the eval harness fixes both the torch seed and
  the env seed, and how many episodes is "enough" to distinguish methods.
  Start with 50 and only revisit if variance is washing out effect sizes.

## 9. What to do when resuming this work

Infra is in place (see Status). The remaining ordered steps:

1. Author a target sim config: copy
   `configs/new_juggle/sysid_best_params_hist4.yaml` to
   `configs/new_juggle/sim2sim_<tag>.yaml` and change only physics keys
   (e.g., `gravity`, `puck_damping`, `paddle_density`). Keep task /
   observation / action space identical. Add a provenance comment at
   the top.
2. Fill placeholders in the three sim2sim training YAMLs
   (`td3_sim2sim_full_ft.yaml`, `td3_sim2sim_from_scratch.yaml`,
   and — once residual is implemented — `td3_sim2sim_residual.yaml`):
   `config`, `model_path`, `log_parent_dir`, `run_name`, `seed`.
3. Run zero-shot first:
   `python scripts/smooth_policy/sim2sim_eval.py ...` (§2 of this doc).
   Establishes the sim2sim gap baseline.
4. Run `full_ft` and `from_scratch` for at least 2 seeds each. `residual`
   is blocked on `residual_rl_plan.md`'s implementation.
5. `python scripts/smooth_policy/sim2sim_compare.py --campaign-dir ...`
   to build `comparison.md`.
6. Only after sim2sim works end-to-end do we plan sim2real.

## 10. Files / locations (quick reference)

- This plan: `notes/scratch/sim2sim_infra_plan.md`
- Residual RL plan: `notes/scratch/residual_rl_plan.md`
- Source sim config (canonical): `scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist4.yaml`
- Target sim configs (future): `scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_*.yaml`
- Training configs (future): `scripts/smooth_policy/amp_history/configs/td3/sim2sim/td3_sim2sim_*.yaml`
- Eval harness (future): `scripts/smooth_policy/sim2sim_eval.py`
- Aggregator (future): `scripts/smooth_policy/sim2sim_compare.py`
- Existing single-env eval (reference): `scripts/smooth_policy/evaluate.py`
- Training entrypoint: `scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py`
- Recommended TD3 config: `scripts/smooth_policy/amp_history/configs/td3/td3_recommended.yaml`
- Metric extractor (pattern to mirror): `notes/scratch/extract_expl_metrics.py`
- Formal doc target (future): `notes/docs/training/sim2sim.md`
