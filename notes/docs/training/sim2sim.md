# Sim2sim transfer testing

A *sim2sim* campaign trains a policy on one Box2D sim ("source") and tests how it transfers to a perturbed Box2D sim ("target"). It is the rehearsal step before sim2real and the home for fine-tuning experiments.

> **Canonical training approach for sim2sim / sim2real (2026-05-11 onward).** Source policies that need to transfer should be trained with **environment-parameter domain randomization** — paddle_density / puck_damping / gravity drawn uniform per-reset (±25 % of sysid). Launch via `scripts/td3/td3_training_dr.py` with sim config [`configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml`](../../../configs/new_juggle/zeroshot_ablations/sim_paramrand_pm25.yaml) and args [`configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml`](../../../configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml). The earlier "engineered randomization" strategy (per-collision strength/direction jitter, action-force attenuation, delay jitter, paddle-density fluctuation) was deprecated and the mechanisms physically removed from the env. The residual fine-tune recipes below are still useful for adapting a source policy to a *specific* perturbed target, but the source they consume should now be a paramrand-trained policy, not an engineered-DR one.

## Canonical big-gap target

[`configs/new_juggle/sim2sim_warp075_p30.yaml`](../../../configs/new_juggle/sim2sim_warp075_p30.yaml) — paddle −30% (mass-preserved) + sine-y puck-obs warp 0.075, all delays / hist_len / restitutions matched to source. Zero-shot return ≈ 48; from-scratch peak ≈ 112 at 400k.

Sibling targets on the same axis (used by the canonical phase-C/D residual recipes):

| Target | Warp · paddle | zs |
|--------|---------------|-----|
| [`sim2sim_warp075_p10.yaml`](../../../configs/new_juggle/sim2sim_warp075_p10.yaml) | 0.075 · −10% | 49 |
| [`sim2sim_warp075_p30.yaml`](../../../configs/new_juggle/sim2sim_warp075_p30.yaml) | 0.075 · −30% | 48 |
| [`sim2sim_warp100_p30.yaml`](../../../configs/new_juggle/sim2sim_warp100_p30.yaml) | 0.10 · −30% | (lower) |

The small-gap target [`sim2sim_combined.yaml`](../../../configs/new_juggle/sim2sim_combined.yaml) — paddle and dynamics deltas without the sine warp — is used by the small-gap recipe ([`td3_sim2sim_residual.yaml`](../../../configs/td3/sim2sim/td3_sim2sim_residual.yaml)).

Sim2sim configs use the same sim env schema as training; the file has a `# Source: <source_yaml>` header for provenance and per-key `# PERTURBED: …` annotations on every changed parameter.

## Source policy

The historical source policy used by the residual recipes below is [`latest_models/canonical/hist2_motion0_v2/`](../../../latest_models/canonical/hist2_motion0_v2/) (trained on `sysid_best_params_hist2.yaml` with the now-deprecated engineered randomization; eval mean 169.72 on the source sim). The earlier `hist2_motion0/` predecessor is also kept on disk for reproducibility but should not be referenced in new work.

**For new work**, train a fresh source policy with env-parameter randomization (see the banner at the top of this doc). The hist2_motion0_v2 checkpoint remains loadable — the env silently ignores its now-unknown config keys — but it represents the deprecated regime.

## Fine-tuning recipes

See [`residual-rl-recipe.md`](residual-rl-recipe.md) for full details. Quick reference:

| Gap size | Recipe | Config |
|----------|--------|--------|
| **Big** (~30% zs drop) | CQL α=20 + `actor_updates_per_iteration=2` | [`configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml`](../../../configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml) |
| **Big, mild paddle** (−10%) | Same recipe, p10 target | [`phaseD_actor2_p10_1M.yaml`](../../../configs/td3/sim2sim/warp075_p30_residual/phaseD_actor2_p10_1M.yaml) |
| **Big, warp 0.10** | Same recipe, `actor_updates_per_iteration=4` | [`phaseD_actor4_w10_1M.yaml`](../../../configs/td3/sim2sim/warp075_p30_residual/phaseD_actor4_w10_1M.yaml) |
| **Small** (<10% zs drop) | `recency_top50` (no CQL) | [`td3_sim2sim_residual.yaml`](../../../configs/td3/sim2sim/td3_sim2sim_residual.yaml) |

Recipe boundary: works through warp 0.10 (with actor=4); warp 0.125 is intractable.

## Running a campaign

### Zero-shot evaluation

Use the standard trainer in eval mode. (Eval works the same way regardless of how the source was trained; the `--args-file` only needs to match the architecture used at training time.)

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/zeroshot_paramrand/td3_paramrand_pm25.yaml \
  --eval-mode \
  --config configs/new_juggle/sim2sim_warp075_p30.yaml \
  --model-path runs/td3/<paramrand_src_run>/checkpoint_<step>/model.pth \
  --total-timesteps 12500 \
  --num-envs 1
```

### Residual fine-tune

```bash
.venv/bin/python -m scripts.td3.td3_training \
  --args-file configs/td3/sim2sim/warp075_p30_residual/phaseC_actor2_1M.yaml \
  --num-envs 1
```

Before launching, edit the recipe YAML's `config:`, `model_path:`, `log_parent_dir:`, `run_name:`, and `seed:`. Run ≥ 3 seeds.

### Per-checkpoint eval and best-checkpoint selection

The trainer writes a checkpoint every 50k steps. Iterate over `runs/td3/sim2sim/<tag>/seedN/checkpoint_*/` dirs, eval each in `--eval-mode`, then ship the checkpoint with the highest `mean_return`.

## Authoring a new target

1. Copy a `sysid_best_params*.yaml` source.
2. Change only the physics keys you want to perturb. Mark each one with `# PERTURBED: <reason>`.
3. Add `# Source: configs/new_juggle/<source>.yaml` as the first line for provenance.
4. Zero-shot the canonical source policy against it to set a baseline (`mean_return ± std`).
5. If the zero-shot drop is < 10%, the small-gap recipe applies. If it's ~20–30%, use the canonical big-gap recipe. If from-scratch on the target plateaus far below zero-shot, the target is structurally untrainable and the gap is too aggressive — back off the perturbations.

## Results directory layout

```
runs/td3/sim2sim/<src_tag>_to_<tgt_tag>/
  zero_shot/        metrics.json + optional eval_rollouts/*.gif
  residual/         seed0/ seed1/ ...   (td3_training.py output dirs)
  from_scratch/     seed0/ seed1/ ...
  full_ft/          seed0/ seed1/ ...   (rarely used)
```

`<src_tag>` and `<tgt_tag>` are short qualitative names (e.g. `hist2_motion0v2`, `warp075_p30`). Each `seedN` directory is a self-contained `td3_training.py` output — never reuse one between runs (the trainer will append `r1` / `r2` and split runs into sibling directories).
