# RMA sim2real baseline (`scripts/rma/`)

Self-contained implementation of **RMA — Rapid Motor Adaptation** (Kumar, Fu,
Pathak, Malik; RSS 2021) as a sim2real baseline for the CoRL-2026 paper, on the
project's canonical ±25 % physics-parameter domain randomization, with the
project's TD3 recipe as the phase-1 RL algorithm. Nothing under `scripts/td3/`
imports this package. Full write-up (design, fidelity to the paper, the
TD3-instead-of-PPO assessment, metrics): [`notes/docs/training/rma-baseline.md`](../../notes/docs/training/rma-baseline.md).

| File | What |
|---|---|
| `networks.py` | `EnvFactorEncoder` μ(e)→z, `RMAActor` π(x, a₋₁, μ(e)), `AdaptationModule` φ(history)→ẑ (1-D conv stack of the paper), `AdaptedActor` π∘φ, `HistoryActor` (long-history TD3 control: the deployable RMA architecture trained end-to-end without privileged input), per-step feature helpers |
| `env_wrapper.py` | `EnvParamNormalizer` (DR ranges → [-1, 1]); `RMAEnvVector` — single-env vector wrapper that appends the normalised privileged factors e_t to every observation; `HistoryEnvVector` — appends the flattened 50-step (x, a) window instead (history control). Both take `goal=True` for goal-conditioned tasks (inner env = the HER trainer's `GoalEnvVector`, flat `[observation, goal]` + the relabelling infos) |
| `history.py` | `StepHistoryBuffer` (online window with repeat-first padding), `AdaptationDataset` (episodes → windows, rolling capacity, train/holdout) |
| `train_base_policy.py` | **Phase 1**: canonical TD3 loop with π + μ trained end-to-end, asymmetric critic Q(x, e, a). `rma_mode: history` → the long-history control baseline (`configs/rma/history_juggle_dr.yaml`). Sim configs with `return_goal_obs: true` (puck-goal tasks) get hindsight experience replay (`her_k` …, same keys as `td3_training_her`) in both modes |
| `train_adaptation_module.py` | **Phase 2**: iterative on-policy supervised regression of φ to μ(e); latent-fit metrics; paired policy evals |
| `evaluate.py` | Agents (`privileged`, `adapted`, `nominal`, `td3_dr`) and the fixed-env multi-env evaluation (same env set / seeds / JSON schema as `td3_training_dr`); `FlatGoalEnv` / `make_eval_env` give goal-conditioned tasks flat `[observation, goal]` rollouts |
| `checkpoint_eval.py` | Background per-checkpoint eval subprocess for phase 1 |
| `bundle.py` | `rma_meta.json` / `adaptation_module.pth` save + load |
| `eval_paired.py` | Paired ID + OOD evaluation of any set of RMA phase-2 / history / plain-DR runs on the same envs and episode seeds → `paired_eval.md` |
| `summarize.py` | Markdown tables: phase-1 curve vs baseline, phase-2 fit history, final paired eval |
| `plot_results.py`, `analyze_latent_usage.py` | Report figure; how much the base policy uses z per checkpoint |
| `tests/` | `pytest scripts/rma/tests -q` (CPU, ~15 s) |

## Run

```bash
# phase 1 (2M steps, ~1-2 h on one GPU; per-checkpoint eval in the background)
.venv/bin/python -m scripts.rma.train_base_policy \
    --args-file configs/rma/rma_juggle_dr_phase1.yaml --device cuda:0 \
    --log-parent-dir runs/rma/juggle_dr_phase1_seed0

# phase 2 (400k on-policy env steps, ~15-30 min, CPU rollouts + GPU regression)
.venv/bin/python -m scripts.rma.train_adaptation_module \
    --args-file configs/rma/rma_juggle_dr_phase2.yaml \
    --phase1-dir runs/rma/juggle_dr_phase1_seed0 --device cuda:0 \
    --log-parent-dir runs/rma/juggle_dr_phase1_seed0/phase2

# long-history control (same recipe, no privileged input)
.venv/bin/python -m scripts.rma.train_base_policy \
    --args-file configs/rma/history_juggle_dr.yaml --device cuda:1 \
    --log-parent-dir runs/rma/history_juggle_dr_seed0

# paired ID + OOD eval of everything on the same episode seeds
.venv/bin/python -m scripts.rma.eval_paired \
    --rma-phase2 runs/rma/juggle_dr_phase1_seed0/phase2 \
    --history-runs runs/rma/history_juggle_dr_seed0 \
    --td3-runs runs/td3/tasks_20260904/dr/juggle_dr --eps-per-env 20 --out-dir runs/rma/paired_eval

# tables
.venv/bin/python -m scripts.rma.summarize \
    --phase1-dirs runs/rma/juggle_dr_phase1_seed0 \
    --baseline-dirs runs/td3/tasks_20260904/dr/juggle_dr \
    --phase2-dirs runs/rma/juggle_dr_phase1_seed0/phase2 --out runs/rma/summary.md
```

Phase 1 also runs through the batch runner: `scripts/td3/run_experiments.py --mode rma` (or `--mode auto`, which picks it from the `rma_latent_dim` key).

## Outputs

Phase 1 run dir = a canonical TD3 run dir (`model.pth` is the `RMAActor` state dict — encoder included — plus `rma_meta.json`; `checkpoint_*/multi_env_eval.json` holds the privileged policy's eval). Phase 2 run dir: `adaptation_module.pth`, `phase2_metrics.jsonl` (one row per iteration), `eval_iterNNN/` and `eval_final/` (`multi_env_eval.json`, `multi_env_eval_ood.json`, GIFs), `phase2_summary.json`.
