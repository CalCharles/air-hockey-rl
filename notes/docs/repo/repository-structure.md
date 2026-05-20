# Repository structure

High-level map of the repository. Paths are relative to the repo root.

## Core package

| Path | Role |
|------|------|
| [`airhockey/`](../../../airhockey) | Installable Box2D + real-UR5 env package. Tasks are registered in `airhockey/__init__.py`. |
| [`airhockey/sims/airhockey_box2d.py`](../../../airhockey/sims/airhockey_box2d.py) | Box2D physics backend (primary for training). |
| [`airhockey/sims/air_hockey_real.py`](../../../airhockey/sims/air_hockey_real.py) | Real UR5 adapter. |
| [`airhockey/airhockey_tasks/`](../../../airhockey/airhockey_tasks/) | Goal-conditioned and reach/score tasks. |
| [`airhockey/airhockey_simple_tasks.py`](../../../airhockey/airhockey_simple_tasks.py) | Puck-juggle, puck-touch, etc. — task registry consumes this. |
| [`airhockey/airhockey_rewards/`](../../../airhockey/airhockey_rewards/) | Reward function implementations. |

## Training code

| Path | Role |
|------|------|
| [`scripts/td3/td3_training.py`](../../../scripts/td3/td3_training.py) | Sim TD3 trainer (entrypoint). |
| [`scripts/td3/{agent,deterministic_agent,residual_agent,encoder}.py`](../../../scripts/td3/) | TD3 actor networks. |
| [`scripts/td3/{evaluate,eval_utils}.py`](../../../scripts/td3/) | TD3 evaluation. |
| [`scripts/td3/helper/`](../../../scripts/td3/helper/) | Runtime support: replay, Q-network, exploration, real-world runners, checkpointing, metrics. |
| [`scripts/td3/extras/`](../../../scripts/td3/extras/) | Real-world entrypoints: `async_td3_real{,_eval,_teleop_eval}.py`. |
| [`scripts/td3/tests/`](../../../scripts/td3/tests/) | Pytest suite. |

## Other scripts

| Path | Role |
|------|------|
| [`scripts/real/`](../../../scripts/real/) | Real-robot rollout helpers (calibration, teleop, ArUco, homography, frozen-policy rollout). |
| [`scripts/visualization/`](../../../scripts/visualization/) | Trajectory rendering, teleop-segment visualization (used by both training and real-world stacks). |
| [`scripts/analysis/`](../../../scripts/analysis/) | Standalone analysis tools (e.g., occlusion-pattern analysis). |
| [`scripts/utils.py`](../../../scripts/utils.py) | Shared utilities (e.g., `save_tensorboard_plots`). |

## Configs

All YAMLs at the repo root under `configs/`:

| Path | Role |
|------|------|
| [`configs/new_juggle/`](../../../configs/new_juggle/) | Sim env configs. **For sim2sim / sim2real transfer**: `zeroshot_ablations/sim_paramrand_pm25.yaml` (env-param DR, canonical). For source-sim-only training / ablations: `sysid_best_params{,_hist2}.yaml`. Sim2sim warp targets: `sim2sim_*.yaml`. |
| [`configs/td3/`](../../../configs/td3/) | TD3 sim training args. **Canonical sim2sim / sim2real source-policy training**: `zeroshot_paramrand/td3_paramrand_pm25.yaml` (launched via `scripts/td3/td3_training_dr.py`). Source-sim-only / ablations: `td3_recommended_top50_hist2.yaml`. |
| [`configs/td3/sim2sim/`](../../../configs/td3/sim2sim/) | Sim2sim residual fine-tune recipes (used on top of a trained source policy: canonical: `warp075_p30_residual/phaseC_actor2_1M.yaml` + phaseD variants; small-gap: `td3_sim2sim_residual.yaml`). |
| [`configs/td3_real_world/`](../../../configs/td3_real_world/) | Real-robot residual fine-tune args (`td3_residual.yaml`). |
| [`configs/real_configs/`](../../../configs/real_configs/) | Real-robot rollout / mouse-teleop configs. |

## Models

| Path | Role |
|------|------|
| [`latest_models/canonical/hist2_motion0_v2/`](../../../latest_models/canonical/hist2_motion0_v2/) | Historical sim-pretrained source policy (trained with the deprecated engineered-randomization stack pre-2026-05-11). Loadable but **not the canonical source for new sim2sim / sim2real work** — retrain via the paramrand path. |
| [`latest_models/canonical/hist2_motion0/`](../../../latest_models/canonical/hist2_motion0/) | Deprecated predecessor; on disk for reproducibility. |
| [`latest_models/ablations/`](../../../latest_models/ablations/) | 16 CoRL 2026 deployment-ready ablation checkpoints. |

## Project metadata

| Path | Role |
|------|------|
| [`CLAUDE.md`](../../../CLAUDE.md) | Agent context, active code paths, conventions. |
| [`README.md`](../../../README.md) | Repo-level intro + quickstart. |
| [`notes/docs/`](../../) | Formal documentation (this tree). |
| [`notes/scratch/`](../../scratch/) | Experiment logs and design notes (read-only history). |
| [`paper/`](../../../paper/) | CoRL 2026 LaTeX setup. |
| [`assets/`](../../../assets) | PNG sprites for Box2D renderer + real-camera homography matrices in `assets/real/`. |
| [`pyproject.toml`](../../../pyproject.toml), [`uv.lock`](../../../uv.lock) | Dependencies and install layout. |

## Run artifacts (gitignored)

These directories live on disk for local runs but are not tracked: `runs/`, `results/`, `trained_models/`, `eval_gifs/`, `real_runs/`, `shared/`, `sysid/`, `dataset_management/`, `tests/`, `wandb/`, `gifs/`, `plots/`, `datasets/`.
