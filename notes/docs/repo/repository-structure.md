# Repository structure

High-level map of the repository. Paths are relative to the repo root.

## Core package

| Path | Role |
|------|------|
| [`airhockey/`](../../../airhockey) | Installable package: Gym-style environments, simulator adapters (`sims/`), tasks, rewards, utilities. |

## Training and experiments

| Path | Role |
|------|------|
| [`scripts/`](../../../scripts) | Active training code (e.g. [`scripts/smooth_policy`](../../../scripts/smooth_policy)), real-robot scripts, utilities. |
| [`configs/`](../../../configs) | YAML configs for scripts and environments. |
| [`agents/`](../../../agents) | Agent-related assets or helpers (project-specific). |
| [`offline_rl_algorithms/`](../../../offline_rl_algorithms) | Offline RL code used by some experiment paths. |

## Real robot and data

| Path | Role |
|------|------|
| [`scripts/real/`](../../../scripts/real) | Teleoperation and real-hardware entrypoints (see also [`environments/real-world/overview.md`](../environments/real-world/overview.md)). |
| [`dataset_management/`](../../../dataset_management) | Dataset handling utilities. |
| [`real_runs/`](../../../real_runs) | Runtime/output artifacts for real sessions (if present in your checkout). |

## Project metadata

| Path | Role |
|------|------|
| [`notes/`](../../) | Formal docs under `notes/docs/`, scratch under `notes/scratch/`. |
| [`assets/`](../../../assets) | Images and media for README and docs. |
| [`pyproject.toml`](../../../pyproject.toml), [`uv.lock`](../../../uv.lock) | Dependencies and install layout (`uv sync`, editable install). |

## Legacy / optional roots

Top-level scripts such as `airhockey2d.py`, `train.py`, and `render.py` (see [README](../../../README.md)) are older entrypoints; prefer the packaged `airhockey` API and `scripts/` flows for new work.
