#!/usr/bin/env python3
"""
analyze_transformer_context_sweep.py

Standalone analysis script (no subprocess / no shelling out to
td3_training_dr). Walks a `runs/td3/` directory tree of `sweep_transformer_*`
training runs, picks the single best checkpoint per seed (by
multi_env_eval.json's aggregate.mean_return_across_envs), runs
compare_performance_ID_OOD() on that checkpoint in-process for ID and one or
more OOD distributions, aggregates across seeds, and produces a table +
a two-panel ("use transformer" vs "no transformer") bar chart per OOD
distribution, with context length on the x-axis and a vanilla-TD3 baseline
included for comparison.

Run from the repo root (so `scripts.*` / `airhockey` imports resolve), e.g.:

    .venv/bin/python -m scripts.td3.analyze_transformer_context_sweep \
        --runs-dir runs/td3 \
        --baseline-run-dir runs/td3/sanity_check \
        --gravity-ood -1.2 -0.8 \
        --paddle-density-ood 3500 4500 \
        --puck-damping-ood 0.05 0.15 \
        --n-workers 56

PARALLELIZATION
---------------
Each call to compare_performance_ID_OOD is independent (no shared state,
no ordering dependencies), so we fan out all (prefix, seed) eval jobs across
a multiprocessing.Pool.  On a 144-core Grace node the whole sweep finishes
in roughly the time of a single eval call.

Worker count is set by --n-workers (default: all logical CPUs via
os.cpu_count()).  Each worker writes its own subdirectory under out_dir
(unchanged from the serial version), so there are no filesystem races.
All results are returned through the Pool as plain dicts; the aggregation
step runs in the main process after all workers finish.

Spawn context is used (not fork) because PyTorch models and gym envs are not
safe to fork after initialisation.  This means each worker re-imports the
repo modules and reconstructs the model from scratch, which is exactly what
the serial version was doing per call anyway.

DEBUG CHECKPOINTS
-----------------
Search this file for "DEBUG CHECKPOINT" to find the spots where you can drop
a `breakpoint()` or print statement to inspect:
  (1) which run folders were discovered and how they were grouped by prefix
  (2) which (args.yaml, training_state.pth) pair was selected as "best" per
      seed, and what mean_return_across_envs each candidate had
  (3) the actual param dicts (ID/OOD ranges) being passed into
      compare_performance_ID_OOD for a given run
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import re
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Repo imports — these only work when run from the repo root / with the repo
# on PYTHONPATH. Kept inside a try/except so this file can still be parsed
# and its pure-python helpers (parsing, aggregation) unit-tested without the
# real torch model code present.
# ---------------------------------------------------------------------------
try:
    from airhockey import AirHockeyEnv
    from scripts.td3.deterministic_agent import DeterministicAgent
    from scripts.transformer.context_encoder import ContextEncoder
    from scripts.transformer.compare_performance_ID_OOD import compare_performance_ID_OOD
    _IMPORTS_OK = True
except ImportError as e:
    _IMPORTS_OK = False
    _IMPORT_ERROR = e

HISTORY_ENTRY_DIM = 4

# Used as the dict key for the vanilla-TD3 baseline in the results structure.
BASELINE_KEY = "__baseline_vanilla_td3__"


# ---------------------------------------------------------------------------
# Step 1: Discover run folders and group by prefix
# ---------------------------------------------------------------------------

# Matches e.g.:
#   sweep_transformer_false_ctx_16_seed_1
#   sweep_transformer_false_ctx_32r1_seed_3
#   sweep_transformer_true_ctx_14          (no seed suffix)
#   sweep_transformer_true_ctx_14r1_seed_4
_PREFIX_RE = re.compile(
    r"^(sweep_transformer_(?:true|false)_ctx_\d+)(?:r\d+)?(?:_seed_\d+)?$"
)


def parse_run_prefix(folder_name: str) -> str:
    """Collapse a run-folder name down to its structural prefix, e.g.
    'sweep_transformer_false_ctx_32r1_seed_3' -> 'sweep_transformer_false_ctx_32'.

    Falls back to splitting on '_seed_' if the strict pattern doesn't match,
    and finally returns the folder name unchanged if neither works (so an
    unexpected naming convention doesn't get silently dropped — it shows up
    as its own singleton group instead, which is easy to spot in the debug
    printout).
    """
    match = _PREFIX_RE.match(folder_name)
    if match:
        return match.group(1)
    if "_seed_" in folder_name:
        return folder_name.split("_seed_")[0]
    return folder_name


def parse_prefix_axes(prefix: str) -> tuple[bool, int] | None:
    """Extract (use_transformer, context_len) from a parsed prefix like
    'sweep_transformer_false_ctx_16'. Returns None if it doesn't match the
    expected sweep naming (e.g. a baseline run folder)."""
    m = re.match(r"^sweep_transformer_(true|false)_ctx_(\d+)$", prefix)
    if not m:
        return None
    use_transformer = (m.group(1) == "true")
    ctx = int(m.group(2))
    return use_transformer, ctx


def discover_run_folders(runs_dir: str, name_filter: str = "sweep_transformer") -> dict[str, list[str]]:
    """Scan `runs_dir` for folders containing `name_filter` in their name and
    group their *full paths* by structural prefix.

    Returns: { prefix: [full_path_to_seed_folder, ...] }
    """
    grouped: dict[str, list[str]] = defaultdict(list)

    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"runs_dir does not exist: {runs_dir}")

    for folder in sorted(os.listdir(runs_dir)):
        if name_filter not in folder:
            continue
        full_path = os.path.join(runs_dir, folder)
        if not os.path.isdir(full_path):
            continue
        prefix = parse_run_prefix(folder)
        grouped[prefix].append(full_path)

    # ------------------------------------------------------------------
    # DEBUG CHECKPOINT (1): run discovery + grouping
    #
    # Uncomment to inspect exactly which folders were found and how they
    # were grouped into prefixes before any checkpoint scanning happens:
    #
    #   for prefix, paths in sorted(grouped.items()):
    #       print(f"[DISCOVER] prefix={prefix!r}  n_seed_folders={len(paths)}")
    #       for p in paths:
    #           print(f"    {p}")
    #   breakpoint()
    # ------------------------------------------------------------------

    return dict(grouped)


# ---------------------------------------------------------------------------
# Step 2: For each seed folder, find the best checkpoint by
#          multi_env_eval.json -> aggregate.mean_return_across_envs
# ---------------------------------------------------------------------------

@dataclass
class CheckpointCandidate:
    seed_folder: str
    checkpoint_dir: str
    checkpoint_step: int
    mean_return_across_envs: float
    args_yaml_path: str
    config_yaml_path: str
    training_state_path: str
    transformer_path: str | None  # None if not use_transformer


def _read_multi_env_eval_return(checkpoint_dir: str) -> float | None:
    """Read aggregate.mean_return_across_envs from
    checkpoint_dir/multi_env_eval.json. Returns None (with a printed warning)
    if the file is missing or doesn't have the expected shape, rather than
    raising — a single malformed checkpoint shouldn't kill the whole sweep.
    """
    path = os.path.join(checkpoint_dir, "multi_env_eval.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [WARN] could not parse {path}: {e}")
        return None

    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict) or "mean_return_across_envs" not in aggregate:
        print(
            f"  [WARN] {path} missing aggregate.mean_return_across_envs "
            f"(top-level keys found: {list(data.keys())}); skipping this checkpoint."
        )
        return None
    return float(aggregate["mean_return_across_envs"])


def find_best_checkpoint_per_seed(seed_folder: str) -> CheckpointCandidate | None:
    """Scan all checkpoint_* subdirs of `seed_folder`, read each
    multi_env_eval.json, and return the CheckpointCandidate with the highest
    mean_return_across_envs. Returns None if the seed folder has no usable
    checkpoints (e.g. the run failed before any checkpoint_interval was hit).
    """
    candidates: list[CheckpointCandidate] = []

    if not os.path.isdir(seed_folder):
        return None

    for entry in sorted(os.listdir(seed_folder)):
        if not entry.startswith("checkpoint_"):
            continue
        ckpt_dir = os.path.join(seed_folder, entry)
        if not os.path.isdir(ckpt_dir):
            continue

        training_state_path = os.path.join(ckpt_dir, "training_state.pth")
        args_yaml_path = os.path.join(ckpt_dir, "args.yaml")
        config_yaml_path = os.path.join(ckpt_dir, "config.yaml")

        if not os.path.exists(training_state_path):
            print(f"  [SKIP] {ckpt_dir}: no training_state.pth (likely failed/partial checkpoint)")
            continue
        # Per-checkpoint args.yaml/config.yaml are preferred (a run's
        # hyperparams shouldn't change between checkpoints, but reading them
        # per-checkpoint is robust to runs resumed with different args).
        # Fall back to the seed-folder-level copy if a checkpoint dir doesn't
        # have its own (matches save_full_checkpoint's behavior, which writes
        # args.yaml/config.yaml into every checkpoint dir it creates).
        if not os.path.exists(args_yaml_path):
            args_yaml_path = os.path.join(seed_folder, "args.yaml")
        if not os.path.exists(config_yaml_path):
            config_yaml_path = os.path.join(seed_folder, "config.yaml")
        if not os.path.exists(args_yaml_path) or not os.path.exists(config_yaml_path):
            print(f"  [SKIP] {ckpt_dir}: missing args.yaml/config.yaml (checked checkpoint dir and seed folder)")
            continue

        mean_return = _read_multi_env_eval_return(ckpt_dir)
        if mean_return is None:
            continue

        step_match = re.match(r"checkpoint_(\d+)$", entry)
        step = int(step_match.group(1)) if step_match else -1

        transformer_path = os.path.join(ckpt_dir, "transformer.pth")
        if not os.path.exists(transformer_path):
            transformer_path = None

        candidates.append(CheckpointCandidate(
            seed_folder=seed_folder,
            checkpoint_dir=ckpt_dir,
            checkpoint_step=step,
            mean_return_across_envs=mean_return,
            args_yaml_path=args_yaml_path,
            config_yaml_path=config_yaml_path,
            training_state_path=training_state_path,
            transformer_path=transformer_path,
        ))

    # ------------------------------------------------------------------
    # DEBUG CHECKPOINT (2): best-checkpoint-per-seed selection
    #
    # Uncomment to see every candidate checkpoint considered for this seed
    # folder and which one wins, before moving on to the next seed:
    #
    #   print(f"[BEST-CKPT] seed_folder={seed_folder}")
    #   for c in candidates:
    #       print(f"    step={c.checkpoint_step:>8}  mean_return={c.mean_return_across_envs:8.2f}  dir={c.checkpoint_dir}")
    #   breakpoint()
    # ------------------------------------------------------------------

    if not candidates:
        return None
    return max(candidates, key=lambda c: c.mean_return_across_envs)


def build_seed_index(grouped_folders: dict[str, list[str]]) -> dict[str, list[CheckpointCandidate]]:
    """For every prefix, find the best checkpoint of every seed folder.
    Returns { prefix: [CheckpointCandidate, ...] } with exactly one entry
    per seed folder that had at least one usable checkpoint.
    """
    seed_index: dict[str, list[CheckpointCandidate]] = {}

    for prefix, seed_folders in grouped_folders.items():
        best_per_seed = []
        for seed_folder in seed_folders:
            best = find_best_checkpoint_per_seed(seed_folder)
            if best is None:
                print(f"  [SKIP] {seed_folder}: no usable checkpoints found, excluding from '{prefix}'")
                continue
            best_per_seed.append(best)
        if not best_per_seed:
            print(f"  [WARN] prefix '{prefix}' has zero usable seeds — it will be omitted from results.")
            continue
        seed_index[prefix] = best_per_seed

    return seed_index


# ---------------------------------------------------------------------------
# Step 3: Build actor / transformer from a checkpoint candidate, run
#          compare_performance_ID_OOD for ID + each OOD distribution
# ---------------------------------------------------------------------------

def _load_args_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _load_config_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def build_actor_and_transformer(candidate: CheckpointCandidate, device: str):
    """Reconstruct the actor (+ transformer, if applicable) from a
    CheckpointCandidate, mirroring td3_training.py's model-building logic
    (policy_obs_dim computation, DeterministicAgent construction, weight
    loading from training_state.pth's 'actor' key).
    """
    if not _IMPORTS_OK:
        raise ImportError(
            "Could not import repo modules (DeterministicAgent / ContextEncoder / "
            f"compare_performance_ID_OOD). Run this script from the repo root with "
            f"the repo on PYTHONPATH. Original error: {_IMPORT_ERROR}"
        )

    args_dict = _load_args_yaml(candidate.args_yaml_path)
    config = _load_config_yaml(candidate.config_yaml_path)

    use_history = bool(args_dict.get("use_history", False))
    use_transformer = bool(args_dict.get("use_transformer", False))
    use_last_action = bool(args_dict.get("use_last_action_in_policy_state", False))
    context_len = int(args_dict.get("context_len", 0))
    context_vector_dim = int(args_dict.get("context_vector_dim", 0))
    agent_hidden_layer_size = int(args_dict.get("agent_hidden_layer_size", 64))
    agent_num_hidden_layers = int(args_dict.get("agent_num_hidden_layers", 2))

    air_hockey_base = config["air_hockey"]

    # raw_obs_dim / act_dim are NEVER saved to args.yaml or config.yaml —
    # td3_training.py computes them at runtime from the env's own
    # observation_space/action_space (see the `raw_obs_dim = ...` lines right
    # after `envs = gym.vector.AsyncVectorEnv(...)`). We do the equivalent
    # here without the vectorization: AirHockeyEnv is a plain (non-vector)
    # gym.Env, so we just instantiate one off air_hockey_base and read its
    # observation_space/action_space directly.
    probe_env = AirHockeyEnv(air_hockey_base)
    raw_obs_dim = int(np.array(probe_env.observation_space.shape).prod())
    act_dim = int(np.prod(probe_env.action_space.shape))
    action_low = probe_env.action_space.low
    action_high = probe_env.action_space.high
    probe_env.close()

    policy_obs_dim = raw_obs_dim + act_dim if use_last_action else raw_obs_dim
    if use_history:
        if use_transformer:
            policy_obs_dim += context_vector_dim
        else:
            policy_obs_dim += context_len * HISTORY_ENTRY_DIM

    from types import SimpleNamespace
    import gymnasium as gym
    policy_env_view = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(policy_obs_dim,), dtype=np.float32
        ),
        single_action_space=gym.spaces.Box(
            low=action_low, high=action_high, shape=(act_dim,), dtype=np.float32
        ),
    )

    actor = DeterministicAgent(
        policy_env_view,
        action_scale=1,
        action_bias=0.0,
        hidden_layer_size=agent_hidden_layer_size,
        num_hidden_layers=agent_num_hidden_layers,
    ).to(device)

    state = torch.load(candidate.training_state_path, map_location=device, weights_only=False)
    if "actor" not in state:
        raise KeyError(
            f"training_state.pth at {candidate.training_state_path} has no 'actor' key. "
            f"Top-level keys found: {list(state.keys())}"
        )
    actor.load_state_dict(state["actor"], strict=False)
    actor.eval()

    transformer = None
    if use_history and use_transformer:
        if candidate.transformer_path is None:
            raise FileNotFoundError(
                f"use_transformer=True but no transformer.pth found in {candidate.checkpoint_dir}"
            )
        transformer = ContextEncoder(
            obs_dim=HISTORY_ENTRY_DIM,
            context_dim=context_vector_dim,
            context_len=context_len,
        ).to(device)
        transformer.load_state_dict(torch.load(candidate.transformer_path, map_location=device))
        transformer.eval()

    model_info = {
        "use_history": use_history,
        "use_transformer": use_transformer,
        "use_last_action": use_last_action,
        "context_len": context_len,
        "raw_obs_dim": raw_obs_dim,
        "act_dim": act_dim,
        "air_hockey_base": air_hockey_base,
    }
    return actor, transformer, model_info


def run_id_ood_eval_for_candidate(
    candidate: CheckpointCandidate,
    ood_overrides: dict[str, list[float]],
    device: str,
    n_envs: int,
    n_eps: int,
    seed: int,
    out_dir: str,
) -> dict[str, dict[str, float]]:
    """Build the actor/transformer from `candidate`, override the OOD ranges
    with `ood_overrides`, and call compare_performance_ID_OOD in-process.

    Returns {"id": {"mean_return":..., "std_return":..., "mean_success_rate":..., "std_success_rate":...},
             "ood": {...}}
    """
    actor, transformer, model_info = build_actor_and_transformer(candidate, device)

    air_hockey_base = copy.deepcopy(model_info["air_hockey_base"])
    air_hockey_base.setdefault("random_variable_ranges_OOD", {})
    for var, (lo, hi) in ood_overrides.items():
        air_hockey_base["random_variable_ranges_OOD"][var] = [lo, hi]

    # ------------------------------------------------------------------
    # DEBUG CHECKPOINT (3): ID/OOD param ranges actually used for this eval
    #
    #   print(f"[EVAL] checkpoint_dir={candidate.checkpoint_dir}")
    #   print(f"    ID  ranges : {air_hockey_base.get('random_variable_ranges')}")
    #   print(f"    OOD ranges : {air_hockey_base.get('random_variable_ranges_OOD')}")
    #   breakpoint()
    # ------------------------------------------------------------------

    candidate_out_dir = os.path.join(
        out_dir, os.path.basename(candidate.seed_folder), f"checkpoint_{candidate.checkpoint_step}"
    )

    compare_performance_ID_OOD(
        actor=actor,
        air_hockey_base=air_hockey_base,
        raw_obs_dim=model_info["raw_obs_dim"],
        act_dim=model_info["act_dim"],
        use_last_action=model_info["use_last_action"],
        use_history=model_info["use_history"],
        use_transformer=model_info["use_transformer"],
        transformer=transformer,
        context_len=model_info["context_len"],
        n_envs=n_envs,
        n_eps=n_eps,
        out_dir=candidate_out_dir,
        device=device,
        seed=seed,
        model_path=candidate.training_state_path,
        params_cache_path="saved/parallel_transformer_context_analysis_samples",  
    )

    summary_path = os.path.join(candidate_out_dir, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    return summary["aggregates"]  # {"id": {...}, "ood": {...}}


# ---------------------------------------------------------------------------
# Parallel worker: top-level function (required for spawn-safe pickling)
# ---------------------------------------------------------------------------

def _eval_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Top-level worker function executed in each subprocess.

    Receives a plain-dict job description (fully picklable), runs
    run_id_ood_eval_for_candidate, and returns a plain dict with the
    prefix, seed_folder path, and either the aggregates or an error string.
    Using a plain dict (rather than a dataclass) keeps pickling simple across
    the spawn boundary.

    The job dict contains:
        prefix          str
        candidate       CheckpointCandidate   (dataclass, picklable)
        ood_overrides   dict
        device          str
        n_envs          int
        n_eps           int
        seed            int
        out_dir         str
    """
    candidate: CheckpointCandidate = job["candidate"]
    try:
        aggregates = run_id_ood_eval_for_candidate(
            candidate=candidate,
            ood_overrides=job["ood_overrides"],
            device=job["device"],
            n_envs=job["n_envs"],
            n_eps=job["n_eps"],
            seed=job["seed"],
            out_dir=job["out_dir"],
        )
        return {
            "prefix": job["prefix"],
            "seed_folder": candidate.seed_folder,
            "checkpoint_step": candidate.checkpoint_step,
            "mean_return_across_envs": candidate.mean_return_across_envs,
            "aggregates": aggregates,
            "error": None,
        }
    except Exception:
        return {
            "prefix": job["prefix"],
            "seed_folder": candidate.seed_folder,
            "checkpoint_step": candidate.checkpoint_step,
            "mean_return_across_envs": candidate.mean_return_across_envs,
            "aggregates": None,
            "error": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Step 4: Aggregate per-seed results -> per-prefix (mean, success, stderr)
# ---------------------------------------------------------------------------
# TODO: make sure this is doing what we want
def stderr_of_mean(values: list[float]) -> float:
    """Standard error of the mean: std(values, ddof=1) / sqrt(n).
    ddof=1 (sample std) because we're estimating the population stderr from
    a finite sample of seeds; matches the sweep-aggregation script's
    np.std(arr, ddof=1) / sqrt(n) convention. Returns 0.0 for n<=1 (no
    variance estimate possible from a single sample)."""
    n = len(values)
    if n <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(n))


def aggregate_across_seeds(per_seed_results: list[dict[str, float]]) -> dict[str, float]:
    """per_seed_results: list of {"mean_return":..., "mean_success_rate":...}
    (one dict per seed, already the per-seed mean over that seed's envs).

    Returns the across-seed mean and stderr-of-the-mean for both metrics.
    This is the "second-level" stderr: stderr computed over the seed-level
    means, NOT a re-use of any single seed's own std_return/std_success_rate
    (those already got folded into "mean_return" per seed and aren't
    otherwise touched here).
    """
    returns = [r["mean_return"] for r in per_seed_results]
    successes = [r["mean_success_rate"] for r in per_seed_results]
    return {
        "mean_return": float(np.mean(returns)),
        "stderr_return": stderr_of_mean(returns),
        "mean_success_rate": float(np.mean(successes)),
        "stderr_success_rate": stderr_of_mean(successes),
        "n_seeds": len(per_seed_results),
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def analyze_transformer_context_sweep(
    gravity_min_OOD: float,
    gravity_max_OOD: float,
    paddle_density_min_OOD: float,
    paddle_density_max_OOD: float,
    puck_damping_min_OOD: float,
    puck_damping_max_OOD: float,
    runs_dir: str = "runs/td3",
    baseline_run_dir: str | None = None,
    device: str = "cpu",
    n_envs: int = 30,
    n_eps: int = 8,
    seed: int = 42,
    out_dir: str = "results/transformer_context_sweep_analysis",
    n_workers: int | None = None,
) -> dict[str, Any]:
    """Main entry point.

    For every `sweep_transformer_{true,false}_ctx_{N}` prefix found under
    `runs_dir`, picks the best checkpoint per seed (by multi_env_eval.json),
    runs compare_performance_ID_OOD on it for ID and the OOD distribution
    described by the *_OOD args, and aggregates results across seeds.

    If `baseline_run_dir` is given (a single run folder, e.g.
    'runs/td3/sanity_check'), it's evaluated the same way and stored under
    BASELINE_KEY so it can be plotted as a reference line/bar.

    n_workers controls the multiprocessing pool size. Defaults to
    os.cpu_count() (all logical CPUs). Pass 1 to run serially (useful for
    debugging with breakpoints, since multiprocessing swallows pdb).

    Returns the raw nested results dict (see `results` below); also writes
    a JSON dump, a text table, and PNG figures (one per OOD distribution —
    currently just the one OOD distribution passed in, but the structure
    supports calling this function multiple times with different OOD
    ranges and combining the on-disk JSON outputs across calls).
    """
    os.makedirs(out_dir, exist_ok=True)

    ood_overrides = {
        "gravity": [gravity_min_OOD, gravity_max_OOD],
        "paddle_density": [paddle_density_min_OOD, paddle_density_max_OOD],
        "puck_damping": [puck_damping_min_OOD, puck_damping_max_OOD],
    }

    effective_workers = n_workers if n_workers is not None else os.cpu_count()

    print(f"\n{'='*70}")
    print("analyze_transformer_context_sweep")
    print(f"  runs_dir : {runs_dir}")
    print(f"  OOD overrides : {ood_overrides}")
    print(f"  n_envs={n_envs}, n_eps={n_eps}, seed={seed}")
    print(f"  n_workers={effective_workers}")
    print(f"{'='*70}\n")

    # --- Step 1: discover + group run folders ---
    grouped_folders = discover_run_folders(runs_dir)
    print(f"Discovered {len(grouped_folders)} distinct sweep prefixes:")
    for prefix, folders in sorted(grouped_folders.items()):
        print(f"  {prefix:<38} n_seed_folders={len(folders)}")

    # --- Step 2: best checkpoint per seed, per prefix ---
    print("\nSelecting best checkpoint per seed (by multi_env_eval.json)...")
    seed_index = build_seed_index(grouped_folders)

    if baseline_run_dir is not None:
        baseline_best = find_best_checkpoint_per_seed(baseline_run_dir)
        if baseline_best is None:
            print(f"  [WARN] baseline_run_dir='{baseline_run_dir}' has no usable checkpoints; "
                  "baseline will be omitted from the figure.")
        else:
            seed_index[BASELINE_KEY] = [baseline_best]

    # --- Step 3: build job list and fan out across workers ---
    # Build a flat list of jobs (one per (prefix, seed) pair) so the Pool
    # can distribute them freely across all available workers.  The job dict
    # is fully picklable: CheckpointCandidate is a dataclass (all fields are
    # plain Python types), ood_overrides is a plain dict of lists, etc.
    jobs: list[dict[str, Any]] = []
    for prefix, candidates in seed_index.items():
        for candidate in candidates:
            jobs.append({
                "prefix": prefix,
                "candidate": candidate,
                "ood_overrides": ood_overrides,
                "device": device,
                "n_envs": n_envs,
                "n_eps": n_eps,
                "seed": seed,
                "out_dir": out_dir,
            })

    total_jobs = len(jobs)
    print(f"\nDispatching {total_jobs} eval jobs across {effective_workers} worker(s)...")

    if effective_workers == 1:
        # Serial path: keeps breakpoints / pdb / print working normally.
        worker_results = [_eval_worker(job) for job in jobs]
    else:
        # Parallel path: spawn (not fork) for PyTorch / gym safety.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=effective_workers) as pool:
            worker_results = pool.map(_eval_worker, jobs)

    # --- Collect worker outputs, report errors, group by prefix ---
    # results: { prefix: [ {"id": {...}, "ood": {...}}, ... ] }
    results: dict[str, list[dict[str, dict[str, float]]]] = defaultdict(list)

    completed = 0
    for r in worker_results:
        prefix = r["prefix"]
        seed_folder_name = os.path.basename(r["seed_folder"])
        step = r["checkpoint_step"]
        if r["error"] is not None:
            print(
                f"  [ERROR] {prefix} / {seed_folder_name} / checkpoint_{step}: "
                f"eval failed\n{r['error']}"
            )
        else:
            completed += 1
            print(
                f"  [OK]    {prefix} / {seed_folder_name} / checkpoint_{step} "
                f"(training mean_return={r['mean_return_across_envs']:.2f})"
            )
            results[prefix].append(r["aggregates"])

    print(f"\n{completed}/{total_jobs} evals completed successfully.")

    # --- Step 4: aggregate across seeds, per prefix, per condition ---
    final: dict[str, dict[str, dict[str, float]]] = {}
    for prefix, per_seed_list in results.items():
        if not per_seed_list:
            continue
        id_per_seed = [r["id"] for r in per_seed_list]
        ood_per_seed = [r["ood"] for r in per_seed_list]
        final[prefix] = {
            "id": aggregate_across_seeds(id_per_seed),
            "ood": aggregate_across_seeds(ood_per_seed),
        }

    # --- Save raw + final results to disk ---
    dump = {
        "ood_overrides": ood_overrides,
        "config": {
            "n_envs": n_envs,
            "n_eps": n_eps,
            "seed": seed,
            "runs_dir": runs_dir,
            "n_workers": effective_workers,
        },
        "per_seed_results": {k: v for k, v in results.items()},
        "final_aggregates": final,
    }
    summary_path = os.path.join(out_dir, "sweep_analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"\nSaved full results -> {summary_path}")

    print_sweep_table(final)
    plot_sweep_figure(final, ood_overrides, out_dir)

    return dump


# ---------------------------------------------------------------------------
# Reporting: text table + two-panel bar chart
# ---------------------------------------------------------------------------

def print_sweep_table(final: dict[str, dict[str, dict[str, float]]]) -> None:
    lines = []
    lines.append(f"\n{'='*95}")
    lines.append(f"{'Transformer / Context-Length Sweep — ID vs OOD':^95}")
    lines.append(f"{'='*95}")
    header = (
        f"{'Run Prefix':<38} | {'Cond':<5} | {'MeanReturn':>11} ± {'StdErr':<7} "
        f"| {'SuccessRate':>11} ± {'StdErr':<6} | {'n_seeds':>7}"
    )
    lines.append(header)
    lines.append("-" * 95)

    def sort_key(prefix):
        if prefix == BASELINE_KEY:
            return (-1, 0)
        axes = parse_prefix_axes(prefix)
        if axes is None:
            return (2, prefix)
        use_transformer, ctx = axes
        return (1 if use_transformer else 0, ctx)

    for prefix in sorted(final.keys(), key=sort_key):
        agg = final[prefix]
        display_name = "BASELINE (vanilla TD3)" if prefix == BASELINE_KEY else prefix
        for cond in ("id", "ood"):
            c = agg[cond]
            lines.append(
                f"{display_name:<38} | {cond.upper():<5} | "
                f"{c['mean_return']:>11.2f} ± {c['stderr_return']:<7.2f} | "
                f"{c['mean_success_rate']:>11.3f} ± {c['stderr_success_rate']:<6.3f} | "
                f"{c['n_seeds']:>7}"
            )
        display_name = ""  # only print run name on the ID row
        lines.append("-" * 95)

    for line in lines:
        print(line)


def plot_sweep_figure(
    final: dict[str, dict[str, dict[str, float]]],
    ood_overrides: dict[str, list[float]],
    out_dir: str,
) -> None:
    """Two-panel figure: left = use_transformer, right = no transformer.
    X-axis = context length. Bars = ID (blue) and OOD (red) mean return
    ± stderr. A horizontal dashed line marks the vanilla-TD3 baseline
    (if available) on both panels for reference.
    """
    points_by_branch: dict[bool, list[tuple[int, str]]] = {True: [], False: []}
    for prefix in final.keys():
        axes = parse_prefix_axes(prefix)
        if axes is None:
            continue
        use_transformer, ctx = axes
        points_by_branch[use_transformer].append((ctx, prefix))
    for branch in points_by_branch:
        points_by_branch[branch].sort(key=lambda t: t[0])

    baseline = final.get(BASELINE_KEY)

    fig, axes_arr = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    branch_titles = {True: "Use Transformer", False: "No Transformer"}

    for ax, branch in zip(axes_arr, (True, False)):
        points = points_by_branch[branch]
        ctx_labels = [str(ctx) for ctx, _ in points]
        x = np.arange(len(points))
        width = 0.35

        id_means = [final[p]["id"]["mean_return"] for _, p in points]
        id_errs = [final[p]["id"]["stderr_return"] for _, p in points]
        ood_means = [final[p]["ood"]["mean_return"] for _, p in points]
        ood_errs = [final[p]["ood"]["stderr_return"] for _, p in points]

        ax.bar(x - width / 2, id_means, width, yerr=id_errs, capsize=4,
               color="#2196F3", alpha=0.85, label="ID")
        ax.bar(x + width / 2, ood_means, width, yerr=ood_errs, capsize=4,
               color="#F44336", alpha=0.85, label="OOD")

        if baseline is not None:
            ax.axhline(
                baseline["id"]["mean_return"], color="#2196F3", linestyle="--",
                linewidth=1.2, alpha=0.7, label="Baseline ID",
            )
            ax.axhline(
                baseline["ood"]["mean_return"], color="#F44336", linestyle="--",
                linewidth=1.2, alpha=0.7, label="Baseline OOD",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(ctx_labels)
        ax.set_xlabel("Context Length")
        ax.set_title(branch_titles[branch])
        ax.grid(axis="y", alpha=0.3)

    axes_arr[0].set_ylabel("Mean Return ± StdErr")
    axes_arr[0].legend(loc="best", fontsize=8)

    ood_str = ", ".join(f"{k}={v}" for k, v in ood_overrides.items())
    fig.suptitle(f"ID vs OOD Mean Return  (OOD ranges: {ood_str})", fontsize=10)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "transformer_context_sweep_figure.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure -> {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs/td3")
    p.add_argument("--baseline-run-dir", default=None,
                    help="e.g. runs/td3/sanity_check — single run folder for the vanilla-TD3 comparison line")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-envs", type=int, default=30)
    p.add_argument("--n-eps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="results/transformer_context_sweep_analysis")
    p.add_argument(
        "--n-workers", type=int, default=None,
        help="Number of parallel worker processes (default: os.cpu_count()). "
             "Pass 1 to run serially, which preserves breakpoints and pdb.",
    )

    p.add_argument("--gravity-ood", type=float, nargs=2, required=True, metavar=("MIN", "MAX"))
    p.add_argument("--paddle-density-ood", type=float, nargs=2, required=True, metavar=("MIN", "MAX"))
    p.add_argument("--puck-damping-ood", type=float, nargs=2, required=True, metavar=("MIN", "MAX"))
    return p.parse_args()


if __name__ == "__main__":
    # Guard required for multiprocessing spawn on all platforms.
    # Without this, spawned workers re-execute the top-level script and
    # recurse infinitely trying to start more workers.
    mp.freeze_support()
    args = _parse_args()
    analyze_transformer_context_sweep(
        gravity_min_OOD=args.gravity_ood[0],
        gravity_max_OOD=args.gravity_ood[1],
        paddle_density_min_OOD=args.paddle_density_ood[0],
        paddle_density_max_OOD=args.paddle_density_ood[1],
        puck_damping_min_OOD=args.puck_damping_ood[0],
        puck_damping_max_OOD=args.puck_damping_ood[1],
        runs_dir=args.runs_dir,
        baseline_run_dir=args.baseline_run_dir,
        device=args.device,
        n_envs=args.n_envs,
        n_eps=args.n_eps,
        seed=args.seed,
        out_dir=args.out_dir,
        n_workers=args.n_workers,
    )