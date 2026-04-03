"""
Generate a bash script that launches a sweep of TD3 training runs across GPUs.

Usage:
    uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/generate_sweep.py \\
        --sweep-file scripts/smooth_policy/amp_history/configs/td3/example_reward_sweep.yaml \\
        --output-file run_reward_sweep.sh

Sweep YAML format:
    base_args_file: path/to/base_config.yaml   # required
    log_parent_dir: /data2/.../sweep_name       # required
    run_name_prefix: td3_sweep                  # optional, default "run"
    gpus: [0, 1, 2, 3]                         # GPU indices, round-robin
    mode: grid                                  # "grid" (cartesian product), "random", or "individual"
    num_samples: 20                             # only used for mode=random
    wandb_project: my-project-name             # optional: enables wandb logging for every run
    wandb_entity: my-team                       # optional: wandb entity/team
    eval_resolve_run_dir: true                 # optional: eval script picks <tag> or <tag>r1.. if td3 bumped dir
    eval_run_dir_suffix: r1                    # optional: if set, eval uses <tag><suffix> and skips resolve
    eval_last_n_policies: 3                   # optional: collect last N TD3 checkpoints + final (collect_policy_data)
    post_min_checkpoints: 5                   # optional: after writing scripts, run cleanup_sweep_runs post-process
    post_cleanup_apply: false                 # if true, delete run dirs below post_min_checkpoints
    post_aggregate_gifs: true                 # copy one eval_0.gif per kept run into log_parent_dir/<post_gifs_subdir>/
    post_gifs_subdir: full_evaluation_gifs    # output folder name under log_parent_dir
    post_remove_noncanonical_run_dirs: false # with post_cleanup_apply: also delete non-eval-resolved tagrN dirs
    post_promote_r_dirs: false               # rename tagrN -> tag when tag has no args.yaml and tag missing or empty
    post_prune_hollow_dirs: true             # rm tag/ trees with no args.yaml and no files (before promote)
    post_prune_stale_tag_dirs: true         # rm tag/ with no args when tagrN/ has args (junk placeholders)
    post_prune_abandoned_tag_dirs: true     # rm expected tag/ with no args at tag/ or any tagrN/ (failed runs)
    post_prune_r_suffix_without_args: true  # rm tagrN/ dirs with no args.yaml
    extra_args:                                 # static overrides always added
      seed: 42
      total_timesteps: 500000
    params:                                     # parameters to sweep
      velocity_reward_weight:
        values: [0.1, 0.5, 1.0]
      jerk_reward_weight:
        linspace: {start: 0.1, stop: 1.0, num: 4}
      motion_reward_weight:
        logspace: {start: -2, stop: 0, num: 3}  # 10^start .. 10^stop
"""

import argparse
import itertools

import os
import random
import sys

import yaml


# ---------------------------------------------------------------------------
# Value range helpers
# ---------------------------------------------------------------------------

def _resolve_param_values(spec: dict) -> list:
    """Turn a param spec dict into a flat list of values."""
    if "values" in spec:
        return list(spec["values"])
    if "linspace" in spec:
        cfg = spec["linspace"]
        start, stop, num = float(cfg["start"]), float(cfg["stop"]), int(cfg["num"])
        if num == 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]
    if "logspace" in spec:
        cfg = spec["logspace"]
        start, stop, num = float(cfg["start"]), float(cfg["stop"]), int(cfg["num"])
        base = float(cfg.get("base", 10.0))
        if num == 1:
            return [base ** start]
        step = (stop - start) / (num - 1)
        return [base ** (start + i * step) for i in range(num)]
    if "range" in spec:
        cfg = spec["range"]
        start, stop, step = float(cfg["start"]), float(cfg["stop"]), float(cfg["step"])
        vals = []
        v = start
        while v <= stop + 1e-9:
            vals.append(v)
            v += step
        return vals
    raise ValueError(f"Unknown param spec keys: {list(spec.keys())}")


# ---------------------------------------------------------------------------
# Run config building
# ---------------------------------------------------------------------------

def _param_to_flag(name: str) -> str:
    """Convert python attribute name to CLI flag (underscores -> hyphens)."""
    return "--" + name.replace("_", "-")


def _value_to_str(v) -> str:
    """Format a value for the CLI, keeping floats readable."""
    if isinstance(v, float):
        # Use up to 6 significant figures, strip trailing zeros
        formatted = f"{v:.6g}"
        return formatted
    return str(v)


def _value_to_tag(v) -> str:
    """Short tag for run names (replace dots/minus to avoid shell issues)."""
    s = _value_to_str(v)
    s = s.replace("-", "n").replace(".", "p")
    return s


def _build_runs(sweep: dict) -> list[dict]:
    """Return a list of run dicts: {params: {name: value}, ...}."""
    raw_params: dict = sweep.get("params", {})
    param_names = list(raw_params.keys())
    param_value_lists = [_resolve_param_values(raw_params[n]) for n in param_names]

    mode = sweep.get("mode", "grid")

    if mode == "grid":
        combos = list(itertools.product(*param_value_lists))
        runs = [{"params": {param_names[i]: combo[i] for i in range(len(param_names))}} for combo in combos]
    elif mode == "random":
        num_samples = int(sweep.get("num_samples", 10))
        all_combos = list(itertools.product(*param_value_lists))
        if num_samples >= len(all_combos):
            combos = all_combos
        else:
            rng = random.Random(sweep.get("random_seed", 0))
            combos = rng.sample(all_combos, num_samples)
        runs = [{"params": {param_names[i]: combo[i] for i in range(len(param_names))}} for combo in combos]
    elif mode == "individual":
        # One run per value per parameter; all other parameters are left at base config defaults
        # (i.e. not passed as CLI overrides at all).
        runs = []
        for name, values in zip(param_names, param_value_lists):
            for value in values:
                runs.append({"params": {name: value}})
    else:
        raise ValueError(f"Unknown sweep mode '{mode}'. Expected 'grid', 'random', or 'individual'.")

    return runs


def sweep_run_tags(sweep: dict) -> list[str]:
    """Directory tags (LOG_BASE/<tag> or <tag>r1, …) for each run in sweep order."""
    runs = _build_runs(sweep)
    tags: list[str] = []
    for idx, run in enumerate(runs):
        params = run["params"]
        tag_parts = [f"{k}_{_value_to_tag(v)}" for k, v in params.items()]
        tag = "__".join(tag_parts) if tag_parts else f"run{idx:03d}"
        tags.append(tag)
    return tags


# ---------------------------------------------------------------------------
# Bash generation
# ---------------------------------------------------------------------------

def _generate_bash(sweep: dict, runs: list[dict], output_path: str) -> None:
    base_args_file = sweep["base_args_file"]
    log_parent_dir = sweep["log_parent_dir"]
    run_name_prefix = sweep.get("run_name_prefix", "run")
    gpus: list = sweep.get("gpus", [0])
    extra_args: dict = sweep.get("extra_args", {}) or {}
    max_parallel: int = int(sweep.get("max_parallel", 0))  # 0 = unlimited
    wandb_project: str | None = sweep.get("wandb_project")
    wandb_entity: str | None = sweep.get("wandb_entity")

    script_path = (
        "scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py"
    )

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'LOG_BASE="{log_parent_dir}"',
        # Ensure sweep root exists for per-run nohup logs. Do NOT mkdir per-run log dirs here:
        # td3_training.py bumps to <tag>r1, r2, ... if the path already exists; pre-creating <tag>
        # only for nohup would leave empty <tag> dirs and break eval paths.
        'mkdir -p "$LOG_BASE"',
        "",
    ]

    if max_parallel > 0:
        lines += [
            f"MAX_PARALLEL={max_parallel}",
            "_job_count=0",
            "",
            "# Wait for one background job to finish before continuing.",
            "_wait_one() {",
            "  wait -n 2>/dev/null || wait",
            "  _job_count=$(( _job_count - 1 ))",
            "}",
            "",
        ]

    lines += ["# ---------- launch all runs ----------"]

    for idx, run in enumerate(runs):
        gpu = gpus[idx % len(gpus)]
        params = run["params"]

        tag_parts = [f"{k}_{_value_to_tag(v)}" for k, v in params.items()]
        tag = "__".join(tag_parts) if tag_parts else f"run{idx:03d}"
        run_name = f"{run_name_prefix}__{tag}__gpu{gpu}"

        run_dir = f'"$LOG_BASE/{tag}"'

        param_flags = []
        for name, value in params.items():
            param_flags.append(f"{_param_to_flag(name)} {_value_to_str(value)}")
        for name, value in extra_args.items():
            param_flags.append(f"{_param_to_flag(name)} {_value_to_str(value)}")

        indent = "    "
        param_lines = [f"{indent}{flag} \\" for flag in param_flags]
        param_block = "\n".join(param_lines)

        throttle_lines = []
        if max_parallel > 0:
            throttle_lines = [
                f"  if (( _job_count >= MAX_PARALLEL )); then _wait_one; fi",
            ]

        wandb_flags = ""
        if wandb_project:
            wandb_flags += f"\n    --wandb-project {wandb_project} \\"
        if wandb_entity:
            wandb_flags += f"\n    --wandb-entity {wandb_entity} \\"

        lines += [
            "",
            f"# run {idx}: {tag}",
            f"if [[ ${{DRY_RUN:-0}} == 1 ]]; then",
            f'  echo "DRY_RUN [{idx}]: {run_name} on cuda:{gpu}"',
            f"else",
            *throttle_lines,
            f"  nohup uv run {script_path} \\",
            f"    --args-file {base_args_file} \\",
            f"    --device cuda:{gpu} \\",
            f"    --log-parent-dir {run_dir} \\",
            f"    --run-name {run_name} \\{wandb_flags}",
            param_block,
            # Log file under LOG_BASE so we never create <tag> before td3 starts (see mkdir note above).
            f'    > "$LOG_BASE/nohup_{tag}.out" 2>&1 &',
            *(["  _job_count=$(( _job_count + 1 ))"] if max_parallel > 0 else []),
            f'  echo "Started [{idx}]: {run_name} on cuda:{gpu} -> {run_dir}"',
            f"fi",
        ]

    wait_remaining = ["  wait", '  echo "All jobs finished."'] if max_parallel > 0 else []
    lines += [
        "",
        'if [[ ${DRY_RUN:-0} == 1 ]]; then',
        f'  echo "Dry run complete. {len(runs)} run(s) would be launched."',
        "else",
        *wait_remaining,
        f'  echo "All {len(runs)} run(s) launched."',
        '  echo "Check status: ps -fu \\"$USER\\" | grep td3_training.py"',
        "fi",
        "",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    os.chmod(output_path, 0o755)


def _generate_eval_bash(sweep: dict, runs: list[dict], output_path: str) -> None:
    """Generate a bash script that evaluates every run in the sweep."""
    log_parent_dir = sweep["log_parent_dir"]
    max_parallel: int = int(sweep.get("eval_max_parallel", sweep.get("max_parallel", 0)))
    eval_num_episodes: int = int(sweep.get("eval_num_episodes", 20))
    eval_n_gifs: int = int(sweep.get("eval_n_gifs", 3))
    eval_last_n_policies: int = int(sweep.get("eval_last_n_policies", 1))
    # If set (e.g. "r1"), use LOG_BASE/<tag><suffix> and skip auto-resolve.
    eval_run_dir_suffix: str | None = sweep.get("eval_run_dir_suffix")
    eval_resolve_run_dir: bool = bool(sweep.get("eval_resolve_run_dir", True))

    eval_script = "scripts/smooth_policy/collect_policy_data.py"

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'LOG_BASE="{log_parent_dir}"',
        "",
    ]

    if max_parallel > 0:
        lines += [
            f"MAX_PARALLEL={max_parallel}",
            "_job_count=0",
            "",
            "_wait_one() {",
            "  wait -n 2>/dev/null || wait",
            "  _job_count=$(( _job_count - 1 ))",
            "}",
            "",
        ]

    if eval_run_dir_suffix is None and eval_resolve_run_dir:
        lines += [
            "# Resolve TD3 log dir: training uses <tag> unless it already existed, then <tag>r1, r2, ...",
            "_resolve_run_dir() {",
            '  local _base="$1" _tag="$2" _d _i',
            '  _d="${_base}/${_tag}"',
            '  if [[ -f "${_d}/args.yaml" ]]; then',
            '    printf "%s\\n" "${_d}"',
            "    return",
            "  fi",
            "  _i=1",
            "  while ((_i <= 50)); do",
            '    if [[ -f "${_d}r${_i}/args.yaml" ]]; then',
            '      printf "%s\\n" "${_d}r${_i}"',
            "      return",
            "    fi",
            "    ((_i++)) || true",
            "  done",
            '  printf "%s\\n" "${_d}"',
            "}",
            "",
        ]

    lines += ["# ---------- evaluate all runs ----------"]

    for idx, run in enumerate(runs):
        params = run["params"]
        tag_parts = [f"{k}_{_value_to_tag(v)}" for k, v in params.items()]
        tag = "__".join(tag_parts) if tag_parts else f"run{idx:03d}"
        if eval_run_dir_suffix is not None:
            run_dir_cmd = f'"$LOG_BASE/{tag}{eval_run_dir_suffix}"'
        elif eval_resolve_run_dir:
            run_dir_cmd = f'"$(_resolve_run_dir "$LOG_BASE" "{tag}")"'
        else:
            run_dir_cmd = f'"$LOG_BASE/{tag}"'

        throttle_lines = []
        if max_parallel > 0:
            throttle_lines = [
                "  if (( _job_count >= MAX_PARALLEL )); then _wait_one; fi",
            ]

        lines += [
            "",
            f"# run {idx}: {tag}",
            f"if [[ ${{DRY_RUN:-0}} == 1 ]]; then",
            f'  echo "DRY_RUN [{idx}]: eval {tag}"',
            f"else",
            *throttle_lines,
            f"  _run_dir={run_dir_cmd}",
            f"  nohup uv run {eval_script} \\",
            f'    --run-dir "$_run_dir" \\',
            f"    --num-episodes {eval_num_episodes} \\",
            f"    --n-gifs {eval_n_gifs} \\",
            f"    --last-n-policies {eval_last_n_policies} \\",
            f'    > "$_run_dir/eval_nohup.out" 2>&1 &',
            *(["  _job_count=$(( _job_count + 1 ))"] if max_parallel > 0 else []),
            (
                f'  echo "Started eval [{idx}]: {tag} -> $_run_dir/rollout (last {eval_last_n_policies} policies)"'
                if eval_last_n_policies > 1
                else f'  echo "Started eval [{idx}]: {tag} -> $_run_dir/rollout"'
            ),
            "fi",
        ]

    wait_remaining = ["  wait", '  echo "All eval jobs finished."'] if max_parallel > 0 else []
    lines += [
        "",
        'if [[ ${DRY_RUN:-0} == 1 ]]; then',
        f'  echo "Dry run complete. {len(runs)} eval(s) would be launched."',
        "else",
        *wait_remaining,
        f'  echo "All {len(runs)} eval(s) launched."',
        '  echo "Check status: ps -fu \\"$USER\\" | grep collect_policy_data.py"',
        "fi",
        "",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    os.chmod(output_path, 0o755)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a bash sweep script for TD3 training runs."
    )
    parser.add_argument(
        "--sweep-file",
        required=True,
        help="Path to the sweep YAML file.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        help="Path for the generated training bash script.",
    )
    parser.add_argument(
        "--eval-output-file",
        default=None,
        help="If given, also write an eval bash script for all runs in the sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing the file.",
    )
    args = parser.parse_args()

    with open(args.sweep_file, "r") as f:
        sweep = yaml.safe_load(f)

    if "base_args_file" not in sweep:
        print("ERROR: sweep YAML must contain 'base_args_file'.", file=sys.stderr)
        sys.exit(1)
    if "log_parent_dir" not in sweep:
        print("ERROR: sweep YAML must contain 'log_parent_dir'.", file=sys.stderr)
        sys.exit(1)

    runs = _build_runs(sweep)
    print(f"Total runs: {len(runs)}")
    for i, run in enumerate(runs):
        gpu = sweep.get("gpus", [0])[i % len(sweep.get("gpus", [0]))]
        param_str = ", ".join(f"{k}={_value_to_str(v)}" for k, v in run["params"].items())
        print(f"  [{i:3d}] cuda:{gpu}  {param_str}")

    if args.dry_run:
        print("(dry run — no file written)")
        return

    _generate_bash(sweep, runs, args.output_file)
    print(f"Wrote: {args.output_file}")
    print(f"Launch with:  bash {args.output_file}")
    print(f"Dry-run with: DRY_RUN=1 bash {args.output_file}")

    if args.eval_output_file:
        _generate_eval_bash(sweep, runs, args.eval_output_file)
        print(f"Wrote eval script: {args.eval_output_file}")
        print(f"Eval with:     bash {args.eval_output_file}")
        print(f"Eval dry-run:  DRY_RUN=1 bash {args.eval_output_file}")

    post_min = sweep.get("post_min_checkpoints")
    if post_min is not None:
        from scripts.smooth_policy.amp_history.amp_training.td3.helper import cleanup_sweep_runs

        sweep_dir = os.path.abspath(sweep["log_parent_dir"])
        print(
            f"post_min_checkpoints={post_min}: running cleanup_sweep_runs.postprocess "
            f"(aggregate_gifs={sweep.get('post_aggregate_gifs', True)}, "
            f"cleanup_apply={bool(sweep.get('post_cleanup_apply', False))}, "
            f"remove_noncanonical={bool(sweep.get('post_remove_noncanonical_run_dirs', False))}, "
            f"promote_r_dirs={bool(sweep.get('post_promote_r_dirs', False))}, "
            f"prune_hollow={bool(sweep.get('post_prune_hollow_dirs', True))}, "
            f"prune_stale_tag={bool(sweep.get('post_prune_stale_tag_dirs', True))}, "
            f"prune_abandoned={bool(sweep.get('post_prune_abandoned_tag_dirs', True))}, "
            f"prune_r_suffix={bool(sweep.get('post_prune_r_suffix_without_args', True))})"
        )
        cleanup_sweep_runs.postprocess_reward_sweep(
            sweep_dir=sweep_dir,
            min_checkpoints=int(post_min),
            sweep_cfg=sweep,
            aggregate_gifs=bool(sweep.get("post_aggregate_gifs", True)),
            gifs_subdir=str(sweep.get("post_gifs_subdir", "full_evaluation_gifs")),
            cleanup_apply=bool(sweep.get("post_cleanup_apply", False)),
            remove_noncanonical_run_dirs=bool(sweep.get("post_remove_noncanonical_run_dirs", False)),
            promote_r_dirs=bool(sweep.get("post_promote_r_dirs", False)),
            prune_hollow_dirs=bool(sweep.get("post_prune_hollow_dirs", True)),
            prune_stale_tag_dirs=bool(sweep.get("post_prune_stale_tag_dirs", True)),
            prune_abandoned_tag_dirs=bool(sweep.get("post_prune_abandoned_tag_dirs", True)),
            prune_r_suffix_without_args=bool(sweep.get("post_prune_r_suffix_without_args", True)),
            eval_run_dir_suffix=sweep.get("eval_run_dir_suffix"),
            eval_resolve_run_dir=bool(sweep.get("eval_resolve_run_dir", True)),
        )


if __name__ == "__main__":
    main()
