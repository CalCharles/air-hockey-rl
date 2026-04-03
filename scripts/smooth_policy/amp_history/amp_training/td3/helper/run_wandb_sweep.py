"""
Create a wandb sweep from a generate_sweep.py YAML and run a wandb agent.

This translates the existing sweep YAML format into a wandb sweep config,
registers it with wandb, and launches an agent that calls td3_training.py
as a subprocess for each trial.

The sweep YAML must include:
    wandb_project: my-project-name   # required
    wandb_entity:  my-team           # optional

Usage:
    # Create sweep and start agent (blocks until count trials are done)
    uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/run_wandb_sweep.py \\
        --sweep-file scripts/smooth_policy/amp_history/configs/td3/example_reward_sweep.yaml \\
        --device cuda:0 \\
        --count 20

    # Just create the sweep and print the agent command (do not run agent)
    uv run ... --sweep-file ... --create-only

    # Resume an existing sweep
    uv run ... --sweep-file ... --sweep-id abc123 --device cuda:1 --count 10

Design notes:
    Each agent trial:
      1. Calls wandb.init() to create a run under the sweep and receive the
         trial's hyperparameters via wandb.config.
      2. Immediately calls wandb.finish() to release the run handle.
      3. Launches td3_training.py as a subprocess with the trial params as
         CLI flags, plus WANDB_RUN_ID and WANDB_RESUME=allow in the environment
         so the training process resumes and populates that same run.
    This avoids having two processes simultaneously attached to one wandb run
    while still keeping all training metrics under a single sweep run.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import yaml


_TRAINING_SCRIPT = "scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py"

# Wandb sweep param spec keys that we expand to explicit value lists.
_EXPAND_MODES = {"linspace", "logspace", "range", "values"}


# ---------------------------------------------------------------------------
# Sweep YAML -> wandb sweep config conversion
# ---------------------------------------------------------------------------

def _resolve_values(spec: dict) -> list:
    """Expand a param spec dict to a flat list of values (same logic as generate_sweep.py)."""
    if "values" in spec:
        return list(spec["values"])
    if "linspace" in spec:
        cfg = spec["linspace"]
        start, stop, num = float(cfg["start"]), float(cfg["stop"]), int(cfg["num"])
        if num == 1:
            return [start]
        step = (stop - start) / (num - 1)
        return [round(start + i * step, 10) for i in range(num)]
    if "logspace" in spec:
        cfg = spec["logspace"]
        start, stop, num = float(cfg["start"]), float(cfg["stop"]), int(cfg["num"])
        base = float(cfg.get("base", 10.0))
        if num == 1:
            return [base ** start]
        step = (stop - start) / (num - 1)
        return [round(base ** (start + i * step), 10) for i in range(num)]
    if "range" in spec:
        cfg = spec["range"]
        start, stop, step = float(cfg["start"]), float(cfg["stop"]), float(cfg["step"])
        vals, v = [], start
        while v <= stop + 1e-9:
            vals.append(round(v, 10))
            v += step
        return vals
    raise ValueError(f"Unknown param spec keys: {list(spec.keys())}")


def _build_wandb_sweep_config(sweep: dict) -> dict:
    """
    Convert a generate_sweep.py YAML into a wandb sweep config dict.

    Supported modes: grid, random.
    'individual' mode is not a standard wandb concept; it is converted to
    'grid' (each parameter sweeps its values independently while others hold
    their base-config defaults — this is not natively representable in wandb
    grid mode, so all combinations will be run instead; use --count to limit).
    """
    mode = sweep.get("mode", "grid")
    if mode == "individual":
        print(
            "WARNING: sweep mode 'individual' is not natively supported by wandb. "
            "Treating as 'grid'. Use --count to limit the number of trials.",
            file=sys.stderr,
        )
        method = "grid"
    elif mode == "random":
        method = "random"
    else:
        method = "grid"

    raw_params: dict = sweep.get("params", {})
    wandb_params = {}
    for name, spec in raw_params.items():
        wandb_params[name] = {"values": _resolve_values(spec)}

    wandb_cfg: dict = {
        "method": method,
        "metric": {
            "goal": "maximize",
            "name": "charts/rolling2k_avg_episode_return",
        },
        "parameters": wandb_params,
    }

    if method == "random" and "num_samples" in sweep:
        wandb_cfg["run_cap"] = int(sweep["num_samples"])

    return wandb_cfg


# ---------------------------------------------------------------------------
# Agent function factory
# ---------------------------------------------------------------------------

def _make_agent_fn(
    sweep: dict,
    device: str,
    project: str,
    entity: str | None,
) -> callable:
    """
    Return the function that wandb.agent will call for each trial.

    For each trial:
      - wandb.init() creates a run and populates wandb.config with the trial params.
      - wandb.finish() releases the wrapper's handle on the run.
      - td3_training.py is launched as a subprocess; it resumes the same run
        via WANDB_RUN_ID + WANDB_RESUME=allow.
    """
    base_args_file: str = sweep["base_args_file"]
    log_parent_dir: str = sweep["log_parent_dir"]
    run_name_prefix: str = sweep.get("run_name_prefix", "run")
    extra_args: dict = sweep.get("extra_args", {}) or {}

    def _run() -> None:
        import wandb

        # Step 1: create the run for this trial and read its config.
        run = wandb.init(project=project, entity=entity)
        trial_params = dict(wandb.config)
        run_id = run.id

        # Step 2: release the wrapper's run handle so the subprocess can
        # resume this same run as the sole active writer.
        wandb.finish()

        # Build a human-readable tag and log dir mirroring generate_sweep.py.
        def _tag(v) -> str:
            s = f"{v:.6g}" if isinstance(v, float) else str(v)
            return s.replace("-", "n").replace(".", "p")

        tag_parts = [f"{k}_{_tag(v)}" for k, v in sorted(trial_params.items())]
        tag = "__".join(tag_parts) if tag_parts else "run000"
        device_label = device.replace(":", "")
        run_name = f"{run_name_prefix}__{tag}__{device_label}"
        run_dir = os.path.join(log_parent_dir, tag)
        os.makedirs(run_dir, exist_ok=True)

        # Assemble the training command.
        cmd = [
            "uv", "run", _TRAINING_SCRIPT,
            "--args-file", base_args_file,
            "--device", device,
            "--log-parent-dir", run_dir,
            "--run-name", run_name,
            "--wandb-project", project,
        ]
        if entity:
            cmd += ["--wandb-entity", entity]

        # Static overrides first, then trial-specific params (trial wins on collision).
        for k, v in {**extra_args, **trial_params}.items():
            cmd += [f"--{k.replace('_', '-')}", str(v)]

        # Step 3: launch the training subprocess, instructing it to resume the
        # run that the wrapper just created.
        env = os.environ.copy()
        env["WANDB_RUN_ID"] = run_id
        env["WANDB_RESUME"] = "allow"

        print(f"[wandb-agent] Starting trial: {run_name}  (run_id={run_id})")
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"Training subprocess exited with code {result.returncode} "
                f"for run {run_id} ({run_name})"
            )

    return _run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a wandb sweep from a generate_sweep.py YAML and run an agent."
    )
    parser.add_argument(
        "--sweep-file",
        required=True,
        help="Path to the sweep YAML file.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device passed to each training run (default: cuda:0). "
             "Run one agent process per GPU, e.g. --device cuda:1.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Maximum number of trials this agent will run. "
             "Omit to run until the sweep is exhausted.",
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Resume an existing sweep instead of creating a new one.",
    )
    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Create (or validate) the wandb sweep and print the agent command, "
             "but do not start the agent.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.sweep_file):
        print(f"ERROR: sweep file not found: {args.sweep_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.sweep_file) as f:
        sweep = yaml.safe_load(f)

    for required_key in ("base_args_file", "log_parent_dir"):
        if required_key not in sweep:
            print(f"ERROR: sweep YAML must contain '{required_key}'.", file=sys.stderr)
            sys.exit(1)

    project: str | None = sweep.get("wandb_project")
    entity: str | None = sweep.get("wandb_entity")

    if not project:
        print(
            "ERROR: sweep YAML must contain 'wandb_project' to use the wandb agent.\n"
            "Add a line like:  wandb_project: my-project-name",
            file=sys.stderr,
        )
        sys.exit(1)

    import wandb

    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Resuming existing sweep: {sweep_id}")
    else:
        wandb_cfg = _build_wandb_sweep_config(sweep)
        sweep_id = wandb.sweep(
            sweep=wandb_cfg,
            project=project,
            entity=entity,
        )
        print(f"Created wandb sweep: {sweep_id}")
        entity_prefix = f"{entity}/" if entity else ""
        print(
            f"To run additional agents:\n"
            f"  wandb agent {entity_prefix}{project}/{sweep_id}"
        )

    if args.create_only:
        print("--create-only set; not starting agent.")
        return

    agent_fn = _make_agent_fn(sweep, device=args.device, project=project, entity=entity)
    print(f"Starting agent on {args.device} (count={args.count or 'unlimited'}) ...")
    wandb.agent(sweep_id, function=agent_fn, count=args.count, project=project, entity=entity)


if __name__ == "__main__":
    main()