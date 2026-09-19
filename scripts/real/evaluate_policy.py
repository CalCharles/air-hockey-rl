"""Single entrypoint for evaluating a trained policy on a specified environment.

One command covers both axes:

* **Environment** — ``--config`` is an ordinary air-hockey config YAML. Its
  ``simulator:`` key decides where the policy runs: ``real`` drives the UR5,
  ``box2d`` runs the identical loop in sim. Nothing else changes.
* **Algorithm** — ``--checkpoint`` is inspected to pick the loader. Plain TD3
  actors, RMA phase-2 adapted policies, long-history policies, and SGCRL all
  dispatch through ``scripts.td3.helper.real_eval_agents.EVAL_AGENT_BUILDERS``.

Everything downstream of agent construction is the existing frozen-policy eval
pipeline (``scripts.td3.extras.async_td3_real_eval``): between-episode reset
FSM, task-specific metrics hooks, episode validation, and the
``eval_summary.json`` / ``eval_per_episode.jsonl`` / HDF5 artifact set. This
module only resolves arguments and delegates; it owns no rollout logic.

Not covered here: paired multi-policy comparison on fixed DR env sets. That is
a different job with its own scripts (``scripts.rma.eval_paired``,
``scripts.td3.extras.eval_sysid_v2_campaign``).

Examples
--------
TD3 on the real robot::

    python -m scripts.real.evaluate_policy \
      --config configs/real_configs/rollout_td3_config_hist2.yaml \
      --checkpoint data/policies_2026-09-19/juggle/sysid \
      --episodes 10 --data-root-dir real_runs/eval_juggle_sysid

RMA phase-2 on the real robot (same command, different bundle)::

    python -m scripts.real.evaluate_policy \
      --config configs/real_configs/rollout_td3_config_hist2.yaml \
      --checkpoint data/policies_2026-09-19/juggle/rma_full \
      --episodes 10 --data-root-dir real_runs/eval_juggle_rma_full

The same policy in Box2D, as a pre-robot smoke test::

    python -m scripts.real.evaluate_policy \
      --config configs/new_juggle/tasks_v2/sim_sysid_juggle.yaml \
      --checkpoint data/policies_2026-09-19/juggle/rma_full \
      --episodes 3 --data-root-dir /tmp/smoke --no-artifacts

Any flag can instead come from a run-config YAML passed with ``--run-config``
(same key names, underscores); explicit command-line flags win.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from scripts.td3.extras.async_td3_real_eval import EvalSpecificArgs
from scripts.td3.extras.async_td3_real_eval import main as run_frozen_policy_eval
from scripts.td3.helper.real_eval_agents import (
    EVAL_AGENT_BUILDERS,
    RMA_META_FILENAME,
    load_rma_meta,
    synthesize_eval_train_args,
)
from scripts.td3.helper.real_td3_runtime import (
    Args,
    TrainArgs,
    _build_args_file_defaults,
    _load_train_args,
    _setup_run_data_dir,
)

# Real-world runtime defaults (transition holds, buffer sizes, device wiring).
# Algorithm-independent: the exploration knobs it carries are forced to zero by
# the eval entrypoint, and the architecture keys are ignored on every path that
# does not read them.
DEFAULT_ARGS_FILE = "configs/td3_real_world/td3_online.yaml"

# Algorithms whose architecture lives in a training-run args.yaml rather than
# inside the checkpoint itself.
_NEEDS_TRAIN_ARGS = ("td3",)

# Keys a --run-config YAML may set. Names match the CLI flags with underscores.
_RUN_CONFIG_KEYS = (
    "config",
    "checkpoint",
    "algo",
    "episodes",
    "max_attempts",
    "data_root_dir",
    "device",
    "seed",
    "train_args",
    "args_file",
    "artifacts",
    "verbose",
)


def detect_algo(checkpoint: str) -> str:
    """Infer the algorithm from what is on disk next to the checkpoint.

    An ``rma_meta.json`` beside the weights is the discriminator, and its
    ``mode`` separates the long-history control from the privileged/adapted
    pair. Everything else falls back on the file extension.
    """
    path = Path(checkpoint)
    bundle_dir = path if path.is_dir() else path.parent
    meta_path = bundle_dir / RMA_META_FILENAME
    if meta_path.is_file():
        with meta_path.open() as f:
            mode = str(json.load(f).get("mode", "privileged"))
        return "history" if mode == "history" else "rma"
    if path.suffix == ".pkl":
        return "sgcrl"
    return "td3"


def resolve_train_args(algo: str, checkpoint: str, explicit_path: Optional[str]) -> tuple[TrainArgs, Optional[str]]:
    """Return the policy-state contract plus the path it came from, if any.

    TD3 needs the training run's ``args.yaml`` because the actor is rebuilt from
    scratch and the layer shapes must match the saved weights; it defaults to
    the ``args.yaml`` sitting beside the checkpoint. RMA reads the equivalent
    facts out of ``rma_meta.json``, and SGCRL out of its own pickle, so neither
    requires the file.
    """
    if algo in _NEEDS_TRAIN_ARGS:
        path = explicit_path
        if path is None:
            candidate = Path(checkpoint)
            candidate = (candidate if candidate.is_dir() else candidate.parent) / "args.yaml"
            if not candidate.is_file():
                raise SystemExit(
                    f"--algo {algo} needs the training run's args.yaml for the network "
                    f"architecture, and none was found at {candidate}. Pass --train-args."
                )
            path = str(candidate)
        return _load_train_args(path), path

    if algo in ("rma", "history"):
        meta = load_rma_meta(checkpoint)
        return synthesize_eval_train_args(use_last_action=bool(meta["use_last_action"])), None

    return synthesize_eval_train_args(use_last_action=False), None


def resolve_model_path(algo: str, checkpoint: str) -> str:
    """Normalize ``--checkpoint`` into what the chosen builder expects.

    RMA needs the bundle *directory* (three files). TD3 needs a single weights
    file, so a directory is resolved to ``training_state.pth`` or ``model.pth``.
    """
    path = Path(checkpoint)
    if algo in ("rma", "history"):
        return str(path if path.is_dir() else path.parent)
    if path.is_dir():
        for name in ("training_state.pth", "model.pth"):
            if (path / name).is_file():
                return str(path / name)
        raise SystemExit(
            f"no training_state.pth or model.pth in {path}; pass the weights file directly."
        )
    return str(path)


def build_args(cli: argparse.Namespace, algo: str, train_args_path: Optional[str]) -> Args:
    """Assemble the runtime ``Args`` the eval pipeline consumes.

    Layered: args-file YAML defaults first, then the resolutions this script is
    responsible for (env config, checkpoint, device, output dir).
    """
    mapped: Dict[str, Any] = {}
    if cli.args_file:
        mapped, applied, ignored = _build_args_file_defaults(cli.args_file)
        print(f"[evaluate_policy] args-file defaults from {cli.args_file} ({len(applied)} applied, {len(ignored)} ignored)")

    mapped["args_file"] = cli.args_file
    mapped["train_args"] = train_args_path
    mapped["config"] = cli.config
    mapped["model_path"] = resolve_model_path(algo, cli.checkpoint)
    mapped["collector_device"] = cli.device
    mapped["seed"] = int(cli.seed)
    mapped["data_root_dir"] = cli.data_root_dir
    if not cli.artifacts:
        mapped["enable_episode_gif"] = False
        mapped["enable_episode_camera_video"] = False
    return Args(**mapped)


def _apply_run_config(cli: argparse.Namespace, argv: list[str]) -> argparse.Namespace:
    """Fill options from a run-config YAML. Explicit command-line flags win."""
    if not cli.run_config:
        return cli
    with open(cli.run_config) as f:
        payload = yaml.load(f, Loader=yaml.FullLoader) or {}
    unknown = sorted(set(payload) - set(_RUN_CONFIG_KEYS))
    if unknown:
        raise SystemExit(
            f"--run-config {cli.run_config} has unsupported keys {unknown}; "
            f"supported: {sorted(_RUN_CONFIG_KEYS)}"
        )
    given = {
        arg.lstrip("-").replace("-", "_").split("=")[0]
        for arg in argv
        if arg.startswith("--")
    }
    for key, value in payload.items():
        if key not in given and f"no_{key}" not in given:
            setattr(cli, key, value)
    return cli


def parse_cli(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-config", default=None, help="YAML supplying any of the options below.")
    parser.add_argument("--config", default=None, help="Air-hockey env config YAML; its `simulator:` picks real vs box2d.")
    parser.add_argument("--checkpoint", default=None, help="Policy bundle directory, or a weights file inside one.")
    parser.add_argument(
        "--algo",
        default="auto",
        choices=("auto", *sorted(EVAL_AGENT_BUILDERS)),
        help="Algorithm. 'auto' infers it from the checkpoint contents.",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Target number of kept (validator-passed) episodes.")
    parser.add_argument("--max-attempts", type=int, default=None, help="Cap on total attempts including discards; 0 = unlimited.")
    parser.add_argument("--data-root-dir", default=None, help="Root for this run's artifacts.")
    parser.add_argument("--device", default=None, help="Torch device for policy inference.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--train-args", default=None, help="TD3 only; defaults to args.yaml beside the checkpoint.")
    parser.add_argument("--args-file", default=None, help="Real-world runtime defaults YAML.")
    parser.add_argument(
        "--no-artifacts",
        dest="artifacts",
        action="store_false",
        default=None,
        help="Skip episode GIFs and camera video (useful for sim smoke tests).",
    )
    parser.add_argument("--verbose", action="store_true", default=None, help="Restore per-step debug prints.")

    argv = list(sys.argv[1:] if argv is None else argv)
    cli = parser.parse_args(argv)

    # Options default to None so a --run-config can supply them; the real
    # defaults land after the YAML merge.
    cli = _apply_run_config(cli, argv)
    for key, default in (
        ("episodes", 10),
        ("max_attempts", 0),
        ("data_root_dir", "real_runs/eval"),
        ("device", "cpu"),
        ("seed", 0),
        ("args_file", DEFAULT_ARGS_FILE),
        ("artifacts", True),
        ("verbose", False),
    ):
        if getattr(cli, key, None) is None:
            setattr(cli, key, default)

    missing = [name for name in ("config", "checkpoint") if not getattr(cli, name, None)]
    if missing:
        raise SystemExit(f"missing required option(s): {', '.join('--' + m for m in missing)}")
    if not os.path.exists(cli.config):
        raise SystemExit(f"--config does not exist: {cli.config}")
    if not os.path.exists(cli.checkpoint):
        raise SystemExit(f"--checkpoint does not exist: {cli.checkpoint}")
    if int(cli.episodes) <= 0:
        raise SystemExit(f"--episodes must be > 0, got {cli.episodes}")
    return cli


def main(argv: Optional[list[str]] = None) -> None:
    cli = parse_cli(argv)

    algo = detect_algo(cli.checkpoint) if cli.algo == "auto" else cli.algo
    train_args, train_args_path = resolve_train_args(algo, cli.checkpoint, cli.train_args)
    args = build_args(cli, algo, train_args_path)

    with open(cli.config) as f:
        simulator = (yaml.load(f, Loader=yaml.FullLoader) or {}).get("air_hockey", {}).get("simulator", "?")
    print(
        f"[evaluate_policy] algo={algo}{' (auto)' if cli.algo == 'auto' else ''} "
        f"simulator={simulator} config={cli.config} model_path={args.model_path} "
        f"train_args={train_args_path or '-'} episodes={cli.episodes}"
    )

    eval_args = EvalSpecificArgs(
        eval_episodes=int(cli.episodes),
        eval_max_attempts=int(cli.max_attempts),
        quiet=not bool(cli.verbose),
        agent=algo,
    )
    _setup_run_data_dir(args, run_note="")
    run_frozen_policy_eval(args, train_args, eval_args)


if __name__ == "__main__":
    main()
