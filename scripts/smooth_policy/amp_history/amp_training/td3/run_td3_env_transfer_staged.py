#!/usr/bin/env python3
"""Run two-stage TD3 environment transfer from an existing checkpoint.

Stage 1: Q-only adaptation on target environment settings.
Stage 2: Normal TD3 training from stage 1 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


SCALAR_Q1_TASK_MEAN = "losses/q1_task_mean"
SCALAR_Q1_MOTION_MEAN = "losses/q1_motion_mean"
CHECKPOINT_NAME = "training_state.pth"
CHECKPOINT_DIR_PATTERN = re.compile(r"checkpoint_(\d+)$")
ALTERNATE_LOG_DIR_PATTERN = re.compile(r"Saving to alternate log directory:\s*(.+)$")


@dataclass
class StageConvergenceStatus:
    converged: bool
    end_step: int | None
    rel_delta_task: float | None
    rel_delta_motion: float | None


@dataclass
class ExplorationSettings:
    pre_learning_steps: int
    pre_learning_chance: float
    chance_start: float
    chance_end: float
    anneal_steps: int


@dataclass
class StageSpec:
    stage_key: str
    stage_name: str
    stage_dir_name: str
    additional_steps: int
    q_only: bool
    stop_on_convergence: bool


@dataclass
class StageResult:
    seed: int | None
    stage_key: str
    stage_name: str
    stage_dir: str
    stop_reason: str
    start_step: int
    target_step: int
    end_step: int | None
    checkpoint_path: str
    rel_delta_task: float | None
    rel_delta_motion: float | None
    started_from_checkpoint: str
    q_only: bool
    motion_reward_weight: float | str
    q_updates: int | str
    target_network_frequency: int | str
    exploration: Dict[str, float | int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged TD3 environment transfer with Q-only adaptation then full training.",
    )
    parser.add_argument("--args-file", required=True, help="Target-environment TD3 args YAML.")
    parser.add_argument(
        "--initial-checkpoint",
        required=True,
        help="Source policy training_state.pth checkpoint to resume from.",
    )
    parser.add_argument("--log-parent-dir", required=True, help="Parent directory for staged runs.")
    parser.add_argument("--device", required=True, help="CUDA/CPU device string for training.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of seeds to run. If provided, runs all stages once per seed "
            "under log-parent-dir/seed_<seed>/..."
        ),
    )
    parser.add_argument(
        "--stage1-max-steps",
        type=int,
        default=500_000,
        help="Maximum additional timesteps for stage 1 Q-only adaptation.",
    )
    parser.add_argument(
        "--stage2-steps",
        type=int,
        default=1_000_000,
        help="Additional timesteps for stage 2 normal training.",
    )
    parser.add_argument(
        "--window-steps",
        type=int,
        default=50_000,
        help="Sliding window size for convergence check.",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.05,
        help="Relative-delta threshold for convergence.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=20.0,
        help="Polling interval for stage monitoring.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used to launch TD3 training.",
    )
    parser.add_argument(
        "--train-script",
        default=(
            "scripts/smooth_policy/amp_history/amp_training/td3/"
            "amp_training_td3.py"
        ),
        help="Path to TD3 train script.",
    )
    parser.add_argument(
        "--motion-reward-weight",
        type=float,
        default=None,
        help=(
            "Optional global motion reward weight override forwarded to training. "
            "If omitted, args-file value is used."
        ),
    )
    parser.add_argument(
        "--stage1-motion-reward-weight",
        type=float,
        default=None,
        help="Optional stage 1 motion reward weight override.",
    )
    parser.add_argument(
        "--stage2-motion-reward-weight",
        type=float,
        default=None,
        help="Optional stage 2 motion reward weight override.",
    )
    parser.add_argument(
        "--q-updates",
        type=int,
        default=None,
        help=(
            "Optional global q_updates override forwarded to training. "
            "If omitted, args-file value is used."
        ),
    )
    parser.add_argument(
        "--stage1-q-updates",
        type=int,
        default=None,
        help="Optional stage 1 q_updates override.",
    )
    parser.add_argument(
        "--stage2-q-updates",
        type=int,
        default=None,
        help="Optional stage 2 q_updates override.",
    )
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=None,
        help=(
            "Optional global target_network_frequency override forwarded to training. "
            "If omitted, args-file value is used."
        ),
    )
    parser.add_argument(
        "--stage1-target-network-frequency",
        type=int,
        default=None,
        help="Optional stage 1 target_network_frequency override.",
    )
    parser.add_argument(
        "--stage2-target-network-frequency",
        type=int,
        default=None,
        help="Optional stage 2 target_network_frequency override.",
    )

    # Global exploration overrides. These are stage-relative values and are converted
    # to absolute global-step thresholds expected by amp_training_td3.py.
    parser.add_argument(
        "--exploration-pre-learning-steps",
        type=int,
        default=30_000,
        help=(
            "Stage-relative steps to keep pre-learning exploration chance active. "
            "Converted to absolute --learning-starts per stage."
        ),
    )
    parser.add_argument(
        "--exploration-primitive-chance-pre-learning-starts",
        type=float,
        default=0.15,
        help="Exploration chance to use during pre-learning phase.",
    )
    parser.add_argument(
        "--exploration-primitive-chance-start",
        type=float,
        default=0.05,
        help="Starting primitive exploration chance for stage annealing.",
    )
    parser.add_argument(
        "--exploration-primitive-chance",
        "--exploration-primitive-chance-final",
        dest="exploration_primitive_chance",
        type=float,
        default=None,
        help=(
            "Final primitive exploration chance after annealing. "
            "Alias: --exploration-primitive-chance-final."
        ),
    )
    parser.add_argument(
        "--exploration-primitive-chance-anneal-steps",
        type=int,
        default=30_000,
        help=(
            "Stage-relative anneal duration in steps. "
            "Converted to absolute --exploration-primitive-chance-anneal-steps per stage."
        ),
    )

    # Optional stage-specific exploration overrides.
    parser.add_argument("--stage1-exploration-pre-learning-steps", type=int, default=None)
    parser.add_argument(
        "--stage1-exploration-primitive-chance-pre-learning-starts",
        type=float,
        default=None,
    )
    parser.add_argument("--stage1-exploration-primitive-chance-start", type=float, default=None)
    parser.add_argument(
        "--stage1-exploration-primitive-chance",
        "--stage1-exploration-primitive-chance-final",
        dest="stage1_exploration_primitive_chance",
        type=float,
        default=None,
    )
    parser.add_argument("--stage1-exploration-primitive-chance-anneal-steps", type=int, default=None)
    parser.add_argument("--stage2-exploration-pre-learning-steps", type=int, default=None)
    parser.add_argument(
        "--stage2-exploration-primitive-chance-pre-learning-starts",
        type=float,
        default=None,
    )
    parser.add_argument("--stage2-exploration-primitive-chance-start", type=float, default=None)
    parser.add_argument(
        "--stage2-exploration-primitive-chance",
        "--stage2-exploration-primitive-chance-final",
        dest="stage2_exploration_primitive_chance",
        type=float,
        default=None,
    )
    parser.add_argument("--stage2-exploration-primitive-chance-anneal-steps", type=int, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.stage1_max_steps < 0:
        raise ValueError("stage1-max-steps must be >= 0")
    if args.stage2_steps <= 0:
        raise ValueError("stage2-steps must be > 0")
    if args.window_steps <= 0:
        raise ValueError("window-steps must be > 0")
    if args.convergence_threshold <= 0:
        raise ValueError("convergence-threshold must be > 0")
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be > 0")
    if args.seeds is not None and len(args.seeds) == 0:
        raise ValueError("if provided, seeds must contain at least one value")

    int_fields = [
        "q_updates",
        "stage1_q_updates",
        "stage2_q_updates",
        "target_network_frequency",
        "stage1_target_network_frequency",
        "stage2_target_network_frequency",
        "exploration_pre_learning_steps",
        "exploration_primitive_chance_anneal_steps",
        "stage1_exploration_pre_learning_steps",
        "stage1_exploration_primitive_chance_anneal_steps",
        "stage2_exploration_pre_learning_steps",
        "stage2_exploration_primitive_chance_anneal_steps",
    ]
    for field_name in int_fields:
        value = getattr(args, field_name)
        if value is not None and int(value) < 0:
            raise ValueError(f"{field_name} must be >= 0")
        if value is not None and field_name in {
            "q_updates",
            "stage1_q_updates",
            "stage2_q_updates",
            "target_network_frequency",
            "stage1_target_network_frequency",
            "stage2_target_network_frequency",
        } and int(value) <= 0:
            raise ValueError(f"{field_name} must be > 0")

    float_fields = [
        "motion_reward_weight",
        "stage1_motion_reward_weight",
        "stage2_motion_reward_weight",
        "exploration_primitive_chance_pre_learning_starts",
        "exploration_primitive_chance_start",
        "exploration_primitive_chance",
        "stage1_exploration_primitive_chance_pre_learning_starts",
        "stage1_exploration_primitive_chance_start",
        "stage1_exploration_primitive_chance",
        "stage2_exploration_primitive_chance_pre_learning_starts",
        "stage2_exploration_primitive_chance_start",
        "stage2_exploration_primitive_chance",
    ]
    for field_name in float_fields:
        value = getattr(args, field_name)
        if value is not None and float(value) < 0.0:
            raise ValueError(f"{field_name} must be >= 0")


def resolve_stage_motion_reward_weight(args: argparse.Namespace, stage_key: str) -> float | None:
    stage_attr = f"{stage_key}_motion_reward_weight"
    stage_value = getattr(args, stage_attr, None)
    if stage_value is not None:
        return float(stage_value)
    if args.motion_reward_weight is not None:
        return float(args.motion_reward_weight)
    return None


def resolve_stage_q_updates(args: argparse.Namespace, stage_key: str) -> int | None:
    stage_attr = f"{stage_key}_q_updates"
    stage_value = getattr(args, stage_attr, None)
    if stage_value is not None:
        return int(stage_value)
    if args.q_updates is not None:
        return int(args.q_updates)
    return None


def resolve_stage_target_network_frequency(args: argparse.Namespace, stage_key: str) -> int | None:
    stage_attr = f"{stage_key}_target_network_frequency"
    stage_value = getattr(args, stage_attr, None)
    if stage_value is not None:
        return int(stage_value)
    if args.target_network_frequency is not None:
        return int(args.target_network_frequency)
    return None


def parse_checkpoint_step_from_path(checkpoint_path: Path) -> int | None:
    for parent in [checkpoint_path.parent, *checkpoint_path.parents]:
        match = CHECKPOINT_DIR_PATTERN.search(parent.name)
        if match:
            return int(match.group(1))
    return None


def load_checkpoint_step(checkpoint_path: Path) -> int:
    parsed = parse_checkpoint_step_from_path(checkpoint_path)
    if parsed is not None:
        return parsed

    checkpoint_obj = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint_obj, dict) or "global_step" not in checkpoint_obj:
        raise ValueError(f"{checkpoint_path} does not contain a global_step")
    return int(checkpoint_obj["global_step"])


def read_scalar_points(log_dir: Path, tag: str) -> List[Tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing import event_accumulator
    except Exception as exc:
        raise ImportError(
            "tensorboard is required to monitor training scalars. Install optional train dependencies."
        ) from exc

    accumulator = event_accumulator.EventAccumulator(str(log_dir))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        return []
    return [(int(evt.step), float(evt.value)) for evt in accumulator.Scalars(tag)]


def window_mean(
    points: Sequence[Tuple[int, float]],
    step_min_exclusive: int,
    step_max_inclusive: int,
) -> float | None:
    values = [
        value
        for step, value in points
        if step_min_exclusive < step <= step_max_inclusive
    ]
    if not values:
        return None
    return float(np.mean(values))


def relative_delta(curr: float, prev: float, eps: float = 1e-8) -> float:
    return abs(curr - prev) / max(abs(prev), eps)


def evaluate_convergence(
    q1_task_points: Sequence[Tuple[int, float]],
    q1_motion_points: Sequence[Tuple[int, float]],
    window_steps: int,
    threshold: float,
) -> StageConvergenceStatus:
    if not q1_task_points or not q1_motion_points:
        return StageConvergenceStatus(False, None, None, None)

    latest_step = min(q1_task_points[-1][0], q1_motion_points[-1][0])
    if latest_step < 2 * window_steps:
        return StageConvergenceStatus(False, latest_step, None, None)

    prev_start = latest_step - 2 * window_steps
    prev_end = latest_step - window_steps
    curr_start = prev_end
    curr_end = latest_step

    task_prev = window_mean(q1_task_points, prev_start, prev_end)
    task_curr = window_mean(q1_task_points, curr_start, curr_end)
    motion_prev = window_mean(q1_motion_points, prev_start, prev_end)
    motion_curr = window_mean(q1_motion_points, curr_start, curr_end)
    if any(v is None for v in (task_prev, task_curr, motion_prev, motion_curr)):
        return StageConvergenceStatus(False, latest_step, None, None)

    task_delta = relative_delta(float(task_curr), float(task_prev))
    motion_delta = relative_delta(float(motion_curr), float(motion_prev))
    return StageConvergenceStatus(
        converged=(task_delta <= threshold and motion_delta <= threshold),
        end_step=latest_step,
        rel_delta_task=task_delta,
        rel_delta_motion=motion_delta,
    )


def find_latest_stage_checkpoint(stage_dir: Path) -> Path:
    checkpoint_candidates: List[Tuple[int, Path]] = []
    for child in stage_dir.iterdir():
        if not child.is_dir():
            continue
        match = CHECKPOINT_DIR_PATTERN.fullmatch(child.name)
        if not match:
            continue
        checkpoint_file = child / CHECKPOINT_NAME
        if checkpoint_file.exists():
            checkpoint_candidates.append((int(match.group(1)), checkpoint_file))

    if checkpoint_candidates:
        checkpoint_candidates.sort(key=lambda item: item[0])
        return checkpoint_candidates[-1][1]

    fallback_checkpoint = stage_dir / CHECKPOINT_NAME
    if fallback_checkpoint.exists():
        return fallback_checkpoint
    raise FileNotFoundError(f"No checkpoint found in stage dir: {stage_dir}")


def terminate_process(
    process: subprocess.Popen,
    soft_timeout: float = 60.0,
    term_timeout: float = 30.0,
) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=soft_timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=term_timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    if process.poll() is None:
        process.kill()
    process.wait()


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def ensure_python_available(python_executable: str) -> None:
    candidate = Path(python_executable)
    if candidate.is_absolute():
        ensure_exists(candidate, "python executable")
        return
    if shutil.which(python_executable) is None:
        raise FileNotFoundError(f"python executable not found in PATH: {python_executable}")


def detect_alternate_log_dir(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    try:
        with log_path.open("r", encoding="utf-8") as log_file:
            for line in reversed(log_file.readlines()):
                match = ALTERNATE_LOG_DIR_PATTERN.search(line.strip())
                if match:
                    return Path(match.group(1)).resolve()
    except Exception:
        return None
    return None


def resolve_stage_exploration_settings(args: argparse.Namespace, stage_key: str) -> ExplorationSettings:
    prefix = f"{stage_key}_"

    def pick(suffix: str, global_attr: str):
        stage_attr = prefix + suffix
        stage_value = getattr(args, stage_attr, None)
        if stage_value is not None:
            return stage_value
        return getattr(args, global_attr)

    chance_end = pick("exploration_primitive_chance", "exploration_primitive_chance")
    if chance_end is None:
        # None means "leave args-file default unchanged", which we do by skipping
        # this arg in command construction.
        chance_end = -1.0

    return ExplorationSettings(
        pre_learning_steps=int(
            pick("exploration_pre_learning_steps", "exploration_pre_learning_steps")
        ),
        pre_learning_chance=float(
            pick(
                "exploration_primitive_chance_pre_learning_starts",
                "exploration_primitive_chance_pre_learning_starts",
            )
        ),
        chance_start=float(
            pick(
                "exploration_primitive_chance_start",
                "exploration_primitive_chance_start",
            )
        ),
        chance_end=float(chance_end),
        anneal_steps=int(
            pick(
                "exploration_primitive_chance_anneal_steps",
                "exploration_primitive_chance_anneal_steps",
            )
        ),
    )


def build_stage_command(
    *,
    args: argparse.Namespace,
    train_script_path: Path,
    stage_dir: Path,
    stage_spec: StageSpec,
    stage_start_checkpoint: Path,
    stage_start_step: int,
    target_total_timesteps: int,
    exploration: ExplorationSettings,
    motion_reward_weight: float | None,
    q_updates: int | None,
    target_network_frequency: int | None,
    seed: int | None,
) -> List[str]:
    command = [
        args.python_executable,
        str(train_script_path),
        "--args-file",
        args.args_file,
        "--device",
        args.device,
        "--log-parent-dir",
        str(stage_dir),
        "--run-name",
        stage_spec.stage_dir_name,
        "--model-path",
        str(stage_start_checkpoint),
        "--total-timesteps",
        str(target_total_timesteps),
        "--learning-starts",
        str(stage_start_step + exploration.pre_learning_steps),
        "--exploration-primitive-chance-pre-learning-starts",
        str(exploration.pre_learning_chance),
        "--exploration-primitive-chance-start",
        str(exploration.chance_start),
        "--exploration-primitive-chance-anneal-steps",
        str(stage_start_step + exploration.anneal_steps),
    ]
    if exploration.chance_end >= 0.0:
        command.extend(["--exploration-primitive-chance", str(exploration.chance_end)])
    if motion_reward_weight is not None:
        command.extend(["--motion-reward-weight", str(motion_reward_weight)])
    if q_updates is not None:
        command.extend(["--q-updates", str(q_updates)])
    if target_network_frequency is not None:
        command.extend(["--target-network-frequency", str(target_network_frequency)])
    if stage_spec.q_only:
        command.extend(["--policy-lr", "0.0"])
    if seed is not None:
        command.extend(["--seed", str(seed)])
        command.extend(["--run-name", f"{stage_spec.stage_dir_name}_seed{seed}"])
    return command


def run_stage(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    train_script_path: Path,
    stage_spec: StageSpec,
    stage_start_checkpoint: Path,
    stage_start_step: int,
    seed: int | None,
    run_log_parent_dir: Path,
) -> StageResult:
    stage_dir = run_log_parent_dir / stage_spec.stage_dir_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_target_step = int(stage_start_step + stage_spec.additional_steps)
    exploration = resolve_stage_exploration_settings(args, stage_spec.stage_key)
    motion_reward_weight = resolve_stage_motion_reward_weight(args, stage_spec.stage_key)
    q_updates = resolve_stage_q_updates(args, stage_spec.stage_key)
    target_network_frequency = resolve_stage_target_network_frequency(args, stage_spec.stage_key)
    command = build_stage_command(
        args=args,
        train_script_path=train_script_path,
        stage_dir=stage_dir,
        stage_spec=stage_spec,
        stage_start_checkpoint=stage_start_checkpoint,
        stage_start_step=stage_start_step,
        target_total_timesteps=stage_target_step,
        exploration=exploration,
        motion_reward_weight=motion_reward_weight,
        q_updates=q_updates,
        target_network_frequency=target_network_frequency,
        seed=seed,
    )

    launcher_log_path = run_log_parent_dir / f"{stage_spec.stage_dir_name}_launcher.log"
    launcher_command_path = run_log_parent_dir / f"{stage_spec.stage_dir_name}_launcher_command.txt"
    with launcher_command_path.open("w", encoding="utf-8") as command_file:
        command_file.write(" ".join(command) + "\n")

    print(
        f"[{stage_spec.stage_key}] start_step={stage_start_step}, target_step={stage_target_step}, "
        f"q_only={stage_spec.q_only}"
    )
    print(
        f"[{stage_spec.stage_key}] exploration: pre_steps={exploration.pre_learning_steps}, "
        f"pre_chance={exploration.pre_learning_chance}, chance_start={exploration.chance_start}, "
        f"chance_end={'args_default' if exploration.chance_end < 0 else exploration.chance_end}, "
        f"anneal_steps={exploration.anneal_steps}"
    )
    print(
        f"[{stage_spec.stage_key}] motion_reward_weight="
        f"{'args_file_default' if motion_reward_weight is None else motion_reward_weight}"
    )
    print(
        f"[{stage_spec.stage_key}] q_updates="
        f"{'args_file_default' if q_updates is None else q_updates}"
    )
    print(
        f"[{stage_spec.stage_key}] target_network_frequency="
        f"{'args_file_default' if target_network_frequency is None else target_network_frequency}"
    )

    active_stage_dir = stage_dir
    with launcher_log_path.open("a", encoding="utf-8") as stage_log_file:
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=stage_log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        convergence_status = StageConvergenceStatus(False, None, None, None)
        stop_reason = "process_exited"

        while True:
            time.sleep(args.poll_seconds)
            process_state = process.poll()
            alternate_dir = detect_alternate_log_dir(launcher_log_path)
            if alternate_dir is not None and alternate_dir != active_stage_dir:
                active_stage_dir = alternate_dir
                print(f"[{stage_spec.stage_key}] detected alternate log dir: {active_stage_dir}")

            try:
                q1_task_points = read_scalar_points(active_stage_dir, SCALAR_Q1_TASK_MEAN)
                q1_motion_points = read_scalar_points(active_stage_dir, SCALAR_Q1_MOTION_MEAN)
                convergence_status = evaluate_convergence(
                    q1_task_points=q1_task_points,
                    q1_motion_points=q1_motion_points,
                    window_steps=args.window_steps,
                    threshold=args.convergence_threshold,
                )
            except Exception:
                pass

            if convergence_status.end_step is not None:
                print(
                    f"[{stage_spec.stage_key}] monitor step={convergence_status.end_step}, "
                    f"task_delta={convergence_status.rel_delta_task}, "
                    f"motion_delta={convergence_status.rel_delta_motion}, "
                    f"converged={convergence_status.converged}"
                )
            else:
                print(f"[{stage_spec.stage_key}] waiting for scalar events...")

            if stage_spec.stop_on_convergence and convergence_status.converged:
                stop_reason = "converged"
                terminate_process(process)
                break

            latest_logged_step = convergence_status.end_step
            if latest_logged_step is not None and latest_logged_step >= stage_target_step:
                stop_reason = "max_stage_steps"
                terminate_process(process)
                break

            if process_state is not None:
                if process_state != 0:
                    raise RuntimeError(
                        f"{stage_spec.stage_key} exited with code {process_state}. "
                        f"Inspect {launcher_log_path}"
                    )
                break

    latest_checkpoint = find_latest_stage_checkpoint(active_stage_dir)
    end_step = convergence_status.end_step
    if end_step is None:
        try:
            end_step = load_checkpoint_step(latest_checkpoint)
        except Exception:
            end_step = None

    chance_end_summary: float | str = (
        exploration.chance_end if exploration.chance_end >= 0.0 else "args_file_default"
    )
    return StageResult(
        seed=seed,
        stage_key=stage_spec.stage_key,
        stage_name=stage_spec.stage_name,
        stage_dir=str(active_stage_dir),
        stop_reason=stop_reason,
        start_step=stage_start_step,
        target_step=stage_target_step,
        end_step=end_step,
        checkpoint_path=str(latest_checkpoint),
        rel_delta_task=convergence_status.rel_delta_task,
        rel_delta_motion=convergence_status.rel_delta_motion,
        started_from_checkpoint=str(stage_start_checkpoint),
        q_only=stage_spec.q_only,
        motion_reward_weight=(
            motion_reward_weight if motion_reward_weight is not None else "args_file_default"
        ),
        q_updates=(q_updates if q_updates is not None else "args_file_default"),
        target_network_frequency=(
            target_network_frequency
            if target_network_frequency is not None
            else "args_file_default"
        ),
        exploration={
            "pre_learning_steps": exploration.pre_learning_steps,
            "pre_learning_chance": exploration.pre_learning_chance,
            "chance_start": exploration.chance_start,
            "chance_end": chance_end_summary,
            "anneal_steps": exploration.anneal_steps,
            "absolute_learning_starts": stage_start_step + exploration.pre_learning_steps,
            "absolute_anneal_steps": stage_start_step + exploration.anneal_steps,
        },
    )


def write_summary(
    run_log_parent_dir: Path,
    args: argparse.Namespace,
    stage_results: Iterable[StageResult],
    seed: int | None,
) -> Path:
    summary_path = run_log_parent_dir / "summary.json"
    payload: Dict[str, object] = {
        "generated_at_unix": time.time(),
        "args_file": args.args_file,
        "initial_checkpoint": args.initial_checkpoint,
        "device": args.device,
        "seed": seed,
        "stage1_max_steps": args.stage1_max_steps,
        "stage2_steps": args.stage2_steps,
        "window_steps": args.window_steps,
        "convergence_threshold": args.convergence_threshold,
        "stages": [stage_result.__dict__ for stage_result in stage_results],
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(payload, summary_file, indent=2, sort_keys=True)
    return summary_path


def main() -> None:
    args = parse_args()
    validate_args(args)

    repo_root = Path.cwd()
    args_file_path = Path(args.args_file).resolve()
    initial_checkpoint = Path(args.initial_checkpoint).resolve()
    log_parent_dir = Path(args.log_parent_dir).resolve()
    train_script_path = Path(args.train_script)
    if not train_script_path.is_absolute():
        train_script_path = (repo_root / train_script_path).resolve()

    ensure_exists(args_file_path, "args file")
    ensure_exists(initial_checkpoint, "initial checkpoint")
    ensure_exists(train_script_path, "train script")
    ensure_python_available(args.python_executable)
    log_parent_dir.mkdir(parents=True, exist_ok=True)

    stage_specs: List[StageSpec] = []
    if args.stage1_max_steps > 0:
        stage_specs.append(
            StageSpec(
                stage_key="stage1",
                stage_name="q_only_adaptation",
                stage_dir_name="stage_01_q_only_adapt",
                additional_steps=int(args.stage1_max_steps),
                q_only=True,
                stop_on_convergence=True,
            )
        )
    else:
        print("stage1-max-steps=0 -> skipping stage 1 Q-only adaptation.")

    stage_specs.append(
        StageSpec(
            stage_key="stage2",
            stage_name="full_training",
            stage_dir_name="stage_02_full_train",
            additional_steps=int(args.stage2_steps),
            q_only=False,
            stop_on_convergence=False,
        )
    )

    all_seed_summaries: List[str] = []
    seed_values: List[int | None]
    if args.seeds is None:
        seed_values = [None]
    else:
        seed_values = list(dict.fromkeys(args.seeds))

    for seed in seed_values:
        run_log_parent_dir = (
            log_parent_dir if seed is None else log_parent_dir / f"seed_{seed}"
        )
        run_log_parent_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"Starting staged transfer for seed={seed if seed is not None else 'args_file_default'} "
            f"at {run_log_parent_dir}"
        )

        stage_results: List[StageResult] = []
        current_checkpoint = initial_checkpoint
        for stage_spec in stage_specs:
            stage_start_step = load_checkpoint_step(current_checkpoint)
            result = run_stage(
                args=args,
                repo_root=repo_root,
                train_script_path=train_script_path,
                stage_spec=stage_spec,
                stage_start_checkpoint=current_checkpoint,
                stage_start_step=stage_start_step,
                seed=seed,
                run_log_parent_dir=run_log_parent_dir,
            )
            stage_results.append(result)
            current_checkpoint = Path(result.checkpoint_path)
            print(f"[{stage_spec.stage_key}] complete. next_checkpoint={current_checkpoint}")

        summary_path = write_summary(run_log_parent_dir, args, stage_results, seed)
        all_seed_summaries.append(str(summary_path))
        print(f"Completed staged transfer for seed={seed}. Summary: {summary_path}")

    print("All staged transfers complete.")
    for summary_path in all_seed_summaries:
        print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
