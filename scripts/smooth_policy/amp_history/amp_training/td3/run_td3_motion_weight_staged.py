#!/usr/bin/env python3
"""Run staged TD3 training while increasing motion reward weight.

This script launches TD3 runs sequentially, monitors TensorBoard scalars from each
run directory, and stops each stage when both q1 metrics stabilize or a hard
step cap is reached.
"""

from __future__ import annotations

import argparse
import json
import re
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
class StageResult:
    stage_index: int
    motion_reward_weight: float
    stage_dir: str
    stop_reason: str
    end_step: int | None
    checkpoint_path: str
    rel_delta_task: float | None
    rel_delta_motion: float | None
    started_from_checkpoint: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged TD3 training with gradually increasing motion reward weight.",
    )
    parser.add_argument(
        "--args-file",
        required=True,
        help="YAML args file passed to amp_training_td3.py",
    )
    parser.add_argument(
        "--initial-checkpoint",
        required=True,
        help="Initial training_state.pth checkpoint to resume from.",
    )
    parser.add_argument(
        "--log-parent-dir",
        required=True,
        help="Parent directory for all staged runs.",
    )
    parser.add_argument(
        "--device",
        required=True,
        help="CUDA/CPU device string passed to training.",
    )
    parser.add_argument(
        "--start-motion-reward-weight",
        type=float,
        default=0.0,
        help="Initial motion reward weight for first stage.",
    )
    parser.add_argument(
        "--final-motion-reward-weight",
        type=float,
        required=True,
        help="Final motion reward weight to stop at.",
    )
    parser.add_argument(
        "--motion-reward-weight-increment",
        type=float,
        required=True,
        help="Positive increment between staged motion reward weights.",
    )
    parser.add_argument(
        "--window-steps",
        type=int,
        default=50_000,
        help="Sliding window size in timesteps for convergence check.",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.05,
        help="Relative-difference threshold for convergence.",
    )
    parser.add_argument(
        "--max-stage-steps",
        type=int,
        default=500_000,
        help="Maximum additional timesteps per stage (deprecated name; same as force-stop-after-timesteps).",
    )
    parser.add_argument(
        "--force-stop-after-timesteps",
        type=int,
        default=None,
        help=(
            "Force-stop threshold per stage in additional timesteps from stage start. "
            "If set, overrides --max-stage-steps."
        ),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=20.0,
        help="Polling interval for reading TensorBoard scalars.",
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
        help="TD3 train script path.",
    )
    parser.add_argument(
        "--stage-learning-starts",
        type=int,
        default=0,
        help=(
            "Override learning_starts for every staged run. "
            "Set to 0 to avoid random-action warmup in resumed stages."
        ),
    )
    parser.add_argument(
        "--stage-exploration-primitive-chance-pre-learning-starts",
        type=float,
        default=0.0,
        help=(
            "Override exploration_primitive_chance_pre_learning_starts for each stage "
            "(lower to avoid aggressive from-scratch exploration behavior)."
        ),
    )
    parser.add_argument(
        "--stage-exploration-primitive-chance-start",
        type=float,
        default=0.08,
        help=(
            "Override exploration_primitive_chance_start for each stage "
            "(lower starting point for annealing)."
        ),
    )
    parser.add_argument(
        "--stage-exploration-primitive-chance-anneal-steps",
        type=int,
        default=50_000,
        help=(
            "Override exploration_primitive_chance_anneal_steps for each stage "
            "(fewer anneal timesteps)."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.force_stop_after_timesteps is not None:
        args.max_stage_steps = args.force_stop_after_timesteps

    if args.final_motion_reward_weight < args.start_motion_reward_weight:
        raise ValueError("final motion reward weight must be >= start motion reward weight")
    if args.motion_reward_weight_increment <= 0.0:
        raise ValueError("motion reward weight increment must be > 0")
    if args.window_steps <= 0:
        raise ValueError("window steps must be > 0")
    if args.convergence_threshold <= 0.0:
        raise ValueError("convergence threshold must be > 0")
    if args.max_stage_steps <= 0:
        raise ValueError("max stage steps must be > 0")
    if args.poll_seconds <= 0.0:
        raise ValueError("poll seconds must be > 0")
    if args.stage_learning_starts < 0:
        raise ValueError("stage learning starts must be >= 0")
    if args.stage_exploration_primitive_chance_anneal_steps < 0:
        raise ValueError("stage exploration primitive chance anneal steps must be >= 0")
    if args.stage_exploration_primitive_chance_pre_learning_starts < 0.0:
        raise ValueError("stage exploration primitive chance pre-learning starts must be >= 0")
    if args.stage_exploration_primitive_chance_start < 0.0:
        raise ValueError("stage exploration primitive chance start must be >= 0")


def build_weight_schedule(start: float, final: float, increment: float) -> List[float]:
    if final < start:
        raise ValueError("final must be >= start")
    if increment <= 0.0:
        raise ValueError("increment must be > 0")

    weights: List[float] = []
    current = float(start)
    # Round to avoid floating-point drift in tags and comparisons.
    while current < final - 1e-12:
        weights.append(round(current, 10))
        current += increment
    weights.append(round(final, 10))
    return weights


def format_weight_tag(weight: float) -> str:
    return f"{weight:.6f}".rstrip("0").rstrip(".").replace("-", "neg").replace(".", "p")


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
            "tensorboard is required to monitor training scalars. "
            "Install optional train dependencies."
        ) from exc

    accumulator = event_accumulator.EventAccumulator(str(log_dir))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        return []
    return [(int(evt.step), float(evt.value)) for evt in accumulator.Scalars(tag)]


def window_mean(points: Sequence[Tuple[int, float]], step_min_exclusive: int, step_max_inclusive: int) -> float | None:
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
    converged = task_delta <= threshold and motion_delta <= threshold
    return StageConvergenceStatus(
        converged=converged,
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


def terminate_process(process: subprocess.Popen, soft_timeout: float = 60.0, term_timeout: float = 30.0) -> None:
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


def build_stage_command(
    *,
    args: argparse.Namespace,
    train_script_path: Path,
    stage_dir: Path,
    run_name: str,
    motion_reward_weight: float,
    stage_start_checkpoint: Path,
    stage_total_timesteps: int,
) -> List[str]:
    return [
        args.python_executable,
        str(train_script_path),
        "--args-file",
        args.args_file,
        "--device",
        args.device,
        "--log-parent-dir",
        str(stage_dir),
        "--run-name",
        run_name,
        "--motion-reward-weight",
        str(motion_reward_weight),
        "--model-path",
        str(stage_start_checkpoint),
        "--total-timesteps",
        str(stage_total_timesteps),
        "--learning-starts",
        str(args.stage_learning_starts),
        "--exploration-primitive-chance-pre-learning-starts",
        str(args.stage_exploration_primitive_chance_pre_learning_starts),
        "--exploration-primitive-chance-start",
        str(args.stage_exploration_primitive_chance_start),
        "--exploration-primitive-chance-anneal-steps",
        str(args.stage_exploration_primitive_chance_anneal_steps),
    ]


def run_stage(
    *,
    stage_index: int,
    motion_reward_weight: float,
    stage_start_checkpoint: Path,
    stage_start_step: int,
    args: argparse.Namespace,
    repo_root: Path,
) -> StageResult:
    stage_weight_tag = format_weight_tag(motion_reward_weight)
    stage_dir = Path(args.log_parent_dir) / f"stage_{stage_index:02d}_w{stage_weight_tag}"

    train_script_path = Path(args.train_script)
    if not train_script_path.is_absolute():
        train_script_path = repo_root / train_script_path

    stage_total_timesteps = int(stage_start_step + args.max_stage_steps)
    run_name = f"td3_stage_{stage_index:02d}_w{stage_weight_tag}"
    print(
        f"[stage {stage_index}] starting: weight={motion_reward_weight}, "
        f"start_step={stage_start_step}, force_stop_step={stage_total_timesteps}"
    )
    print(f"[stage {stage_index}] log_dir={stage_dir}")
    print(f"[stage {stage_index}] resume_checkpoint={stage_start_checkpoint}")
    command = build_stage_command(
        args=args,
        train_script_path=train_script_path,
        stage_dir=stage_dir,
        run_name=run_name,
        motion_reward_weight=motion_reward_weight,
        stage_start_checkpoint=stage_start_checkpoint,
        stage_total_timesteps=stage_total_timesteps,
    )

    launcher_log_path = Path(args.log_parent_dir) / f"{stage_dir.name}_launcher.log"
    launcher_command_path = Path(args.log_parent_dir) / f"{stage_dir.name}_launcher_command.txt"
    with launcher_command_path.open("w", encoding="utf-8") as command_file:
        command_file.write(" ".join(command) + "\n")
    print(f"[stage {stage_index}] command saved to {launcher_command_path}")
    print(f"[stage {stage_index}] launching training process...")

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
                print(
                    f"[stage {stage_index}] detected alternate training log dir: {alternate_dir}"
                )
                active_stage_dir = alternate_dir

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
                # Keep polling while events may still be initializing.
                pass

            if convergence_status.end_step is not None:
                task_delta_msg = (
                    f"{convergence_status.rel_delta_task:.4f}"
                    if convergence_status.rel_delta_task is not None
                    else "n/a"
                )
                motion_delta_msg = (
                    f"{convergence_status.rel_delta_motion:.4f}"
                    if convergence_status.rel_delta_motion is not None
                    else "n/a"
                )
                print(
                    f"[stage {stage_index}] monitor: step={convergence_status.end_step}, "
                    f"task_delta={task_delta_msg}, motion_delta={motion_delta_msg}, "
                    f"converged={convergence_status.converged}"
                )
            else:
                print(f"[stage {stage_index}] monitor: waiting for TensorBoard scalars...")

            if convergence_status.converged:
                stop_reason = "converged"
                print(f"[stage {stage_index}] convergence reached; stopping stage process.")
                terminate_process(process)
                break

            latest_logged_step = convergence_status.end_step
            if latest_logged_step is not None and latest_logged_step >= stage_total_timesteps:
                stop_reason = "max_stage_steps"
                print(
                    f"[stage {stage_index}] reached force-stop threshold "
                    f"(latest_logged_step={latest_logged_step}); stopping stage process."
                )
                terminate_process(process)
                break

            if process_state is not None:
                if process_state != 0:
                    print(
                        f"[stage {stage_index}] process exited unexpectedly "
                        f"with code {process_state}. Check {launcher_log_path}."
                    )
                    raise RuntimeError(
                        f"Training process exited with code {process_state} for stage {stage_index}"
                    )
                print(f"[stage {stage_index}] process exited normally.")
                break

    latest_checkpoint = find_latest_stage_checkpoint(active_stage_dir)
    print(f"[stage {stage_index}] latest checkpoint: {latest_checkpoint}")
    print(f"[stage {stage_index}] stop_reason={stop_reason}")
    return StageResult(
        stage_index=stage_index,
        motion_reward_weight=motion_reward_weight,
        stage_dir=str(active_stage_dir),
        stop_reason=stop_reason,
        end_step=convergence_status.end_step,
        checkpoint_path=str(latest_checkpoint),
        rel_delta_task=convergence_status.rel_delta_task,
        rel_delta_motion=convergence_status.rel_delta_motion,
        started_from_checkpoint=str(stage_start_checkpoint),
    )


def write_summary(log_parent_dir: Path, stage_results: Iterable[StageResult]) -> Path:
    summary_path = log_parent_dir / "summary.json"
    payload: Dict[str, object] = {
        "generated_at_unix": time.time(),
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

    ensure_exists(args_file_path, "args file")
    ensure_exists(initial_checkpoint, "initial checkpoint")
    ensure_exists(Path(args.python_executable), "python executable")

    train_script_path = Path(args.train_script)
    if not train_script_path.is_absolute():
        train_script_path = repo_root / train_script_path
    ensure_exists(train_script_path, "train script")

    log_parent_dir.mkdir(parents=True, exist_ok=True)

    weights = build_weight_schedule(
        start=args.start_motion_reward_weight,
        final=args.final_motion_reward_weight,
        increment=args.motion_reward_weight_increment,
    )
    print(f"Resolved weight schedule ({len(weights)} stages): {weights}")
    print(
        "Stage settings: "
        f"window_steps={args.window_steps}, threshold={args.convergence_threshold}, "
        f"force_stop_after={args.max_stage_steps}, poll_seconds={args.poll_seconds}"
    )
    print(
        "Exploration overrides per stage: "
        f"learning_starts={args.stage_learning_starts}, "
        f"pre_learning_chance={args.stage_exploration_primitive_chance_pre_learning_starts}, "
        f"anneal_start={args.stage_exploration_primitive_chance_start}, "
        f"anneal_steps={args.stage_exploration_primitive_chance_anneal_steps}"
    )

    stage_results: List[StageResult] = []
    current_checkpoint = initial_checkpoint
    for stage_index, motion_reward_weight in enumerate(weights):
        stage_start_step = load_checkpoint_step(current_checkpoint)
        stage_result = run_stage(
            stage_index=stage_index,
            motion_reward_weight=motion_reward_weight,
            stage_start_checkpoint=current_checkpoint,
            stage_start_step=stage_start_step,
            args=args,
            repo_root=repo_root,
        )
        stage_results.append(stage_result)
        current_checkpoint = Path(stage_result.checkpoint_path)
        print(
            f"[stage {stage_index}] complete. Next stage will resume from: {current_checkpoint}"
        )

    summary_path = write_summary(log_parent_dir, stage_results)
    print(f"Completed {len(stage_results)} stage(s). Summary: {summary_path}")


if __name__ == "__main__":
    main()
