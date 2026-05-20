"""Shared runtime library for real-world async TD3 training.

This module hosts the dataclasses (``Args``, ``TrainArgs``, ``LearnerRuntimeState``),
config/args-file plumbing (``_build_args_file_defaults``, ``_load_train_args``),
checkpoint helpers, episode/replay utilities, and the synchronous learner
step (``_init_sync_learner_state`` / ``_run_sync_learner_iteration`` /
``_finalize_sync_learner_state``) used by the real-world TD3 stack.

The runnable entrypoint lives in ``extras/async_td3_real.py``; this file is
imported by it, by the eval / teleop-eval entrypoints, and by the
residual/args-mapping tests, but is not itself executable.
"""
from __future__ import annotations

import copy
import math
import os
import re
import json
import shutil
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Literal, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import tyro
import yaml
from torch.utils.tensorboard import SummaryWriter

from airhockey import AirHockeyEnv
from scripts.td3.helper.episode_artifacts import (
    clean_episode_hdf5,
    generate_episode_camera_video,
    generate_episode_gif,
    save_split_episode_hdf5,
)
from scripts.td3.helper.q_network import TD3QNetwork
from scripts.td3.helper.exploration_selector import (
    PrimitiveExplorationSelector,
)
from scripts.td3.helper.shared_replay import SharedTD3Replay
from scripts.td3.helper.real_collector_factories import (
    build_primitive_exploration_selector_for_real_collector,
)
from scripts.td3.helper.real_episode_buffers import (
    vector_with_width,
)
from scripts.td3.helper.real_stop_state import _classify_stop_event
from scripts.td3.helper.real_warm_start import _warm_start_replay_from_hdf5
from scripts.td3.helper.td3_episode_collection import EpisodeTrajectory
from scripts.td3.helper.td3_replay_sampling import (
    critic_success_failure_counts,
)
from scripts.td3.helper.td3_checkpointing import (
    get_rng_states,
    set_rng_states,
)
from scripts.td3.deterministic_agent import DeterministicAgent
from scripts.td3.residual_agent import ResidualActor, zero_init_residual_head

ROLLING_PERF_WINDOW_EPISODES = 50


# ---------------------------------------------------------------------------
# Quiet-print filter (shared by extras/async_td3_real.py and
# extras/async_td3_real_eval.py). Many helpers in this stack print per-step or
# per-reset diagnostics (control-gate moveL/servoStop, force-wrench
# worker, second-hit capture, reset FSM, etc.) that are useful while
# debugging the real-robot pipeline but pure noise once it's healthy.
# When ``--quiet`` is set, ``install_quiet_print_filter`` monkey-patches
# ``builtins.print`` to drop lines whose stripped text starts with one
# of ``QUIET_SUPPRESS_PREFIXES`` or contains one of
# ``QUIET_SUPPRESS_SUBSTRS``. Everything else (rolling-window summaries,
# checkpoints, warm-start, errors, run-startup banners) still prints.
#
# Default policy: ``--quiet`` is on by default for both the trainer and
# the eval. Pass ``--verbose`` (eval) or ``--no-quiet`` (trainer) to
# restore the full chatter when something looks wrong on the real rig.
# Keep ``[collector_rolling…]`` OUT of the suppression list — that's the
# main qualitative signal someone monitoring training watches over time.
# ---------------------------------------------------------------------------


QUIET_SUPPRESS_PREFIXES: tuple[str, ...] = (
    # real_transition_hold.py — per-reset hold begin / hold_step / hold_complete
    "[collector_transition]",
    # real_reset_runner.py + scripts/real/rollout_reset_policy_real.py
    "[reset_fsm]",
    "[reset_fsm_force]",
    "[collector_fallback_reset]",
    "[fallback_reset]",
    # real_reset_runner.py — per-episode reset path / soft-reset stats
    "[collector] episode_id=",
    # real_policy_runner.py — robot readiness recoveries / failures
    "[collector_safety]",
    # scripts/real/rollout_reset_policy_real.py — UR5 control gating
    "[control_gate]",
    "[control_debug]",
    # async render background worker
    "[async_render]",
    # rollout_reset_policy_real second-hit capture / path viz
    "[second_hit_capture]",
    "[reset_path]",
    # trajectory_merging.py per-episode HDF5 merge log
    "[trajectory]",
)

QUIET_SUPPRESS_SUBSTRS: tuple[str, ...] = (
    # episode_artifacts.py — per-episode camera-video render block
    "Generating camera video",
    "Total camera frames:",
    "Frames to render:",
    "Subsample factor:",
    "Frame size:",
    "Playback FPS:",
    "Fallback codec:",
    "Writing frame ",
    "Saving camera video",
    # rollout_reset_policy_real / air_hockey_real.py per-reset chatter
    "reset to initial pose:",
    "Press space to start",
    "Saving trajectory and resetting...",
    "Resetting without saving...",
    "Force-triggering reset mode...",
)


def install_quiet_print_filter() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Replace ``builtins.print`` with a prefix/substring filter.

    Returns the (prefixes, substrings) pair so the caller can echo the
    contents to the user, making it auditable what was silenced.
    Idempotent: if already installed, returns the lists without
    re-wrapping.
    """
    import builtins

    if getattr(builtins.print, "_air_hockey_quiet_filter_installed", False):
        return QUIET_SUPPRESS_PREFIXES, QUIET_SUPPRESS_SUBSTRS

    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        if args:
            text = " ".join(str(a) for a in args).lstrip()
            for prefix in QUIET_SUPPRESS_PREFIXES:
                if text.startswith(prefix):
                    return
            for substr in QUIET_SUPPRESS_SUBSTRS:
                if substr in text:
                    return
        original_print(*args, **kwargs)

    filtered_print._air_hockey_quiet_filter_installed = True  # type: ignore[attr-defined]
    builtins.print = filtered_print
    return QUIET_SUPPRESS_PREFIXES, QUIET_SUPPRESS_SUBSTRS


def _prepare_air_hockey_config(config: dict, seed: int = 0, return_goal_obs: bool = False) -> dict:
    """Build an air_hockey config dict with required top-level fields merged in."""
    ah = dict(config["air_hockey"])
    if "seed" not in ah:
        seed_cfg = config.get("seed", seed)
        if isinstance(seed_cfg, (list, tuple)):
            seed_cfg = seed_cfg[0] if len(seed_cfg) > 0 else 0
        ah["seed"] = int(seed_cfg)
    if "n_training_steps" not in ah:
        ah["n_training_steps"] = config.get("n_training_steps", 1000000)
    if "return_goal_obs" not in ah:
        ah["return_goal_obs"] = return_goal_obs
    return ah


def h_transform(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + eps * x


def h_inverse(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    abs_x = torch.abs(x)
    inner = 1 + 4 * eps * (abs_x + 1 + eps)
    sqrt_inner = torch.sqrt(inner)
    quotient = (sqrt_inner - 1) / (2 * eps)
    return torch.sign(x) * (quotient**2 - 1)


def augment_policy_observation(
    observation: torch.Tensor,
    last_action: torch.Tensor,
    use_last_action_in_policy_state: bool,
) -> torch.Tensor:
    if not use_last_action_in_policy_state:
        return observation
    return torch.cat([observation, last_action], dim=-1)


def deterministic_actor_action(actor: DeterministicAgent, policy_obs: torch.Tensor) -> torch.Tensor:
    if hasattr(actor, "get_action"):
        return actor.get_action(policy_obs)
    raise TypeError(f"Unsupported actor type for deterministic action: {type(actor)}")


def extract_deterministic_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    deterministic_state = {}
    for key, value in state_dict.items():
        if key.startswith("actor.") or key.startswith("actor_mean_head."):
            deterministic_state[key] = value
        if key in ("action_scale", "action_bias"):
            deterministic_state[key] = value
    if not deterministic_state:
        raise ValueError("No deterministic actor weights found in provided state dict.")
    return deterministic_state


def build_policy_env_view(obs_dim: int, act_dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        single_observation_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        single_action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32),
    )


def _build_collector_actor(
    args: "Args",
    train_args: "TrainArgs",
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    device: torch.device,
) -> DeterministicAgent:
    """Construct the collector-side actor.

    Returns a `DeterministicAgent` in standard mode and a `ResidualActor`
    wrapping a frozen base + zero-init residual head in residual mode. The
    base/residual weights are populated by the caller via a `state_dict()`
    copy from `learner_state.actor` — this helper only builds the shell so
    the architecture matches the learner's actor.
    """
    policy_obs_dim = obs_dim + act_dim if train_args.use_last_action_in_policy_state else obs_dim
    policy_env_view = build_policy_env_view(policy_obs_dim, act_dim)
    base = DeterministicAgent(
        policy_env_view,
        action_scale=train_args.action_scale,
        action_bias=0.0,
        hidden_layer_size=train_args.agent_hidden_layer_size,
        num_hidden_layers=train_args.agent_num_hidden_layers,
    ).to(device)
    if args.full_checkpoint_load not in ("residual", "residual_resume"):
        base.eval()
        return base
    residual_head = DeterministicAgent(
        policy_env_view,
        action_scale=args.residual_scale,
        action_bias=0.0,
        hidden_layer_size=train_args.agent_hidden_layer_size,
        num_hidden_layers=train_args.agent_num_hidden_layers,
    ).to(device)
    zero_init_residual_head(residual_head)
    actor = ResidualActor(
        base_actor=base,
        residual_actor=residual_head,
        action_low=torch.as_tensor(action_low_np, dtype=torch.float32, device=device),
        action_high=torch.as_tensor(action_high_np, dtype=torch.float32, device=device),
    ).to(device)
    actor.eval()
    return actor


def linear_anneal(start: float, end: float, step: int, anneal_steps: int) -> float:
    if anneal_steps <= 0:
        return end
    progress = min(max(step, 0) / float(anneal_steps), 1.0)
    return start + progress * (end - start)


def primitive_exploration_chance_for_step(args: "Args", step: int) -> float:
    return linear_anneal(
        args.exploration_primitive_chance_start,
        args.exploration_primitive_chance,
        step,
        args.exploration_primitive_chance_anneal_steps,
    )


def _primitive_state_tensor(values: object, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(1, -1)[:, :2]


def _extract_primitive_state_tensors(
    env: AirHockeyEnv,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zeros = torch.zeros((1, 2), dtype=torch.float32, device=device)
    state_info = None
    simulator = getattr(env, "simulator", None)
    if simulator is not None and hasattr(simulator, "get_current_state"):
        try:
            state_info = simulator.get_current_state()
        except Exception:
            state_info = None
    if state_info is None:
        state_info = getattr(env, "current_state", None)
    if not isinstance(state_info, dict):
        return zeros.clone(), zeros.clone(), zeros.clone()
    try:
        paddle_position = _primitive_state_tensor(
            state_info["paddles"]["paddle_ego"]["position"],
            device=device,
        )
        puck_position = _primitive_state_tensor(
            state_info["pucks"][0]["position"],
            device=device,
        )
        puck_velocity = _primitive_state_tensor(
            state_info["pucks"][0]["velocity"],
            device=device,
        )
        return paddle_position, puck_position, puck_velocity
    except Exception:
        return zeros.clone(), zeros.clone(), zeros.clone()


def _reset_primitive_rollout_state(
    primitive_selector: PrimitiveExplorationSelector | None,
) -> None:
    if primitive_selector is not None:
        done_mask = torch.ones(primitive_selector.num_envs, dtype=torch.bool, device=primitive_selector.device)
        primitive_selector.reset(done_mask)


def _normalize_replay_source_priority(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"warmstart_only", "checkpoint_only", "checkpoint_then_append"}:
        return normalized
    return "warmstart_only"


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _checkpoint_root_from_tb(tb_log_dir: str, override_root: str | None) -> Path:
    if override_root is not None and str(override_root).strip():
        return Path(override_root).expanduser().resolve()
    # learner_tb and collector_tb are children of the same run root.
    return Path(tb_log_dir).expanduser().resolve().parent


def _coerce_float_list(values: object, *, max_items: int | None = None) -> list[float]:
    if not isinstance(values, (list, tuple)):
        return []
    coerced = [float(item) for item in values]
    if max_items is not None and max_items > 0 and len(coerced) > max_items:
        coerced = coerced[-int(max_items) :]
    return coerced


def _build_async_training_state(
    *,
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    actor,
    actor_target,
    qfs,
    qfs_target,
    q_optimizer,
    actor_optimizer,
    total_updates: int,
    total_actor_updates: int,
    latest_train_metrics: Dict[str, float],
    collector_total_steps: int,
    last_checkpoint_collector_steps: int,
    run_elapsed_total_s: float,
    rolling50_reward_values: Sequence[float],
    rolling50_episode_length_values: Sequence[float],
    rolling50_estop_episode_flags: Sequence[float],
    rolling50_episode_return_values: Sequence[float] = (),
    rolling50_episode_juggles_values: Sequence[float] = (),
    rolling50_episode_contacts_values: Sequence[float] = (),
    include_non_vital_training_state_fields: bool,
) -> Dict[str, object]:
    replay_state = replay.state_dict()
    payload: Dict[str, object] = {
        "checkpoint_version": 2,
        "actor": actor.state_dict(),
        "actor_target": actor_target.state_dict(),
        "success_replay_buffer": replay_state["success"],
        "failure_replay_buffer": replay_state["failure"],
        "rng_states": get_rng_states(),
    }
    # Save N critics under qf1..qfN / qf1_target..qfN_target. N=2 is bit-identical
    # to the legacy two-key layout, so old readers (eval scripts, sim2sim ckpts)
    # keep working unchanged.
    for i, (q, qt) in enumerate(zip(qfs, qfs_target), start=1):
        payload[f"qf{i}"] = q.state_dict()
        payload[f"qf{i}_target"] = qt.state_dict()
    if include_non_vital_training_state_fields:
        payload.update(
            {
                "global_step": int(total_updates),
                "iteration": int(total_updates),
                "q_optimizer": q_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "train_metrics": dict(latest_train_metrics),
                "learner_q_updates": int(total_updates),
                "learner_actor_updates": int(total_actor_updates),
                "collector_total_steps": int(collector_total_steps),
                "last_checkpoint_collector_steps": int(last_checkpoint_collector_steps),
                "run_elapsed_total_s": float(run_elapsed_total_s),
                "rolling_window_size": int(ROLLING_PERF_WINDOW_EPISODES),
                "rolling50_reward_values": list(rolling50_reward_values),
                "rolling50_episode_length_values": list(rolling50_episode_length_values),
                "rolling50_estop_episode_flags": list(rolling50_estop_episode_flags),
                "rolling50_episode_return_values": list(rolling50_episode_return_values),
                "rolling50_episode_juggles_values": list(rolling50_episode_juggles_values),
                "rolling50_episode_contacts_values": list(rolling50_episode_contacts_values),
                # Metadata only; runtime args always come from external CLI/args_file.
                "args": {**asdict(args), **asdict(train_args)},
            }
        )
    return payload


def _save_async_checkpoint(
    *,
    checkpoint_root: Path,
    checkpoint_tag: str,
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    actor,
    actor_target,
    qfs,
    qfs_target,
    q_optimizer,
    actor_optimizer,
    total_updates: int,
    total_actor_updates: int,
    latest_train_metrics: Dict[str, float],
    stats: Dict[str, object],
    actor_ema=None,
) -> Path:
    checkpoint_dir = checkpoint_root / f"checkpoint_{checkpoint_tag}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(args.config, "r") as f:
            checkpoint_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(checkpoint_dir / "config.yaml", "w") as f:
            yaml.dump(checkpoint_config, f)
    except Exception as exc:
        print(f"[learner_checkpoint] failed to save config.yaml: {exc}")
    try:
        # Merge TrainArgs into args.yaml so the saved file is itself a valid
        # --train-args source for downstream rollouts.
        with open(checkpoint_dir / "args.yaml", "w") as f:
            yaml.dump({**asdict(args), **asdict(train_args)}, f)
    except Exception as exc:
        print(f"[learner_checkpoint] failed to save args.yaml: {exc}")
    torch.save(actor.state_dict(), checkpoint_dir / "model.pth")
    torch.save(actor_target.state_dict(), checkpoint_dir / "actor_target.pth")
    if actor_ema is not None:
        torch.save(actor_ema.state_dict(), checkpoint_dir / "model_ema.pth")
    # Per-critic flat .pth dumps — qf1.pth..qfN.pth and matching _target. Eval
    # scripts that only care about qf1/qf2 keep working; ensemble runs (N>2)
    # additionally drop qf3.pth, qf4.pth, ... in the same directory.
    for i, (q, qt) in enumerate(zip(qfs, qfs_target), start=1):
        torch.save(q.state_dict(), checkpoint_dir / f"qf{i}.pth")
        torch.save(qt.state_dict(), checkpoint_dir / f"qf{i}_target.pth")
    training_state = _build_async_training_state(
        args=args,
        train_args=train_args,
        replay=replay,
        actor=actor,
        actor_target=actor_target,
        qfs=qfs,
        qfs_target=qfs_target,
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        total_updates=total_updates,
        total_actor_updates=total_actor_updates,
        latest_train_metrics=latest_train_metrics,
        collector_total_steps=int(stats.get("collector_total_steps", stats.get("collector_steps", 0.0))),
        last_checkpoint_collector_steps=int(float(stats.get("last_checkpoint_collector_steps", 0.0))),
        run_elapsed_total_s=float(stats.get("run_elapsed_total_s", 0.0)),
        rolling50_reward_values=_coerce_float_list(
            stats.get(
                "rolling50_reward_values",
                stats.get("rolling50_task_reward_values", []),
            ),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        rolling50_episode_length_values=_coerce_float_list(
            stats.get("rolling50_episode_length_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        rolling50_estop_episode_flags=_coerce_float_list(
            stats.get("rolling50_estop_episode_flags", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        rolling50_episode_return_values=_coerce_float_list(
            stats.get("rolling50_episode_return_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        rolling50_episode_juggles_values=_coerce_float_list(
            stats.get("rolling50_episode_juggles_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        rolling50_episode_contacts_values=_coerce_float_list(
            stats.get("rolling50_episode_contacts_values", []),
            max_items=ROLLING_PERF_WINDOW_EPISODES,
        ),
        include_non_vital_training_state_fields=args.include_non_vital_training_state_fields,
    )
    _atomic_torch_save(training_state, checkpoint_dir / "training_state.pth")
    return checkpoint_dir


def _save_checkpoint_from_learner_state(
    *,
    state: LearnerRuntimeState,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    checkpoint_tag: str,
    args: Args,
    train_args: TrainArgs,
) -> Path:
    return _save_async_checkpoint(
        checkpoint_root=state.checkpoint_root,
        checkpoint_tag=checkpoint_tag,
        args=args,
        train_args=train_args,
        replay=replay,
        actor=state.actor,
        actor_target=state.actor_target,
        qfs=state.qfs,
        qfs_target=state.qfs_target,
        q_optimizer=state.q_optimizer,
        actor_optimizer=state.actor_optimizer,
        total_updates=state.total_updates,
        total_actor_updates=state.total_actor_updates,
        latest_train_metrics=state.latest_train_metrics,
        stats=stats,
        actor_ema=state.actor_ema,
    )


@dataclass
class TrainArgs:
    """Architecture spec sourced from a td3_training.py-style args.yaml.

    These fields describe the actor/critic network shape, ensemble size, and
    policy-state contract used during training. They must be read from the
    training run's args.yaml — NOT from the online `--args-file` or CLI — so
    the rebuilt actor/critic layers match the saved checkpoint exactly.

    `num_critics` and `target_critic_subset_size` were added 2026-05-04 to
    support v27-style Maxmin-N / REDQ-N-M ensembles in the async real-world
    learner. Old args.yaml files predate these keys and resolve to the
    backwards-compatible vanilla-TD3 default (N=2, subset=None).
    """

    action_scale: float
    agent_hidden_layer_size: int
    agent_num_hidden_layers: int
    q_hidden_layer_size: int
    q_num_hidden_layers: int
    use_last_action_in_policy_state: bool
    # Ensemble size (≥2). 2 = vanilla twin-TD3 (default; backwards-compat).
    # >2 with subset=None → Maxmin-N. >2 with subset=M<N → REDQ-N-M.
    num_critics: int = 2
    # Subset size used when computing the target Q (random M-of-N at every
    # update). None or ≥num_critics → Maxmin (use all critics).
    target_critic_subset_size: int | None = None


TRAIN_ARGS_FIELD_NAMES: Tuple[str, ...] = tuple(f.name for f in fields(TrainArgs))
# Required architecture keys — every td3_training.py args.yaml has these.
# Ensemble keys are optional with safe defaults so older args.yaml files keep
# loading; see _load_train_args (and feedback_loader_defaults memory).
_TRAIN_ARGS_REQUIRED_FIELDS: Tuple[str, ...] = (
    "action_scale",
    "agent_hidden_layer_size",
    "agent_num_hidden_layers",
    "q_hidden_layer_size",
    "q_num_hidden_layers",
    "use_last_action_in_policy_state",
)


def _load_train_args(train_args_path: str) -> TrainArgs:
    """Load architecture fields from a td3_training.py-style args.yaml.

    Only canonical field names are accepted; deprecated aliases
    (`agent_hidden_size`, `q_hidden_size`, ...) are not remapped.
    Extra keys in the file are ignored.
    """
    if not os.path.exists(train_args_path):
        raise FileNotFoundError(f"--train-args file does not exist: {train_args_path}")
    with open(train_args_path, "r") as f:
        loaded = yaml.load(f, Loader=yaml.FullLoader)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected --train-args to contain a YAML mapping, got: {type(loaded)}"
        )
    missing = [name for name in _TRAIN_ARGS_REQUIRED_FIELDS if name not in loaded]
    if missing:
        raise KeyError(
            f"--train-args file {train_args_path} is missing required fields: {missing}. "
            f"Expected canonical td3_training.py args.yaml field names."
        )
    raw_subset = loaded.get("target_critic_subset_size", None)
    target_subset: int | None = None if raw_subset is None else int(raw_subset)
    return TrainArgs(
        action_scale=float(loaded["action_scale"]),
        agent_hidden_layer_size=int(loaded["agent_hidden_layer_size"]),
        agent_num_hidden_layers=int(loaded["agent_num_hidden_layers"]),
        q_hidden_layer_size=int(loaded["q_hidden_layer_size"]),
        q_num_hidden_layers=int(loaded["q_num_hidden_layers"]),
        use_last_action_in_policy_state=bool(loaded["use_last_action_in_policy_state"]),
        num_critics=int(loaded.get("num_critics", 2)),
        target_critic_subset_size=target_subset,
    )


@dataclass
class Args:
    # Required: training args.yaml (architecture source; see TrainArgs).
    train_args: str | None = None
    # Optional: online-behavior defaults YAML (e.g. td3_residual.yaml).
    args_file: str | None = None
    config: str = "configs/real_configs/rollout_config.yaml"
    model_path: str | None = None
    # Checkpoint-load mode for `model_path`. Mirrors `td3_training.py:Args.full_checkpoint_load`.
    # - "full_resume":     load all training state (weights + optimizer + replay + RNG).
    # - "weights_only":    load network weights only; fresh optimizer + replay + RNG.
    # - "fine_tune":       load weights + optimizer; reset replay + RNG.
    # - "residual":        load source actor as frozen base, build fresh residual head + fresh critic.
    # - "residual_resume": resume a prior RESIDUAL run from its training_state.pth.
    #                      Loads the wrapped ResidualActor (base + trained residual head) +
    #                      the N-critic ensemble + (with load_replay_from_checkpoint) the
    #                      replay buffer + (with include_non_vital_training_state_fields)
    #                      the optimizer state. Use this — NOT "residual" — when continuing
    #                      a previously-checkpointed residual training run. See
    #                      notes/docs/training/residual-rl-recipe.md "Resuming a residual run".
    full_checkpoint_load: Literal[
        "full_resume", "weights_only", "fine_tune", "residual", "residual_resume"
    ] = "full_resume"
    # Residual RL: max magnitude of the residual action component (used when
    # full_checkpoint_load=="residual"). Combined action is clipped to env
    # action bounds, so residual_scale > 0 caps |residual|_inf via tanh.
    residual_scale: float = 0.15
    # Residual RL: Adam weight_decay on the residual actor's parameters. Default
    # 0.0 (off). Set to 1e-3 / 1e-2 for mild / strong L2 if drift is observed.
    residual_weight_decay: float = 0.0
    # Residual RL: optional EMA decay for the residual head (e.g. 0.9999). When
    # set, an EMA-averaged copy of the residual is maintained alongside the
    # online actor and saved as `model_ema.pth` in each checkpoint.
    residual_ema_decay: float | None = None
    # Residual RL: if > 0, add `lambda * mean(residual_action^2)` to the actor
    # loss. Penalises the residual *output* directly (vs. residual_weight_decay
    # which penalises parameters). Default 0.0 (off).
    residual_action_l2: float = 0.0
    collector_device: str = "cpu"
    learner_device: str = "cuda:0"
    seed: int = 0

    # Shared replay sizes
    success_buffer_size: int = int(2e5)
    failure_buffer_size: int = int(8e5)
    recent_episode_window_size: int = 500
    success_top_fraction: float = 0.2
    warm_start_hdf5_dirs: Tuple[str, ...] = ()
    warm_start_hdf5_recursive: bool = True

    # TD3 core
    gamma: float = 0.975
    tau: float = 0.005
    batch_size: int = 256
    # Gates the learner on TOTAL replay size (success + failure). Engaged
    # whenever total replay < this value — useful when there is no warm-start.
    min_replay_size_before_learning: int = 5000
    # Gates the learner on FRESH post-launch collector steps only, independent
    # of warm-start replay size. Mirrors td3_training.py's `learning_starts`:
    # for the first N fresh steps the agent collects (frozen base + zero-init
    # residual + exploration_noise) and pushes transitions into replay, but
    # the critic does NOT update. After N steps the critic has seen pure
    # on-policy data before its first gradient step. v27 sim2sim uses 2000;
    # 0 disables (default — preserves legacy async-real behavior). Run-
    # relative: on resume, fresh-step counting restarts from 0, which means
    # if you set this >0 and resume mid-fill, the gate re-engages until N
    # *new* fresh steps land.
    learning_starts_fresh_steps: int = 0
    q_lr: float = 1e-3
    q_weight_decay: float = 1e-4
    policy_lr: float = 3e-4
    q_updates: int = 1
    actor_updates_per_iteration: int = 1
    target_network_frequency: int = 1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    h_transform_eps: float = 1e-3
    # CQL (Conservative Q-Learning) penalty on the task-Q head. Mirrors
    # td3_training.py:Args.cql_alpha / cql_n_random. Default 0.0 → no penalty,
    # critic loss is identical to pre-CQL behavior. The canonical big-gap
    # residual recipe (sim2sim winner, 2026-05-08) sets cql_alpha=20.0.
    # See notes/docs/training/residual-rl-recipe.md.
    cql_alpha: float = 0.0
    cql_n_random: int = 10
    critic_success_sample_fraction: float = 0.3
    critic_failure_sample_fraction: float = 0.7

    # Collector behavior
    exploration_noise: float = 0.1
    exploration_primitive_chance: float = 0.0
    exploration_primitive_chance_start: float = 0.0
    exploration_primitive_chance_anneal_steps: int = 200000
    exploration_primitive_steps: int = 5
    exploration_primitive_weight_stand_still: float = 1.0
    exploration_primitive_weight_same_direction: float = 1.0
    exploration_primitive_weight_y_aligned: float = 1.0
    exploration_primitive_weight_target_position_directional: float = 1.0
    # Legacy action-space skew knob kept for compatibility with older configs.
    exploration_direction_y_component_weight: float = 2.0
    # Simulator-space per-step displacement ranges for directional primitives.
    exploration_same_direction_min_angle_deg: float = -180.0
    exploration_same_direction_max_angle_deg: float = 180.0
    exploration_same_direction_min_magnitude: float = 0.012
    exploration_same_direction_max_magnitude: float = 0.26
    exploration_y_aligned_min_angle_deg: float = 45.0
    exploration_y_aligned_max_angle_deg: float = 135.0
    exploration_y_aligned_min_magnitude: float = 0.012
    exploration_y_aligned_max_magnitude: float = 0.12
    exploration_target_position_directional_min_angle_deg: float = -180.0
    exploration_target_position_directional_max_angle_deg: float = 180.0
    exploration_target_position_directional_min_magnitude: float = 0.2
    exploration_target_position_directional_max_magnitude: float = 0.5
    exploration_target_position_min_distance: float = 0.2
    exploration_target_position_max_distance: float = 0.5
    exploration_target_position_delta_x: float = 0.26
    exploration_target_position_delta_y: float = 0.12
    exploration_target_position_steps: int = 5
    collector_policy_stand_still: bool = False
    transition_hold_steps_post_reset: int = 8
    transition_hold_steps_post_estop_enter: int = 0
    transition_hold_steps_post_estop_clear: int = 8
    transition_hold_steps_post_actor_sync: int = 3
    transition_hold_steps_post_safety_rearm: int = 3
    transition_disable_exploration_noise: bool = True
    transition_last_action_mode: str = "zero"
    transition_hold_log_every_step: bool = False
    actor_sync_check_every_episode: bool = True
    collector_log_interval_sec: float = 60.0
    learner_log_interval_sec: float = 60.0
    # Single root for all collected per-episode data (HDF5s, GIFs, camera videos).
    # `_setup_run_data_dir` creates the actual run folder at:
    #   <data_root_dir>/data_<YYYYMMDD-HHMMSS>/
    # and populates `episode_artifact_dir`, `reset_artifact_dir`, `episode_gif_dir`,
    # and `episode_camera_video_dir` as runtime attributes pointing at subfolders.
    data_root_dir: str = "runs/async_td3/data"
    enable_episode_gif: bool = True
    episode_gif_fps: int = 20
    episode_gif_subsample: int = 1
    # Set to 0 to disable cap.
    episode_gif_max_frames: int = 0
    episode_gif_require_puck: bool = False
    enable_episode_camera_video: bool = True
    episode_camera_video_fps: int = 20
    episode_camera_video_subsample: int = 1
    # Set to 0 to disable cap.
    episode_camera_video_max_frames: int = 0
    episode_camera_video_codec: str = "mp4v"
    log_parent_dir: str | None = None
    run_name: str = "async_td3_real"
    enable_periodic_checkpointing: bool = True
    checkpoint_every_collector_steps: int = 5000
    # Total valid (kept-episode) collector steps to train for. 0 disables
    # the cap (run until Ctrl-C). On resume the counter is read from the
    # checkpoint, so a 50k-step checkpoint resumed with total_timesteps=100k
    # runs for 50k more steps.
    total_timesteps: int = 100000
    checkpoint_root_dir: str | None = None
    load_replay_from_checkpoint: bool = True
    replay_source_priority: str = "warmstart_only"
    include_non_vital_training_state_fields: bool = False

    # Optional smoke-test mode (0 disables)
    smoke_test_seconds: float = 0.0
    enable_latency_profiling: bool = False
    latency_profile_output_dir: str | None = None
    latency_profile_hist_bins: int = 40


def _build_args_file_defaults(
    args_file_path: str,
) -> tuple[dict, list[str], list[str]]:
    """Load defaults from a td3_training.py-style args.yaml.

    Only canonical Args field names are accepted; any other keys in the YAML
    are returned as `ignored_source_keys` and not applied. Deprecated legacy
    aliases (e.g. `agent_hidden_size`, `q_hidden_size`, `learning_starts`,
    `device`) are intentionally NOT remapped — use the canonical names.
    """
    with open(args_file_path, "r") as f:
        loaded_yaml = yaml.load(f, Loader=yaml.FullLoader)
    if loaded_yaml is None:
        return {}, [], []
    if not isinstance(loaded_yaml, dict):
        raise ValueError(f"Expected args_file YAML to be a mapping, got {type(loaded_yaml)}")

    valid_async_keys = {field.name for field in fields(Args)}
    mapped_defaults: dict = {}
    applied_source_keys: list[str] = []
    ignored_source_keys: list[str] = []

    for source_key, source_value in loaded_yaml.items():
        if source_key in valid_async_keys:
            mapped_defaults[source_key] = source_value
            applied_source_keys.append(source_key)
        else:
            ignored_source_keys.append(source_key)

    return mapped_defaults, sorted(applied_source_keys), sorted(ignored_source_keys)


def _episode_to_tensors(episode_trajectory: EpisodeTrajectory) -> Dict[str, torch.Tensor]:
    return {
        "observations": torch.stack(episode_trajectory.observations, dim=0),
        "next_observations": torch.stack(episode_trajectory.next_observations, dim=0),
        "actions": torch.stack(episode_trajectory.actions, dim=0),
        "prev_actions": torch.stack(episode_trajectory.prev_actions, dim=0),
        "rewards": torch.stack(episode_trajectory.rewards, dim=0).view(-1),
        # Same semantics as td3_training replay `dones`: env terminations (+ stop / last-step
        # signals), not time-limit truncation alone.
        "dones": torch.stack(episode_trajectory.dones, dim=0).view(-1),
    }


def _compute_episode_return_success_threshold(
    recent_episode_returns: deque[float],
    success_top_fraction: float,
) -> float:
    if len(recent_episode_returns) <= 0:
        return 0.0
    quantile = 1.0 - float(success_top_fraction)
    return float(np.quantile(np.asarray(recent_episode_returns, dtype=np.float32), quantile))


def _add_episode_to_shared_replay(
    replay: SharedTD3Replay,
    episode_trajectory: EpisodeTrajectory,
    recent_episode_returns: deque[float],
    success_top_fraction: float,
) -> tuple[str, float, float, int]:
    episode_return = float(episode_trajectory.episode_return)
    recent_episode_returns.append(episode_return)
    episode_return_success_threshold = _compute_episode_return_success_threshold(
        recent_episode_returns=recent_episode_returns,
        success_top_fraction=success_top_fraction,
    )
    partition = "success" if episode_return >= episode_return_success_threshold else "failure"
    inserted_steps = replay.add_episode(partition, _episode_to_tensors(episode_trajectory))
    return partition, episode_return, episode_return_success_threshold, inserted_steps


_VITAL_TRAINING_STATE_KEYS = (
    "actor",
    "actor_target",
    "qf1",
    "qf2",
    "qf1_target",
    "qf2_target",
    "success_replay_buffer",
    "failure_replay_buffer",
    "rng_states",
)
# Default values for non-vital training_state.pth fields. Cross-source
# checkpoints (e.g., a sim-trained `td3_training.py` training_state used as a
# residual base) only contain a subset of these; missing keys are filled with
# the defaults below so downstream readers can dict-access them unconditionally.
# `q_optimizer` / `actor_optimizer` are intentionally NOT defaulted — their
# presence is the gate the learner uses to decide whether to restore optimizer
# state at all (`real_td3_runtime._init_sync_learner_state`).
_NON_VITAL_TRAINING_STATE_DEFAULTS: Dict[str, object] = {
    "learner_q_updates": 0,
    "learner_actor_updates": 0,
    "train_metrics": {},
    "collector_total_steps": 0,
    "run_elapsed_total_s": 0.0,
    "rolling50_reward_values": [],
    "rolling50_episode_length_values": [],
    "rolling50_estop_episode_flags": [],
    "rolling50_episode_juggles_values": [],
    "rolling50_episode_contacts_values": [],
}


def _load_training_state_checkpoint(model_path: str) -> Dict[str, object]:
    """Load a training_state.pth dict, with strict vital-key validation and
    per-key defaults for missing non-vital fields.

    Vital keys (actor/critic/target weights, replay buffers, RNG state) are
    required and trigger a hard failure if absent.

    Non-vital keys (learner counters, rolling perf stats, train metrics) are
    filled with safe defaults from `_NON_VITAL_TRAINING_STATE_DEFAULTS` when
    missing, and a single log line lists which keys were defaulted. This
    accommodates cross-source resumes where a sim-trained `td3_training.py`
    training_state is loaded as a base for real-world residual / fine-tune
    flows — sim sources contain only a subset of the real-world non-vital
    fields. `q_optimizer` / `actor_optimizer` are intentionally not in the
    defaults: their presence is the gate the learner uses to decide whether
    to restore optimizer state."""
    loaded_obj = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(loaded_obj, dict):
        raise TypeError(
            f"Expected training_state.pth at {model_path} to be a dict, "
            f"got {type(loaded_obj).__name__}."
        )
    missing_vital = [k for k in _VITAL_TRAINING_STATE_KEYS if k not in loaded_obj]
    if missing_vital:
        raise KeyError(
            f"training_state.pth at {model_path} is missing required keys: "
            f"{missing_vital}. Expected a dict produced by _build_async_training_state."
        )
    defaulted: list[str] = []
    for key, default in _NON_VITAL_TRAINING_STATE_DEFAULTS.items():
        if key not in loaded_obj:
            # Copy mutable defaults so the module-level dict is never aliased.
            if isinstance(default, dict):
                loaded_obj[key] = dict(default)
            elif isinstance(default, list):
                loaded_obj[key] = list(default)
            else:
                loaded_obj[key] = default
            defaulted.append(key)
    if defaulted:
        print(
            f"[resume] training_state.pth at {model_path} missing non-vital "
            f"keys; filled with defaults: {defaulted}"
        )
    return loaded_obj


def _finite_or_default(value: float, default: float = -1.0) -> float:
    value_f = float(value)
    return value_f if np.isfinite(value_f) else float(default)


def _latest_camera_frame(env: AirHockeyEnv) -> np.ndarray | None:
    """Fetch the latest raw camera frame if available.

    Two sources are supported:

    1. ``simulator.images`` — populated by the main-process ``save_collect()``
       branch in ``AirHockeyReal.step()`` (gated on ``self.cap is not None``,
       i.e. ``control_mode in {'RL', 'BC', 'IQL', 'observe', ...}``).
    2. ``simulator.shared_camera_frame`` — published by the camera subprocess
       when ``control_mode in {'mouse', 'mimic'}`` and the main process has no
       ``self.cap`` of its own. Read once per call; the subprocess overwrites
       the buffer on every camera tick (~20 Hz), so this gives the latest
       available frame.
    """
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return None

    images = getattr(simulator, "images", None)
    if isinstance(images, list) and len(images) > 0 and images[-1] is not None:
        frame = np.asarray(images[-1])
        if frame.ndim == 3:
            return np.array(frame, copy=True)

    shared = getattr(simulator, "shared_camera_frame", None)
    ready = getattr(simulator, "shared_camera_frame_ready", None)
    shape = getattr(simulator, "_shared_camera_frame_shape", None)
    if shared is None or shape is None:
        return None
    if ready is not None and not bool(ready.value):
        return None
    try:
        with shared.get_lock():
            buf = np.frombuffer(shared.get_obj(), dtype=np.uint8).copy()
    except Exception:
        return None
    expected = int(np.prod(shape))
    if buf.size != expected:
        return None
    return buf.reshape(shape)


def _next_available_episode_id(output_dir: str | Path) -> int:
    """Return one plus the largest saved trajectory_data*.hdf5 index (recursive)."""
    artifact_dir = Path(output_dir).expanduser().resolve()
    if not artifact_dir.exists():
        return 0
    max_seen = -1
    pattern = re.compile(r"^trajectory_data(\d+)\.hdf5$")
    for path in artifact_dir.rglob("trajectory_data*.hdf5"):
        match = pattern.match(path.name)
        if match is None:
            continue
        max_seen = max(max_seen, int(match.group(1)))
    return max_seen + 1


def _episode_length_bucket_name(episode_steps: int) -> str:
    """Map episode step count to a quality bucket label."""
    if episode_steps < 50:
        return "<50"
    if episode_steps <= 100:
        return "50-100"
    if episode_steps <= 200:
        return "100-200"
    return ">200"


def _bucketed_output_dir(base_dir: str | Path, episode_steps: int) -> str:
    """Return a length-bucketed output directory under the given parent."""
    return str(Path(base_dir).expanduser().resolve() / _episode_length_bucket_name(episode_steps))


def _stop_output_dir(base_dir: str | Path, stop_label: str) -> str:
    """Return the additional stop-specific output directory under the parent."""
    return str(Path(base_dir).expanduser().resolve() / str(stop_label))


def _reset_output_dir(base_dir: str | Path, partition: str, episode_steps: int) -> str:
    """Return the reset artifact directory under success/failure partition buckets."""
    partition_root = Path(base_dir).expanduser().resolve() / str(partition)
    return _bucketed_output_dir(partition_root, episode_steps)


def _copy_to_stop_dir(file_path: str | Path, stop_root: str | Path) -> Path:
    """Copy one artifact file into the stop-specific directory and return destination."""
    src = Path(file_path).expanduser().resolve()
    stop_root_path = Path(stop_root).expanduser().resolve()
    if src.suffix == ".hdf5":
        dst_dir = stop_root_path
    else:
        # Keep per-episode media grouping (trajectory_data{N}/...).
        dst_dir = stop_root_path / src.parent.name
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def _safe_nonnegative_ms(value: float) -> float:
    value_f = float(value)
    if not np.isfinite(value_f):
        return 0.0
    return max(0.0, value_f)


def _env_timing_info(env: AirHockeyEnv) -> dict:
    state_info = getattr(env, "current_state", None)
    if not isinstance(state_info, dict):
        return {}
    timing_info = state_info.get("timing", {})
    if not isinstance(timing_info, dict):
        return {}
    return timing_info


def _latency_bucket_stats(values_ms: list[float]) -> dict:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean_ms": float("nan"),
            "median_ms": float("nan"),
            "p90_ms": float("nan"),
            "p99_ms": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def _write_latency_profile_episode(
    output_dir: str | Path,
    episode_id: int,
    puck_detection_ms: list[float],
    model_inference_ms: list[float],
    block_sleep_ms: list[float],
    other_ms: list[float],
    hist_bins: int,
) -> tuple[Path, Path, dict]:
    # Keep matplotlib import local so normal training path has no extra import cost.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    bucket_arrays = {
        "puck_detection_ms": np.asarray(puck_detection_ms, dtype=np.float64),
        "model_inference_ms": np.asarray(model_inference_ms, dtype=np.float64),
        "block_sleep_ms": np.asarray(block_sleep_ms, dtype=np.float64),
        "other_ms": np.asarray(other_ms, dtype=np.float64),
    }
    summary = {
        "episode_id": int(episode_id),
        "step_count": int(max((arr.size for arr in bucket_arrays.values()), default=0)),
        "buckets": {name: _latency_bucket_stats(arr.tolist()) for name, arr in bucket_arrays.items()},
        "per_step_ms": {name: arr.tolist() for name, arr in bucket_arrays.items()},
    }
    json_path = output_path / f"episode_{int(episode_id):06d}_latency.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"Episode {int(episode_id)} latency (ms)")
    ordered_names = ("puck_detection_ms", "model_inference_ms", "block_sleep_ms", "other_ms")
    bins = max(5, int(hist_bins))
    for axis, name in zip(axes.flatten(), ordered_names):
        values = bucket_arrays[name]
        if values.size > 0:
            axis.hist(values, bins=bins)
        axis.set_title(name)
        axis.set_xlabel("ms")
        axis.set_ylabel("count")
    fig.tight_layout()
    png_path = output_path / f"episode_{int(episode_id):06d}_latency.png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    return json_path, png_path, summary


def _build_split_episode_row(
    env: AirHockeyEnv,
    action_xy: np.ndarray,
    episode_id: int,
    episode_step_idx: int,
    protective_stop_active: bool,
    controller_disconnected: bool,
    reset_stage_id: int | None = None,
    *,
    reward: float | None = None,
    done: float | None = None,
) -> Dict[str, np.ndarray]:
    """Build one HDF5 row for a single env step.

    Optional ``reward`` / ``done`` are written when provided (the policy
    runner passes them every step) and omitted otherwise (the reset-FSM
    runner has no policy reward stream). Together with ``policy_action``
    (the raw [-1, 1] action vector recorded alongside the post-transform
    ``desired_pose``), these fields make a policy-collected HDF5
    self-sufficient for offline policy replay.
    """
    state_info = getattr(env, "current_state", None)
    if not isinstance(state_info, dict):
        state_info = env.simulator.get_current_state()
    paddle = state_info["paddles"]["paddle_ego"]
    puck_info = state_info["pucks"][0]
    paddle_pos = np.asarray(paddle.get("position", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    paddle_vel = np.asarray(paddle.get("velocity", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    move_lims = np.asarray(getattr(env.simulator, "move_lims", (1.0, 1.0)), dtype=np.float64).reshape(-1)
    if move_lims.shape[0] < 2:
        move_lims = np.array([1.0, 1.0], dtype=np.float64)
    desired_xy = paddle_pos[:2] + np.asarray(action_xy[:2], dtype=np.float64) * move_lims[:2]
    puck_position = np.asarray(puck_info.get("position", [0.0, 0.0]), dtype=np.float64).reshape(-1)
    puck_occluded = float(np.asarray(puck_info.get("occluded", [0.0]), dtype=np.float64).reshape(-1)[0])

    pose = vector_with_width(np.concatenate([paddle_pos[:2], np.zeros(4, dtype=np.float64)]), 6)
    speed = vector_with_width(np.concatenate([paddle_vel[:2], np.zeros(4, dtype=np.float64)]), 6)
    # TODO: Source force/acc/safety/estop directly from robot telemetry for real deployment.
    force = np.zeros((6,), dtype=np.float64)
    acc = np.zeros((3,), dtype=np.float64)
    desired_pose = vector_with_width(np.concatenate([desired_xy, np.zeros(4, dtype=np.float64)]), 6)
    puck = vector_with_width(np.concatenate([puck_position[:2], np.array([puck_occluded])]), 3)
    timing_info = state_info.get("timing", {}) if isinstance(state_info, dict) else {}
    paddle_actual_pose = np.asarray(state_info.get("paddle_actual_pose", np.zeros(6)), dtype=np.float64).reshape(-1)
    paddle_actual_speed = np.asarray(state_info.get("paddle_actual_speed", np.zeros(6)), dtype=np.float64).reshape(-1)
    paddle_target_pre = np.asarray(
        state_info.get("paddle_target_pose_pre_filter", np.zeros(6)), dtype=np.float64
    ).reshape(-1)
    paddle_target_post = np.asarray(
        state_info.get("paddle_target_pose_post_filter", np.zeros(6)), dtype=np.float64
    ).reshape(-1)
    paddle_actual = vector_with_width(np.concatenate([paddle_actual_pose[:2], paddle_actual_speed[:2], np.zeros(2)]), 6)
    paddle_cmd = vector_with_width(np.concatenate([paddle_target_pre[:6], paddle_target_post[:6]]), 12)
    timing = vector_with_width(
        np.array(
            [
                time.time(),
                _finite_or_default(timing_info.get("step_start_s", -1.0)),
                _finite_or_default(timing_info.get("telemetry_read_s", -1.0)),
                # Post-homography puck timestamp used by delay analysis.
                _finite_or_default(timing_info.get("puck_detection_done_s", -1.0)),
                _finite_or_default(timing_info.get("command_sent_s", -1.0)),
                _finite_or_default(timing_info.get("step_end_s", -1.0)),
                _finite_or_default(timing_info.get("sleep_before_step_s", -1.0)),
                _finite_or_default(timing_info.get("loop_runtime_before_sleep_s", -1.0)),
                _finite_or_default(timing_info.get("camera_frame_received_s", -1.0)),
            ],
            dtype=np.float64,
        ),
        9,
    )
    puck_meta = vector_with_width(
        np.array(
            [
                puck_occluded,
                1.0 if bool(state_info.get("puck_detector_used_fallback", puck_occluded > 0.5)) else 0.0,
            ],
            dtype=np.float64,
        ),
        2,
    )
    stop_flags = vector_with_width(
        np.array(
            [
                1.0 if protective_stop_active else 0.0,
                1.0 if controller_disconnected else 0.0,
                1.0 if (protective_stop_active or controller_disconnected) else 0.0,
            ],
            dtype=np.float64,
        ),
        3,
    )

    row = {
        "cur_time": np.array([time.time()], dtype=np.float64),
        "tidx": np.array([float(episode_id)], dtype=np.float64),
        "i": np.array([float(episode_step_idx)], dtype=np.float64),
        "estop": np.array([1.0 if protective_stop_active else 0.0], dtype=np.float64),
        "safety": np.array([1.0], dtype=np.float64),
        "pose": pose,
        "speed": speed,
        "force": force,
        "acc": acc,
        "desired_pose": desired_pose,
        "puck": puck,
        "timing": timing,
        "paddle_actual": paddle_actual,
        "paddle_cmd": paddle_cmd,
        "puck_meta": puck_meta,
        "stop_flags": stop_flags,
    }
    if reset_stage_id is not None:
        row["reset_stage_id"] = np.array([float(reset_stage_id)], dtype=np.float64)
    if reward is not None:
        row["policy_action"] = np.asarray(action_xy, dtype=np.float64).reshape(-1)[:2]
        row["reward"] = np.array([float(reward)], dtype=np.float64)
        row["done"] = np.array([float(done) if done is not None else 0.0], dtype=np.float64)
    return row


def _mixed_sample_from_shared(
    replay: SharedTD3Replay,
    batch_size: int,
    success_fraction: float,
    device: str,
) -> tuple[Dict[str, torch.Tensor] | None, int, int]:
    success_count, failure_count = critic_success_failure_counts(
        batch_size=batch_size,
        success_fraction=success_fraction,
        success_available=replay.len("success") > 0,
        failure_available=replay.len("failure") > 0,
    )
    if success_count + failure_count == 0:
        return None, success_count, failure_count

    chunks = []
    if success_count > 0:
        chunks.append(replay.sample("success", success_count, device=device))
    if failure_count > 0:
        chunks.append(replay.sample("failure", failure_count, device=device))
    if len(chunks) == 1:
        batch = chunks[0]
    else:
        batch = {
            key: torch.cat([chunk[key] for chunk in chunks], dim=0)
            for key in (
                "observations",
                "next_observations",
                "actions",
                "prev_actions",
                "rewards",
                "dones",
            )
        }
    return batch, success_count, failure_count


def _simulator_step_readiness(env: AirHockeyEnv) -> tuple[bool, str]:
    simulator = getattr(env, "simulator", None)
    if simulator is None:
        return True, "no_simulator"
    readiness_fn = getattr(simulator, "robot_command_readiness", None)
    if callable(readiness_fn):
        try:
            readiness = readiness_fn()
            step_ready = bool(readiness.get("step_ready", True))
            reason = str(readiness.get("reason", "ready"))
            return step_ready, reason
        except Exception as exc:
            return False, f"readiness_exception:{exc.__class__.__name__}"
    legacy_estop = _classify_stop_event(env).protective_stop
    return (not legacy_estop), "legacy_estop_fallback"


@dataclass
class LearnerRuntimeState:
    actor: DeterministicAgent
    actor_target: DeterministicAgent
    # Critic ensemble — N online critics + N targets (N = train_args.num_critics).
    # N=2 reproduces vanilla twin-TD3; N>2 enables Maxmin-N / REDQ-N-M (v27).
    # `qf1`, `qf2`, `qf1_target`, `qf2_target` are properties that alias
    # qfs[0]/qfs[1] for backwards compatibility with checkpoint helpers and
    # sites that predate the ensemble path. Anything new should index `qfs`.
    qfs: list[TD3QNetwork] = field(default_factory=list)
    qfs_target: list[TD3QNetwork] = field(default_factory=list)
    # M = target_critic_subset_size (None → Maxmin: use all N targets).
    target_critic_subset_size: int | None = None
    q_optimizer: optim.Optimizer = None  # type: ignore[assignment]
    actor_optimizer: optim.Optimizer = None  # type: ignore[assignment]
    action_low: torch.Tensor = None  # type: ignore[assignment]
    action_high: torch.Tensor = None  # type: ignore[assignment]
    writer: SummaryWriter = None  # type: ignore[assignment]
    checkpoint_root: Path = None  # type: ignore[assignment]
    last_log_time: float = 0.0
    learner_start_time: float = 0.0
    total_updates: int = 0
    total_actor_updates: int = 0
    last_handled_checkpoint_request_id: int = 0
    latest_train_metrics: Dict[str, float] = field(default_factory=dict)
    # Optional EMA copy of the actor, populated when residual_ema_decay is set.
    # In residual mode this is a `ResidualActor` wrapping the same frozen base
    # plus an EMA-averaged copy of the residual head.
    actor_ema: DeterministicAgent | None = None

    @property
    def qf1(self) -> TD3QNetwork:
        return self.qfs[0]

    @property
    def qf2(self) -> TD3QNetwork:
        return self.qfs[1]

    @property
    def qf1_target(self) -> TD3QNetwork:
        return self.qfs_target[0]

    @property
    def qf2_target(self) -> TD3QNetwork:
        return self.qfs_target[1]


def _make_qf(
    obs_dim: int,
    act_dim: int,
    hidden_layer_size: int,
    num_hidden_layers: int,
    device: torch.device,
) -> TD3QNetwork:
    return TD3QNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_layer_size=hidden_layer_size,
        num_hidden_layers=num_hidden_layers,
    ).to(device)


def _init_sync_learner_state(
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    tb_log_dir: str,
    resume_checkpoint: Dict[str, object] | None = None,
) -> LearnerRuntimeState:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.learner_device)
    writer = SummaryWriter(tb_log_dir)

    # Critic ensemble — N online critics + N targets (Maxmin-N when
    # target_critic_subset_size is None or ≥N; REDQ-N-M for smaller M). N=2
    # reproduces vanilla twin-TD3 (default). N=5 with subset=None is the
    # canonical v27 recipe. See `notes/docs/training/residual-rl-recipe.md`.
    num_critics = int(getattr(train_args, "num_critics", 2))
    if num_critics < 2:
        raise ValueError(f"num_critics must be >=2, got {num_critics}")
    target_subset = getattr(train_args, "target_critic_subset_size", None)
    if target_subset is not None and not (1 <= int(target_subset) <= num_critics):
        raise ValueError(
            f"target_critic_subset_size must be in [1, num_critics={num_critics}], "
            f"got {target_subset}"
        )
    policy_obs_dim = obs_dim + act_dim if train_args.use_last_action_in_policy_state else obs_dim
    policy_env_view = build_policy_env_view(policy_obs_dim, act_dim)
    actor = DeterministicAgent(
        policy_env_view,
        action_scale=train_args.action_scale,
        action_bias=0.0,
        hidden_layer_size=train_args.agent_hidden_layer_size,
        num_hidden_layers=train_args.agent_num_hidden_layers,
    ).to(device)
    actor_target = DeterministicAgent(
        policy_env_view,
        action_scale=train_args.action_scale,
        action_bias=0.0,
        hidden_layer_size=train_args.agent_hidden_layer_size,
        num_hidden_layers=train_args.agent_num_hidden_layers,
    ).to(device)
    actor_target.load_state_dict(actor.state_dict())
    # Build N critics + N targets. Distinct nn.Module instances → diverse inits.
    qfs: list[TD3QNetwork] = [
        _make_qf(
            obs_dim,
            act_dim,
            train_args.q_hidden_layer_size,
            train_args.q_num_hidden_layers,
            device,
        )
        for _ in range(num_critics)
    ]
    qfs_target: list[TD3QNetwork] = [
        _make_qf(
            obs_dim,
            act_dim,
            train_args.q_hidden_layer_size,
            train_args.q_num_hidden_layers,
            device,
        )
        for _ in range(num_critics)
    ]
    for q, qt in zip(qfs, qfs_target):
        qt.load_state_dict(q.state_dict())
    if num_critics > 2:
        subset_str = (
            "Maxmin"
            if target_subset is None or int(target_subset) >= num_critics
            else f"REDQ-{num_critics}-{int(target_subset)}"
        )
        print(
            f"[learner] critic ensemble: num_critics={num_critics} "
            f"target_subset={target_subset} ({subset_str})"
        )
    # Legacy aliases — kept so the rest of this function (residual setup,
    # resume path, target sync site) and existing checkpoint helpers can
    # still reference qf1/qf2 by name. New code should index `qfs` directly.
    qf1, qf2 = qfs[0], qfs[1]
    qf1_target, qf2_target = qfs_target[0], qfs_target[1]

    actor_ema: DeterministicAgent | None = None
    if args.full_checkpoint_load == "residual":
        # Residual RL: load source actor weights only (frozen base), build a
        # fresh zero-init residual head on top, and keep the critic untouched
        # from initialization. Mirrors td3_training.py:848.
        if args.model_path is None:
            raise ValueError(
                "full_checkpoint_load='residual' requires model_path to point at "
                "the source actor checkpoint."
            )
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model path {args.model_path} does not exist.")
        print(f"[learner] residual mode: loading source actor from {args.model_path}")
        loaded_obj = torch.load(args.model_path, map_location=device, weights_only=False)
        is_full_state = (
            isinstance(loaded_obj, dict) and "actor" in loaded_obj and "qf1" in loaded_obj
        )
        base_state = loaded_obj["actor"] if is_full_state else loaded_obj
        actor.load_state_dict(extract_deterministic_state_dict(base_state), strict=False)
        residual_online = DeterministicAgent(
            policy_env_view,
            action_scale=args.residual_scale,
            action_bias=0.0,
            hidden_layer_size=train_args.agent_hidden_layer_size,
            num_hidden_layers=train_args.agent_num_hidden_layers,
        ).to(device)
        residual_target = DeterministicAgent(
            policy_env_view,
            action_scale=args.residual_scale,
            action_bias=0.0,
            hidden_layer_size=train_args.agent_hidden_layer_size,
            num_hidden_layers=train_args.agent_num_hidden_layers,
        ).to(device)
        zero_init_residual_head(residual_online)
        zero_init_residual_head(residual_target)
        residual_target.load_state_dict(residual_online.state_dict())
        residual_action_low = torch.as_tensor(
            action_low_np, dtype=torch.float32, device=device
        )
        residual_action_high = torch.as_tensor(
            action_high_np, dtype=torch.float32, device=device
        )
        # Wrap actor + actor_target. Both targets share the same frozen base
        # instance so target sync only touches the residual head — matches
        # td3_training.py:885.
        actor = ResidualActor(
            base_actor=actor,
            residual_actor=residual_online,
            action_low=residual_action_low,
            action_high=residual_action_high,
        ).to(device)
        actor_target = ResidualActor(
            base_actor=actor.base,
            residual_actor=residual_target,
            action_low=residual_action_low,
            action_high=residual_action_high,
        ).to(device)
        if args.residual_ema_decay is not None:
            residual_ema = copy.deepcopy(residual_online)
            for p in residual_ema.parameters():
                p.requires_grad_(False)
            actor_ema = ResidualActor(
                base_actor=actor.base,
                residual_actor=residual_ema,
                action_low=residual_action_low,
                action_high=residual_action_high,
            ).to(device)
            print(
                f"[learner] residual EMA: decay={args.residual_ema_decay} — "
                "saving model_ema.pth alongside each checkpoint"
            )
        print(
            f"[learner] residual mode: base frozen, residual_scale={args.residual_scale}, "
            "critic from scratch."
        )
    elif args.full_checkpoint_load == "residual_resume":
        # Resume a previously-checkpointed RESIDUAL run. The saved actor
        # state_dict is a wrapped ResidualActor (keys: `base.*`, `residual.*`,
        # `action_low`, `action_high`) — `extract_deterministic_state_dict`
        # would silently strip them all, so we don't go through that filter.
        # Build the same ResidualActor wrapper as residual mode, then load
        # the saved state_dict directly into it (base inside the wrapper picks
        # up the original frozen weights from the saved dict; residual head
        # picks up the trained weights). N critics + targets + optimizer +
        # replay + RNG come from the same training_state.pth, same paths as
        # full_resume.
        if resume_checkpoint is None:
            raise ValueError(
                "full_checkpoint_load='residual_resume' requires model_path to "
                "point at the prior residual training_state.pth."
            )
        if "actor" not in resume_checkpoint or "qf1" not in resume_checkpoint:
            raise KeyError(
                "residual_resume: checkpoint is missing required keys 'actor' "
                "and/or 'qf1'. Pass a training_state.pth from a residual run."
            )
        residual_online = DeterministicAgent(
            policy_env_view,
            action_scale=args.residual_scale,
            action_bias=0.0,
            hidden_layer_size=train_args.agent_hidden_layer_size,
            num_hidden_layers=train_args.agent_num_hidden_layers,
        ).to(device)
        residual_target = DeterministicAgent(
            policy_env_view,
            action_scale=args.residual_scale,
            action_bias=0.0,
            hidden_layer_size=train_args.agent_hidden_layer_size,
            num_hidden_layers=train_args.agent_num_hidden_layers,
        ).to(device)
        residual_action_low = torch.as_tensor(
            action_low_np, dtype=torch.float32, device=device
        )
        residual_action_high = torch.as_tensor(
            action_high_np, dtype=torch.float32, device=device
        )
        # Wrap the placeholder base + placeholder residual; the saved
        # state_dict overwrites both below. Both targets share the frozen
        # base instance — same invariant as residual mode.
        actor = ResidualActor(
            base_actor=actor,
            residual_actor=residual_online,
            action_low=residual_action_low,
            action_high=residual_action_high,
        ).to(device)
        actor_target = ResidualActor(
            base_actor=actor.base,
            residual_actor=residual_target,
            action_low=residual_action_low,
            action_high=residual_action_high,
        ).to(device)
        actor.load_state_dict(resume_checkpoint["actor"], strict=False)
        if "actor_target" in resume_checkpoint:
            actor_target.load_state_dict(resume_checkpoint["actor_target"], strict=False)
        else:
            actor_target.load_state_dict(actor.state_dict(), strict=False)
        # Re-freeze the base. ResidualActor.__init__ flips requires_grad off,
        # but load_state_dict above doesn't touch the requires_grad flag, so
        # this is defensive — kept in case a future state_dict serializes the
        # flag differently.
        for p in actor.base.parameters():
            p.requires_grad_(False)
        for p in actor_target.base.parameters():
            p.requires_grad_(False)
        # Critics — same growth-tolerant logic as the standard resume branch.
        n_in_ckpt = sum(
            1
            for k in resume_checkpoint
            if k.startswith("qf") and not k.endswith("_target") and k[2:].isdigit()
        )
        if n_in_ckpt == 0:
            n_in_ckpt = 2
        if n_in_ckpt > num_critics:
            raise ValueError(
                f"residual_resume mismatch: checkpoint has {n_in_ckpt} critics "
                f"but num_critics={num_critics} is smaller. Cannot drop critics."
            )
        n_to_load = min(n_in_ckpt, num_critics)
        for i in range(1, n_to_load + 1):
            qfs[i - 1].load_state_dict(resume_checkpoint[f"qf{i}"])
            qfs_target[i - 1].load_state_dict(resume_checkpoint[f"qf{i}_target"])
        if n_in_ckpt < num_critics:
            print(
                f"[learner] residual_resume: checkpoint has {n_in_ckpt} critics "
                f"but num_critics={num_critics}. Cloning qf1 into "
                f"qf{n_to_load+1}..qf{num_critics}."
            )
            for i in range(n_to_load + 1, num_critics + 1):
                qfs[i - 1].load_state_dict(qfs[0].state_dict())
                qfs_target[i - 1].load_state_dict(qfs_target[0].state_dict())
        set_rng_states(resume_checkpoint["rng_states"])
        if args.residual_ema_decay is not None:
            # Initialize EMA from the (just-restored) online residual head;
            # if the prior run was saving model_ema.pth there isn't a
            # canonical key in training_state.pth to restore from, so EMA
            # restarts from the current online head — acceptable given EMA
            # is a smoothing tool, not a load-bearing checkpoint artifact.
            residual_ema = copy.deepcopy(actor.residual)
            for p in residual_ema.parameters():
                p.requires_grad_(False)
            actor_ema = ResidualActor(
                base_actor=actor.base,
                residual_actor=residual_ema,
                action_low=residual_action_low,
                action_high=residual_action_high,
            ).to(device)
            print(
                f"[learner] residual_resume: EMA re-initialized from restored "
                f"online residual head (decay={args.residual_ema_decay})."
            )
        print(
            f"[learner] residual_resume: restored ResidualActor (base+residual) "
            f"+ {n_to_load} critic(s) from {args.model_path}."
        )
    elif resume_checkpoint is not None:
        actor.load_state_dict(
            extract_deterministic_state_dict(resume_checkpoint["actor"]), strict=False
        )
        actor_target.load_state_dict(actor.state_dict())
        # Resume N critics. Old checkpoints only have qf1/qf2; newer ones
        # (num_critics>2) add qf3, qf4, ... and matching _target keys. Load
        # whatever's there, leave any extra critics at fresh init when growing
        # the ensemble; refuse to shrink it. Mirrors td3_training.py:957–991.
        n_in_ckpt = sum(
            1
            for k in resume_checkpoint
            if k.startswith("qf") and not k.endswith("_target") and k[2:].isdigit()
        )
        if n_in_ckpt == 0:
            n_in_ckpt = 2  # legacy ckpts always have qf1/qf2
        if n_in_ckpt > num_critics:
            raise ValueError(
                f"Resume mismatch: checkpoint has {n_in_ckpt} critics but "
                f"num_critics={num_critics} is smaller. Cannot drop critics "
                "from an ensemble checkpoint."
            )
        n_to_load = min(n_in_ckpt, num_critics)
        for i in range(1, n_to_load + 1):
            qfs[i - 1].load_state_dict(resume_checkpoint[f"qf{i}"])
            qfs_target[i - 1].load_state_dict(resume_checkpoint[f"qf{i}_target"])
        if n_in_ckpt < num_critics:
            print(
                f"[learner] resume: checkpoint has {n_in_ckpt} critics but "
                f"num_critics={num_critics}. Cloning qf1 into qf{n_to_load+1}.."
                f"qf{num_critics} so the ensemble doesn't dominate the "
                "min-target with fresh-init critics."
            )
            for i in range(n_to_load + 1, num_critics + 1):
                qfs[i - 1].load_state_dict(qfs[0].state_dict())
                qfs_target[i - 1].load_state_dict(qfs_target[0].state_dict())
        set_rng_states(resume_checkpoint["rng_states"])
    q_params: list = []
    for q in qfs:
        q_params.extend(q.parameters())
    q_optimizer = optim.Adam(
        q_params,
        lr=args.q_lr,
        weight_decay=args.q_weight_decay,
    )
    if args.full_checkpoint_load in ("residual", "residual_resume"):
        # Both residual modes: actor_optimizer covers ONLY the trainable
        # residual head — base is frozen. Optimizer state from a residual
        # checkpoint matches this param set (same head architecture), so
        # the resume load below is shape-compatible.
        actor_optimizer = optim.Adam(
            actor.residual.parameters(),
            lr=args.policy_lr,
            weight_decay=args.residual_weight_decay,
        )
        if args.residual_weight_decay > 0:
            print(
                f"[learner] residual actor optimizer: Adam(lr={args.policy_lr}, "
                f"weight_decay={args.residual_weight_decay}) — residual head L2 active"
            )
    else:
        actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    total_updates = 0
    total_actor_updates = 0
    latest_train_metrics: Dict[str, float] = {}
    # Optimizer state is mode-specific:
    #   - "residual" / "weights_only": optimizer covers a DIFFERENT param set
    #     than the source (residual head only / fresh critic ensemble), so
    #     loading source optimizer state would fail with a shape mismatch
    #     (e.g., source num_critics=2 → run num_critics=5). Skip the load —
    #     the optimizer starts fresh, learner counters start at 0. Mirrors
    #     td3_training.py's behavior.
    #   - "residual_resume": optimizer covers the SAME param set as the saved
    #     state (same residual head + same critic ensemble), so the load is
    #     shape-compatible.
    #   - "full_resume" / "fine_tune": same architecture as source, load works.
    optimizer_load_modes = {"full_resume", "fine_tune", "residual_resume"}
    if (
        args.full_checkpoint_load in optimizer_load_modes
        and resume_checkpoint is not None
        and "q_optimizer" in resume_checkpoint
    ):
        q_optimizer.load_state_dict(resume_checkpoint["q_optimizer"])
        actor_optimizer.load_state_dict(resume_checkpoint["actor_optimizer"])
        total_updates = int(
            resume_checkpoint.get(
                "learner_q_updates", resume_checkpoint.get("global_step", 0)
            )
        )
        total_actor_updates = int(resume_checkpoint.get("learner_actor_updates", 0))
        if isinstance(resume_checkpoint.get("train_metrics"), dict):
            latest_train_metrics = {
                str(key): float(value)
                for key, value in resume_checkpoint["train_metrics"].items()
            }
        print(
            "[learner] resumed optimizer state "
            f"q_updates={total_updates} actor_updates={total_actor_updates}"
        )
    elif (
        args.full_checkpoint_load in ("residual", "weights_only")
        and resume_checkpoint is not None
        and "q_optimizer" in resume_checkpoint
    ):
        print(
            f"[learner] {args.full_checkpoint_load} mode: skipping source "
            "optimizer state restore (param set differs from source — fresh "
            "Adam momentum + counters at 0)."
        )
    return LearnerRuntimeState(
        actor=actor,
        actor_target=actor_target,
        qfs=qfs,
        qfs_target=qfs_target,
        target_critic_subset_size=(None if target_subset is None else int(target_subset)),
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        action_low=torch.as_tensor(action_low_np, dtype=torch.float32, device=device).unsqueeze(0),
        action_high=torch.as_tensor(action_high_np, dtype=torch.float32, device=device).unsqueeze(0),
        writer=writer,
        checkpoint_root=_checkpoint_root_from_tb(tb_log_dir, args.checkpoint_root_dir),
        last_log_time=time.time(),
        learner_start_time=time.time(),
        total_updates=total_updates,
        total_actor_updates=total_actor_updates,
        last_handled_checkpoint_request_id=int(stats.get("checkpoint_save_request_id", 0)),
        latest_train_metrics=latest_train_metrics,
        actor_ema=actor_ema,
    )


def _run_sync_learner_iteration(
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    state: LearnerRuntimeState,
) -> bool:
    current_checkpoint_request_id = int(stats.get("checkpoint_save_request_id", 0))
    if (
        args.enable_periodic_checkpointing
        and current_checkpoint_request_id > state.last_handled_checkpoint_request_id
    ):
        trigger_steps = int(
            stats.get(
                "checkpoint_trigger_total_steps",
                stats.get("collector_total_steps", 0),
            )
        )
        checkpoint_tag = f"step_{trigger_steps}"
        try:
            checkpoint_dir = _save_checkpoint_from_learner_state(
                state=state,
                replay=replay,
                stats=stats,
                checkpoint_tag=checkpoint_tag,
                args=args,
                train_args=train_args,
            )
            stats["last_checkpoint_dir"] = str(checkpoint_dir)
            stats["last_checkpoint_collector_steps"] = float(trigger_steps)
            stats["last_checkpoint_q_updates"] = float(state.total_updates)
            stats["last_checkpoint_request_id"] = float(current_checkpoint_request_id)
            print(
                "[learner_checkpoint] "
                f"request_id={current_checkpoint_request_id} "
                f"steps={trigger_steps} q_updates={state.total_updates} path={checkpoint_dir}"
            )
        except Exception:
            print(f"[learner_checkpoint] save FAILED:\n{traceback.format_exc()}")
        state.last_handled_checkpoint_request_id = current_checkpoint_request_id

    total_replay_size = replay.len("success") + replay.len("failure")
    if total_replay_size < int(args.min_replay_size_before_learning):
        return False

    # Fresh-buffer-fill phase. Mirrors td3_training.py's `learning_starts`:
    # gate the learner on fresh post-launch collector steps only, so the
    # critic's first gradient step lands on a buffer that has seen pure
    # on-policy data — independent of how full warm-start replay is. The
    # orchestrator publishes `fresh_collector_steps_this_run` to stats
    # before each learner call (run-relative, so resumes auto-skip when
    # the prior run already crossed the gate). 0 disables (default).
    if int(args.learning_starts_fresh_steps) > 0:
        fresh_steps_this_run = int(stats.get("fresh_collector_steps_this_run", 0))
        threshold = int(args.learning_starts_fresh_steps)
        if fresh_steps_this_run < threshold:
            stats["learning_starts_pending"] = float(threshold - fresh_steps_this_run)
            # Log once when the gate first engages, so it's visible without
            # spamming every episode boundary.
            if not bool(stats.get("learning_starts_logged_engage", False)):
                print(
                    f"[learner] buffer-fill gate engaged: waiting for "
                    f"{threshold} fresh collector steps before first "
                    f"learner update (currently at {fresh_steps_this_run})."
                )
                stats["learning_starts_logged_engage"] = True
            return False
        # Gate just opened — log once and stop reporting `learning_starts_pending`.
        if stats.pop("learning_starts_pending", None) is not None:
            print(
                f"[learner] buffer-fill gate cleared at "
                f"{fresh_steps_this_run} fresh collector steps "
                f"(threshold={threshold}). Starting critic / actor updates."
            )

    actor_updated = False
    for q_update_idx in range(args.q_updates):
        batch, success_batch_count, failure_batch_count = _mixed_sample_from_shared(
            replay=replay,
            batch_size=args.batch_size,
            success_fraction=args.critic_success_sample_fraction,
            device=args.learner_device,
        )
        if batch is None:
            continue
        sampled_observations = batch["observations"]
        sampled_next_observations = batch["next_observations"]
        sampled_actions = batch["actions"]
        sampled_rewards = batch["rewards"]
        sampled_dones = batch["dones"]
        sampled_next_prev_actions = sampled_actions * (1.0 - sampled_dones.unsqueeze(-1))
        sampled_next_policy_observations = augment_policy_observation(
            sampled_next_observations,
            sampled_next_prev_actions,
            train_args.use_last_action_in_policy_state,
        )
        n_critics = len(state.qfs)
        with torch.no_grad():
            target_next_action = deterministic_actor_action(state.actor_target, sampled_next_policy_observations)
            noise = torch.randn_like(target_next_action) * float(args.policy_noise)
            noise = torch.clamp(noise, -float(args.noise_clip), float(args.noise_clip))
            target_next_action = torch.clamp(target_next_action + noise, state.action_low, state.action_high)
            # Target Q = min over the chosen subset (Maxmin if subset is None or
            # ≥N; REDQ-style random M-of-N otherwise).
            subset_size = state.target_critic_subset_size
            if subset_size is None or int(subset_size) >= n_critics:
                target_indices = list(range(n_critics))
            else:
                target_indices = (
                    torch.randperm(n_critics)[: int(subset_size)].tolist()
                )
            next_q_h_list = [
                state.qfs_target[ti](sampled_next_observations, target_next_action)
                for ti in target_indices
            ]
            if len(next_q_h_list) == 1:
                min_next_q_h = next_q_h_list[0]
            elif len(next_q_h_list) == 2:
                min_next_q_h = torch.min(next_q_h_list[0], next_q_h_list[1])
            else:
                min_next_q_h = torch.stack(next_q_h_list, dim=0).min(dim=0).values
            min_next_q = h_inverse(min_next_q_h, eps=float(args.h_transform_eps)).view(-1)
            bellman = (
                sampled_rewards
                + (1.0 - sampled_dones) * float(args.gamma) * min_next_q
            )
            target_h = h_transform(bellman, eps=float(args.h_transform_eps))
        # Pre-compute CQL random actions and policy action once per minibatch.
        cql_enabled = float(args.cql_alpha) > 0.0
        cql_penalty_value: float | None = None
        if cql_enabled:
            bsz = sampled_observations.shape[0]
            n_rand = int(args.cql_n_random)
            act_dim = sampled_actions.shape[-1]
            cql_random_actions = torch.empty(
                n_rand * bsz, act_dim, device=args.learner_device
            ).uniform_(-1.0, 1.0)
            cql_obs_repeat = sampled_observations.unsqueeze(0).expand(
                n_rand, -1, -1
            ).reshape(n_rand * bsz, -1)
            sampled_prev_actions = batch.get("prev_actions")
            if sampled_prev_actions is None:
                sampled_prev_actions = sampled_actions
            cql_policy_obs = augment_policy_observation(
                sampled_observations,
                sampled_prev_actions,
                train_args.use_last_action_in_policy_state,
            )
            with torch.no_grad():
                cql_policy_action = deterministic_actor_action(
                    state.actor, cql_policy_obs
                )
        # Forward pass over all N critics; train each against the shared target.
        qi_h_list = []
        qi_loss_list = []
        cql_penalty_per_critic: list[torch.Tensor] = []
        for q in state.qfs:
            qi_h = q(sampled_observations, sampled_actions)
            qi_h_list.append(qi_h)
            loss_i = torch.nn.functional.mse_loss(qi_h.view(-1), target_h)
            if cql_enabled:
                q_rand_h = q(cql_obs_repeat, cql_random_actions).view(n_rand, bsz)
                q_pi_h = q(sampled_observations, cql_policy_action).view(-1)
                cql_logsumexp = (
                    torch.logsumexp(q_rand_h, dim=0) - math.log(float(n_rand))
                )
                cql_penalty = (cql_logsumexp - q_pi_h).mean()
                loss_i = loss_i + float(args.cql_alpha) * cql_penalty
                cql_penalty_per_critic.append(cql_penalty.detach())
            qi_loss_list.append(loss_i)
        if cql_enabled and cql_penalty_per_critic:
            cql_penalty_value = float(
                torch.stack(cql_penalty_per_critic).mean().item()
            )
        q1_h = qi_h_list[0]
        q_loss = sum(qi_loss_list)
        state.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        state.q_optimizer.step()
        state.total_updates += 1
        positive_reward_mask = sampled_rewards > 0
        positive_reward_count = float(positive_reward_mask.sum().item())
        minibatch_size = max(int(sampled_rewards.numel()), 1)
        positive_rewards = sampled_rewards[positive_reward_mask]
        state.latest_train_metrics.update(
            {
                "losses/q_loss": float(
                    (sum(qi_loss_list) / float(n_critics)).item()
                ),
                "losses/q_total_loss": float(q_loss.item()),
                "losses/q1_mean": float(q1_h.mean().item()),
                "rewards/sampled_reward_mean": float(sampled_rewards.mean().item()),
                "rewards/sampled_reward_min": float(sampled_rewards.min().item()),
                "rewards/sampled_reward_std": float(sampled_rewards.std(unbiased=False).item()),
                "rewards/sampled_reward_positive_count": positive_reward_count,
                "rewards/sampled_reward_positive_fraction": (
                    positive_reward_count / float(minibatch_size)
                ),
                "rewards/sampled_reward_positive_mean": (
                    float(positive_rewards.mean().item()) if positive_rewards.numel() > 0 else 0.0
                ),
                "rewards/sampled_reward_positive_std": (
                    float(positive_rewards.std(unbiased=False).item())
                    if positive_rewards.numel() > 0
                    else 0.0
                ),
                "replay/success_buffer_size": float(replay.len("success")),
                "replay/failure_buffer_size": float(replay.len("failure")),
                "replay/critic_success_sample_count": float(success_batch_count),
                "replay/critic_failure_sample_count": float(failure_batch_count),
                "replay/critic_success_sample_fraction": (
                    float(success_batch_count) / float(max(args.batch_size, 1))
                ),
                "replay/critic_failure_sample_fraction": (
                    float(failure_batch_count) / float(max(args.batch_size, 1))
                ),
            }
        )
        if cql_penalty_value is not None:
            state.latest_train_metrics["losses/cql_penalty"] = cql_penalty_value
            state.latest_train_metrics["losses/cql_penalty_weighted"] = (
                cql_penalty_value * float(args.cql_alpha)
            )
        if (q_update_idx + 1) % int(args.target_network_frequency) == 0:
            with torch.no_grad():
                # Polyak-average all N critic-target pairs + the actor target.
                # In residual mode the wrapped ResidualActor target shares the
                # frozen base with the online ResidualActor, so this only
                # updates the residual head (and clamp buffers, which are
                # parameters with no grad) — matches td3_training.py.
                sync_pairs: list[tuple] = [
                    (q, qt) for q, qt in zip(state.qfs, state.qfs_target)
                ]
                sync_pairs.append((state.actor, state.actor_target))
                for source, target in sync_pairs:
                    for param, target_param in zip(source.parameters(), target.parameters()):
                        target_param.data.copy_(
                            float(args.tau) * param.data + (1.0 - float(args.tau)) * target_param.data
                        )
    for _ in range(args.actor_updates_per_iteration):
        actor_batch, _, _ = _mixed_sample_from_shared(
            replay=replay,
            batch_size=args.batch_size,
            success_fraction=args.critic_success_sample_fraction,
            device=args.learner_device,
        )
        if actor_batch is None:
            continue
        actor_obs = actor_batch["observations"]
        actor_prev_actions = actor_batch["prev_actions"]
        actor_policy_obs = augment_policy_observation(
            actor_obs,
            actor_prev_actions,
            train_args.use_last_action_in_policy_state,
        )
        policy_actions = deterministic_actor_action(state.actor, actor_policy_obs)
        q1_h = state.qf1(actor_obs, policy_actions)
        q1 = h_inverse(q1_h, eps=float(args.h_transform_eps)).view(-1)
        actor_loss = -q1.mean()
        residual_action_l2_loss: float | None = None
        if (
            args.full_checkpoint_load in ("residual", "residual_resume")
            and args.residual_action_l2 > 0.0
        ):
            residual_action = state.actor.residual.get_action(actor_policy_obs)
            l2_term = (residual_action ** 2).mean()
            actor_loss = actor_loss + args.residual_action_l2 * l2_term
            residual_action_l2_loss = float(l2_term.item())
        state.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        state.actor_optimizer.step()
        state.total_actor_updates += 1
        actor_updated = True
        if state.actor_ema is not None and args.full_checkpoint_load in ("residual", "residual_resume"):
            decay = float(args.residual_ema_decay)
            with torch.no_grad():
                for ema_param, online_param in zip(
                    state.actor_ema.residual.parameters(),
                    state.actor.residual.parameters(),
                ):
                    ema_param.data.mul_(decay).add_(online_param.data, alpha=1.0 - decay)
        norm_q = (1.0 - float(args.gamma)) * q1
        state.latest_train_metrics.update(
            {
                "losses/actor_loss": float(actor_loss.item()),
                "losses/actor_norm_q_mean": float(norm_q.mean().item()),
            }
        )
        if residual_action_l2_loss is not None:
            state.latest_train_metrics["losses/residual_action_l2"] = residual_action_l2_loss
    now = time.time()
    if now - state.last_log_time >= float(args.learner_log_interval_sec):
        stats["learner_q_updates"] = float(state.total_updates)
        stats["learner_actor_updates"] = float(state.total_actor_updates)
        stats["learner_replay_size"] = float(total_replay_size)
        step_index = max(state.total_updates, 1)
        for metric_name, metric_value in state.latest_train_metrics.items():
            state.writer.add_scalar(metric_name, float(metric_value), step_index)
        state.writer.add_scalar(
            "charts/SPS",
            float(state.total_updates) / max(now - state.learner_start_time, 1e-6),
            step_index,
        )
        state.writer.add_scalar("replay/success_buffer_size", float(replay.len("success")), step_index)
        state.writer.add_scalar("replay/failure_buffer_size", float(replay.len("failure")), step_index)
        print(
            "[learner] "
            f"q_updates={state.total_updates} actor_updates={state.total_actor_updates} replay_size={total_replay_size}"
        )
        state.last_log_time = now
    return actor_updated


def _prompt_optional_run_note() -> str:
    """Prompt the user for an optional note describing this run.

    Returns the entered note (stripped). Returns empty string if the user
    skips it or the prompt is run in a non-interactive context.
    """
    try:
        note = input("Optional run note (press Enter to skip): ").strip()
    except EOFError:
        note = ""
    return note


def _setup_run_data_dir(args: Args, run_note: str) -> Path:
    """Create the unified per-run folder and route every artifact into it.

    Layout (all under `<data_root_dir>/data_<TIMESTAMP>/`):

        episode_hdf5/             ← per-step trajectories
        reset_hdf5/               ← reset-FSM trajectories
        episode_gifs/              ← side-by-side Box2D + camera GIFs
        episode_camera_videos/     ← raw camera MP4s
        collector_tb/              ← collector TensorBoard scalars
        learner_tb/                ← learner TensorBoard scalars
        checkpoint_*/              ← periodic checkpoints (when enabled)
        latency_profiles/          ← when --enable-latency-profiling is set
        run_note.txt               ← optional human note

    Side effects on `args`:
      - `episode_artifact_dir`, `reset_artifact_dir`, `episode_gif_dir`,
        `episode_camera_video_dir` are populated to subdirs of run_data_dir.
      - `checkpoint_root_dir` is forced to `run_data_dir` so TB logs and
        checkpoints share the run folder. Any prior value (CLI or args-file)
        is overridden.
      - `log_parent_dir` is forced to `None` so the precedence chain in
        `main()` resolves cleanly to `checkpoint_root_dir`.

    To direct artifacts to a different location, change `--data-root-dir`.
    """
    data_root_base = Path(args.data_root_dir).expanduser().resolve()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_data_dir = data_root_base / f"data_{timestamp}"
    run_data_dir.mkdir(parents=True, exist_ok=True)

    # Episode/reset artifact subdirs are populated dynamically (not declared
    # on Args) so the rest of the collector / learner code can keep reading
    # the same attribute names.
    args.episode_artifact_dir = str(run_data_dir / "episode_hdf5")
    args.reset_artifact_dir = str(run_data_dir / "reset_hdf5")
    args.episode_gif_dir = str(run_data_dir / "episode_gifs")
    args.episode_camera_video_dir = str(run_data_dir / "episode_camera_videos")

    # Force TB logs + checkpoints under the same root as the episode data so
    # a single folder holds everything for the run.
    prior_checkpoint_root = args.checkpoint_root_dir
    prior_log_parent = args.log_parent_dir
    args.checkpoint_root_dir = str(run_data_dir)
    args.log_parent_dir = None

    note_path = run_data_dir / "run_note.txt"
    if run_note:
        note_path.write_text(run_note + "\n", encoding="utf-8")

    print(f"[run_data] all artifacts unified under: {run_data_dir}")
    if (prior_checkpoint_root and str(prior_checkpoint_root).strip()) or (
        prior_log_parent and str(prior_log_parent).strip()
    ):
        print(
            "[run_data] ignoring prior checkpoint_root_dir="
            f"{prior_checkpoint_root!r} log_parent_dir={prior_log_parent!r} "
            "(unified under run_data_dir; change --data-root-dir to relocate)"
        )
    if run_note:
        print(f"[run_data] note: {run_note}")

    return run_data_dir


def _finalize_sync_learner_state(
    args: Args,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    state: LearnerRuntimeState,
) -> None:
    if args.enable_periodic_checkpointing:
        final_steps = int(stats.get("collector_total_steps", 0))
        final_tag = f"final_step_{final_steps}"
        try:
            final_checkpoint_dir = _save_checkpoint_from_learner_state(
                state=state,
                replay=replay,
                stats=stats,
                checkpoint_tag=final_tag,
                args=args,
                train_args=train_args,
            )
            stats["last_checkpoint_dir"] = str(final_checkpoint_dir)
            stats["last_checkpoint_collector_steps"] = float(final_steps)
            stats["last_checkpoint_q_updates"] = float(state.total_updates)
            print(f"[learner_checkpoint] final steps={final_steps} path={final_checkpoint_dir}")
        except Exception:
            print(f"[learner_checkpoint] final save FAILED:\n{traceback.format_exc()}")
    state.writer.close()
