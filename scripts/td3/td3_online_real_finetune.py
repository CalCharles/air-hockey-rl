"""Online real-robot fine-tuning of a sim-trained TD3 policy (sim-to-online recipe).

What this is
------------
Real-robot entrypoint that takes a TD3 checkpoint trained in the Box2D sim
(``scripts/td3/td3_training.py`` → ``training_state.pth``, e.g. the sysid-v2
hist2 / hist4 juggle policies) and keeps training it online on the UR5, using
the recipe from As et al., "What Matters for Simulation-to-Online
Reinforcement Learning on Real Robots" (arXiv:2602.20220):

1. **In-place load.** Actor, actor target, every critic, every critic target,
   and both Adam states come from the sim ``training_state.pth``. The online
   networks are the sim ones (no residual head, no fresh critic). After the
   optimizer load the learning rates / weight decay are re-applied from the
   args file (``load_state_dict`` would otherwise restore the sim actor LR of
   3e-4): actor LR = ``policy_lr`` (1e-5), critic LR = ``q_lr`` (1e-3).
2. **One real replay buffer, no sim data.** Transitions go into a single
   online buffer that starts empty; the sim replay buffer in the checkpoint is
   ignored. No D0 mixing, no α annealing, no success / failure split, no CQL.
3. **Warm start.** The first ``warm_start_episodes`` kept episodes (20) run
   the loaded sim actor unchanged and are only stored — the learner is idle.
4. **Per-episode UTD.** After every later kept episode of T transitions the
   learner runs K = round(``utd_ratio`` × T) critic updates (η = 5, so a
   180-step episode → 900 critic updates), sampling the online buffer only.
5. **Delayed actor (M).** Inside those K critic updates, one actor update
   follows every ``actor_update_every``-th critic update (M = 20), i.e.
   ⌊K / M⌋ actor updates per episode; the K − M⌊K / M⌋ leftover critic
   updates get no actor update (paper Eq. 6). Polyak (τ) updates of all
   targets stay inside the critic loop, every ``target_network_frequency``
   critic updates.
6. **Episodic ("asynchronous") learning.** Learning happens between
   episodes, decoupled from the 1-step control loop — the robot is idle while
   the K updates run, then the updated actor is synced to the collector.

Everything else (policy / reset runners, safety stops, transition holds,
HDF5 / GIF / video artifacts, JSONL run logs, periodic checkpoints) is the
existing real-world stack: this file reuses the
``scripts/td3/extras/async_td3_real.py`` orchestrator and swaps in its own
replay push and learner step through that orchestrator's hooks.

Checkpoints written here use the standard real-world layout (``model.pth``,
``qf*.pth``, ``training_state.pth`` with optimizer state when
``include_non_vital_training_state_fields: true``), so a saved checkpoint can
be passed back as ``--checkpoint`` (networks + optimizers carry over; the
online buffer and the warm start start fresh) or evaluated with
``scripts/td3/extras/async_td3_real_eval.py``.

Usage (on the real-robot machine, from the repo root)
-----------------------------------------------------
hist2 sim policy:

    python -m scripts.td3.td3_online_real_finetune \\
        --checkpoint runs/td3/sysid_v2_hist_len_2_train_2M/juggle_sysid_v2_hist2/training_state.pth \\
        --args-file  configs/td3/td3_online_real_finetune/juggle_hist2.yaml

hist4 sim policy:

    python -m scripts.td3.td3_online_real_finetune \\
        --checkpoint runs/td3/sysid_v2_hist_len_4_train_step_2M/juggle_sysid_v2_hist4/training_state.pth \\
        --args-file  configs/td3/td3_online_real_finetune/juggle_hist4.yaml

``--checkpoint`` must be a full ``training_state.pth`` (not ``model.pth``):
the critics and optimizer states are needed. The args file carries the recipe
knobs and the real rollout config (``config:``); ``--config`` overrides the
latter. Network architecture is read from the checkpoint's stored ``args``
(fallback: ``args.yaml`` next to the checkpoint; override: ``--train-args``).
Any ``Args`` field can be overridden on the CLI, e.g. ``--warm-start-episodes
0`` or ``--total-timesteps 0`` (run until Ctrl-C). ``--no-quiet`` restores the
per-step robot debug prints.
"""
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.optim as optim
import tyro
import yaml

from airhockey import AirHockeyEnv
from scripts.td3.deterministic_agent import DeterministicAgent
from scripts.td3.extras.async_td3_real import (
    _parse_modular_specific_args,
    collector_process_modular,
)
from scripts.td3.helper.real_td3_runtime import (
    ROLLING_PERF_WINDOW_EPISODES,
    Args,
    LearnerRuntimeState,
    TrainArgs,
    _checkpoint_root_from_tb,
    _episode_to_tensors,
    _finalize_sync_learner_state,
    _make_qf,
    _prepare_air_hockey_config,
    _prompt_optional_run_note,
    _save_checkpoint_from_learner_state,
    _setup_run_data_dir,
    augment_policy_observation,
    build_policy_env_view,
    deterministic_actor_action,
    h_inverse,
    h_transform,
    install_quiet_print_filter,
)
from scripts.td3.helper.run_event_log import append_run_event
from scripts.td3.helper.shared_replay import SharedTD3Replay
from scripts.td3.helper.td3_episode_collection import EpisodeTrajectory

# ``SharedTD3Replay`` always has a success and a failure partition. The online
# buffer is the "success" partition; "failure" is a 1-slot stub that is never
# written or sampled. Episode summaries label the partition "online".
_ONLINE_PARTITION = "success"
_ONLINE_PARTITION_LABEL = "online"

# Base ``Args`` fields that belong to the residual / two-buffer / fixed-count
# learner of async_td3_real.py and have no effect here. Setting them in the
# args file is reported as ignored instead of being silently applied.
_UNUSED_BASE_FIELDS = frozenset(
    {
        "model_path",
        "full_checkpoint_load",
        "residual_scale",
        "residual_weight_decay",
        "residual_ema_decay",
        "residual_action_l2",
        "q_updates",
        "actor_updates_per_iteration",
        "cql_n_random",
        "success_buffer_size",
        "failure_buffer_size",
        "success_top_fraction",
        "critic_success_sample_fraction",
        "critic_failure_sample_fraction",
        "warm_start_hdf5_recursive",
        "replay_source_priority",
        "load_replay_from_checkpoint",
        "min_replay_size_before_learning",
        "learning_starts_fresh_steps",
    }
)

_REQUIRED_CHECKPOINT_KEYS = ("actor", "actor_target", "q_optimizer", "actor_optimizer")


@dataclass
class OnlineFinetuneArgs(Args):
    # Sim-trained training_state.pth to fine-tune (required).
    checkpoint: str | None = None
    # η: critic updates per real transition. K = round(utd_ratio * T) after
    # each post-warm-start episode of T transitions.
    utd_ratio: float = 5.0
    # M: one actor update after every M-th critic update within an episode's
    # K critic updates (⌊K / M⌋ actor updates per episode).
    actor_update_every: int = 20
    # N*: kept episodes collected with the loaded sim actor before the first
    # update. Their transitions seed the online buffer.
    warm_start_episodes: int = 20
    # Capacity of the single online replay buffer (transitions).
    online_buffer_size: int = 100_000


# ---------------------------------------------------------------------------
# Args / checkpoint loading
# ---------------------------------------------------------------------------


def _build_finetune_args_file_defaults(args_file_path: str) -> tuple[dict, list[str], list[str]]:
    """Map an online-finetune args YAML onto ``OnlineFinetuneArgs`` defaults.

    Returns ``(defaults, applied_keys, ignored_keys)``. Keys that are not
    ``OnlineFinetuneArgs`` fields, or that only drive the async_td3_real
    learner (``_UNUSED_BASE_FIELDS``), are ignored.
    """
    with open(args_file_path, "r") as f:
        loaded = yaml.load(f, Loader=yaml.FullLoader)
    if loaded is None:
        return {}, [], []
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected args file {args_file_path} to be a YAML mapping, got {type(loaded)}")
    valid = {f.name for f in fields(OnlineFinetuneArgs)} - _UNUSED_BASE_FIELDS
    defaults: dict = {}
    applied: list[str] = []
    ignored: list[str] = []
    for key, value in loaded.items():
        if key in valid:
            defaults[key] = value
            applied.append(key)
        else:
            ignored.append(key)
    return defaults, sorted(applied), sorted(ignored)


def _load_checkpoint(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"--checkpoint does not exist: {path}")
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(
            f"--checkpoint {path} is a {type(loaded).__name__}, expected a training_state.pth dict "
            "(a bare model.pth has no critics or optimizer state)."
        )
    missing = [key for key in _REQUIRED_CHECKPOINT_KEYS if key not in loaded]
    if "qf1" not in loaded or "qf1_target" not in loaded:
        missing.append("qf1/qf1_target")
    if missing:
        raise KeyError(
            f"--checkpoint {path} is missing {missing}. In-place fine-tuning needs the actor, "
            "actor target, critics, critic targets, and both optimizer states of a full "
            "training_state.pth."
        )
    return loaded


def _train_args_from_mapping(mapping: dict, source: str) -> TrainArgs:
    required = (
        "agent_hidden_layer_size",
        "agent_num_hidden_layers",
        "q_hidden_layer_size",
        "q_num_hidden_layers",
        "use_last_action_in_policy_state",
    )
    missing = [key for key in required if key not in mapping]
    if missing:
        raise KeyError(f"Architecture source {source} is missing {missing}.")
    raw_subset = mapping.get("target_critic_subset_size")
    return TrainArgs(
        agent_hidden_layer_size=int(mapping["agent_hidden_layer_size"]),
        agent_num_hidden_layers=int(mapping["agent_num_hidden_layers"]),
        q_hidden_layer_size=int(mapping["q_hidden_layer_size"]),
        q_num_hidden_layers=int(mapping["q_num_hidden_layers"]),
        use_last_action_in_policy_state=bool(mapping["use_last_action_in_policy_state"]),
        num_critics=int(mapping.get("num_critics", 2) or 2),
        target_critic_subset_size=None if raw_subset is None else int(raw_subset),
    )


def _resolve_train_args(args: OnlineFinetuneArgs, checkpoint: Dict[str, object]) -> tuple[TrainArgs, str]:
    """Architecture: ``--train-args`` > checkpoint ``args`` > sibling ``args.yaml``."""
    if args.train_args is not None:
        with open(args.train_args, "r") as f:
            return _train_args_from_mapping(yaml.load(f, Loader=yaml.FullLoader), args.train_args), args.train_args
    ckpt_args = checkpoint.get("args")
    if isinstance(ckpt_args, dict) and "agent_hidden_layer_size" in ckpt_args:
        return _train_args_from_mapping(ckpt_args, "checkpoint['args']"), "checkpoint['args']"
    sibling = Path(str(args.checkpoint)).parent / "args.yaml"
    if sibling.exists():
        with open(sibling, "r") as f:
            return _train_args_from_mapping(yaml.load(f, Loader=yaml.FullLoader), str(sibling)), str(sibling)
    raise FileNotFoundError(
        "Could not find the network architecture: the checkpoint has no stored args and there is "
        f"no {sibling}. Pass --train-args <training run args.yaml>."
    )


def _hist_len_of(config_path: str | None) -> int | None:
    if not config_path or not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader) or {}
    hist_len = config.get("air_hockey", {}).get("simulator_params", {}).get("hist_len")
    return None if hist_len is None else int(hist_len)


def _check_hist_len_matches(args: OnlineFinetuneArgs, checkpoint: Dict[str, object]) -> None:
    """Refuse a hist2 policy on a hist4 real config (and vice versa)."""
    ckpt_args = checkpoint.get("args") if isinstance(checkpoint.get("args"), dict) else {}
    sim_config = ckpt_args.get("config")
    sim_hist = _hist_len_of(sim_config)
    real_hist = _hist_len_of(args.config)
    if sim_hist is None or real_hist is None:
        print(
            f"[online_finetune] hist_len check skipped (sim config={sim_config!r} -> {sim_hist}, "
            f"real config={args.config!r} -> {real_hist})."
        )
        return
    if sim_hist != real_hist:
        raise ValueError(
            f"hist_len mismatch: the checkpoint was trained with hist_len={sim_hist} ({sim_config}) "
            f"but the real config {args.config} uses hist_len={real_hist}. Use the matching "
            "configs/td3/td3_online_real_finetune/juggle_hist{2,4}.yaml or --config."
        )
    print(f"[online_finetune] hist_len={real_hist} matches the sim training config ({sim_config}).")


def _validate_args(args: OnlineFinetuneArgs) -> None:
    if args.utd_ratio <= 0:
        raise ValueError("utd_ratio must be > 0.")
    if args.actor_update_every <= 0:
        raise ValueError("actor_update_every must be > 0.")
    if args.warm_start_episodes < 0:
        raise ValueError("warm_start_episodes must be >= 0.")
    if args.online_buffer_size <= 0:
        raise ValueError("online_buffer_size must be > 0.")
    if args.target_network_frequency <= 0:
        raise ValueError("target_network_frequency must be > 0.")
    if float(args.cql_alpha) != 0.0:
        raise ValueError("The sim-to-online recipe has no CQL term: cql_alpha must be 0.")
    if args.full_checkpoint_load in ("residual", "residual_resume"):
        raise ValueError("No residual head in this recipe: full_checkpoint_load must not be residual.")
    if len(args.warm_start_hdf5_dirs) > 0:
        raise ValueError(
            "warm_start_hdf5_dirs is not supported: this recipe starts from an empty online buffer "
            "and warm-starts by running the sim policy for warm_start_episodes episodes."
        )
    if args.enable_periodic_checkpointing and int(args.checkpoint_every_collector_steps) <= 0:
        raise ValueError("checkpoint_every_collector_steps must be > 0 when checkpointing is enabled.")
    if args.enable_periodic_checkpointing and not args.include_non_vital_training_state_fields:
        print(
            "[online_finetune] WARNING: include_non_vital_training_state_fields=False — checkpoints "
            "will not contain optimizer state, so they cannot be passed back as --checkpoint."
        )


# ---------------------------------------------------------------------------
# Learner construction (in-place load of the sim networks + optimizers)
# ---------------------------------------------------------------------------


def _override_adam_hyperparams(optimizer: optim.Optimizer, *, lr: float, weight_decay: float) -> None:
    """Re-apply LR / weight decay after ``load_state_dict`` and drop the sim's
    CUDA-graph flags.

    ``Optimizer.load_state_dict`` restores the saved param-group options, so the
    sim's ``lr`` (actor 3e-4), ``weight_decay`` and ``capturable=True`` /
    ``fused=True`` (td3_training.py captures its updates in CUDA graphs) would
    win over the constructor arguments. Adam moments (``exp_avg`` /
    ``exp_avg_sq``) and step counts are kept. ``fused`` / ``foreach`` go back
    to ``None`` (not ``False``): only then does Adam auto-pick the multi-tensor
    path — ``fused=False`` silently selects the ~6x slower per-tensor loop.
    """
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
        group["weight_decay"] = float(weight_decay)
        group["capturable"] = False
        group["fused"] = None
        group["foreach"] = None
    for param_state in optimizer.state.values():
        step = param_state.get("step")
        if torch.is_tensor(step):
            param_state["step"] = step.detach().to("cpu", torch.float32)


def _build_learner_from_checkpoint(
    *,
    args: OnlineFinetuneArgs,
    train_args: TrainArgs,
    checkpoint: Dict[str, object],
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    tb_log_dir: str,
) -> LearnerRuntimeState:
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.learner_device)

    policy_obs_dim = obs_dim + act_dim if train_args.use_last_action_in_policy_state else obs_dim
    policy_env_view = build_policy_env_view(policy_obs_dim, act_dim)

    def _new_actor() -> DeterministicAgent:
        return DeterministicAgent(
            policy_env_view,
            action_scale=1.0,
            action_bias=0.0,
            hidden_layer_size=train_args.agent_hidden_layer_size,
            num_hidden_layers=train_args.agent_num_hidden_layers,
        ).to(device)

    actor = _new_actor()
    actor_target = _new_actor()
    # strict=True: a residual / differently-shaped checkpoint must fail loudly
    # rather than silently half-load.
    actor.load_state_dict(checkpoint["actor"], strict=True)
    actor_target.load_state_dict(checkpoint["actor_target"], strict=True)

    num_critics = int(train_args.num_critics)
    n_in_ckpt = sum(
        1 for key in checkpoint if key.startswith("qf") and not key.endswith("_target") and key[2:].isdigit()
    )
    if n_in_ckpt != num_critics:
        raise ValueError(
            f"Checkpoint has {n_in_ckpt} critics but the architecture says num_critics={num_critics}."
        )
    qfs = [
        _make_qf(obs_dim, act_dim, train_args.q_hidden_layer_size, train_args.q_num_hidden_layers, device)
        for _ in range(num_critics)
    ]
    qfs_target = [
        _make_qf(obs_dim, act_dim, train_args.q_hidden_layer_size, train_args.q_num_hidden_layers, device)
        for _ in range(num_critics)
    ]
    for i, (q, qt) in enumerate(zip(qfs, qfs_target), start=1):
        q.load_state_dict(checkpoint[f"qf{i}"], strict=True)
        qt.load_state_dict(checkpoint[f"qf{i}_target"], strict=True)

    # Same parameter order as td3_training.py, so the saved Adam state maps 1:1.
    q_optimizer = optim.Adam([p for q in qfs for p in q.parameters()], lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    q_optimizer.load_state_dict(checkpoint["q_optimizer"])
    actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
    loaded_q_lr = float(q_optimizer.param_groups[0]["lr"])
    loaded_actor_lr = float(actor_optimizer.param_groups[0]["lr"])
    _override_adam_hyperparams(q_optimizer, lr=args.q_lr, weight_decay=args.q_weight_decay)
    _override_adam_hyperparams(actor_optimizer, lr=args.policy_lr, weight_decay=0.0)
    print(
        "[online_finetune] loaded in place from checkpoint: actor + actor_target + "
        f"{num_critics} critics + {num_critics} critic targets + q/actor Adam state. "
        f"actor lr {loaded_actor_lr:g} -> {args.policy_lr:g}, critic lr {loaded_q_lr:g} -> {args.q_lr:g}, "
        f"critic weight_decay -> {args.q_weight_decay:g}. Sim replay buffer NOT loaded."
    )

    target_subset = train_args.target_critic_subset_size
    return LearnerRuntimeState(
        actor=actor,
        actor_target=actor_target,
        qfs=qfs,
        qfs_target=qfs_target,
        target_critic_subset_size=None if target_subset is None else int(target_subset),
        q_optimizer=q_optimizer,
        actor_optimizer=actor_optimizer,
        action_low=torch.as_tensor(action_low_np, dtype=torch.float32, device=device).unsqueeze(0),
        action_high=torch.as_tensor(action_high_np, dtype=torch.float32, device=device).unsqueeze(0),
        writer=SummaryWriter(tb_log_dir),
        checkpoint_root=_checkpoint_root_from_tb(tb_log_dir, args.checkpoint_root_dir),
        last_log_time=time.time(),
        learner_start_time=time.time(),
        total_updates=0,
        total_actor_updates=0,
        last_handled_checkpoint_request_id=0,
    )


# ---------------------------------------------------------------------------
# Replay push + learner step (plugged into collector_process_modular)
# ---------------------------------------------------------------------------


def _make_online_replay_push(stats: Dict[str, object]):
    """Replay push with the ``_add_episode_to_shared_replay`` signature that
    writes every kept episode into the single online buffer and records the
    episode's transition count for the learner's K = η·T budget."""

    def _push(
        replay: SharedTD3Replay,
        episode_trajectory: EpisodeTrajectory,
        recent_episode_returns,
        success_top_fraction: float,
    ) -> tuple[str, float, float, int]:
        del success_top_fraction  # no success / failure split
        episode_return = float(episode_trajectory.episode_return)
        recent_episode_returns.append(episode_return)
        inserted = int(replay.add_episode(_ONLINE_PARTITION, _episode_to_tensors(episode_trajectory)))
        stats["online_kept_episodes"] = float(int(stats.get("online_kept_episodes", 0)) + 1)
        stats["online_pending_episode_steps"] = float(inserted)
        return _ONLINE_PARTITION_LABEL, episode_return, 0.0, inserted

    return _push


def _handle_checkpoint_request(
    args: OnlineFinetuneArgs,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    state: LearnerRuntimeState,
) -> None:
    """Service the orchestrator's periodic-checkpoint request (same contract as
    ``_run_sync_learner_iteration``: request id in ``stats``)."""
    request_id = int(stats.get("checkpoint_save_request_id", 0))
    if not args.enable_periodic_checkpointing or request_id <= state.last_handled_checkpoint_request_id:
        return
    trigger_steps = int(stats.get("checkpoint_trigger_total_steps", stats.get("collector_total_steps", 0)))
    try:
        checkpoint_dir = _save_checkpoint_from_learner_state(
            state=state,
            replay=replay,
            stats=stats,
            checkpoint_tag=f"step_{trigger_steps}",
            args=args,
            train_args=train_args,
        )
        stats["last_checkpoint_dir"] = str(checkpoint_dir)
        stats["last_checkpoint_collector_steps"] = float(trigger_steps)
        stats["last_checkpoint_q_updates"] = float(state.total_updates)
        stats["last_checkpoint_request_id"] = float(request_id)
        print(
            f"[learner_checkpoint] request_id={request_id} steps={trigger_steps} "
            f"q_updates={state.total_updates} path={checkpoint_dir}"
        )
    except Exception:
        print(f"[learner_checkpoint] save FAILED:\n{traceback.format_exc()}")
    state.last_handled_checkpoint_request_id = request_id


class _DeviceReplaySnapshot:
    """The online buffer copied to the learner device once per episode.

    No transitions arrive while an episode's K updates run, so one copy serves
    all of them; sampling indices on-device avoids the CPU fancy-index +
    host-to-device copy of ``SharedTD3Replay.sample`` on every update (the
    dominant per-update cost at K ≈ 1000).
    """

    def __init__(self, replay: SharedTD3Replay, device: str | torch.device):
        partition = replay.state_dict()[_ONLINE_PARTITION]
        self.size = int(partition["size"])
        self.device = torch.device(device)
        self.tensors = {
            key: partition[key].to(self.device)
            for key in ("observations", "next_observations", "actions", "prev_actions", "rewards", "dones")
        }

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = torch.randint(0, self.size, (int(batch_size),), device=self.device)
        return {key: value[indices] for key, value in self.tensors.items()}


def _polyak_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    torch._foreach_lerp_(
        [p.data for p in target.parameters()],
        [p.data for p in source.parameters()],
        float(tau),
    )


def _critic_update(
    args: OnlineFinetuneArgs,
    train_args: TrainArgs,
    buffer: _DeviceReplaySnapshot,
    state: LearnerRuntimeState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One TD3 critic step on the online buffer with the transformed Bellman
    target (same math as td3_training.py / async_td3_real, minus CQL)."""
    batch = buffer.sample(int(args.batch_size))
    observations = batch["observations"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    dones = batch["dones"]
    next_policy_observations = augment_policy_observation(
        batch["next_observations"],
        actions * (1.0 - dones.unsqueeze(-1)),
        train_args.use_last_action_in_policy_state,
    )
    n_critics = len(state.qfs)
    with torch.no_grad():
        next_action = deterministic_actor_action(state.actor_target, next_policy_observations)
        noise = torch.clamp(
            torch.randn_like(next_action) * float(args.policy_noise),
            -float(args.noise_clip),
            float(args.noise_clip),
        )
        next_action = torch.clamp(next_action + noise, state.action_low, state.action_high)
        subset_size = state.target_critic_subset_size
        if subset_size is None or int(subset_size) >= n_critics:
            target_indices = range(n_critics)
        else:
            target_indices = torch.randperm(n_critics)[: int(subset_size)].tolist()
        next_q_h = torch.stack(
            [state.qfs_target[i](batch["next_observations"], next_action) for i in target_indices], dim=0
        ).min(dim=0).values
        next_q = h_inverse(next_q_h, eps=float(args.h_transform_eps)).view(-1)
        target_h = h_transform(
            rewards + (1.0 - dones) * float(args.gamma) * next_q,
            eps=float(args.h_transform_eps),
        )
    q_h_list = [q(observations, actions) for q in state.qfs]
    q_loss = sum(torch.nn.functional.mse_loss(q_h.view(-1), target_h) for q_h in q_h_list)
    state.q_optimizer.zero_grad(set_to_none=True)
    q_loss.backward()
    state.q_optimizer.step()
    state.total_updates += 1
    if state.total_updates % int(args.target_network_frequency) == 0:
        # target ← (1 − τ)·target + τ·online for every critic and the actor.
        with torch.no_grad():
            for source, target in [*zip(state.qfs, state.qfs_target), (state.actor, state.actor_target)]:
                _polyak_update(source, target, float(args.tau))
    return (q_loss / float(n_critics)).detach(), q_h_list[0].mean().detach()


def _actor_update(
    args: OnlineFinetuneArgs,
    train_args: TrainArgs,
    buffer: _DeviceReplaySnapshot,
    state: LearnerRuntimeState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One deterministic policy-gradient step through Q1 on the online buffer."""
    batch = buffer.sample(int(args.batch_size))
    policy_observations = augment_policy_observation(
        batch["observations"],
        batch["prev_actions"],
        train_args.use_last_action_in_policy_state,
    )
    policy_actions = deterministic_actor_action(state.actor, policy_observations)
    q1 = h_inverse(state.qf1(batch["observations"], policy_actions), eps=float(args.h_transform_eps)).view(-1)
    actor_loss = -q1.mean()
    state.actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    state.actor_optimizer.step()
    state.total_actor_updates += 1
    return actor_loss.detach(), ((1.0 - float(args.gamma)) * q1).mean().detach()


def _online_learner_step(
    args: OnlineFinetuneArgs,
    train_args: TrainArgs,
    replay: SharedTD3Replay,
    stats: Dict[str, object],
    state: LearnerRuntimeState,
) -> bool:
    """Post-episode learner (``_run_sync_learner_iteration`` signature).

    Warm start: idle for the first ``warm_start_episodes`` kept episodes.
    After that: K = round(η·T) critic updates for the episode's T transitions,
    with one actor update after every M-th critic update. Returns True when the
    actor changed (the orchestrator then syncs it to the collector).
    """
    _handle_checkpoint_request(args, train_args, replay, stats, state)

    episode_steps = int(stats.pop("online_pending_episode_steps", 0))
    kept_episodes = int(stats.get("online_kept_episodes", 0))
    buffer_size = replay.len(_ONLINE_PARTITION)
    if kept_episodes <= int(args.warm_start_episodes):
        print(
            f"[online_warm_start] episode {kept_episodes}/{args.warm_start_episodes} stored "
            f"(T={episode_steps}, buffer={buffer_size}); learner idle, sim actor on the robot."
        )
        if kept_episodes == int(args.warm_start_episodes):
            print("[online_warm_start] warm start complete — updates start after the next episode.")
        return False
    if episode_steps <= 0 or buffer_size <= 0:
        return False

    critic_budget = max(1, int(round(float(args.utd_ratio) * episode_steps)))
    every = int(args.actor_update_every)
    start = time.time()
    buffer = _DeviceReplaySnapshot(replay, args.learner_device)
    q_losses, q1_means, actor_losses, actor_norm_qs = [], [], [], []
    for k in range(1, critic_budget + 1):
        q_loss, q1_mean = _critic_update(args, train_args, buffer, state)
        q_losses.append(q_loss)
        q1_means.append(q1_mean)
        # Paper Eq. 6: blocks of M critic updates each followed by one actor
        # update; the K mod M leftover critic updates get no actor update.
        if k % every == 0:
            actor_loss, actor_norm_q = _actor_update(args, train_args, buffer, state)
            actor_losses.append(actor_loss)
            actor_norm_qs.append(actor_norm_q)
    learner_s = time.time() - start

    actor_updates = len(actor_losses)
    metrics: Dict[str, float] = {
        "losses/q_loss": float(torch.stack(q_losses).mean().item()),
        "losses/q1_mean": float(torch.stack(q1_means).mean().item()),
        "online/episode_transitions_T": float(episode_steps),
        "online/critic_updates_K": float(critic_budget),
        "online/actor_updates": float(actor_updates),
        "online/replay_size": float(buffer_size),
        "online/learner_wall_s": float(learner_s),
        "online/actor_lr": float(state.actor_optimizer.param_groups[0]["lr"]),
        "online/critic_lr": float(state.q_optimizer.param_groups[0]["lr"]),
    }
    if actor_updates:
        metrics["losses/actor_loss"] = float(torch.stack(actor_losses).mean().item())
        metrics["losses/actor_norm_q_mean"] = float(torch.stack(actor_norm_qs).mean().item())
    state.latest_train_metrics.update(metrics)
    step_index = max(state.total_updates, 1)
    for name, value in metrics.items():
        state.writer.add_scalar(name, value, step_index)
    state.writer.add_scalar("online/total_actor_updates", float(state.total_actor_updates), step_index)
    stats["learner_q_updates"] = float(state.total_updates)
    stats["learner_actor_updates"] = float(state.total_actor_updates)
    stats["learner_replay_size"] = float(buffer_size)
    print(
        f"[online_learner] episode={kept_episodes} T={episode_steps} K={critic_budget} "
        f"actor_updates={actor_updates} (M={every}) q_loss={metrics['losses/q_loss']:.4f} "
        f"q1_mean={metrics['losses/q1_mean']:.3f} "
        f"actor_loss={metrics.get('losses/actor_loss', float('nan')):.4f} "
        f"total_q={state.total_updates} total_actor={state.total_actor_updates} "
        f"buffer={buffer_size} wall={learner_s:.1f}s"
    )
    return actor_updates > 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _probe_env_spaces(args: OnlineFinetuneArgs) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Obs / action shapes from a Box2D twin of the real config (no robot I/O)."""
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    probe_config = _prepare_air_hockey_config(config, seed=args.seed, return_goal_obs=False)
    probe_config["simulator"] = "box2d"
    probe_sim_params = dict(probe_config.get("simulator_params", {}))
    for key in ("control_mode", "wait_for_space_to_start", "save_path", "debug_control", "debug_control_every"):
        probe_sim_params.pop(key, None)
    probe_config["simulator_params"] = probe_sim_params
    probe_env = AirHockeyEnv(probe_config)
    try:
        return (
            int(np.prod(probe_env.observation_space.shape)),
            int(np.prod(probe_env.action_space.shape)),
            np.asarray(probe_env.action_space.low, dtype=np.float32),
            np.asarray(probe_env.action_space.high, dtype=np.float32),
        )
    finally:
        probe_env.close()


def _initial_stats() -> Dict[str, object]:
    stats: Dict[str, object] = {
        "successful_online_episodes_kept": 0.0,
        "checkpoint_save_request_id": 0.0,
        "last_checkpoint_collector_steps": 0.0,
        "collector_total_steps": 0.0,
        "run_elapsed_total_s": 0.0,
        "rolling50_window_size": float(ROLLING_PERF_WINDOW_EPISODES),
        "rolling50_window_count": 0.0,
        "rolling50_reward_avg": 0.0,
        "rolling50_episode_length_avg": 0.0,
        "rolling50_estop_episode_count": 0.0,
        "online_kept_episodes": 0.0,
    }
    for key in (
        "rolling50_reward_values",
        "rolling50_episode_length_values",
        "rolling50_estop_episode_flags",
        "rolling50_episode_return_values",
        "rolling50_episode_juggles_values",
        "rolling50_episode_contacts_values",
    ):
        stats[key] = []
    return stats


def main(
    args: OnlineFinetuneArgs,
    train_args: TrainArgs,
    checkpoint: Dict[str, object],
    *,
    quiet: bool = True,
) -> None:
    if quiet:
        install_quiet_print_filter()
        print("[main_quiet] per-step / per-reset robot debug prints suppressed (--no-quiet restores them).")
    _validate_args(args)
    _check_hist_len_matches(args, checkpoint)

    obs_dim, act_dim, action_low_np, action_high_np = _probe_env_spaces(args)
    replay = SharedTD3Replay(
        success_capacity=int(args.online_buffer_size),
        failure_capacity=1,
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
    )
    stats = _initial_stats()

    base_log_dir = str(Path(args.checkpoint_root_dir).expanduser().resolve())
    collector_tb_dir = os.path.join(base_log_dir, "collector_tb")
    learner_tb_dir = os.path.join(base_log_dir, "learner_tb")
    os.makedirs(collector_tb_dir, exist_ok=True)
    os.makedirs(learner_tb_dir, exist_ok=True)
    print(f"TensorBoard logs: {base_log_dir}")

    learner_state = _build_learner_from_checkpoint(
        args=args,
        train_args=train_args,
        checkpoint=checkpoint,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        tb_log_dir=learner_tb_dir,
    )
    print(
        "[online_finetune] recipe: "
        f"warm_start_episodes={args.warm_start_episodes} utd_ratio(η)={args.utd_ratio:g} "
        f"actor_update_every(M)={args.actor_update_every} tau={args.tau:g} "
        f"target_network_frequency={args.target_network_frequency} batch_size={args.batch_size} "
        f"gamma={args.gamma:g} exploration_noise={args.exploration_noise:g} "
        f"online_buffer_size={args.online_buffer_size}"
    )

    run_end_reason = "completed"
    try:
        collector_process_modular(
            args,
            train_args,
            replay,
            stats,
            learner_state,
            obs_dim,
            act_dim,
            action_low_np,
            action_high_np,
            collector_tb_dir,
            add_episode_to_replay_fn=_make_online_replay_push(stats),
            learner_step_fn=_online_learner_step,
        )
    except KeyboardInterrupt:
        print("[main] interrupted by user; shutting down.")
        run_end_reason = "keyboard_interrupt"
    except BaseException:
        run_end_reason = "exception"
        raise
    finally:
        _finalize_sync_learner_state(
            args=args,
            train_args=train_args,
            replay=replay,
            stats=stats,
            state=learner_state,
        )
        if int(stats.get("last_checkpoint_request_id", 0)) > 0 or stats.get("last_checkpoint_dir"):
            append_run_event(
                args,
                "checkpoint_saved",
                checkpoint_dir=str(stats.get("last_checkpoint_dir", "")),
                total_steps=int(float(stats.get("last_checkpoint_collector_steps", 0.0))),
                q_updates=int(float(stats.get("last_checkpoint_q_updates", 0.0))),
                trigger="final_on_shutdown",
            )
        append_run_event(
            args,
            "run_end",
            reason=run_end_reason,
            collector_total_steps=int(float(stats.get("collector_total_steps", 0.0))),
            run_elapsed_total_s=float(stats.get("run_elapsed_total_s", 0.0)),
            online_kept_episodes=int(float(stats.get("online_kept_episodes", 0.0))),
            learner_q_updates=int(learner_state.total_updates),
            learner_actor_updates=int(learner_state.total_actor_updates),
            last_checkpoint_dir=str(stats.get("last_checkpoint_dir", "")) or None,
        )
        print("Final stats:", {k: v for k, v in stats.items() if not isinstance(v, list)})


if __name__ == "__main__":
    modular_extra_args = _parse_modular_specific_args()
    temp_args = tyro.cli(OnlineFinetuneArgs)
    if temp_args.checkpoint is None:
        raise SystemExit("td3_online_real_finetune.py requires --checkpoint <sim training_state.pth>.")
    if temp_args.args_file is None:
        raise SystemExit(
            "td3_online_real_finetune.py requires --args-file "
            "(e.g. configs/td3/td3_online_real_finetune/juggle_hist2.yaml)."
        )
    defaults, applied_keys, ignored_keys = _build_finetune_args_file_defaults(temp_args.args_file)
    defaults["args_file"] = temp_args.args_file
    args = tyro.cli(OnlineFinetuneArgs, default=OnlineFinetuneArgs(**defaults))
    # Recorded in run_events.jsonl / checkpoint metadata as the source policy.
    args.model_path = args.checkpoint
    print(f"[args_file] loaded defaults from: {args.args_file}")
    print("[args_file] applied keys:", ", ".join(applied_keys) if applied_keys else "none")
    if ignored_keys:
        print("[args_file] ignored keys (not used by online fine-tuning):", ", ".join(ignored_keys))

    checkpoint = _load_checkpoint(args.checkpoint)
    train_args, train_args_source = _resolve_train_args(args, checkpoint)
    print(
        f"[train_args] architecture from {train_args_source}: "
        f"actor {train_args.agent_num_hidden_layers}x{train_args.agent_hidden_layer_size} "
        f"critic {train_args.q_num_hidden_layers}x{train_args.q_hidden_layer_size} "
        f"num_critics={train_args.num_critics} "
        f"use_last_action_in_policy_state={train_args.use_last_action_in_policy_state}"
    )
    run_note = _prompt_optional_run_note()
    _setup_run_data_dir(args, run_note)
    main(args, train_args, checkpoint, quiet=bool(modular_extra_args.quiet))
