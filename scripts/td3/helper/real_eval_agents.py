"""Agent build dispatcher for the real-world TD3 eval entrypoint.

Decouples the eval loop from any one agent class. The orchestrator
(``extras/async_td3_real_eval.py``) hands this module an agent kind
string (from ``--agent``) and gets back an ``EvalAgent`` bundle:

  * ``actor`` — anything that exposes ``.get_action(policy_obs_tensor) ->
    action_tensor`` and ``.eval()``. ``PolicyRunner`` calls these via
    ``deterministic_actor_action`` from ``real_td3_runtime``.
  * ``train_args`` — the policy-state contract the runner uses. Only
    ``use_last_action_in_policy_state`` is read on the eval path; the
    architecture fields are filler for the dataclass.
  * ``metadata`` — surfaced in ``eval_summary.json`` / ``episode_summaries.jsonl``.
    TD3 fills ``q_updates`` / ``actor_updates`` from the checkpoint;
    SGCRL leaves them at 0 and stashes the source path.

Implementations that ship:

  * ``"td3"``     — historical default; reuses ``_build_collector_actor`` +
                    the ``training_state.pth`` schema, so the entire
                    ResidualActor / Maxmin-N / REDQ stack keeps working.
  * ``"sgcrl"``   — wraps ``scripts.real.sgcrl_policy.load_sgcrl_deterministic_policy``
                    behind a tensor-IO adapter. ``TrainArgs`` are synthesized
                    with ``use_last_action_in_policy_state=False``.
  * ``"rma"`` /
    ``"history"`` — loads an RMA bundle through ``scripts.rma.bundle`` and wraps
                    ``AdaptedRMAAgent`` / ``HistoryAgent`` from
                    ``scripts.rma.evaluate``. These carry per-episode state, so
                    the adapter also implements ``on_episode_start``.

Adding a new agent = register a builder in ``EVAL_AGENT_BUILDERS``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import torch

from scripts.td3.eval_utils import unwrap_eval_state_dict

from .real_td3_runtime import (
    Args,
    TrainArgs,
    _build_collector_actor,
)


# ---------------------------------------------------------------------------
# Result bundle.
# ---------------------------------------------------------------------------


@dataclass
class EvalAgent:
    """What every agent builder hands back to the eval orchestrator."""

    actor: Any  # must expose .get_action(tensor) -> tensor and .eval()
    train_args: TrainArgs
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# TrainArgs synthesis for non-TD3 agents.
# ---------------------------------------------------------------------------


def synthesize_eval_train_args(*, use_last_action: bool = False) -> TrainArgs:
    """Build a minimal ``TrainArgs`` for non-TD3 agents.

    The eval loop only reads ``use_last_action_in_policy_state`` from
    ``train_args`` (``augment_policy_observation`` + the two
    ``transition_hold.begin`` calls). Architecture fields are inert
    because ``_build_collector_actor`` is not invoked; harmless defaults
    keep the dataclass happy.
    """
    return TrainArgs(
        agent_hidden_layer_size=256,
        agent_num_hidden_layers=2,
        q_hidden_layer_size=256,
        q_num_hidden_layers=2,
        use_last_action_in_policy_state=bool(use_last_action),
    )


# ---------------------------------------------------------------------------
# TD3 builder (matches the pre-refactor _load_actor_for_eval body).
# ---------------------------------------------------------------------------


def _load_td3_eval_actor_state(model_path: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Load actor weights from a sim or real checkpoint without requiring replay.

    Frozen eval only needs the actor. Sim ``training_state.pth`` files (and
    the exported policy bundles) often omit ``success_replay_buffer`` /
    ``failure_replay_buffer`` — those keys are required by the async
    *resume* loader, not by eval.
    """
    loaded_obj = torch.load(model_path, map_location="cpu", weights_only=False)
    actor_state = unwrap_eval_state_dict(loaded_obj)
    metadata_src = loaded_obj if isinstance(loaded_obj, dict) else {}
    metadata = {
        "q_updates": int(metadata_src.get("learner_q_updates", 0) or 0),
        "actor_updates": int(metadata_src.get("learner_actor_updates", 0) or 0),
        "model_path": str(model_path),
    }
    return actor_state, metadata


def build_td3_eval_agent(
    *,
    args: Args,
    train_args: TrainArgs,
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    device: torch.device,
) -> EvalAgent:
    """Build a TD3 eval actor from a ``training_state.pth`` or ``model.pth``.

    Same architecture builder and ``strict=False`` weight-load as before, but
    actor extraction does not go through ``_load_training_state_checkpoint``
    (that helper is for async resume and rejects stripped sim checkpoints).
    """
    if args.model_path is None:
        raise SystemExit(
            "--agent td3 requires --model-path pointing to a "
            "training_state.pth or model.pth produced by td3_training.py or an "
            "async-real run. Eval mode cannot run against a fresh / random "
            "actor — there is nothing to evaluate."
        )
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"--model-path does not exist: {args.model_path}")

    actor_state, metadata = _load_td3_eval_actor_state(args.model_path)
    actor = _build_collector_actor(
        args=args,
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        device=device,
    )
    load_result = actor.load_state_dict(actor_state, strict=False)
    n_actor_keys = len(actor.state_dict())
    n_loaded = n_actor_keys - len(load_result.missing_keys)
    if n_loaded == 0:
        raise ValueError(
            f"Loading actor weights into the eval actor produced 0 matching "
            f"keys. Likely a mode mismatch — was the source checkpoint trained "
            f"with full_checkpoint_load={args.full_checkpoint_load!r}? "
            f"first_missing={list(load_result.missing_keys)[:5]} "
            f"first_unexpected={list(load_result.unexpected_keys)[:5]}"
        )
    actor.eval()
    print(
        f"[eval_actor] loaded td3 actor from {args.model_path} "
        f"residual_mode={args.full_checkpoint_load in ('residual', 'residual_resume')} "
        f"loaded_keys={n_loaded}/{n_actor_keys} "
        f"missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)} "
        f"q_updates={metadata['q_updates']} "
        f"actor_updates={metadata['actor_updates']}"
    )
    return EvalAgent(
        actor=actor,
        train_args=train_args,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# SGCRL builder + adapter.
# ---------------------------------------------------------------------------


class _SGCRLActorAdapter:
    """Wraps ``SGCRLDeterministicPolicy`` to expose the runner's actor contract.

    The runner calls ``deterministic_actor_action(actor, policy_obs_tensor)``
    which forwards to ``actor.get_action(policy_obs_tensor) -> action_tensor``
    (see ``real_td3_runtime.deterministic_actor_action``). SGCRL's
    ``PolicyAgent`` is callable on numpy and returns numpy; this adapter
    bridges the tensor IO and adds a no-op ``.eval()`` so the runner's
    standard initialization works unchanged.
    """

    def __init__(self, policy: Any, device: torch.device) -> None:
        self._policy = policy
        self._device = device

    def eval(self) -> None:
        return None

    def get_action(self, policy_obs: torch.Tensor) -> torch.Tensor:
        # policy_obs: (B=1, obs_dim). ``augment_policy_observation`` passes
        # through unchanged when ``use_last_action_in_policy_state`` is
        # False — which the SGCRL builder enforces via synthesized TrainArgs.
        obs_np = policy_obs.squeeze(0).detach().cpu().numpy()
        action_np = self._policy(obs_np)
        return torch.as_tensor(
            action_np, dtype=torch.float32, device=self._device
        ).unsqueeze(0)


def build_sgcrl_eval_agent(
    *,
    args: Args,
    train_args: TrainArgs,
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    device: torch.device,
) -> EvalAgent:
    """Load an SGCRL ``.pkl`` checkpoint and wrap it as an ``EvalAgent``.

    Imports the loader lazily so the SGCRL dependency tree (which pulls in
    ``scripts.real.sgcrl_policy`` and its pickle-tolerant unpickler) is
    only paid for when the SGCRL agent is actually requested.
    """
    # Local import so non-SGCRL paths don't pay the import cost.
    from scripts.real.sgcrl_policy import load_sgcrl_deterministic_policy

    if args.model_path is None:
        raise SystemExit(
            "--agent sgcrl requires --model-path pointing to a .pkl "
            "checkpoint produced by the SGCRL trainer."
        )
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"--model-path does not exist: {args.model_path}")

    # SGCRL doesn't augment obs with the last action; the eval loop must
    # not either. Refuse a synthesized TrainArgs that says otherwise so a
    # mis-set flag surfaces here instead of as a silent obs-shape mismatch.
    if train_args.use_last_action_in_policy_state:
        raise SystemExit(
            "--agent sgcrl requires use_last_action_in_policy_state=False; "
            "the SGCRL actor expects raw env obs."
        )

    del action_low_np, action_high_np  # SGCRL self-clips via tanh; runner re-clamps.

    policy = load_sgcrl_deterministic_policy(
        model_path=args.model_path,
        env_obs_dim=int(obs_dim),
        env_act_dim=int(act_dim),
        device=device,
    )
    adapter = _SGCRLActorAdapter(policy, device=device)
    return EvalAgent(
        actor=adapter,
        train_args=train_args,
        metadata={
            "q_updates":     0,
            "actor_updates": 0,
            "model_path":    str(args.model_path),
        },
    )


# ---------------------------------------------------------------------------
# RMA builder + adapter (phase-2 adapted policy and long-history control).
# ---------------------------------------------------------------------------


RMA_META_FILENAME = "rma_meta.json"
ADAPTATION_FILENAME = "adaptation_module.pth"


def resolve_rma_bundle_dir(model_path: str) -> str:
    """Accept a bundle dir or any file inside it; return the bundle dir."""
    path = os.path.abspath(model_path)
    bundle_dir = path if os.path.isdir(path) else os.path.dirname(path)
    if not os.path.isfile(os.path.join(bundle_dir, RMA_META_FILENAME)):
        raise FileNotFoundError(
            f"{os.path.join(bundle_dir, RMA_META_FILENAME)} not found — an RMA / "
            f"long-history bundle must carry {RMA_META_FILENAME} next to model.pth."
        )
    return bundle_dir


def load_rma_meta(model_path: str) -> Dict[str, Any]:
    """Read ``rma_meta.json`` for a bundle without building any network.

    The top-level runner needs ``use_last_action`` before the agent exists, to
    synthesize the policy-state contract the eval loop applies.
    """
    import json

    with open(os.path.join(resolve_rma_bundle_dir(model_path), RMA_META_FILENAME)) as f:
        return json.load(f)


def _resolve_adaptation_module_path(bundle_dir: str) -> str:
    """Phase-2 weights sit either in the bundle dir or in its ``phase2/`` subdir."""
    candidates = (
        os.path.join(bundle_dir, ADAPTATION_FILENAME),
        os.path.join(bundle_dir, "phase2", ADAPTATION_FILENAME),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        f"no {ADAPTATION_FILENAME} under {bundle_dir} (looked in the bundle dir and "
        f"phase2/). A privileged phase-1 actor is a sim-only oracle — deployment "
        f"needs the phase-2 adaptation module."
    )


class _RMAActorAdapter:
    """Bridges an ``scripts.rma.evaluate.EvalAgent`` to the runner's contract.

    Two impedance mismatches to absorb:

    * **IO shape.** The runner hands over one ``policy_obs`` tensor with the
      last action already concatenated; the RMA agents take ``(obs,
      last_action)`` as separate numpy arrays. The split is exact because
      ``obs_dim`` comes from ``rma_meta.json``.
    * **Statefulness.** These policies carry a history window across steps and
      must be cleared per episode, which ``on_episode_start`` does via
      ``PolicyRunner._notify_actor_episode_start``.

    Known limitation: ``act()`` pushes the action it *returned* into its own
    history, but the runner may substitute a hold action afterwards (post-reset
    holds, e-stop recovery). During those steps the window records an intent the
    robot did not execute.
    """

    def __init__(
        self,
        agent: Any,
        *,
        obs_dim: int,
        act_dim: int,
        use_last_action: bool,
        device: torch.device,
    ) -> None:
        self._agent = agent
        self._obs_dim = int(obs_dim)
        self._act_dim = int(act_dim)
        self._use_last_action = bool(use_last_action)
        self._device = device
        self._episode_started = False

    def eval(self) -> None:
        return None

    def on_episode_start(self, obs: np.ndarray) -> None:
        # No privileged env params on the robot: pass None so the agent skips
        # the target-latent diagnostic instead of demanding the true physics.
        self._agent.reset(np.asarray(obs, dtype=np.float32).reshape(-1), None)
        self._episode_started = True

    def get_action(self, policy_obs: torch.Tensor) -> torch.Tensor:
        if not self._episode_started:
            raise RuntimeError(
                "RMA actor queried before on_episode_start — its history window "
                "is unseeded. The caller must invoke PolicyRunner.seed_initial / "
                "seed_after_reset before running an episode."
            )
        flat = policy_obs.detach().reshape(-1).cpu().numpy().astype(np.float32)
        obs = flat[: self._obs_dim]
        if self._use_last_action:
            last_action = flat[self._obs_dim : self._obs_dim + self._act_dim]
        else:
            last_action = np.zeros(self._act_dim, dtype=np.float32)
        action = np.asarray(self._agent.act(obs, last_action), dtype=np.float32).reshape(-1)
        return torch.as_tensor(action, dtype=torch.float32, device=self._device).unsqueeze(0)


def build_rma_eval_agent(
    *,
    args: Args,
    train_args: TrainArgs,
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    device: torch.device,
) -> EvalAgent:
    """Build a deployable RMA policy from an exported bundle directory.

    ``rma_meta.json``'s ``mode`` selects the agent: ``history`` runs the
    long-history control (``HistoryAgent``, ``drlong_*`` bundles), ``privileged``
    pairs the phase-1 actor with its phase-2 adaptation module
    (``AdaptedRMAAgent``, ``rma_*`` bundles). The bare privileged and
    nominal-latent agents are deliberately unreachable — both need the true env
    parameters and are sim-only oracles.
    """
    from scripts.rma.bundle import load_adaptation_module, load_phase1
    from scripts.rma.evaluate import AdaptedRMAAgent, HistoryAgent

    if args.model_path is None:
        raise SystemExit(
            "--agent rma requires --model-path pointing at the bundle directory "
            "holding model.pth + rma_meta.json (+ adaptation_module.pth for "
            "phase-2 policies)."
        )
    bundle_dir = resolve_rma_bundle_dir(args.model_path)
    bundle = load_phase1(bundle_dir)
    meta = bundle["meta"]

    # Catch a hist_len / task mismatch at load time rather than as garbage
    # actions: goal tasks legitimately differ (30 vs 32 vs 33) because the goal
    # is appended to the observation.
    if int(meta["obs_dim"]) != int(obs_dim):
        raise ValueError(
            f"bundle obs_dim={int(meta['obs_dim'])} but the env produces "
            f"obs_dim={int(obs_dim)}. The env config's hist_len / task does not "
            f"match what this policy was trained on."
        )
    if int(meta["act_dim"]) != int(act_dim):
        raise ValueError(
            f"bundle act_dim={int(meta['act_dim'])} but the env expects {int(act_dim)}."
        )
    if bool(meta["use_last_action"]) != bool(train_args.use_last_action_in_policy_state):
        raise ValueError(
            f"bundle use_last_action={bool(meta['use_last_action'])} but the eval "
            f"policy-state contract says "
            f"{bool(train_args.use_last_action_in_policy_state)}. The observation "
            f"split would be wrong; synthesize TrainArgs from rma_meta.json."
        )

    mode = str(bundle.get("mode", "privileged"))
    adaptation_path = None
    if mode == "history":
        agent = HistoryAgent(bundle["actor"])
    else:
        adaptation_path = _resolve_adaptation_module_path(bundle_dir)
        phi, phi_meta = load_adaptation_module(adaptation_path)
        agent = AdaptedRMAAgent(bundle["actor"], phi, phi_meta["step_features"])

    print(
        f"[eval_actor] loaded rma agent from {bundle_dir} "
        f"mode={mode} agent={type(agent).__name__} "
        f"obs_dim={int(meta['obs_dim'])} latent_dim={int(meta['latent_dim'])} "
        f"use_last_action={bool(meta['use_last_action'])} "
        f"adaptation_module={adaptation_path or '-'}"
    )
    return EvalAgent(
        actor=_RMAActorAdapter(
            agent,
            obs_dim=obs_dim,
            act_dim=act_dim,
            use_last_action=bool(meta["use_last_action"]),
            device=device,
        ),
        train_args=train_args,
        metadata={
            "q_updates":         0,
            "actor_updates":     0,
            "model_path":        str(bundle["model_path"]),
            "rma_mode":          mode,
            "rma_agent":         type(agent).__name__,
            "adaptation_module": str(adaptation_path) if adaptation_path else None,
        },
    )


# ---------------------------------------------------------------------------
# Registry + dispatcher.
# ---------------------------------------------------------------------------


EVAL_AGENT_BUILDERS: Dict[str, Callable[..., EvalAgent]] = {
    "td3":     build_td3_eval_agent,
    "sgcrl":   build_sgcrl_eval_agent,
    # Both RMA kinds share one builder; `rma_meta.json` picks the agent class.
    "rma":     build_rma_eval_agent,
    "history": build_rma_eval_agent,
}


def build_eval_agent(
    kind: str,
    *,
    args: Args,
    train_args: TrainArgs,
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    device: torch.device,
) -> EvalAgent:
    """Dispatch on ``kind`` to the registered builder. ``SystemExit`` on unknown."""
    builder = EVAL_AGENT_BUILDERS.get(str(kind))
    if builder is None:
        raise SystemExit(
            f"--agent {kind!r} not registered; known: "
            f"{sorted(EVAL_AGENT_BUILDERS.keys())}"
        )
    return builder(
        args=args,
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        device=device,
    )
