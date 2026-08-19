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

Two implementations ship:

  * ``"td3"``   — historical default; reuses ``_build_collector_actor`` +
                  the ``training_state.pth`` schema, so the entire
                  ResidualActor / Maxmin-N / REDQ stack keeps working.
  * ``"sgcrl"`` — wraps ``scripts.real.sgcrl_policy.load_sgcrl_deterministic_policy``
                  behind a tensor-IO adapter. ``TrainArgs`` are synthesized
                  with ``use_last_action_in_policy_state=False``.
  * ``"iwr"``   — same actor/obs contract as SGCRL; loads IWR checkpoints
                  (``algorithm_name='interaction_weighted_sampling'``).
  * ``"crtr"``, ``"sac-gcrl"``, ``"sac-her"``, ``"sac-weighted-her"``, ``"ppo-gcrl"`` — same
                  GCRL-style contract; load pickled actors from the external
                  trainer with algorithm-specific ``algorithm_name`` checks.

Adding a new agent = register a builder in ``EVAL_AGENT_BUILDERS``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import torch

from .real_td3_runtime import (
    Args,
    TrainArgs,
    _build_collector_actor,
    _load_training_state_checkpoint,
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
        # action_scale=1.0,
        agent_hidden_layer_size=256,
        agent_num_hidden_layers=2,
        q_hidden_layer_size=256,
        q_num_hidden_layers=2,
        use_last_action_in_policy_state=bool(use_last_action),
    )


# ---------------------------------------------------------------------------
# TD3 builder (matches the pre-refactor _load_actor_for_eval body).
# ---------------------------------------------------------------------------


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
    """Build a TD3 eval actor from a ``training_state.pth`` checkpoint.

    Mirrors the historical ``_load_actor_for_eval`` body bit-for-bit: same
    architecture builder, same ``strict=False`` weight-load, same metadata
    surfacing. Lifting this here keeps the eval entrypoint agent-blind
    without changing TD3 behavior.
    """
    if args.model_path is None:
        raise SystemExit(
            "--agent td3 requires --model-path pointing to a "
            "training_state.pth checkpoint produced by td3_training.py or an "
            "async-real run. Eval mode cannot run against a fresh / random "
            "actor — there is nothing to evaluate."
        )
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"--model-path does not exist: {args.model_path}")

    checkpoint = _load_training_state_checkpoint(args.model_path)
    actor = _build_collector_actor(
        args=args,
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        device=device,
    )
    load_result = actor.load_state_dict(checkpoint["actor"], strict=False)
    n_actor_keys = len(actor.state_dict())
    n_loaded = n_actor_keys - len(load_result.missing_keys)
    if n_loaded == 0:
        raise ValueError(
            f"Loading checkpoint['actor'] into the eval actor produced 0 matching "
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
        f"q_updates={int(checkpoint.get('learner_q_updates', 0))} "
        f"actor_updates={int(checkpoint.get('learner_actor_updates', 0))}"
    )
    return EvalAgent(
        actor=actor,
        train_args=train_args,
        metadata={
            "q_updates":     int(checkpoint.get("learner_q_updates", 0)),
            "actor_updates": int(checkpoint.get("learner_actor_updates", 0)),
            "model_path":    str(args.model_path),
        },
    )


# ---------------------------------------------------------------------------
# SGCRL builder + adapter.
# ---------------------------------------------------------------------------


class _GCRLStyleActorAdapter:
    """Wraps a GCRL-style ``PolicyAgent`` to expose the runner's actor contract.

    The runner calls ``deterministic_actor_action(actor, policy_obs_tensor)``
    which forwards to ``actor.get_action(policy_obs_tensor) -> action_tensor``
    (see ``real_td3_runtime.deterministic_actor_action``). The underlying
    policy is callable on numpy and returns numpy; this adapter bridges the
    tensor IO and adds a no-op ``.eval()`` so the runner's standard
    initialization works unchanged.
    """

    def __init__(self, policy: Any, device: torch.device) -> None:
        self._policy = policy
        self._device = device

    def eval(self) -> None:
        return None

    def get_action(self, policy_obs: torch.Tensor) -> torch.Tensor:
        obs_np = policy_obs.squeeze(0).detach().cpu().numpy()
        action_np = self._policy(obs_np)
        return torch.as_tensor(
            action_np, dtype=torch.float32, device=self._device
        ).unsqueeze(0)


_SGCRLActorAdapter = _GCRLStyleActorAdapter


def _build_gcrl_style_eval_agent(
    *,
    args: Args,
    train_args: TrainArgs,
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    device: torch.device,
    agent_kind: str,
    loader_fn: Callable[..., Any],
) -> EvalAgent:
    """Shared eval builder for SGCRL / IWR pickled goal-conditioned actors."""
    if args.model_path is None:
        raise SystemExit(
            f"--agent {agent_kind} requires --model-path pointing to a .pkl "
            "checkpoint produced by the external GCRL trainer."
        )
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"--model-path does not exist: {args.model_path}")

    if train_args.use_last_action_in_policy_state:
        raise SystemExit(
            f"--agent {agent_kind} requires use_last_action_in_policy_state=False; "
            "the actor expects raw env obs."
        )

    del action_low_np, action_high_np  # tanh-squashed actor; runner re-clamps.

    policy = loader_fn(
        model_path=args.model_path,
        env_obs_dim=int(obs_dim),
        env_act_dim=int(act_dim),
        device=device,
    )
    adapter = _GCRLStyleActorAdapter(policy, device=device)
    return EvalAgent(
        actor=adapter,
        train_args=train_args,
        metadata={
            "q_updates":     0,
            "actor_updates": 0,
            "model_path":    str(args.model_path),
        },
    )


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
    """Load an SGCRL ``.pkl`` checkpoint and wrap it as an ``EvalAgent``."""
    from scripts.real.sgcrl_policy import load_sgcrl_deterministic_policy

    return _build_gcrl_style_eval_agent(
        args=args,
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        device=device,
        agent_kind="sgcrl",
        loader_fn=load_sgcrl_deterministic_policy,
    )


def build_iwr_eval_agent(
    *,
    args: Args,
    train_args: TrainArgs,
    obs_dim: int,
    act_dim: int,
    action_low_np: np.ndarray,
    action_high_np: np.ndarray,
    device: torch.device,
) -> EvalAgent:
    """Load an IWR ``.pkl`` checkpoint and wrap it as an ``EvalAgent``."""
    from scripts.real.iwr_policy import load_iwr_deterministic_policy

    return _build_gcrl_style_eval_agent(
        args=args,
        train_args=train_args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        action_low_np=action_low_np,
        action_high_np=action_high_np,
        device=device,
        agent_kind="iwr",
        loader_fn=load_iwr_deterministic_policy,
    )


def _make_gcrl_variant_eval_agent_builder(
    agent_kind: str,
    loader_fn: Callable[..., Any],
) -> Callable[..., EvalAgent]:
    """Factory for GCRL-style eval builders that share the SGCRL/IWR contract."""

    def builder(
        *,
        args: Args,
        train_args: TrainArgs,
        obs_dim: int,
        act_dim: int,
        action_low_np: np.ndarray,
        action_high_np: np.ndarray,
        device: torch.device,
    ) -> EvalAgent:
        return _build_gcrl_style_eval_agent(
            args=args,
            train_args=train_args,
            obs_dim=obs_dim,
            act_dim=act_dim,
            action_low_np=action_low_np,
            action_high_np=action_high_np,
            device=device,
            agent_kind=agent_kind,
            loader_fn=loader_fn,
        )

    return builder


# ---------------------------------------------------------------------------
# Registry + dispatcher.
# ---------------------------------------------------------------------------

from scripts.real.gcrl_variant_policies import (
    load_crtr_deterministic_policy,
    load_ppo_gcrl_deterministic_policy,
    load_sac_gcrl_deterministic_policy,
    load_sac_her_deterministic_policy,
    load_sac_weighted_her_deterministic_policy,
)


EVAL_AGENT_BUILDERS: Dict[str, Callable[..., EvalAgent]] = {
    "td3":   build_td3_eval_agent,
    "sgcrl": build_sgcrl_eval_agent,
    "iwr":   build_iwr_eval_agent,
    "crtr": _make_gcrl_variant_eval_agent_builder(
        "crtr", load_crtr_deterministic_policy
    ),
    "sac-gcrl": _make_gcrl_variant_eval_agent_builder(
        "sac-gcrl", load_sac_gcrl_deterministic_policy
    ),
    "sac-her": _make_gcrl_variant_eval_agent_builder(
        "sac-her", load_sac_her_deterministic_policy
    ),
    "sac-weighted-her": _make_gcrl_variant_eval_agent_builder(
        "sac-weighted-her", load_sac_weighted_her_deterministic_policy
    ),
    "ppo-gcrl": _make_gcrl_variant_eval_agent_builder(
        "ppo-gcrl", load_ppo_gcrl_deterministic_policy
    ),
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
    from scripts.real.agent_kinds import normalize_agent_kind

    canonical_kind = normalize_agent_kind(kind)
    builder = EVAL_AGENT_BUILDERS.get(canonical_kind)
    if builder is None:
        raise SystemExit(
            f"--agent {kind!r} not registered; known: "
            f"{sorted(EVAL_AGENT_BUILDERS.keys())} "
            f"(underscore and hyphen spellings are equivalent)"
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
