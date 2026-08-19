"""SGCRL deterministic policy wrapper for ``scripts/real/run_policy.py``.

The checkpoint at ``gcrl/03500032_sgcrl_AirHockeyPuckGoalPosition-v0.pkl``
was produced by an external training repo (``unsupervised_manipulation``)
which we do NOT import here. We only need the actor ``state_dict`` plus a
handful of scalar fields (``state_dim``, ``goal_dim``, ``hidden_dims``) to
rebuild the network locally, so this module:

  1. Unpickles the file with a tolerant ``Unpickler`` that swaps missing
     classes for placeholders (so the ``config`` subtree, which references
     types from the external repo, can be skipped).
  2. Forces tensor storages to load on CPU regardless of what device they
     were saved on, then ``.to(device)`` afterward.
  3. Rebuilds the actor as ``_SgcrlActor`` whose parameter names match the
     loaded keys (``backbone.network.0/2``, ``loc_layer``, ``scale_layer``).

Architectural assumptions baked in (cannot be derived from the state_dict):

  * Hidden activation is ``ReLU`` — SGCRL's standard for ``architecture='mlp'``.
  * Action squash is ``tanh(loc)`` — standard tanh-squashed Gaussian
    convention (consistent with the ``use_action_entropy=True`` flag stored
    in the checkpoint).
"""
from __future__ import annotations

import contextlib
import pickle
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Network rebuild.
# ---------------------------------------------------------------------------


class _SgcrlBackbone(nn.Module):
    """Hidden-layer MLP that exposes its ``Sequential`` as ``self.network``.

    The submodule name ``network`` is significant: it makes the loaded keys
    line up as ``backbone.network.0.weight``, ``backbone.network.2.weight``,
    ... (with ``network.1`` / ``network.3`` being ReLU, which carry no
    parameters and so do not appear in the ``state_dict``).
    """

    def __init__(self, in_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.ReLU())
            prev = int(h)
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class _SgcrlActor(nn.Module):
    """SGCRL tanh-squashed Gaussian actor (deterministic ``forward`` = loc).

    ``scale_layer`` is rebuilt so the loaded ``state_dict`` matches strictly,
    but its output is unused by ``forward`` — we only need the mean for
    deterministic rollouts.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dims: tuple[int, ...],
        action_dim: int,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        self.backbone = _SgcrlBackbone(in_dim, hidden_dims)
        last = int(hidden_dims[-1])
        self.loc_layer = nn.Linear(last, int(action_dim))
        self.scale_layer = nn.Linear(last, int(action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.loc_layer(self.backbone(x))


# ---------------------------------------------------------------------------
# Tolerant unpickler.
# ---------------------------------------------------------------------------


class _MissingClassPlaceholder:
    """Stand-in for classes whose source module is unavailable.

    The SGCRL pkl pickles its full ``TrainConfig`` (a dataclass tree under
    ``unsupervised_manipulation.config``). We don't depend on that package,
    so any reference resolves to one of these placeholders. The ``agent``
    subtree we actually care about is plain dicts + tensors, so this only
    affects the discarded ``config`` subtree.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _TolerantUnpickler(pickle.Unpickler):
    """Replaces unimportable classes with ``_MissingClassPlaceholder``."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError, ModuleNotFoundError):
            placeholder = type(
                name,
                (_MissingClassPlaceholder,),
                {"__module__": module, "__qualname__": name},
            )
            return placeholder


@contextlib.contextmanager
def _force_cpu_tensor_storage() -> Iterator[None]:
    """Make every tensor storage deserialize on CPU regardless of its tag.

    The pkl was saved on a CUDA host; without this patch loading on a
    CPU-only machine fails with "Attempting to deserialize object on a CUDA
    device but torch.cuda.is_available() is False". Tensors can be moved to
    the desired device after load.
    """
    import torch.serialization as _ts

    original = _ts.default_restore_location
    _ts.default_restore_location = lambda storage, location: storage
    try:
        yield
    finally:
        _ts.default_restore_location = original


def _infer_hidden_dims_from_actor_sd(actor_sd: dict[str, Any]) -> tuple[int, ...]:
    """Derive MLP hidden widths from ``backbone.network.*.weight`` keys.

    SAC-family checkpoints omit ``hidden_dims`` in agent metadata but use the
    same backbone naming convention as SGCRL/IWR.
    """
    layer_weights: list[tuple[int, int]] = []
    prefix = "backbone.network."
    for key, tensor in actor_sd.items():
        if not key.startswith(prefix) or not key.endswith(".weight"):
            continue
        idx_str = key[len(prefix) : -len(".weight")]
        if not idx_str.isdigit():
            continue
        layer_weights.append((int(idx_str), int(tensor.shape[0])))
    if not layer_weights:
        raise ValueError(
            "Cannot infer hidden_dims: no backbone.network.*.weight keys in actor state_dict."
        )
    layer_weights.sort(key=lambda item: item[0])
    return tuple(width for _, width in layer_weights)


def _load_sgcrl_pkl(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle, _force_cpu_tensor_storage():
        payload = _TolerantUnpickler(handle).load()
    if not isinstance(payload, dict):
        raise ValueError(
            f"SGCRL checkpoint at {path} must unpickle to a dict, got {type(payload).__name__}."
        )
    return payload


# ---------------------------------------------------------------------------
# Policy wrapper.
# ---------------------------------------------------------------------------


class SGCRLDeterministicPolicy:
    """Wraps a frozen ``_SgcrlActor`` as a ``PolicyAgent`` (obs -> action).

    Stateless: no history augmentation, no last-action memory, so no
    ``reset()`` hook. ``run_policy._maybe_reset_agent`` already handles its
    absence.

    Input obs is expected to be the env's already-concatenated 32-D vector
    ``concat(state_30, desired_goal_2)`` — i.e. a ``puck_goal_position``
    task with ``return_goal_obs=False``.
    """

    def __init__(self, *, actor: nn.Module, device: torch.device) -> None:
        self._actor = actor
        self._device = device

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)
        loc = self._actor(obs_t)
        action = torch.tanh(loc).squeeze(0).cpu().numpy().astype(np.float32)
        return action


# ---------------------------------------------------------------------------
# Top-level loader.
# ---------------------------------------------------------------------------


def load_gcrl_style_deterministic_policy(
    *,
    model_path: str | Path,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
    agent_label: str = "gcrl",
    expected_algorithm_name: str | None = None,
) -> SGCRLDeterministicPolicy:
    """Build a deterministic policy from a GCRL-style ``.pkl`` checkpoint.

    SGCRL and IWR (interaction-weighted sampling) share the same actor
    architecture and ``agent`` dict layout; they differ only in the saved
    ``algorithm_name`` and filename convention from the external trainer.

    Reads ``state_dim`` / ``goal_dim`` / ``hidden_dims`` from the embedded
    ``agent`` metadata so the network topology is derived from the checkpoint
    itself (no external config required).
    """
    model_path = Path(model_path)
    payload = _load_sgcrl_pkl(model_path)

    agent_state = payload.get("agent")
    if not isinstance(agent_state, dict):
        raise ValueError(
            f"{agent_label} checkpoint at {model_path} is missing 'agent' dict "
            f"(got top-level keys: {list(payload.keys())})."
        )

    actual_algorithm_name = agent_state.get("algorithm_name")
    if expected_algorithm_name is not None:
        if actual_algorithm_name != expected_algorithm_name:
            raise SystemExit(
                f"--agent {agent_label} expected checkpoint "
                f"algorithm_name={expected_algorithm_name!r}, "
                f"got {actual_algorithm_name!r} in {model_path}."
            )

    actor_sd = agent_state.get("actor")
    if not isinstance(actor_sd, dict) or "loc_layer.weight" not in actor_sd:
        raise ValueError(
            f"{agent_label} checkpoint at {model_path} does not contain a "
            f"recognisable 'agent.actor' state_dict."
        )

    try:
        state_dim = int(agent_state["state_dim"])
        goal_dim = int(agent_state["goal_dim"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{agent_label} checkpoint at {model_path} missing required metadata "
            f"(state_dim / goal_dim): {exc}"
        )

    raw_hidden_dims = agent_state.get("hidden_dims")
    if raw_hidden_dims is None:
        hidden_dims = _infer_hidden_dims_from_actor_sd(actor_sd)
    else:
        try:
            hidden_dims = tuple(int(h) for h in raw_hidden_dims)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{agent_label} checkpoint at {model_path} has invalid hidden_dims: {exc}"
            )

    actor_in_dim = state_dim + goal_dim
    actor_out_dim = int(actor_sd["loc_layer.weight"].shape[0])

    if int(env_obs_dim) != actor_in_dim:
        raise SystemExit(
            f"--agent {agent_label} needs env_obs_dim == state_dim + goal_dim "
            f"({state_dim} + {goal_dim} = {actor_in_dim}), got env_obs_dim={env_obs_dim}. "
            f"Use a puck_goal_position task with return_goal_obs=False so the env "
            f"appends the 2-D desired_goal to the 30-D state."
        )
    if int(env_act_dim) != actor_out_dim:
        raise SystemExit(
            f"--agent {agent_label} needs env_act_dim == {actor_out_dim} "
            f"(from loc_layer.weight), got env_act_dim={env_act_dim}."
        )

    actor = _SgcrlActor(
        in_dim=actor_in_dim,
        hidden_dims=hidden_dims,
        action_dim=actor_out_dim,
    )
    actor.load_state_dict(actor_sd, strict=True)
    actor.to(device).eval()

    algo_suffix = (
        f", algorithm_name={actual_algorithm_name!r}"
        if actual_algorithm_name is not None
        else ""
    )
    print(
        f"[run_policy] loaded {agent_label} actor from {model_path} "
        f"(state_dim={state_dim}, goal_dim={goal_dim}, "
        f"hidden_dims={hidden_dims}, action_dim={actor_out_dim}{algo_suffix})"
    )
    return SGCRLDeterministicPolicy(actor=actor, device=device)


def load_sgcrl_deterministic_policy(
    *,
    model_path: str | Path,
    env_obs_dim: int,
    env_act_dim: int,
    device: torch.device,
) -> SGCRLDeterministicPolicy:
    """Build a ``SGCRLDeterministicPolicy`` from a saved SGCRL ``.pkl`` checkpoint."""
    return load_gcrl_style_deterministic_policy(
        model_path=model_path,
        env_obs_dim=env_obs_dim,
        env_act_dim=env_act_dim,
        device=device,
        agent_label="sgcrl",
        expected_algorithm_name=None,
    )
