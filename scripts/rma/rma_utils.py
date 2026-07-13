"""Helpers for RMA privileged environment properties.

Self-contained under scripts/rma (does not import from scripts.td3).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch


def privileged_keys_from_config(air_hockey_config: Dict[str, Any]) -> List[str]:
    """Return privileged properties in the config's declared ordering."""
    keys = air_hockey_config.get("random_variables") if isinstance(air_hockey_config, dict) else None
    if not keys:
        raise ValueError("RMA requires a non-empty air_hockey.random_variables list.")
    return list(keys)


def extract_env_props_from_info(info: Dict[str, Any], keys: Sequence[str]) -> np.ndarray:
    """Extract float32 vector of len(keys) from a gym info dict (single env).

    Missing or non-scalar keys are an error; active RMA must never train on
    fabricated privileged values.
    """
    values = []
    for key in keys:
        if not isinstance(info, dict) or key not in info:
            raise KeyError(f"Missing RMA privileged property {key!r} in environment info.")
        val = info[key]
        try:
            values.append(float(val))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid RMA privileged property {key!r}: {val!r}") from exc
    return np.asarray(values, dtype=np.float32)


def extract_env_props_from_vec_info(
    infos: Any,
    keys: Sequence[str],
    n_envs: int,
) -> np.ndarray:
    """Shape (n_envs, len(keys)).

    Vectorized gym infos may be dict-of-arrays OR list-of-dicts depending on
    gym / gymnasium version — handle both.
    """
    n_keys = len(keys)
    out = np.empty((int(n_envs), n_keys), dtype=np.float32)
    found = np.zeros((int(n_envs), n_keys), dtype=bool)

    if isinstance(infos, (list, tuple)):
        for i in range(int(n_envs)):
            if i < len(infos) and isinstance(infos[i], dict):
                out[i] = extract_env_props_from_info(infos[i], keys)
                found[i, :] = True

    if isinstance(infos, dict):
        for j, key in enumerate(keys):
            if key not in infos:
                continue
            vals = infos[key]
            if isinstance(vals, np.ndarray):
                flat = vals.reshape(-1)
                n = min(int(n_envs), flat.shape[0])
                out[:n, j] = flat[:n].astype(np.float32)
                found[:n, j] = True
            elif isinstance(vals, (list, tuple)):
                n = min(int(n_envs), len(vals))
                for i in range(n):
                    try:
                        out[i, j] = float(vals[i])
                        found[i, j] = True
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"Invalid vector RMA property {key!r} at env {i}: {vals[i]!r}"
                        ) from exc
            else:
                # Scalar broadcast to all envs.
                try:
                    out[:, j] = float(vals)
                    found[:, j] = True
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid vector RMA property {key!r}: {vals!r}") from exc

    if not found.all():
        missing = [
            f"env {i}: {keys[j]}"
            for i, j in zip(*np.where(~found))
        ]
        raise KeyError("Missing RMA privileged properties: " + ", ".join(missing))
    return out


def read_env_props_from_vector_env(envs, keys: Sequence[str]) -> np.ndarray:
    """Explicit fallback for vector auto-reset info formats.

    Gymnasium environments expose ``get_wrapper_attr`` through vector ``call``;
    simulator_params is a dataclass/SimpleNamespace on AirHockeyEnv.
    """
    try:
        params = envs.call("get_wrapper_attr", "simulator_params")
    except Exception as exc:
        raise RuntimeError(
            "Could not recover RMA privileged properties from vector environment."
        ) from exc
    rows = []
    for item in params:
        mapping = vars(item) if not isinstance(item, dict) else item
        rows.append(extract_env_props_from_info(mapping, keys))
    return np.stack(rows, axis=0)


def build_prop_normalizer(
    random_variable_ranges: Dict[str, Sequence[float]],
    keys: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lows, highs) float32 arrays shape (len(keys),) from ranges dict
    where each value is [low, high].
    """
    lows = np.empty(len(keys), dtype=np.float32)
    highs = np.empty(len(keys), dtype=np.float32)
    for i, key in enumerate(keys):
        if key not in random_variable_ranges:
            raise ValueError(f"RMA privileged property {key!r} has no training range.")
        lo_hi = random_variable_ranges[key]
        if len(lo_hi) != 2 or float(lo_hi[1]) <= float(lo_hi[0]):
            raise ValueError(f"Invalid RMA training range for {key!r}: {lo_hi!r}")
        lows[i] = float(lo_hi[0])
        highs[i] = float(lo_hi[1])
    return lows, highs


def normalize_env_props(
    props: torch.Tensor,
    lows,
    highs,
) -> torch.Tensor:
    """Map raw props to [-1, 1] using lows/highs (broadcast). Clamp to [-1, 1]."""
    lows_t = torch.as_tensor(lows, dtype=props.dtype, device=props.device)
    highs_t = torch.as_tensor(highs, dtype=props.dtype, device=props.device)
    denom = (highs_t - lows_t).clamp_min(1e-8)
    normalized = 2.0 * (props - lows_t) / denom - 1.0
    return normalized.clamp(-1.0, 1.0)
