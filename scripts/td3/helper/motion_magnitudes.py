"""Per-env motion-magnitude parsing from vectorized env infos.

Extracted from the deleted PPO+AMP trainer (`amp_training.py`). Used by
TD3 training to read paddle velocity / acceleration / jerk magnitudes
emitted by the Box2D env, with mask-aware fallback handling and
terminal-step lookups via `motion_data` in `final_info`.
"""

import numpy as np
import torch


def parse_scalar_info_from_infos(infos, key, num_envs, device, fallback_values):
    """Read vectorized scalar infos with mask-aware fallback handling."""
    if not (isinstance(infos, dict) and key in infos):
        return fallback_values

    raw = infos[key]
    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    if values.shape[0] == num_envs:
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    mask_key = f"_{key}"
    mask = infos.get(mask_key)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if values.shape[0] == int(mask.sum()):
            out = fallback_values.clone()
            out[torch.as_tensor(mask, dtype=torch.bool, device=device)] = torch.as_tensor(
                values, dtype=torch.float32, device=device
            )
            return out
    return fallback_values


def _extract_last_terminal_motion_value(info, motion_key):
    """Return last terminal motion sample from final_info.motion_data when available."""
    if not (isinstance(info, dict) and "motion_data" in info):
        return None
    values = info["motion_data"].get(motion_key)
    if values is None or len(values) == 0:
        return None
    return float(values[-1])


def parse_motion_magnitudes_from_infos(
    infos,
    num_envs,
    device,
    fallback_velocity_mag,
    fallback_acceleration_mag,
    fallback_jerk_mag,
):
    """Extract per-env motion magnitudes from infos, including terminal-step fallbacks."""
    velocity_mag = parse_scalar_info_from_infos(
        infos, "paddle_velocity_mag", num_envs, device, fallback_velocity_mag
    )
    acceleration_mag = parse_scalar_info_from_infos(
        infos, "paddle_acceleration_mag", num_envs, device, fallback_acceleration_mag
    )
    jerk_mag = parse_scalar_info_from_infos(
        infos, "paddle_jerk_mag", num_envs, device, fallback_jerk_mag
    )

    if isinstance(infos, dict) and "final_info" in infos:
        terminal_mask = infos.get("_final_info")
        if terminal_mask is None:
            terminal_mask = np.ones(num_envs, dtype=bool)
        else:
            terminal_mask = np.asarray(terminal_mask, dtype=bool).reshape(-1)

        final_infos = infos["final_info"]
        if isinstance(final_infos, np.ndarray):
            final_infos = final_infos.tolist()

        for env_idx, info in enumerate(final_infos):
            if env_idx >= num_envs or not terminal_mask[env_idx] or not info:
                continue
            terminal_velocity = _extract_last_terminal_motion_value(info, "velocity_mags")
            terminal_acceleration = _extract_last_terminal_motion_value(info, "acceleration_mags")
            terminal_jerk = _extract_last_terminal_motion_value(info, "jerk_mags")
            if terminal_velocity is not None:
                velocity_mag[env_idx] = terminal_velocity
            if terminal_acceleration is not None:
                acceleration_mag[env_idx] = terminal_acceleration
            if terminal_jerk is not None:
                jerk_mag[env_idx] = terminal_jerk

    return velocity_mag, acceleration_mag, jerk_mag
