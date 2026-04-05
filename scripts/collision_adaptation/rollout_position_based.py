"""
Position-based episode rollout for collision adaptation.

Like rollout.py but estimates pre- and post-collision puck speeds from the
observed position trajectory (env.simulator.puck_history) rather than reading
privileged Box2D body velocities via get_episode_collision_stats().

This simulates the real-world case where only noisy, optionally-delayed puck
positions are available (as from a camera tracker), not true velocities.

Key differences from rollout.py:
- Uses env.simulator.puck_history (noisy, optionally-delayed base-frame positions)
  instead of get_episode_collision_stats() which uses privileged Box2D velocities.
- Detects paddle-puck collisions via get_collision_forces() (incremental tracking).
- Estimates velocities with fit_velocity_from_positions from velocity_estimator.py.
- Buckets by estimated pre-collision puck speed (proxy for approach speed).
- Skips estimates with SNR < min_snr (noisy/slow collisions).

Returns the same stats dict format as rollout.py so adapt.py can be reused unchanged.

puck_history layout:
  env.simulator.puck_history is a plain list of [x, y, occluded] entries.
  spawn_puck() prepends 5 padding entries, so episode step k → puck_history[5+k].
  The position at step col_idx is already post-collision (physics runs before the
  position is appended), so the pre-collision window excludes col_idx.
"""

from __future__ import annotations

import numpy as np
import torch

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from airhockey.sims.real.velocity_estimator import fit_velocity_from_positions

_TIERS = ("low", "mid", "high")
_SPEED_BREAKPOINTS = (0.25, 0.75)  # m/s — must match CollisionForceListener defaults
_PUCK_HISTORY_PAD = 5  # spawn_puck() prepends this many padding entries


def _speed_tier(speed: float) -> str:
    lo, hi = _SPEED_BREAKPOINTS
    if speed < lo:
        return "low"
    if speed < hi:
        return "mid"
    return "high"


def _is_paddle_puck_collision(cf: dict) -> bool:
    """Return True if this collision_forces entry is a paddle ↔ puck contact."""
    a = str(cf.get("bodyA", ""))
    b = str(cf.get("bodyB", ""))
    has_paddle = "paddle" in a or "paddle" in b
    has_puck = "puck" in a or "puck" in b
    return has_paddle and has_puck


def _estimate_collision_speeds(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    times: np.ndarray,
    col_idx: int,
    window_frames: int,
    min_snr: float,
    gravity: tuple[float, float],
) -> tuple[float, float] | None:
    """
    Estimate pre- and post-collision puck speeds at a single collision event.

    Parameters
    ----------
    positions   : (N, 2) full-episode position array (after padding removed)
    valid_mask  : (N,) bool — True for non-occluded frames
    times       : (N,) timestamps in seconds
    col_idx     : episode step index of the collision (positions[col_idx] is post-impact)
    window_frames : frames to use for each regression window
    min_snr     : minimum SNR to accept an estimate (both windows must pass)
    gravity     : (gx, gy) deceleration in m/s²

    Returns
    -------
    (speed_before, speed_after) or None if either window fails.
    """
    N = len(positions)

    # Pre-collision window: [col_idx - window_frames, col_idx)
    # positions[col_idx] is already post-impact, so it is excluded.
    pre_start = col_idx - window_frames
    pre_end = col_idx
    if pre_start < 0:
        return None

    # Post-collision window: [col_idx, col_idx + window_frames)
    # positions[col_idx] is the first free-flight frame after impact.
    post_start = col_idx
    post_end = col_idx + window_frames
    if post_end > N:
        return None

    pre_result = fit_velocity_from_positions(
        positions[pre_start:pre_end],
        times[pre_start:pre_end],
        valid_mask[pre_start:pre_end],
        gravity=gravity,
    )
    if pre_result is None or pre_result["snr"] < min_snr:
        return None

    post_result = fit_velocity_from_positions(
        positions[post_start:post_end],
        times[post_start:post_end],
        valid_mask[post_start:post_end],
        gravity=gravity,
    )
    if post_result is None or post_result["snr"] < min_snr:
        return None

    speed_before = float(np.linalg.norm(pre_result["v_at_end"]))
    # v_at_times[0] is the velocity at the first frame of the post window (col_idx)
    speed_after = float(np.linalg.norm(post_result["v_at_times"][0]))

    return speed_before, speed_after


def rollout_episodes_position_based(
    env,
    actor,
    n_episodes: int,
    device: str,
    use_last_action: bool = True,
    window_frames: int = 10,
    min_snr: float = 10.0,
    timestep: float = 0.05,
    gravity: tuple[float, float] = (-0.65, 0.0),
) -> dict:
    """
    Run n_episodes and return aggregate paddle-only collision stats estimated
    from position histories.

    Parameters
    ----------
    env           : AirHockeyEnv instance (configured with correct collision scales)
    actor         : DeterministicAgent with .get_action(obs_tensor) method
    n_episodes    : number of full episodes to roll out
    device        : torch device string
    use_last_action : whether to append last action to obs before actor forward pass
    window_frames : frames used for each velocity regression window (pre and post)
    min_snr       : minimum signal-to-noise ratio to accept a velocity estimate
    timestep      : seconds per env step (default 0.05 s = 20 Hz)
    gravity       : (gx, gy) deceleration in m/s² — use (-0.65, 0.0) for Box2D base coords

    Returns
    -------
    dict with same structure as rollout.rollout_episodes:
        {
          "paddle": {
            "low":  {"count": int, "mean_speed_in": float, "mean_speed_out": float},
            "mid":  {...},
            "high": {...},
          }
        }
    """
    act_dim = int(np.prod(env.action_space.shape))
    action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

    total: dict[str, dict[str, dict]] = {
        "paddle": {
            tier: {"count": 0, "speed_in_sum": 0.0, "speed_out_sum": 0.0}
            for tier in _TIERS
        }
    }

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        last_action = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
        done = False
        step_idx = 0
        prev_force_count = len(env.simulator.get_collision_forces())
        collision_steps: list[int] = []

        while not done:
            with torch.no_grad():
                if use_last_action:
                    policy_obs = torch.cat([obs_tensor, last_action], dim=-1)
                else:
                    policy_obs = obs_tensor
                action_tensor = actor.get_action(policy_obs)
                action_tensor = torch.clamp(action_tensor, action_low, action_high)

            action_np = action_tensor.squeeze(0).detach().cpu().numpy()
            next_obs, _, terminated, truncated, _ = env.step(action_np)
            done = bool(terminated or truncated)
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_action = action_tensor.detach().clone()
            if done:
                last_action.zero_()

            # Detect paddle-puck collisions added during this step.
            forces = env.simulator.get_collision_forces()
            new_paddle_puck = any(
                _is_paddle_puck_collision(cf) for cf in forces[prev_force_count:]
            )
            if new_paddle_puck:
                # Deduplicate: only one entry per step even if multiple sub-step contacts.
                if not collision_steps or collision_steps[-1] != step_idx:
                    collision_steps.append(step_idx)
            prev_force_count = len(forces)
            step_idx += 1

        # Reset privileged stats counters (result discarded — we use position-based estimates).
        env.simulator.get_episode_collision_stats()

        # Build position arrays from puck_history (skip the 5 padding entries).
        raw_hist = env.simulator.puck_history
        ep_hist = raw_hist[_PUCK_HISTORY_PAD:]  # list of [x, y, occluded]
        if len(ep_hist) < 2 * window_frames:
            continue

        positions = np.array([[h[0], h[1]] for h in ep_hist], dtype=float)
        valid_mask = np.array([not bool(h[2]) for h in ep_hist], dtype=bool)
        times = np.arange(len(ep_hist), dtype=float) * timestep

        last_col_idx: int | None = None
        for col_idx in collision_steps:
            # Skip if this collision is too close to the previous one (windows overlap).
            if last_col_idx is not None and col_idx - last_col_idx < window_frames:
                continue

            result = _estimate_collision_speeds(
                positions, valid_mask, times, col_idx, window_frames, min_snr, gravity
            )
            if result is None:
                continue

            speed_in, speed_out = result
            if speed_in < 1e-6:
                continue

            tier = _speed_tier(speed_in)
            total["paddle"][tier]["count"] += 1
            total["paddle"][tier]["speed_in_sum"] += speed_in
            total["paddle"][tier]["speed_out_sum"] += speed_out
            last_col_idx = col_idx

    # Convert sums to means.
    result_stats: dict[str, dict[str, dict]] = {"paddle": {}}
    for tier in _TIERS:
        count = total["paddle"][tier]["count"]
        result_stats["paddle"][tier] = {
            "count": count,
            "mean_speed_in": (
                total["paddle"][tier]["speed_in_sum"] / count if count > 0 else 0.0
            ),
            "mean_speed_out": (
                total["paddle"][tier]["speed_out_sum"] / count if count > 0 else 0.0
            ),
        }

    return result_stats
