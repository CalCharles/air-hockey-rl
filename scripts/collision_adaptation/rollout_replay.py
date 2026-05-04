"""
Oracle-scenario replay rollout for collision adaptation.

Solves the distribution-shift problem in rollout_position_based.py: instead of
running independent oracle and learner rollouts (which have different collision
speed distributions when oracle scales are non-uniform), this module:

  1. Collects collision scenarios from oracle episodes using position-based velocity
     estimation (simulating real-world camera data).
  2. Replays each exact scenario in the learner sim and reads the learner's
     post-collision puck speed via privileged access.
  3. Both oracle and learner therefore evaluate the same physical inputs →
     no distribution shift.

Oracle collection:
  - Puck velocity: estimated from puck_history via fit_velocity_from_positions
  - Paddle velocity: read privileged from oracle sim at collision detection time
  - Oracle post-collision puck speed: estimated from puck_history (position-based)
  - Direction filter: scenarios where estimated puck_vel points > max_angle_deg
    away from the paddle are discarded (unreliable direction estimate)

Learner replay:
  - State injected via direct Box2D assignment (same pattern as render_scenarios.py)
  - Learner post-collision puck speed: read privileged from Box2D body velocity

Returns stats dicts in the same format as rollout.py so adapt.py is reused unchanged.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from airhockey.sims.real.velocity_estimator import fit_velocity_from_positions
from scripts.collision_adaptation.collision_detection import (
    TIERS, PUCK_HISTORY_PAD, speed_tier,
    StepCollisionDetector,
)


def _b2d_to_base(vb) -> tuple[float, float]:
    """Convert Box2D (x, y) velocity to base-frame (vx, vy)."""
    return -float(vb[1]), float(vb[0])


def _estimated_approach_speed(
    puck_vel: np.ndarray,
    paddle_vel: tuple[float, float],
    puck_pos: np.ndarray,
    paddle_pos: np.ndarray,
) -> float:
    """
    Estimate approach speed from position-based puck velocity and privileged paddle velocity.

    Mirrors _presolve_paddle_puck's approach_speed = max(0, -dot(v_puck - v_paddle, n))
    where n = normalize(paddle_pos - puck_pos).

    NOTE: Box2D's _presolve_paddle_puck uses the contact normal from shape geometry,
    while this function uses the center-to-center direction.  For circular bodies
    (puck + paddle) these are identical.  If body shapes ever change to non-circular,
    the two definitions will diverge and this function must be updated.

    Positive return value means the pair is closing — a collision is expected.
    """
    diff = paddle_pos - puck_pos
    dist = float(np.linalg.norm(diff))
    if dist < 1e-9:
        return 0.0
    n = diff / dist
    v_rel = puck_vel - np.array(paddle_vel, dtype=float)
    return float(-np.dot(v_rel, n))


# ---------------------------------------------------------------------------
# Phase A: collect oracle collision scenarios
# ---------------------------------------------------------------------------

def collect_oracle_scenarios(
    oracle_env,
    actor,
    n_episodes: int,
    device: str,
    use_last_action: bool = True,
    window_frames: int = 10,
    min_snr: float = 8.0,
    timestep: float = 0.05,
    gravity: tuple[float, float] = (-0.65, 0.0),
    min_approach_speed: float = 0.1,
) -> list[dict]:
    """
    Run n_episodes in oracle_env and extract collision scenarios using
    position-based velocity estimation.

    Parameters
    ----------
    oracle_env         : AirHockeyEnv with oracle collision scales set
    actor              : DeterministicAgent
    n_episodes         : number of episodes to roll out
    device             : torch device string
    use_last_action    : whether to append last action to obs
    window_frames      : frames used for each velocity regression window
    min_snr            : minimum SNR to accept a velocity estimate
    timestep           : seconds per env step (default 0.05 s = 20 Hz)
    gravity            : (gx, gy) deceleration in m/s² for puck trajectory
    min_approach_speed : minimum estimated approach speed (m/s) to keep a scenario.
                         Uses relative velocity: approach = -dot(v_puck - v_paddle, n),
                         correctly handling paddle-catches-puck scenarios where the puck
                         may not be heading toward the paddle center at all.

    Returns
    -------
    List of scenario dicts:
        puck_pos         (2,) base coords — puck position at col_idx-1
        puck_vel         (2,) base coords — estimated pre-collision puck velocity
        paddle_pos       (2,) base coords — paddle position at col_idx-1
        paddle_vel       (2,) base coords — privileged paddle velocity at collision
        puck_speed_pre   float m/s        — |puck_vel|, used for tier bucketing
        oracle_speed_out float m/s        — estimated post-collision puck speed
        tier             str              — "low" / "mid" / "high"
        approach_speed_est float m/s      — estimated approach speed for diagnostics
    """
    act_dim = int(np.prod(oracle_env.action_space.shape))
    action_low = torch.tensor(oracle_env.action_space.low, dtype=torch.float32, device=device)
    action_high = torch.tensor(oracle_env.action_space.high, dtype=torch.float32, device=device)

    scenarios: list[dict] = []

    for _ in range(n_episodes):
        obs, _ = oracle_env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        last_action = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
        done = False
        step_idx = 0
        detector = StepCollisionDetector(oracle_env.simulator)
        detector.reset()
        collision_steps: list[int] = []
        collision_paddle_vels: dict[int, tuple[float, float]] = {}

        while not done:
            with torch.no_grad():
                policy_obs = torch.cat([obs_tensor, last_action], dim=-1) if use_last_action else obs_tensor
                action_tensor = actor.get_action(policy_obs)
                action_tensor = torch.clamp(action_tensor, action_low, action_high)

            action_np = action_tensor.squeeze(0).detach().cpu().numpy()
            next_obs, _, terminated, truncated, _ = oracle_env.step(action_np)
            done = bool(terminated or truncated)
            obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_action = action_tensor.detach().clone()
            if done:
                last_action.zero_()

            if detector.step():
                if not collision_steps or collision_steps[-1] != step_idx:
                    collision_steps.append(step_idx)
                    # Read privileged paddle velocity at this exact step
                    vb = oracle_env.simulator.paddles["paddle_ego"].linearVelocity
                    collision_paddle_vels[step_idx] = _b2d_to_base(vb)
            step_idx += 1

        # Reset privileged stats counters (result discarded)
        oracle_env.simulator.get_episode_collision_stats()

        # Build position arrays (skip 5 spawn-padding entries)
        raw_puck = oracle_env.simulator.puck_history
        raw_paddle = oracle_env.simulator.paddle_history
        puck_ep = raw_puck[PUCK_HISTORY_PAD:]
        paddle_ep = raw_paddle[PUCK_HISTORY_PAD:]

        if len(puck_ep) < 2 * window_frames:
            continue

        positions = np.array([[h[0], h[1]] for h in puck_ep], dtype=float)
        valid_mask = np.array([not bool(h[2]) for h in puck_ep], dtype=bool)
        times = np.arange(len(puck_ep), dtype=float) * timestep

        paddle_positions = np.array([[h[0], h[1]] for h in paddle_ep], dtype=float)

        last_col_idx: int | None = None

        for col_idx in collision_steps:
            # Skip if too close to a previous collision (windows would overlap)
            if last_col_idx is not None and col_idx - last_col_idx < window_frames:
                continue
            # Bounds check
            if col_idx < window_frames or col_idx + window_frames > len(positions):
                continue
            if col_idx < 1 or col_idx - 1 >= len(paddle_positions):
                continue

            # --- Pre-collision puck velocity ---
            pre_result = fit_velocity_from_positions(
                positions[col_idx - window_frames: col_idx],
                times[col_idx - window_frames: col_idx],
                valid_mask[col_idx - window_frames: col_idx],
                gravity=gravity,
            )
            if pre_result is None or pre_result["snr"] < min_snr:
                continue
            puck_vel_pre = pre_result["v_at_end"]          # (vx, vy) base
            puck_speed_pre = float(np.linalg.norm(puck_vel_pre))
            if puck_speed_pre < 1e-6:
                continue

            # --- Post-collision oracle puck speed ---
            post_result = fit_velocity_from_positions(
                positions[col_idx: col_idx + window_frames],
                times[col_idx: col_idx + window_frames],
                valid_mask[col_idx: col_idx + window_frames],
                gravity=gravity,
            )
            if post_result is None or post_result["snr"] < min_snr:
                continue
            oracle_speed_out = float(np.linalg.norm(post_result["v_at_times"][0]))

            # --- Puck and paddle positions at col_idx-1 ---
            puck_pos_pre = positions[col_idx - 1]
            paddle_pos_pre = paddle_positions[col_idx - 1]

            # --- Privileged paddle velocity (recorded during episode loop) ---
            paddle_vel_pre = collision_paddle_vels.get(col_idx, (0.0, 0.0))

            # --- Approach speed filter ---
            # Uses relative velocity (v_puck - v_paddle) projected onto puck→paddle.
            # This correctly handles paddle-catches-puck scenarios where the puck alone
            # may not be heading toward the paddle center.
            approach_speed_est = _estimated_approach_speed(
                puck_vel_pre, paddle_vel_pre, puck_pos_pre, paddle_pos_pre
            )
            if approach_speed_est < min_approach_speed:
                continue

            scenarios.append({
                "puck_pos": puck_pos_pre.tolist(),
                "puck_vel": puck_vel_pre.tolist(),
                "paddle_pos": paddle_pos_pre.tolist(),
                "paddle_vel": list(paddle_vel_pre),
                "puck_speed_pre": puck_speed_pre,
                "oracle_speed_out": oracle_speed_out,
                "tier": speed_tier(puck_speed_pre),
                "approach_speed_est": approach_speed_est,
            })
            last_col_idx = col_idx

    return scenarios


# ---------------------------------------------------------------------------
# Phase B: replay scenarios in learner, return paired stats
# ---------------------------------------------------------------------------

def replay_scenarios(
    learner_env,
    scenarios: list[dict],
    max_replay_steps: int = 10,
) -> tuple[dict, dict]:
    """
    Replay each oracle scenario in the learner sim and compare speeds.

    For each scenario:
      - Inject the exact puck/paddle state into the learner.
      - Step with zero action until a paddle-puck collision is detected.
      - Read learner post-collision puck speed via privileged access.
      - Accumulate per-tier stats for both oracle (from scenario["oracle_speed_out"])
        and learner (from privileged read).

    Parameters
    ----------
    learner_env      : AirHockeyEnv with current learner collision scales
    scenarios        : list of dicts from collect_oracle_scenarios
    max_replay_steps : maximum env steps to wait for collision before skipping

    Returns
    -------
    (oracle_stats, learner_stats) — both in the same dict format as rollout.rollout_episodes:
        {"paddle": {"low": {"count", "mean_speed_in", "mean_speed_out"}, "mid": ..., "high": ...}}
    """
    sim = learner_env.simulator
    puck_name = list(sim.pucks.keys())[0]
    act_dim = int(np.prod(learner_env.action_space.shape))
    zero_action = np.zeros(act_dim, dtype=np.float32)

    oracle_total = {"paddle": {t: {"count": 0, "speed_in_sum": 0.0, "speed_out_sum": 0.0} for t in TIERS}}
    learner_total = {"paddle": {t: {"count": 0, "speed_in_sum": 0.0, "speed_out_sum": 0.0} for t in TIERS}}

    for sc in scenarios:
        tier = sc["tier"]
        puck_speed_pre = float(sc["puck_speed_pre"])
        oracle_speed_out = float(sc["oracle_speed_out"])

        # Set up learner state
        learner_env.reset()
        sim.pucks[puck_name].position = sim.base_coord_to_box2d(sc["puck_pos"])
        sim.pucks[puck_name].linearVelocity = sim.base_coord_to_box2d(sc["puck_vel"])
        sim.paddles["paddle_ego"].position = sim.base_coord_to_box2d(sc["paddle_pos"])
        sim.paddles["paddle_ego"].linearVelocity = sim.base_coord_to_box2d(sc["paddle_vel"])

        replay_detector = StepCollisionDetector(sim)
        replay_detector.reset()
        learner_speed_out = None
        for _ in range(max_replay_steps):
            learner_env.step(zero_action)
            if replay_detector.step():
                vb = sim.pucks[puck_name].linearVelocity  # privileged
                vx, vy = _b2d_to_base(vb)
                learner_speed_out = math.sqrt(vx * vx + vy * vy)
                break

        if learner_speed_out is None or learner_speed_out < 1e-6:
            continue  # puck missed paddle or zero output — discard

        # Accumulate
        for total, speed_out in [(oracle_total, oracle_speed_out), (learner_total, learner_speed_out)]:
            total["paddle"][tier]["count"] += 1
            total["paddle"][tier]["speed_in_sum"] += puck_speed_pre
            total["paddle"][tier]["speed_out_sum"] += speed_out

    # Convert to mean stats
    def _to_means(total):
        result = {"paddle": {}}
        for tier in TIERS:
            count = total["paddle"][tier]["count"]
            result["paddle"][tier] = {
                "count": count,
                "mean_speed_in": total["paddle"][tier]["speed_in_sum"] / count if count > 0 else 0.0,
                "mean_speed_out": total["paddle"][tier]["speed_out_sum"] / count if count > 0 else 0.0,
            }
        return result

    return _to_means(oracle_total), _to_means(learner_total)
