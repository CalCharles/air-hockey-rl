"""
Validate position-based collision detection against ground-truth Box2D callbacks.

Rolls out episodes with a trained TD3 policy, runs both StepCollisionDetector
(ground truth) and detect_collisions_from_positions (position-based), compares
them, and reports precision/recall/F1.

Saves trajectory data + ground truth + detections to a pickle file so future
runs can reload trajectories without re-rolling episodes.

Usage:
    python scripts/collision_adaptation/test_position_detection.py \
        --config scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params.yaml \
        --model-path ex_model/heavy_td3_model/checkpoint_100000/model.pth \
        --n-episodes 50 \
        --output-dir runs/position_detection_validation

    # Re-run detection on saved trajectories with different thresholds:
    python scripts/collision_adaptation/test_position_detection.py \
        --load-trajectories runs/position_detection_validation/trajectory_data.pkl \
        --speed-change-threshold 0.08 --angle-change-threshold 25.0
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from airhockey import AirHockeyEnv
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from scripts.collision_adaptation.collision_detection import (
    PUCK_HISTORY_PAD,
    CollisionEvent,
    StepCollisionDetector,
    detect_collisions_from_positions,
    is_paddle_puck_collision,
    is_wall_puck_collision,
)


# ---------------------------------------------------------------------------
# Env / actor helpers (same pattern as run_adaptation_position_based.py)
# ---------------------------------------------------------------------------

def _build_env(config_path: str, paddle_scales: list[float] | None = None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["simulator_params"]["puck_density"] = 3000
    cfg["simulator_params"]["paddle_density"] = 3000
    env = AirHockeyEnv(cfg)
    if paddle_scales is not None:
        env.simulator.set_collision_scales(
            wall_scales=[1.0, 1.0, 1.0],
            paddle_scales=paddle_scales,
        )
    return env


def _load_actor(model_path: str, device: str, obs_dim: int = 32, act_dim: int = 2):
    class _EnvView:
        class single_observation_space:
            shape = (obs_dim,)
        class single_action_space:
            shape = (act_dim,)

    actor = DeterministicAgent(
        _EnvView(),
        action_scale=1.0,
        hidden_layer_size=64,
        num_hidden_layers=5,
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt)
    actor.eval()
    return actor.to(device)


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectories(env, actor, n_episodes, device, use_last_action=True, timestep=0.05):
    """Roll out episodes and collect positions + ground-truth collision indices.

    Ground-truth collision indices are stored in FRAME space (not env-step space).
    The sim may record multiple puck_history entries per env step (e.g. 2x when
    observation delay creates sub-step breakpoints). We detect the frames_per_step
    ratio and convert step indices to frame indices.
    """
    sim = env.simulator if hasattr(env, "simulator") else env.unwrapped.simulator
    detector = StepCollisionDetector(sim)
    episodes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        detector.reset()
        last_action = np.zeros(2)

        gt_paddle_steps = []
        gt_wall_steps = []
        prev_cf_count = len(sim.get_collision_forces())

        done = False
        step_idx = 0
        while not done:
            # Build policy input
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if use_last_action:
                la_t = torch.tensor(last_action, dtype=torch.float32, device=device).unsqueeze(0)
                obs_t = torch.cat([obs_t, la_t], dim=-1)

            with torch.no_grad():
                action = actor.get_action(obs_t).cpu().numpy().squeeze(0)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            last_action = action

            # Ground-truth collision detection
            forces = sim.get_collision_forces()
            new_entries = forces[prev_cf_count:]
            prev_cf_count = len(forces)

            has_paddle = any(is_paddle_puck_collision(cf) for cf in new_entries)
            has_wall = any(is_wall_puck_collision(cf) for cf in new_entries)
            if has_paddle:
                gt_paddle_steps.append(step_idx)
            if has_wall:
                gt_wall_steps.append(step_idx)

            step_idx += 1

        # Extract position arrays (skip PUCK_HISTORY_PAD padding entries)
        puck_hist = sim.puck_history[PUCK_HISTORY_PAD:]
        paddle_hist = sim.paddle_history[PUCK_HISTORY_PAD:]

        n_frames = min(len(puck_hist), len(paddle_hist))
        n_env_steps = step_idx

        # Compute frames-per-step ratio and actual frame dt
        frames_per_step = n_frames / n_env_steps if n_env_steps > 0 else 1
        frame_dt = timestep / frames_per_step

        puck_pos = np.array([[h[0], h[1]] for h in puck_hist[:n_frames]])
        paddle_pos = np.array([[h[0], h[1]] for h in paddle_hist[:n_frames]])
        valid_mask = np.array([not bool(h[2]) for h in puck_hist[:n_frames]])
        times = np.arange(n_frames) * frame_dt

        # Convert ground-truth step indices to frame indices
        # Each env step corresponds to frames_per_step frames; map to the
        # last frame of that step (where the collision state is recorded).
        fps_int = round(frames_per_step)
        gt_paddle_frames = [s * fps_int + (fps_int - 1) for s in gt_paddle_steps
                           if s * fps_int + (fps_int - 1) < n_frames]
        gt_wall_frames = [s * fps_int + (fps_int - 1) for s in gt_wall_steps
                         if s * fps_int + (fps_int - 1) < n_frames]

        episodes.append({
            "puck_positions": puck_pos,
            "paddle_positions": paddle_pos,
            "valid_mask": valid_mask,
            "times": times,
            "gt_paddle_collisions": gt_paddle_frames,
            "gt_wall_collisions": gt_wall_frames,
            "n_env_steps": n_env_steps,
            "n_frames": n_frames,
            "frames_per_step": fps_int,
            "frame_dt": frame_dt,
        })

        n_paddle = len(gt_paddle_frames)
        n_wall = len(gt_wall_frames)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  Episode {ep+1}/{n_episodes}: {n_frames} frames ({fps_int}x), "
                  f"{n_paddle} paddle, {n_wall} wall collisions")

    return episodes


# ---------------------------------------------------------------------------
# Run position-based detection on collected trajectories
# ---------------------------------------------------------------------------

def run_detection(episodes, **detect_kwargs):
    """Run detect_collisions_from_positions on each episode."""
    for ep_data in episodes:
        events = detect_collisions_from_positions(
            ep_data["puck_positions"],
            ep_data["paddle_positions"],
            ep_data["times"],
            valid_mask=ep_data["valid_mask"],
            **detect_kwargs,
        )
        ep_data["detected_events"] = events
    return episodes


# ---------------------------------------------------------------------------
# Matching and metrics
# ---------------------------------------------------------------------------

def _dedup_gt(indices, gap=3):
    """Merge consecutive ground-truth indices within `gap` frames.

    Multiple collision_forces entries from the same physical collision
    can span adjacent env steps.  Returns one representative index
    (the first) per physical collision.
    """
    if not indices:
        return []
    indices = sorted(indices)
    result = [indices[0]]
    for idx in indices[1:]:
        if idx - result[-1] > gap:
            result.append(idx)
    return result


def _match_events(gt_indices, detected_events, collision_type, tolerance, n_frames=None,
                  window_frames=5):
    """Greedy match ground-truth indices to detected events of given type.

    GT indices are first deduplicated (same physical collision across steps)
    and boundary events (within window_frames of start/end) are excluded
    since the detector cannot see them.

    Returns (tp, fp, fn, matched_pairs).
    """
    type_events = [ev for ev in detected_events if ev.collision_type == collision_type]

    # Dedup GT and exclude boundary events
    deduped_gt = _dedup_gt(gt_indices, gap=tolerance)
    if n_frames is not None:
        deduped_gt = [idx for idx in deduped_gt
                      if window_frames <= idx < n_frames - window_frames]

    matched_gt = set()
    matched_det = set()
    pairs = []

    for gt_idx in deduped_gt:
        best_dist = tolerance + 1
        best_j = None
        for j, ev in enumerate(type_events):
            if j in matched_det:
                continue
            dist = abs(ev.frame_idx - gt_idx)
            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is not None:
            matched_gt.add(gt_idx)
            matched_det.add(best_j)
            pairs.append((gt_idx, type_events[best_j]))

    tp = len(matched_gt)
    fp = len(type_events) - len(matched_det)
    fn = len(deduped_gt) - len(matched_gt)
    return tp, fp, fn, pairs


def compute_metrics(episodes, tolerance=3, window_frames=5):
    """Compute precision/recall/F1 across all episodes."""
    totals = {}
    for ctype in ("paddle", "wall", "overall"):
        totals[ctype] = {"tp": 0, "fp": 0, "fn": 0}

    for ep_data in episodes:
        detected = ep_data.get("detected_events", [])
        n_frames = len(ep_data["puck_positions"])

        for ctype, gt_key in [("paddle", "gt_paddle_collisions"), ("wall", "gt_wall_collisions")]:
            gt = ep_data[gt_key]
            tp, fp, fn, _ = _match_events(gt, detected, ctype, tolerance,
                                           n_frames=n_frames, window_frames=window_frames)
            totals[ctype]["tp"] += tp
            totals[ctype]["fp"] += fp
            totals[ctype]["fn"] += fn
            totals["overall"]["tp"] += tp
            totals["overall"]["fp"] += fp
            totals["overall"]["fn"] += fn

    for ctype in totals:
        t = totals[ctype]
        tp, fp, fn = t["tp"], t["fp"], t["fn"]
        t["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        t["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p, r = t["precision"], t["recall"]
        t["f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return totals


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(metrics):
    print("\n" + "=" * 60)
    print("Position-Based Collision Detection — Validation Report")
    print("=" * 60)
    for ctype in ("paddle", "wall", "overall"):
        m = metrics[ctype]
        print(f"\n  {ctype.upper()}")
        print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
        print(f"    Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Validate position-based collision detection against Box2D ground truth."
    )
    p.add_argument("--config",
        default="scripts/smooth_policy/amp_history/configs/new_juggle/"
                "sysid_best_params.yaml")
    p.add_argument("--model-path",
        default="ex_model/heavy_td3_model/checkpoint_100000/model.pth")
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--output-dir", default="runs/position_detection_validation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--paddle-scales", nargs=3, type=float, default=[1.0, 1.0, 1.0])

    # Detection parameters
    p.add_argument("--window-frames", type=int, default=4)
    p.add_argument("--min-snr", type=float, default=3.0)
    p.add_argument("--speed-change-threshold", type=float, default=0.08)
    p.add_argument("--angle-change-threshold", type=float, default=20.0)
    p.add_argument("--paddle-proximity-radius", type=float, default=0.10)
    p.add_argument("--timestep", type=float, default=0.05)

    # Matching
    p.add_argument("--match-tolerance", type=int, default=3)

    # Reload mode
    p.add_argument("--load-trajectories", type=str, default=None,
        help="Path to a saved trajectory_data.pkl to re-run detection without re-rolling.")

    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    detect_kwargs = dict(
        gravity=(-0.65, 0.0),
        window_frames=args.window_frames,
        min_snr=args.min_snr,
        speed_change_threshold=args.speed_change_threshold,
        angle_change_threshold=args.angle_change_threshold,
        paddle_proximity_radius=args.paddle_proximity_radius,
    )

    if args.load_trajectories:
        # Reload saved trajectories
        print(f"Loading trajectories from {args.load_trajectories}")
        with open(args.load_trajectories, "rb") as f:
            saved = pickle.load(f)
        episodes = saved["episodes"]
        print(f"Loaded {len(episodes)} episodes")
    else:
        # Collect fresh trajectories
        print(f"Config:     {args.config}")
        print(f"Model:      {args.model_path}")
        print(f"Episodes:   {args.n_episodes}")
        print(f"Device:     {args.device}")

        env = _build_env(args.config, args.paddle_scales)
        actor = _load_actor(args.model_path, args.device)

        print("\nCollecting trajectories...")
        episodes = collect_trajectories(
            env, actor, args.n_episodes, args.device,
            timestep=args.timestep,
        )

    # Run position-based detection
    print("\nRunning position-based detection...")
    print(f"  window_frames={args.window_frames}  min_snr={args.min_snr}")
    print(f"  speed_change_threshold={args.speed_change_threshold}  "
          f"angle_change_threshold={args.angle_change_threshold}")
    episodes = run_detection(episodes, **detect_kwargs)

    # Compute and print metrics
    metrics = compute_metrics(episodes, tolerance=args.match_tolerance,
                              window_frames=args.window_frames)
    print_report(metrics)

    # Print per-episode summary
    total_gt_paddle = sum(len(ep["gt_paddle_collisions"]) for ep in episodes)
    total_gt_wall = sum(len(ep["gt_wall_collisions"]) for ep in episodes)
    total_det_paddle = sum(
        sum(1 for ev in ep.get("detected_events", []) if ev.collision_type == "paddle")
        for ep in episodes
    )
    total_det_wall = sum(
        sum(1 for ev in ep.get("detected_events", []) if ev.collision_type == "wall")
        for ep in episodes
    )
    print(f"\nGround truth:  {total_gt_paddle} paddle, {total_gt_wall} wall collisions")
    print(f"Detected:      {total_det_paddle} paddle, {total_det_wall} wall collisions")

    # Save trajectories + metrics
    save_data = {
        "episodes": episodes,
        "config": {
            "n_episodes": len(episodes),
            **detect_kwargs,
            "paddle_scales": args.paddle_scales if not args.load_trajectories else None,
            "timestep": args.timestep,
            "match_tolerance": args.match_tolerance,
        },
        "metrics": metrics,
    }

    save_path = os.path.join(args.output_dir, "trajectory_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved trajectory data + metrics -> {save_path}")

    # Also write a human-readable summary
    report_path = os.path.join(args.output_dir, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write("Position-Based Collision Detection — Validation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Episodes: {len(episodes)}\n")
        f.write(f"Ground truth: {total_gt_paddle} paddle, {total_gt_wall} wall\n")
        f.write(f"Detected:     {total_det_paddle} paddle, {total_det_wall} wall\n\n")
        f.write(f"Detection params:\n")
        for k, v in detect_kwargs.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"  match_tolerance: {args.match_tolerance}\n\n")
        for ctype in ("paddle", "wall", "overall"):
            m = metrics[ctype]
            f.write(f"{ctype.upper()}:\n")
            f.write(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}\n")
            f.write(f"  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  F1={m['f1']:.3f}\n\n")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
