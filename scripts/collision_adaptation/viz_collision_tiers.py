"""
Visualize low- and high-speed collision scenarios from real oracle rollouts,
showing oracle (left) and learner (right) side by side.

Collects scenarios via policy rollout, picks the clearest examples from low and
high tiers, replays each in both sims, and saves one GIF per scenario.

Usage:
    python scripts/collision_adaptation/viz_collision_tiers.py \
        --config scripts/smooth_policy/amp_history/configs/new_juggle/pid_noise_constant_upper_half_custom_sim_params_heavy.yaml \
        --model-path runs/td3/final/task_only/checkpoint_350000/model.pth \
        --oracle-paddle-scales 0.7 1.0 1.2 \
        --n-collect-episodes 30 \
        --n-examples 4 \
        --output-dir runs/collision_tier_viz \
        --fps 10
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import imageio
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from airhockey import AirHockeyEnv
from airhockey.renderers import AirHockeyRenderer
from scripts.smooth_policy.deterministic_agent import DeterministicAgent
from airhockey.sims.real.velocity_estimator import fit_velocity_from_positions
from scripts.collision_adaptation.collision_detection import (
    TIERS, PUCK_HISTORY_PAD, speed_tier, is_paddle_puck_collision,
)

_GRAVITY = (-0.65, 0.0)


def _b2d_to_base(vb):
    return -float(vb[1]), float(vb[0])


def _build_env(config_path, paddle_scales):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    cfg["simulator_params"]["puck_density"] = 3000
    cfg["simulator_params"]["paddle_density"] = 3000
    env = AirHockeyEnv(cfg)
    env.simulator.set_collision_scales(
        wall_scales=[1.0, 1.0, 1.0], paddle_scales=paddle_scales
    )
    return env


def _load_actor(model_path, device):
    class _EV:
        class single_observation_space: shape = (32,)
        class single_action_space: shape = (2,)

    actor = DeterministicAgent(_EV(), action_scale=1.0, hidden_layer_size=64, num_hidden_layers=5)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    actor.load_state_dict(ckpt)
    actor.eval()
    return actor.to(device)


# ---------------------------------------------------------------------------
# Collect scenarios from oracle rollouts (no approach filter — keep everything)
# ---------------------------------------------------------------------------

def collect_scenarios(oracle_env, actor, n_episodes, device, window_frames=10, min_snr=6.0, timestep=0.05):
    act_dim = int(np.prod(oracle_env.action_space.shape))
    action_low  = torch.tensor(oracle_env.action_space.low,  dtype=torch.float32, device=device)
    action_high = torch.tensor(oracle_env.action_space.high, dtype=torch.float32, device=device)
    scenarios = []

    for _ in range(n_episodes):
        obs, _ = oracle_env.reset()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        last_a = torch.zeros((1, act_dim), dtype=torch.float32, device=device)
        done = False; step_idx = 0; prev_fc = len(oracle_env.simulator.get_collision_forces())
        col_steps = []; col_paddle_vels = {}

        while not done:
            with torch.no_grad():
                pol = torch.cat([obs_t, last_a], dim=-1)
                act = torch.clamp(actor.get_action(pol), action_low, action_high)
            next_obs, _, term, trunc, _ = oracle_env.step(act.squeeze(0).numpy())
            done = bool(term or trunc)
            obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_a = act.detach().clone()
            if done: last_a.zero_()

            forces = oracle_env.simulator.get_collision_forces()
            new_pp = any(is_paddle_puck_collision(cf) for cf in forces[prev_fc:])
            if new_pp and (not col_steps or col_steps[-1] != step_idx):
                col_steps.append(step_idx)
                vb = oracle_env.simulator.paddles["paddle_ego"].linearVelocity
                col_paddle_vels[step_idx] = _b2d_to_base(vb)
            prev_fc = len(forces); step_idx += 1

        oracle_env.simulator.get_episode_collision_stats()

        puck_ep   = oracle_env.simulator.puck_history[PUCK_HISTORY_PAD:]
        paddle_ep = oracle_env.simulator.paddle_history[PUCK_HISTORY_PAD:]
        N = len(puck_ep)
        if N < 2 * window_frames:
            continue

        positions    = np.array([[h[0], h[1]] for h in puck_ep])
        valid_mask   = np.array([not bool(h[2]) for h in puck_ep])
        times        = np.arange(N) * timestep
        paddle_pos_a = np.array([[h[0], h[1]] for h in paddle_ep])

        last_ci = None
        for col_idx in col_steps:
            if last_ci is not None and col_idx - last_ci < window_frames: continue
            if col_idx < window_frames or col_idx + window_frames > N or col_idx < 1: continue

            pre = fit_velocity_from_positions(
                positions[col_idx - window_frames: col_idx],
                times[col_idx - window_frames: col_idx],
                valid_mask[col_idx - window_frames: col_idx],
                gravity=_GRAVITY,
            )
            if pre is None or pre["snr"] < min_snr: continue
            puck_vel = pre["v_at_end"]
            puck_spd = float(np.linalg.norm(puck_vel))
            if puck_spd < 1e-6: continue

            post = fit_velocity_from_positions(
                positions[col_idx: col_idx + window_frames],
                times[col_idx: col_idx + window_frames],
                valid_mask[col_idx: col_idx + window_frames],
                gravity=_GRAVITY,
            )
            if post is None or post["snr"] < min_snr: continue
            oracle_out = float(np.linalg.norm(post["v_at_times"][0]))

            puck_pos   = positions[col_idx - 1]
            paddle_pos = paddle_pos_a[col_idx - 1]
            paddle_vel = col_paddle_vels.get(col_idx, (0.0, 0.0))

            # Estimated approach speed (relative velocity, accounts for paddle motion)
            diff = paddle_pos - puck_pos
            dist = float(np.linalg.norm(diff))
            approach_spd = 0.0
            if dist > 1e-9:
                n = diff / dist
                v_rel = puck_vel - np.array(paddle_vel)
                approach_spd = float(-np.dot(v_rel, n))

            scenarios.append({
                "puck_pos":    puck_pos.tolist(),
                "puck_vel":    puck_vel.tolist(),
                "paddle_pos":  paddle_pos.tolist(),
                "paddle_vel":  list(paddle_vel),
                "puck_speed_pre":   puck_spd,
                "oracle_speed_out": oracle_out,
                "tier":             speed_tier(puck_spd),
                "snr":              pre["snr"],
                "approach_speed":   approach_spd,
            })
            last_ci = col_idx

    return scenarios


# ---------------------------------------------------------------------------
# Replay one scenario in one env, return annotated frames
# ---------------------------------------------------------------------------

def replay_and_render(env, scenario, scale_label, n_pre_steps=5, n_post_steps=20, fps_label=""):
    sim = env.simulator
    puck_name = list(sim.pucks.keys())[0]
    act_dim = int(np.prod(env.action_space.shape))
    zero_action = np.zeros(act_dim, dtype=np.float32)

    env.reset()
    sim.pucks[puck_name].position    = sim.base_coord_to_box2d(scenario["puck_pos"])
    sim.pucks[puck_name].linearVelocity = sim.base_coord_to_box2d(scenario["puck_vel"])
    sim.paddles["paddle_ego"].position  = sim.base_coord_to_box2d(scenario["paddle_pos"])
    sim.paddles["paddle_ego"].linearVelocity = sim.base_coord_to_box2d(scenario["paddle_vel"])

    renderer = AirHockeyRenderer(
        env,
        orientation="vertical",
        show_target_position=False,
        show_acceleration_arrow=False,
    )

    frames = []
    collision_step = None
    prev_fc = len(sim.get_collision_forces())
    tier_color = {"low": (70, 130, 230), "mid": (230, 165, 30), "high": (220, 80, 80)}
    tc = tier_color.get(scenario["tier"], (0, 0, 0))
    tier = scenario["tier"]

    total_steps = n_pre_steps + n_post_steps + 5  # a few extra to detect collision

    for step_i in range(total_steps):
        raw = renderer.get_frame()
        frame = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        aspect = frame.shape[1] / frame.shape[0]
        frame = cv2.resize(frame, (160, int(160 / aspect)))

        # Current puck speed (privileged, for annotation)
        vb = sim.pucks[puck_name].linearVelocity
        vx, vy = _b2d_to_base(vb)
        speed = float(np.sqrt(vx * vx + vy * vy))

        # Annotation
        cv2.putText(frame, scale_label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, tc, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{tier} tier", (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.35, tc, 1, cv2.LINE_AA)
        cv2.putText(frame, f"|v|={speed:.2f}", (4, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        if collision_step is not None:
            rel = step_i - collision_step
            cv2.putText(frame, f"+{rel} post", (4, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)
        frames.append(frame)

        env.step(zero_action)

        forces = sim.get_collision_forces()
        if collision_step is None and any(is_paddle_puck_collision(cf) for cf in forces[prev_fc:]):
            collision_step = step_i
        prev_fc = len(forces)

        if collision_step is not None and (step_i - collision_step) >= n_post_steps:
            break

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--oracle-paddle-scales", nargs=3, type=float, default=[0.7, 1.0, 1.2],
                   metavar=("LOW", "MID", "HIGH"))
    p.add_argument("--learner-paddle-scales", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                   metavar=("LOW", "MID", "HIGH"))
    p.add_argument("--n-collect-episodes", type=int, default=30)
    p.add_argument("--n-examples", type=int, default=4,
                   help="Number of GIFs to produce per tier (low + high).")
    p.add_argument("--tiers", nargs="+", default=["low", "high"],
                   help="Tiers to visualize (default: low high).")
    p.add_argument("--output-dir", default="runs/collision_tier_viz")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    oracle_scales  = list(args.oracle_paddle_scales)
    learner_scales = list(args.learner_paddle_scales)

    print(f"Collecting scenarios from oracle (scales={oracle_scales}) ...")
    oracle_env  = _build_env(args.config, oracle_scales)
    learner_env = _build_env(args.config, learner_scales)
    actor = _load_actor(args.model_path, args.device)

    scenarios = collect_scenarios(oracle_env, actor, args.n_collect_episodes, args.device)

    for tier in TIERS:
        count = sum(1 for s in scenarios if s["tier"] == tier)
        print(f"  {tier}: {count} scenarios collected")

    duration_ms = int(1000 / max(args.fps, 1))

    for tier in args.tiers:
        tier_sc = [s for s in scenarios if s["tier"] == tier]
        if not tier_sc:
            print(f"No {tier}-tier scenarios found — skipping.")
            continue

        # Sort by absolute approach_speed (most clear-cut examples first)
        tier_sc.sort(key=lambda s: abs(s["approach_speed"]), reverse=True)
        examples = tier_sc[: args.n_examples]

        print(f"\nRendering {len(examples)} {tier}-tier examples ...")

        for idx, sc in enumerate(examples):
            label_o = f"oracle {oracle_scales}"
            label_l = f"learner {learner_scales}"

            oracle_frames  = replay_and_render(oracle_env,  sc, label_o)
            learner_frames = replay_and_render(learner_env, sc, label_l)

            # Pad shorter sequence to equal length
            n = max(len(oracle_frames), len(learner_frames))
            while len(oracle_frames)  < n: oracle_frames.append(oracle_frames[-1])
            while len(learner_frames) < n: learner_frames.append(learner_frames[-1])

            # Side-by-side: add a thin separator column
            sep = np.full((oracle_frames[0].shape[0], 3, 3), 200, dtype=np.uint8)
            combined = [
                np.concatenate([o, sep, l], axis=1)
                for o, l in zip(oracle_frames, learner_frames)
            ]

            # Add header bar with scenario metadata
            header_h = 22
            bar = np.ones((header_h, combined[0].shape[1], 3), dtype=np.uint8) * 240
            meta = (
                f"tier={tier}  pre={sc['puck_speed_pre']:.2f}m/s  "
                f"oracle_out={sc['oracle_speed_out']:.2f}  "
                f"approach={sc['approach_speed']:.2f}  snr={sc['snr']:.0f}"
            )
            cv2.putText(bar, meta, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40, 40, 40), 1, cv2.LINE_AA)
            combined = [np.concatenate([bar, f], axis=0) for f in combined]

            gif_path = os.path.join(args.output_dir, f"{tier}_example_{idx + 1:02d}.gif")
            imageio.mimsave(gif_path, combined, format="GIF", loop=0, duration=duration_ms)
            print(f"  [{tier}] example {idx+1}: pre={sc['puck_speed_pre']:.2f} m/s  "
                  f"approach={sc['approach_speed']:.2f}  oracle_out={sc['oracle_speed_out']:.2f}  → {gif_path}")


if __name__ == "__main__":
    main()
