"""Paired GIFs of the CRL goal eval: same env seed, goal A commanded (left) vs the control goal B commanded (right).

Each panel draws the commanded goal in magenta (circle = the 10 cm hit radius, line = goal velocity × 0.3 s)
and the other goal in grey, plus the step, the current / closest puck distance to goal A, and END once that
episode has terminated (its last frame is held). A goal-agnostic policy gives two identical panels.
Goals are picked from a paired-eval episodes file: the best, the median and the worst paired gap.

    python -m scripts.goal_conditioned_rl.render_crl_gifs --policy crl:runs/crl/crl_paper_s3/final.pt \
        --episodes runs/crl/paired_eval_paper/paper2000000_s3_episodes.json --label paper2M_s3 --out runs/crl/gifs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from scripts.goal_conditioned_rl.crl_env import load_sim_config, make_env, puck_goal
from scripts.goal_conditioned_rl.eval_crl import EVAL_SEED_BASE, RADIUS, STILL_SPEED, _make_policy, _reset_policy

CMD, GREY = (230, 0, 230), (120, 120, 120)  # magenta: the puck sprite is green


def _draw_goal(frame, renderer, goal, color, scale):
    x, y = renderer.world_xy_to_output_pixel(float(goal[0]), float(goal[1]))
    x2, y2 = renderer.world_xy_to_output_pixel(float(goal[0] + 0.3 * goal[2]), float(goal[1] + 0.3 * goal[3]))
    p, q = (int(x * scale), int(y * scale)), (int(x2 * scale), int(y2 * scale))
    cv2.circle(frame, p, max(2, int(RADIUS * renderer.ppm * scale)), color, 1, cv2.LINE_AA)
    cv2.line(frame, p, q, color, 1, cv2.LINE_AA)


def _panel(renderer, cmd_goal, other_goal, title, step, d, dmin, ended, speed=None, stop_step=None):
    from airhockey.renderers import AirHockeyRenderer  # noqa: F401  (renderer built by caller)

    frame = cv2.cvtColor(renderer.get_frame(), cv2.COLOR_BGR2RGB)
    scale = 160 / frame.shape[1]
    frame = cv2.resize(frame, (160, int(round(frame.shape[0] * scale))))
    _draw_goal(frame, renderer, other_goal, GREY, scale)
    _draw_goal(frame, renderer, cmd_goal, CMD, scale)
    font, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.32
    cv2.putText(frame, title, (3, 11), font, fs, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"t {step}  dA {100 * d:.0f}cm min {100 * dmin:.0f}", (3, 22), font, fs, (0, 0, 0), 1,
                cv2.LINE_AA)
    if speed is not None:
        cv2.putText(frame, f"v {100 * speed:.0f}cm/s", (3, 33), font, fs, (0, 0, 0), 1, cv2.LINE_AA)
    if stop_step is not None:  # latched: puck was within RADIUS of goal A and slower than STILL_SPEED
        cv2.putText(frame, f"STOPPED @ {stop_step}", (3, 45), font, 0.36, (0, 140, 0), 1, cv2.LINE_AA)
    if ended:
        cv2.putText(frame, "END", (3, 57), font, 0.4, (200, 0, 0), 1, cv2.LINE_AA)
    return frame


def _save_gif(frames, out_path, fps=20, colors=96):
    """One shared undithered palette: the table photo dithers into ~100 kB/frame otherwise."""
    from PIL import Image

    sample = np.concatenate(frames[:: max(1, len(frames) // 8)], axis=0)
    # The puck / paddle sprites and the overlay colours are tiny regions: give them palette slots explicitly.
    sprites = sample[(sample.max(-1).astype(int) - sample.min(-1)) > 90]  # saturated pixels only
    keys = np.array([CMD, (200, 0, 0), (0, 0, 0)], dtype=np.uint8)
    swatch = np.concatenate([np.repeat(sprites, 4, 0), np.repeat(keys, 4000, 0)])[:, None, :]
    pal = Image.fromarray(np.concatenate([sample.reshape(-1, 1, 3), swatch])).quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT)
    ims = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    ims[0].save(out_path, save_all=True, append_images=ims[1:], loop=0, duration=int(1000 / fps), optimize=False)


def render_pair(spec, cfg, goal_a, goal_b, seed, max_steps, out_path, label, show_stop=False):
    from airhockey.renderers import AirHockeyRenderer

    runs = []
    for cmd, other, name in ((goal_a, goal_b, "cmd A"), (goal_b, goal_a, "cmd B")):
        env = make_env(dict(cfg, max_timesteps=max_steps + 1), seed)
        reset, act = _make_policy(spec, env)
        renderer = AirHockeyRenderer(env, orientation="vertical", show_target_position=False,
                                     show_acceleration_arrow=False)
        obs, _ = env.reset(seed=seed)
        _reset_policy(reset, obs, env)
        last = np.zeros(2, dtype=np.float32)
        frames, dmin, stop = [], np.inf, None
        for t in range(1, max_steps + 1):
            a = np.clip(np.asarray(act(obs, last, cmd), dtype=np.float32), -1.0, 1.0)
            last = a
            obs, _, term, trunc, _ = env.step(a)
            pg = puck_goal(env)
            d = float(np.linalg.norm(pg[:2] - goal_a[:2]))
            v = float(np.linalg.norm(pg[2:]))
            dmin = min(dmin, d)
            if stop is None and d < RADIUS and v < STILL_SPEED:
                stop = t
            sp, st = (v, stop) if show_stop else (None, None)
            frames.append(_panel(renderer, cmd, other, f"{label} {name}", t, d, dmin, False, sp, st))
            if term or trunc:
                frames.append(_panel(renderer, cmd, other, f"{label} {name}", t, d, dmin, True, sp, st))
                break
        env.close()
        runs.append(frames)
    n = max(len(f) for f in runs)
    runs = [f + [f[-1]] * (n - len(f)) for f in runs]
    sep = np.full((runs[0][0].shape[0], 2, 3), 255, dtype=np.uint8)
    frames = [np.concatenate([a, sep, b], axis=1) for a, b in zip(*runs)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_gif(frames, out_path)
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True)
    p.add_argument("--episodes", default=None,
                   help="paired-eval <label>_episodes.json for this policy (optional with --goal-indices)")
    p.add_argument("--label", required=True)
    p.add_argument("--goals", default="runs/crl/eval_goals.npz")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--out", default="runs/crl/gifs")
    p.add_argument("--zero-velocity", action="store_true",
                   help="command (x, y, 0, 0) goals and show puck speed + a STOPPED marker (within 10 cm, < 0.1 m/s)")
    p.add_argument("--puck-noise-std", type=float, default=None,
                   help="override simulator_params.puck_noise_std (default: the sim config's 0.01; 0 = no position noise)")
    p.add_argument("--goal-indices", default="", help="comma list of goal indices to render instead of best/median/worst")
    a = p.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    goals = np.load(a.goals)["goals"].astype(np.float32)
    if a.zero_velocity:
        goals[:, 2:] = 0.0
    n = len(goals)
    eps = json.load(open(a.episodes)) if a.episodes else []
    if a.goal_indices:
        by_i = {e["goal_index"]: e for e in eps}
        picks = {f"pick{k}": by_i.get(int(i), {"goal_index": int(i)}) for k, i in enumerate(a.goal_indices.split(","))}
    else:
        order = sorted(eps, key=lambda e: -e["paired_gap"])
        picks = {"best": order[0], "median": order[len(order) // 2], "worst": order[-1]}
    cfg = load_sim_config()
    if a.puck_noise_std is not None:
        cfg["simulator_params"]["puck_noise_std"] = float(a.puck_noise_std)
    for tag, e in picks.items():
        i = e["goal_index"]
        out = Path(a.out) / f"{a.label}_{tag}_goal{i}.gif"
        k = render_pair(a.policy, cfg, goals[i], goals[(i + n // 2) % n], EVAL_SEED_BASE + i, a.max_steps, out,
                        a.label, show_stop=a.zero_velocity)
        if "paired_gap" in e:
            print(f"{out}  frames {k}  paired_gap {100 * e['paired_gap']:+.1f} cm  "
                  f"minA cmdA {100 * e['min_dist']:.1f} / cmdB {100 * e['min_dist_other_cmd']:.1f} cm", flush=True)
        else:
            print(f"{out}  frames {k}", flush=True)


if __name__ == "__main__":
    main()
