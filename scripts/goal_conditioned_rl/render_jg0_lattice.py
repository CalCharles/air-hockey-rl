"""Lattice overlay GIF for a jg0 policy: one puck launch (same env seed) commanded toward each of the 5 x 7
upper-half goals, all rollouts drawn on one table (puck = filled disc, paddle = ring, puck trail), with the
running "hit so far" count (puck centre within 10 cm of its goal at some step). Same goals, seeds, horizon and
deterministic head vote as the in-training eval of train_jg0.py. Positions drawn are the true Box2D state.

    .venv-jax/bin/python -m scripts.goal_conditioned_rl.render_jg0_lattice \
        --policy runs/crl/jg0_best_s0/best_policy.msgpack --label "best @ 22M" --out runs/crl/gifs/jg0/lattice_best.gif
"""
from __future__ import annotations

import argparse
import colorsys
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU by default: never touch a training GPU
os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--launch", type=int, default=0, help="eval launch index (env seed 900000 + launch)")
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--trail", type=int, default=60, help="trail length in steps")
    a = ap.parse_args()

    import jax.numpy as jnp
    from airhockey.renderers import AirHockeyRenderer
    from scripts.goal_conditioned_rl import jg0
    from scripts.goal_conditioned_rl.crl_env import load_sim_config, make_env, policy_state, puck_goal
    from scripts.goal_conditioned_rl.render_crl_gifs import _save_gif
    from scripts.goal_conditioned_rl.train_jg0 import RADIUS, lattice_goals

    actor = jg0.Learner.load(a.policy)
    goals = np.asarray(lattice_goals(5, 7), np.float32)
    n, seed = len(goals), 900_000 + a.launch
    cfg = load_sim_config(max_timesteps=7000)
    envs, obs = [], []
    for _ in range(n):
        e = make_env(dict(cfg, max_timesteps=a.horizon + 1), seed, 2)
        o, _ = e.reset(seed=seed)
        envs.append(e)
        obs.append(o)
    rend = AirHockeyRenderer(envs[0], orientation="vertical", show_target_position=False,
                             show_acceleration_arrow=False)
    bg = cv2.cvtColor(cv2.rotate(rend.air_hockey_table_img.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE), cv2.COLOR_BGR2RGB)
    sc = a.width / bg.shape[1]
    bg = cv2.resize(bg, (a.width, int(round(bg.shape[0] * sc))))
    head = 52
    hz = int(getattr(envs[0].simulator, "time_frequency", 20))
    cols = [tuple(int(255 * c) for c in colorsys.hsv_to_rgb(i / n, 0.85, 0.9)) for i in range(n)]

    def px(x, y):
        u, v = rend.world_xy_to_output_pixel(float(x), float(y))
        return int(round(u * sc)), int(round(v * sc)) + head

    r_goal, r_puck = int(RADIUS * rend.ppm * sc), max(2, int(envs[0].puck_radius * rend.ppm * sc))
    r_pad = max(3, int(envs[0].paddle_radius * rend.ppm * sc))
    last = np.zeros((n, 2), np.float32)
    alive, hit, dropped = np.ones(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    trails = [[] for _ in range(n)]
    paddles = [None] * n
    frames = []
    for t in range(1, a.horizon + 1):
        S = np.stack([policy_state(obs[i], last[i]) for i in range(n)]).astype(np.float32)
        A = np.clip(np.asarray(actor.sample_actions(jnp.asarray(S), jnp.asarray(goals))), -1, 1).astype(np.float32)
        for i in np.flatnonzero(alive):
            o, _, term, trunc, _ = envs[i].step(A[i])
            obs[i], last[i] = o, A[i]
            pg = puck_goal(envs[i])
            trails[i].append(px(pg[0], pg[1]))
            pad = next(iter(envs[i].simulator.paddles.values()))  # Box2D (bx, by) -> env (-by, bx)
            paddles[i] = px(-pad.position[1], pad.position[0])
            hit[i] |= float(np.linalg.norm(pg[:2] - goals[i, :2])) < RADIUS
            if term or trunc:
                alive[i] = False
                dropped[i] = bool(term)
        f = np.full((bg.shape[0] + head, bg.shape[1], 3), 255, np.uint8)
        f[head:] = bg
        for i in range(n):
            cv2.circle(f, px(*goals[i, :2]), r_goal, cols[i], 2 if hit[i] else 1, cv2.LINE_AA)
        for i in range(n):
            tr = trails[i][-a.trail:]
            if len(tr) > 1:
                cv2.polylines(f, [np.array(tr, np.int32)], False, cols[i], 1, cv2.LINE_AA)
        for i in range(n):
            if not trails[i]:
                continue
            if dropped[i]:
                p = trails[i][-1]
                cv2.drawMarker(f, p, cols[i], cv2.MARKER_TILTED_CROSS, 8, 2)
                continue
            cv2.circle(f, paddles[i], r_pad, cols[i], 1, cv2.LINE_AA)
            cv2.circle(f, trails[i][-1], r_puck, cols[i], -1, cv2.LINE_AA)
            cv2.circle(f, trails[i][-1], r_puck, (0, 0, 0), 1, cv2.LINE_AA)
        fn = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(f, f"jg0 {a.label} | launch {a.launch} toward all {n} goals", (4, 14), fn, 0.42, (0, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(f, f"t = {t / hz:5.2f} s   hit so far: {hit.sum()}/{n}   dropped: {dropped.sum()}/{n}",
                    (4, 32), fn, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(f, "puck = disc, paddle = ring, x = dropped, thick ring = hit", (4, 47), fn, 0.34,
                    (80, 80, 80), 1, cv2.LINE_AA)
        frames.append(f)
        if not alive.any():
            break
    frames += [frames[-1]] * 20
    for e in envs:
        e.close()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_gif(frames, out, fps=hz)
    print(f"{out}: hit {hit.sum()}/{n}, dropped {dropped.sum()}/{n}, frames {len(frames)}")
    print("hit goals:", np.flatnonzero(hit).tolist(), " dropped:", np.flatnonzero(dropped).tolist())


if __name__ == "__main__":
    main()
