#!/usr/bin/env python3
"""Sim eval of the same policies from (a) the sim's own random reset and (b) the real-robot episode starts.

    .venv/bin/python -m scripts.td3.extras.eval_real_start_sim2sim \
        --runs runs/td3/sysid_v2_real_workspace/hist4_seed*/juggle_sysid_v2_hist4/checkpoint_1975000 \
        --config configs/new_juggle/sysid_v2_hist4.yaml \
        --real-runs results/eval_4_tasks_of_policies_hist4_2026-09-19/{juggle_sysid/data_20261014-201743,puck_goal_sysid/data_20261014-210939,puck_goal_vel_sysid/data_20261014-211942,puck_vel_sysid/data_20261014-220834} \
        --out-dir runs/td3/sysid_v2_real_workspace/real_start_sim2sim

Real start states come from every unique episode HDF5 (md5-deduplicated) of the ``--real-runs`` folders.
All four tasks begin from the same reset policy, so every episode is a valid start. The camera frames
behind consecutive steps are alternately ~33 and ~66 ms apart (per-step puck speed alternates ~0.6 / 1.2 m/s),
so a two-frame difference is biased; instead, over the first ``--vel-frames`` visible frames (the first
within ``--max-first-vis`` steps) the sim's constant down-table acceleration ``--accel`` is removed from x
and a line is fitted:
- puck position = the fit at the first visible frame;
- puck velocity = the fitted velocity at that frame;
- paddle position = telemetry pose at step 0 (at rest).
Starts whose first detection is the BRUNSWICK phantom (x > ``--phantom-x``; the puck is never there after a reset)
are dropped, and later phantom frames are left out of the fit. Positions / velocities are in the observation (= sim base) frame, so they go straight into
``spawn_puck`` / ``spawn_paddle``.

Conditions, each run by every policy with paired reset seeds ``--seed-base`` + i (one per start, so the
noise / occlusion draws match across conditions):
- ``sim``: the sim's own reset (puck uniform over the far 2/3, 0-0.5 m/s; paddle uniform in workspace);
- ``real_puck``: real puck position + velocity, sim-random paddle;
- ``real_puck_paddle``: real puck position + velocity and real paddle position.
``--hold-steps N`` reproduces the real post-reset transition hold: the first N steps of every episode (all
conditions) send a zero action, i.e. the paddle holds its pose while the puck moves; they count toward the
episode return and length as on the robot, and the policy's last-action input is zero when it takes over.
The real evals use N = 8; the default 0 lets the policy act from step 0.

Writes ``<out-dir>/real_starts.json``, ``results.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import gymnasium as gym  # noqa: E402

from airhockey import AirHockeyEnv  # noqa: E402
from scripts.td3.eval_utils import (  # noqa: E402
    augment_policy_observation,
    build_policy_env_view,
    load_policy_for_evaluation,
)
from scripts.td3.extras.eval_occlusion_map_sim2sim import load_cfg, mean_se  # noqa: E402

STEP_S = 0.049
CONDITIONS = ("sim", "real_puck", "real_puck_paddle")


def extract_real_starts(real_runs, max_first_vis, vel_frames, phantom_x, accel):
    starts, seen = [], set()
    paths = [p for r in real_runs for p in sorted(glob.glob(os.path.join(r, "episode_hdf5", "*", "*.hdf5")))]
    for path in paths:
        digest = hashlib.md5(open(path, "rb").read()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        with h5py.File(path, "r") as f:
            puck, pose = f["puck"][:], f["pose"][:, :2]
        head = puck[:max_first_vis + 2 * vel_frames]
        rel = os.path.relpath(path, REPO)
        if (head[:, 2] < 0.5).any() and head[np.argmax(head[:, 2] < 0.5), 0] > phantom_x:
            print(f"skip (phantom first detection): {rel}")
            continue
        # later phantom frames (BRUNSWICK lettering) are left out of the fit
        vis = np.where((head[:, 2] < 0.5) & (head[:, 0] <= phantom_x))[0]
        if len(vis) < 3 or vis[0] > max_first_vis:
            print(f"skip (no early detection): {rel}")
            continue
        idx = vis[:vel_frames]
        t = (idx - idx[0]) * STEP_S
        xy = puck[idx, :2].copy()
        xy[:, 0] -= 0.5 * accel * t ** 2
        vel, pos = np.polyfit(t, xy, 1)
        starts.append(dict(file=rel, first_vis_step=int(idx[0]), puck_pos=pos.tolist(),
                           puck_vel=vel.tolist(), paddle_pos=pose[0].tolist()))
    return starts


def run_episodes(policy, cfg, use_last, seeds, starts, condition, hold_steps=0):
    env = AirHockeyEnv(cfg)
    default_puck, default_paddle = env.get_puck_configuration, env.get_paddle_configuration
    out = []
    for i, s in enumerate(seeds):
        st = starts[i]
        if condition == "sim":
            env.get_puck_configuration, env.get_paddle_configuration = default_puck, default_paddle
        else:
            env.get_puck_configuration = lambda *a, st=st, **k: (tuple(st["puck_pos"]), tuple(st["puck_vel"]))
            if condition == "real_puck_paddle":
                env.get_paddle_configuration = (
                    lambda name, st=st: (tuple(st["paddle_pos"]), (0.0, 0.0)) if name == "paddle_ego" else default_paddle(name))
            else:
                env.get_paddle_configuration = default_paddle
        obs, _ = env.reset(seed=int(s))
        last_action = torch.zeros((1, 2), dtype=torch.float32)
        ret, steps, done = 0.0, 0, False
        while not done:
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            if steps < hold_steps:
                action = np.zeros(2, dtype=np.float32)
            else:
                action = policy(augment_policy_observation(o, last_action, use_last)).numpy().squeeze()
            obs, rew, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            ret += float(rew)
            steps += 1
            last_action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)
        out.append(dict(seed=int(s), start=st["file"], ret=ret, steps=steps))
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="checkpoint dirs (model.pth + args.yaml), one per seed")
    ap.add_argument("--config", required=True)
    ap.add_argument("--real-runs", nargs="+", required=True, help="real eval run folders (each with episode_hdf5/)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--max-first-vis", type=int, default=4)
    ap.add_argument("--vel-frames", type=int, default=5)
    ap.add_argument("--accel", type=float, default=0.73, help="down-table (+x) puck acceleration, m/s^2 (sysid v2 |g|)")
    ap.add_argument("--phantom-x", type=float, default=0.8)
    ap.add_argument("--hold-steps", type=int, default=0, help="zero-action steps at the start of every episode (real: 8)")
    ap.add_argument("--out-dir", required=True)
    cli = ap.parse_args()

    starts = extract_real_starts(cli.real_runs, cli.max_first_vis, cli.vel_frames, cli.phantom_x, cli.accel)
    seeds = [cli.seed_base + i for i in range(len(starts))]
    cfg = load_cfg(cli.config)
    os.makedirs(cli.out_dir, exist_ok=True)
    Path(cli.out_dir, "real_starts.json").write_text(json.dumps(starts, indent=1))
    sp = np.linalg.norm([s["puck_vel"] for s in starts], axis=1)
    print(f"{len(starts)} real starts; puck speed median {np.median(sp):.2f} m/s")

    results = {}
    for run in cli.runs:
        run = os.path.normpath(run)
        with open(os.path.join(run, "args.yaml")) as f:
            args_yaml = yaml.safe_load(f)
        use_last = bool(args_yaml["use_last_action_in_policy_state"])
        probe = gym.vector.SyncVectorEnv([lambda: AirHockeyEnv(cfg)])
        view = build_policy_env_view(probe, use_last)
        probe.close()
        policy = load_policy_for_evaluation(
            model_path=os.path.join(run, "model.pth"),
            policy_env_view=view,
            action_scale=1,
            agent_hidden_layer_size=int(args_yaml["agent_hidden_layer_size"]),
            agent_num_hidden_layers=int(args_yaml["agent_num_hidden_layers"]),
        )
        name = Path(run).parts[-3] if len(Path(run).parts) >= 3 else run
        results[name] = {c: run_episodes(policy, cfg, use_last, seeds, starts, c, cli.hold_steps) for c in CONDITIONS}
        print(name, "  ".join(f"{c} {np.mean([e['ret'] for e in results[name][c]]):.1f}" for c in CONDITIONS), flush=True)

    names = list(results)
    per = {c: [np.mean([e["ret"] for e in results[n][c]]) for n in names] for c in CONDITIONS}
    lens = {c: [np.mean([e["steps"] for e in results[n][c]]) for n in names] for c in CONDITIONS}
    heads = {"sim": "sim reset", "real_puck": "real puck start", "real_puck_paddle": "real puck + paddle start"}
    lines = [
        f"# Sim eval from real-robot start states ({len(names)} policies × {len(starts)} episodes per condition)",
        "",
        f"- sim config: `{cli.config}`",
        f"- real starts: {len(starts)} unique episodes from {len(cli.real_runs)} real run folders (`real_starts.json`)",
        f"- reset seeds {seeds[0]}…{seeds[-1]}, identical across policies and conditions; "
        + (f"{cli.hold_steps}-step zero-action hold at every episode start" if cli.hold_steps else "policy acts from step 0"),
        "",
        "## Return: mean ± standard error across policies (each policy = mean of its episodes)",
        "",
        "| | " + " | ".join(heads[c] for c in CONDITIONS) + " |",
        "|---" * (len(CONDITIONS) + 1) + "|",
    ]
    fmt = lambda xs: "{:.1f} ± {:.1f}".format(*mean_se(xs))
    lines.append("| **return** | " + " | ".join(f"**{fmt(per[c])}**" for c in CONDITIONS) + " |")
    lines.append("| Δ vs sim reset (paired) | – | " + " | ".join(
        fmt([b - a for a, b in zip(per["sim"], per[c])]) for c in CONDITIONS[1:]) + " |")
    lines.append("| episode length | " + " | ".join(fmt(lens[c]) for c in CONDITIONS) + " |")
    lines += ["", "## Per policy (mean ± SE over its episodes)", "",
              "| policy | " + " | ".join(heads[c] for c in CONDITIONS) + " |", "|---" * (len(CONDITIONS) + 1) + "|"]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(fmt([e["ret"] for e in results[n][c]]) for c in CONDITIONS) + " |")
    text = "\n".join(lines) + "\n"
    Path(cli.out_dir, "summary.md").write_text(text)
    Path(cli.out_dir, "results.json").write_text(json.dumps(results, indent=1))
    print(text)


if __name__ == "__main__":
    main()
