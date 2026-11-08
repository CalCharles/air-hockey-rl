"""Evaluate juggling policies on the new sim (real-plant proxy + action latency) with survival and hit-style metrics.

The new sim is ``configs/new_juggle/sysid_v2_hist4_realplant.yaml`` (unchanged) with every action applied
``--action-latency`` control steps late (2 by default; the policy's own last action is fed back undelayed, as
on the robot). Per policy it reports return, length, the fraction of episodes reaching 5k / 10k steps (or the
cap) and, at each paddle-puck contact, the hit metrics of ``scripts/ceiling/hit_motion.py``: distance of the
paddle from the top workspace edge (edge hit = within 2 cm), forward paddle speed and puck exit speed, plus the
fraction of time the paddle spends within 2 cm of the edge. All are mean ± std err over episodes.

    python -m scripts.analysis.new_sim_eval --label L --episodes 20 --horizon 10000 <policy_dir> ...
    python -m scripts.analysis.new_sim_eval --label L --select 2 --min-step 800000 <run_dir> ...
        # --select k: evaluate every checkpoint >= --min-step (+ the final model) of each run at --select-horizon
        # (2000) on --select-episodes, then the k best per run at --horizon on fresh seeds.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
NEW_SIM = REPO / "configs/new_juggle/sysid_v2_hist4_realplant.yaml"
OUT_ROOT = REPO / "runs/analysis/new_sim_eval"
NEAR_EDGE = 0.02


def run_episode(act, reset_agent, cfg, seed, horizon, latency):
    import torch
    from airhockey import AirHockeyEnv

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    params = copy.deepcopy(cfg)
    params.update(seed=seed, max_timesteps=horizon + 1)
    params["simulator_params"]["seed"] = seed
    env = AirHockeyEnv(params)
    obs, _ = env.reset(seed=seed)
    reset_agent(obs)
    sim = env.simulator
    paddle = sim.paddles["paddle_ego"]
    xtop = float(cfg["simulator_params"]["x_min_lim"])
    queue = deque([np.zeros(2)] * latency)
    last = np.zeros(2)
    x_prev = -paddle.position[1] - sim.center_offset_constant
    ret, steps, near, hits = 0.0, 0, 0, []
    for step in range(1, horizon + 1):
        a = np.clip(act(obs, last), env.action_space.low, env.action_space.high)
        last = a.copy()
        if latency:
            queue.append(a)
            a = queue.popleft()
        obs, r, term, trunc, info = env.step(a)
        ret += float(np.asarray(r).reshape(-1)[0])
        steps = step
        x = -paddle.position[1] - sim.center_offset_constant
        near += int(x - xtop < NEAR_EDGE)
        if int(info.get("paddle_puck_collision_count", 0)) > 0:
            puck = next(iter(sim.pucks.values()))
            hits.append((x - xtop, -(x - x_prev) / sim.time_per_step, float(np.linalg.norm(puck.linearVelocity))))
        x_prev = x
        if term or trunc:
            break
    env.close()
    return dict(seed=seed, ret=ret, steps=steps, near_edge_frac=near / max(steps, 1), hits=hits)


def run_policy(policy_dir, cfg, latency, episodes, horizon, seed_base):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from airhockey import AirHockeyEnv
    from scripts.ceiling.hit_motion import _load_agent

    torch.set_num_threads(1)
    probe = AirHockeyEnv(copy.deepcopy(cfg))
    reset_agent, act = _load_agent(Path(policy_dir), probe)
    probe.close()
    return [run_episode(act, reset_agent, cfg, seed_base + e, horizon, latency) for e in range(episodes)]


def _mse(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0]


def summarize(eps, horizon):
    steps = [e["steps"] for e in eps]
    hits = [h for e in eps for h in e["hits"]]
    dtop = np.array([h[0] for h in hits]) if hits else np.zeros(0)
    out = dict(n=len(eps), ret=_mse([e["ret"] for e in eps]), len=_mse(steps),
               reach_cap=_mse([s >= horizon for s in steps]),
               near_edge_frac=_mse([e["near_edge_frac"] for e in eps]), n_hits=len(hits),
               edge_hit_frac=float(np.mean(dtop < NEAR_EDGE)) if hits else float("nan"),
               dtop_median_cm=float(np.median(dtop) * 100) if hits else float("nan"),
               vfwd=_mse([h[1] for h in hits]), pout=_mse([h[2] for h in hits]))
    for k in (5000, 10000):
        if horizon >= k:
            out[f"reach_{k}"] = _mse([s >= k for s in steps])
    return out


def line(name, s, horizon):
    reach = " ".join(f"{k // 1000}k {s[f'reach_{k}'][0] * 100:3.0f}±{s[f'reach_{k}'][1] * 100:2.0f}%"
                     for k in (5000, 10000) if f"reach_{k}" in s)
    if not reach:
        reach = f"cap {s['reach_cap'][0] * 100:3.0f}±{s['reach_cap'][1] * 100:2.0f}%"
    return (f"{name:70s} ret {s['ret'][0]:7.0f}±{s['ret'][1]:4.0f} len {s['len'][0]:6.0f}±{s['len'][1]:4.0f} {reach} | "
            f"edge hits {s['edge_hit_frac'] * 100:3.0f}% dtop {s['dtop_median_cm']:4.1f}cm vfwd {s['vfwd'][0]:.2f} "
            f"pout {s['pout'][0]:.2f} near-edge {s['near_edge_frac'][0] * 100:3.0f}%")


def _job(job):
    return job[0], run_policy(*job)


def evaluate(policies, cfg, latency, episodes, horizon, seed_base, workers, label_prefix=""):
    jobs = [(p, cfg, latency, episodes, horizon, seed_base) for p in policies]
    res = {}
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        for p, eps in ex.map(_job, jobs):
            name = os.path.relpath(p, REPO)
            s = summarize(eps, horizon)
            res[name] = dict(summary=s, episodes=[{k: v for k, v in e.items() if k != "hits"} for e in eps])
            print(label_prefix + line(name, s, horizon), flush=True)
    return res


def candidates(run_dir, min_step):
    run_dir = Path(run_dir)
    out = []
    for c in run_dir.glob("checkpoint_*"):
        m = re.fullmatch(r"checkpoint_(\d+)", c.name)
        if m and int(m.group(1)) >= min_step and (c / "model.pth").exists():
            out.append(c)
    out.sort(key=lambda c: int(c.name.split("_")[1]))
    if (run_dir / "model.pth").exists():
        out.append(run_dir)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--label", required=True)
    ap.add_argument("--action-latency", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=10000)
    ap.add_argument("--seed-base", type=int, default=900000)
    ap.add_argument("--select", type=int, default=0, help="per run: keep the k best checkpoints of a short screen")
    ap.add_argument("--min-step", type=int, default=0)
    ap.add_argument("--select-horizon", type=int, default=2000)
    ap.add_argument("--select-episodes", type=int, default=20)
    ap.add_argument("--select-seed-base", type=int, default=800000)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    cfg = yaml.safe_load(NEW_SIM.read_text())["air_hockey"]
    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    out = dict(args=vars(args))
    policies = args.paths
    if args.select:
        screen_list = [c for r in args.paths for c in candidates(r, args.min_step)]
        screen = evaluate(screen_list, cfg, args.action_latency, args.select_episodes, args.select_horizon,
                          args.select_seed_base, args.workers, "[screen] ")
        out["screen"] = screen
        policies = []
        for r in args.paths:
            rel = os.path.relpath(r, REPO)
            mine = {k: v for k, v in screen.items() if k == rel or k.startswith(rel + "/checkpoint_")}
            ranked = sorted(mine, key=lambda k: (mine[k]["summary"]["len"][0], mine[k]["summary"]["ret"][0]), reverse=True)
            policies += [str(REPO / k) for k in ranked[:args.select]]
    out["results"] = evaluate(policies, cfg, args.action_latency, args.episodes, args.horizon, args.seed_base,
                              args.workers, f"[{args.label}] ")
    (out_dir / "results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
