"""Evaluate juggling policies on a modified sim meant to stand in for the real robot.

Starts from ``configs/new_juggle/sysid_v2_hist4.yaml``, merges ``--sim-override`` into
``simulator_params`` and optionally delays every action by ``--action-latency`` control steps
(the policy's own last action is still fed back undelayed, as on the robot), and optionally scales the
executed action per axis (``--action-scale``, as in the real x2.8 / x2.35 test). Reports return,
length and juggles-proxy per policy, mean ± std err over episodes, for comparison with the real
eval logs (``eval_summary.json``).

    python -m scripts.analysis.proxy_sim_eval --label L --sim-override '{"enable_observation_delay": false}' \
        --action-latency 2 --episodes 20 --horizon 1000 <policy_dir> ...
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "runs/analysis/proxy_sim_eval"


def run_policy(policy_dir, cfg, latency, episodes, horizon, seed_base, scale=(1.0, 1.0)):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    from airhockey import AirHockeyEnv
    from scripts.ceiling.hit_motion import _load_agent

    torch.set_num_threads(1)
    probe = AirHockeyEnv(copy.deepcopy(cfg))
    reset_agent, act = _load_agent(Path(policy_dir), probe)
    probe.close()
    eps = []
    for e in range(episodes):
        seed = seed_base + e
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        params = copy.deepcopy(cfg)
        params.update(seed=seed, max_timesteps=horizon + 1)
        params["simulator_params"]["seed"] = seed
        env = AirHockeyEnv(params)
        obs, _ = env.reset(seed=seed)
        reset_agent(obs)
        sim = env.simulator
        paddle = sim.paddles["paddle_ego"]
        queue = deque([np.zeros(2)] * latency)
        last = np.zeros(2)
        ret, steps, hits, speed = 0.0, 0, 0, []
        prev = np.array(paddle.position, float)
        for step in range(1, horizon + 1):
            a = np.clip(act(obs, last), env.action_space.low, env.action_space.high)
            last = a.copy()
            a = np.clip(a * np.asarray(scale), env.action_space.low, env.action_space.high)
            if latency:
                queue.append(a)
                a = queue.popleft()
            obs, r, term, trunc, info = env.step(a)
            ret += float(np.asarray(r).reshape(-1)[0])
            steps = step
            hits += int(int(info.get("paddle_puck_collision_count", 0)) > 0)
            pos = np.array(paddle.position, float)
            speed.append(np.linalg.norm(pos - prev) / sim.time_per_step)
            prev = pos
            if term or trunc:
                break
        env.close()
        eps.append(dict(seed=seed, ret=ret, steps=steps, hit_steps=hits, speed=float(np.mean(speed))))
    return eps


def _mse(x):
    x = np.asarray(x, float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def _job(job):
    d = job[0]
    return d, run_policy(*job)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policies", nargs="+")
    ap.add_argument("--label", required=True)
    ap.add_argument("--sim-config", default=str(REPO / "configs/new_juggle/sysid_v2_hist4.yaml"))
    ap.add_argument("--sim-override", default="{}")
    ap.add_argument("--action-latency", type=int, default=0)
    ap.add_argument("--action-scale", type=float, nargs=2, default=[1.0, 1.0],
                    help="multiply the executed action per axis (the policy still sees its own unscaled last action)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=1000)
    ap.add_argument("--seed-base", type=int, default=800000)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.sim_config).read_text())["air_hockey"]
    cfg["simulator_params"].update(json.loads(args.sim_override))
    out_dir = OUT_ROOT / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(d, cfg, args.action_latency, args.episodes, args.horizon, args.seed_base, tuple(args.action_scale))
            for d in args.policies]
    res = {}
    with ProcessPoolExecutor(min(args.workers, len(jobs))) as ex:
        for d, eps in ex.map(_job, jobs):
            name = os.path.relpath(d, REPO)
            ret, ret_se = _mse([e["ret"] for e in eps])
            ln, ln_se = _mse([e["steps"] for e in eps])
            sp, _ = _mse([e["speed"] for e in eps])
            cap = float(np.mean([e["steps"] >= args.horizon for e in eps]))
            res[name] = dict(episodes=eps, ret=ret, ret_se=ret_se, len=ln, len_se=ln_se, speed=sp, reached_cap=cap)
            print(f"[{args.label}] {name:80s} return {ret:7.1f} ± {ret_se:5.1f} | length {ln:6.1f} ± {ln_se:5.1f} "
                  f"| reached {args.horizon}: {cap*100:3.0f}% | paddle speed {sp:.2f} m/s", flush=True)
    (out_dir / "results.json").write_text(json.dumps(dict(args=vars(args), results=res), indent=1))


if __name__ == "__main__":
    main()
