"""Goal-reaching evaluation for the contrastive-RL policies on the new sim.

Eval goals are true puck states (x, y, vx, vy) from held-out rollouts of the reference juggler
(``latest_models/ceiling/newsim_scratch_cap7000_swa/seed0``): one state per episode at a random step, so the
goals are states the system actually visits while juggling. Per goal one episode is run with the goal
commanded (deterministic actions, fixed seed). Every step the true puck position is compared with the
commanded goal and with a *control goal* (another eval goal that was not commanded): a policy that ignores
its goal reaches both equally, so ``gap = min_dist(control) - min_dist(commanded)`` > 0 is the
goal-conditioning signal. Velocity is not scored here (the "in the radius with vx = 0" eval comes later).

    python -m scripts.goal_conditioned_rl.eval_crl goals --out runs/crl/eval_goals.npz --n 100
    python -m scripts.goal_conditioned_rl.eval_crl eval --goals runs/crl/eval_goals.npz \
        --policy crl:runs/crl/paper_s0/final.pt --policy juggler:latest_models/ceiling/newsim_scratch_cap7000_swa/seed0 \
        --policy random --horizon 7000 --out runs/crl/final_eval
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from scripts.goal_conditioned_rl.crl_env import load_sim_config, make_env, policy_state, puck_goal

JUGGLER = "latest_models/ceiling/newsim_scratch_cap7000_swa/seed0"
RADIUS = 0.10
STILL_SPEED = 0.10  # "stationary" success: within RADIUS and |v| < STILL_SPEED at the same step
GOAL_SEED_BASE = 3_000_000
EVAL_SEED_BASE = 4_000_000


# ----------------------------------------------------------------------------- policies
def _make_policy(spec: str, env, device="cpu"):
    """Returns (reset(obs), act(obs, last, goal) -> action)."""
    import torch

    kind, _, path = spec.partition(":")
    if kind == "random":
        rng = np.random.default_rng(0)
        return (lambda obs: None), (lambda obs, last, goal: rng.uniform(-1, 1, 2))
    if kind == "juggler":
        from scripts.ceiling.hit_motion import _load_agent

        reset, act = _load_agent(Path(path), env)
        return reset, (lambda obs, last, goal: act(obs, last))
    if kind == "crl":
        from scripts.goal_conditioned_rl.crl_agent import CRLAgent

        agent = CRLAgent.from_checkpoint(torch.load(path, map_location=device, weights_only=False), device)
        agent.eval()

        def act(obs, last, goal):
            s = torch.as_tensor(policy_state(obs, last), device=device).unsqueeze(0)
            g = torch.as_tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
            return agent.act(s, g, deterministic=True).cpu().numpy().reshape(2)

        return (lambda obs: None), act
    raise ValueError(f"unknown policy spec {spec!r}")


# ----------------------------------------------------------------------------- episodes
def run_goal_episode(reset, act, cfg, goal, control_goal, seed, horizon, latency=2):
    env = make_env(dict(cfg, max_timesteps=horizon + 1), seed, latency)
    obs, _ = env.reset(seed=seed)
    reset(obs)
    last = np.zeros(2, dtype=np.float32)
    g_xy, c_xy = np.asarray(goal[:2]), np.asarray(control_goal[:2])
    min_d, min_c, within, within_c, first, steps = np.inf, np.inf, 0, 0, None, 0
    still, still_c, speed_at_min = 0.0, 0.0, np.nan
    for t in range(1, horizon + 1):
        a = np.clip(np.asarray(act(obs, last, goal), dtype=np.float32), -1.0, 1.0)
        last = a
        obs, _, term, trunc, _ = env.step(a)
        pg = puck_goal(env)
        p, speed = pg[:2], float(np.linalg.norm(pg[2:]))
        d, dc = float(np.linalg.norm(p - g_xy)), float(np.linalg.norm(p - c_xy))
        if d < min_d:
            speed_at_min = speed
        min_d, min_c = min(min_d, d), min(min_c, dc)
        if speed < STILL_SPEED:
            still = max(still, float(d < RADIUS))
            still_c = max(still_c, float(dc < RADIUS))
        within += d < RADIUS
        within_c += dc < RADIUS
        if first is None and d < RADIUS:
            first = t
        steps = t
        if term or trunc:
            break
    env.close()
    return dict(seed=seed, steps=steps, min_dist=min_d, min_dist_control=min_c, hit=float(min_d < RADIUS),
                hit_control=float(min_c < RADIUS), hit_still=still, hit_still_control=still_c,
                speed_at_min=speed_at_min, frac_within=within / steps, frac_within_control=within_c / steps,
                first_hit=first)


def _eval_job(job):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)
    spec, cfg, goals, idx, horizon = job
    probe = make_env(cfg, 0, 0)
    reset, act = _make_policy(spec, probe)
    probe.close()
    n = len(goals)
    out = []
    for i in idx:
        j = (i + n // 2) % n
        r = run_goal_episode(reset, act, cfg, goals[i], goals[j], EVAL_SEED_BASE + i, horizon)
        r["goal_index"] = int(i)
        # Paired counterfactual: same seed, the control goal commanded instead. A goal-agnostic deterministic
        # policy replays the identical episode, so paired_gap is exactly 0 for it.
        sw = run_goal_episode(reset, act, cfg, goals[j], goals[i], EVAL_SEED_BASE + i, horizon)
        r["min_dist_other_cmd"] = sw["min_dist_control"]
        r["hit_other_cmd"] = sw["hit_control"]
        r["hit_still_other_cmd"] = sw["hit_still_control"]
        r["paired_gap"] = sw["min_dist_control"] - r["min_dist"]
        out.append(r)
    return out


def summarize(eps):
    def ms(key):
        x = np.asarray([e[key] for e in eps], dtype=float)
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    def ms_arr(x):
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0

    gap = np.asarray([e["min_dist_control"] - e["min_dist"] for e in eps])
    s = {k: ms(k) for k in ("min_dist", "min_dist_control", "hit", "hit_control", "hit_still", "speed_at_min",
                            "frac_within",
                            "frac_within_control", "steps")}
    s["gap"] = (float(gap.mean()), float(gap.std(ddof=1) / np.sqrt(len(gap))) if len(gap) > 1 else 0.0)
    if "paired_gap" in eps[0]:
        s["paired_gap"] = ms("paired_gap")
        s["hit_other_cmd"] = ms("hit_other_cmd")
        s["hit_lift"] = ms_arr(np.asarray([e["hit"] - e["hit_other_cmd"] for e in eps]))
        s["hit_still_lift"] = ms_arr(np.asarray([e["hit_still"] - e["hit_still_other_cmd"] for e in eps]))
    s["episodes"] = len(eps)
    return s


def evaluate(specs, goals, horizon, workers, cfg=None):
    cfg = cfg or load_sim_config()
    n = len(goals)
    chunks = [list(range(n))[i::max(1, min(workers, n))] for i in range(max(1, min(workers, n)))]
    jobs = [(spec, cfg, goals, c, horizon) for spec in specs for c in chunks]
    results = {spec: [] for spec in specs}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (spec, *_), out in zip(jobs, pool.map(_eval_job, jobs)):
            results[spec].extend(out)
    return results


# ----------------------------------------------------------------------------- goal set
def _goal_job(job):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    torch.set_num_threads(1)
    cfg, idx, horizon, min_step, max_speed = job
    probe = make_env(cfg, 0, 0)
    reset, act = _make_policy(f"juggler:{JUGGLER}", probe)
    probe.close()
    goals = []
    for i in idx:
        seed = GOAL_SEED_BASE + i
        rng = np.random.default_rng(seed)
        env = make_env(dict(cfg, max_timesteps=horizon + 1), seed, 2)
        obs, _ = env.reset(seed=seed)
        reset(obs)
        last, states = np.zeros(2, dtype=np.float32), []
        for t in range(horizon):
            a = np.clip(np.asarray(act(obs, last, None), dtype=np.float32), -1, 1)
            last = a
            obs, _, term, trunc, _ = env.step(a)
            states.append(puck_goal(env))
            if term or trunc:
                break
        env.close()
        pool = states[min_step:] if len(states) > min_step else states
        if max_speed is not None:  # stationary-goal set: only states where the juggler's puck was this slow
            slow = [g for g in pool if np.linalg.norm(g[2:]) < max_speed]
            pool = slow or [min(pool, key=lambda g: float(np.linalg.norm(g[2:])))]
        goals.append(pool[int(rng.integers(len(pool)))])
    return goals


def make_goals(n, horizon, min_step, workers, max_speed=None):
    cfg = load_sim_config()
    chunks = [list(range(n))[i::workers] for i in range(workers)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(_goal_job, [(cfg, c, horizon, min_step, max_speed) for c in chunks]))
    order = [i for c in chunks for i in c]
    goals = np.zeros((n, 4), dtype=np.float32)
    for i, g in zip(order, [g for p in parts for g in p]):
        goals[i] = g
    return goals


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("goals")
    g.add_argument("--out", default="runs/crl/eval_goals.npz")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--horizon", type=int, default=2000)
    g.add_argument("--min-step", type=int, default=50)
    g.add_argument("--workers", type=int, default=20)
    g.add_argument("--max-speed", type=float, default=None,
                   help="pick only states with puck speed below this (m/s), e.g. 0.1 for juggler apex goals")
    e = sub.add_parser("eval")
    e.add_argument("--goals", default="runs/crl/eval_goals.npz")
    e.add_argument("--policy", action="append", required=True, help="crl:<ckpt.pt> | juggler:<dir> | random")
    e.add_argument("--label", action="append", help="one label per --policy (default: the spec)")
    e.add_argument("--horizon", type=int, default=7000)
    e.add_argument("--workers", type=int, default=40)
    e.add_argument("--out", required=True)
    e.add_argument("--zero-velocity", action="store_true", help="command (x, y, 0, 0): goal velocity set to 0")
    a = p.parse_args()
    if a.cmd == "goals":
        goals = make_goals(a.n, a.horizon, a.min_step, a.workers, a.max_speed)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, goals=goals)
        print(f"saved {len(goals)} goals to {a.out}; x range {goals[:, 0].min():.2f}..{goals[:, 0].max():.2f}")
        return
    goals = np.load(a.goals)["goals"].astype(np.float32)
    if a.zero_velocity:
        goals[:, 2:] = 0.0
    labels = a.label or a.policy
    results = evaluate(a.policy, goals, a.horizon, a.workers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for label, spec in zip(labels, a.policy):
        summary[label] = dict(spec=spec, **summarize(results[spec]))
        with open(out / f"{label.replace('/', '_').replace(':', '_')}_episodes.json", "w") as f:
            json.dump(results[spec], f)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    for label, s in summary.items():
        print(f"{label:30s} min_dist {s['min_dist'][0]:.3f}±{s['min_dist'][1]:.3f}  control {s['min_dist_control'][0]:.3f}"
              f"  gap {s['gap'][0]:+.3f}±{s['gap'][1]:.3f}  hit10 {s['hit'][0]:.2f}±{s['hit'][1]:.2f}"
              f"  still {s['hit_still'][0]:.2f}±{s['hit_still'][1]:.2f} lift {s['hit_still_lift'][0]:+.2f}"
              f"  paired {s['paired_gap'][0]:+.3f}±{s['paired_gap'][1]:.3f}  hitlift {s['hit_lift'][0]:+.2f}±{s['hit_lift'][1]:.2f}"
              f"  frac10 {s['frac_within'][0]:.3f}  steps {s['steps'][0]:.0f}±{s['steps'][1]:.0f}")


if __name__ == "__main__":
    main()
