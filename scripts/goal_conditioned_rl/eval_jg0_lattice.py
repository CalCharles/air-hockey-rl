"""Offline lattice eval of a jg0 policy (CPU): the 5 x 7 upper-half goals x N launches, same env, goals, launch seeds
(900000 + launch), horizon and deterministic head vote as the in-training eval of train_jg0.py, plus stricter
stop metrics. Distances and speeds use the true Box2D puck state; within = puck centre within RADIUS (10 cm).

    .venv-jax/bin/python -m scripts.goal_conditioned_rl.eval_jg0_lattice --policy runs/crl/jg0_best_s0/best_policy.msgpack \
        --launches 20 --out runs/crl/jg0_eval/best_22M.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--launches", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--hold-steps", type=int, default=10, help="consecutive slow-in-circle steps for 'held'")
    a = ap.parse_args()

    import jax.numpy as jnp
    from scripts.goal_conditioned_rl import jg0
    from scripts.goal_conditioned_rl.crl_env import load_sim_config, make_env, policy_state, puck_goal
    from scripts.goal_conditioned_rl.train_jg0 import RADIUS, STILL_SPEED, lattice_goals

    actor = jg0.Learner.load(a.policy)
    lattice = np.asarray(lattice_goals(5, 7), np.float32)
    items = [(g, 900_000 + l) for l in range(a.launches) for g in lattice]
    cfg = load_sim_config(max_timesteps=7000)
    n = len(items)
    envs, obs = [], []
    for _, sd in items:
        e = make_env(dict(cfg, max_timesteps=a.horizon + 1), sd, 2)
        o, _ = e.reset(seed=sd)
        envs.append(e)
        obs.append(o)
    G = np.stack([g for g, _ in items]).astype(np.float32)
    last = np.zeros((n, 2), np.float32)
    alive = np.ones(n, bool)
    min_d, min_v = np.full(n, np.inf), np.full(n, np.inf)
    steps, within, slow_in, run, best_run = (np.zeros(n, int) for _ in range(5))
    dropped = np.zeros(n, bool)
    for _ in range(a.horizon):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        # Fixed batch shape (all n lanes, finished ones ignored): a shrinking batch recompiles the policy for every
        # new size and exhausts XLA's CPU JIT memory.
        S = np.stack([policy_state(obs[i], last[i]) for i in range(n)]).astype(np.float32)
        A = np.clip(np.asarray(actor.sample_actions(jnp.asarray(S), jnp.asarray(G))), -1, 1).astype(np.float32)
        for i in idx:
            o, _, term, trunc, _ = envs[i].step(A[i])
            obs[i], last[i] = o, A[i]
            pg = puck_goal(envs[i])
            d, v = float(np.linalg.norm(pg[:2] - G[i, :2])), float(np.linalg.norm(pg[2:]))
            min_d[i] = min(min_d[i], d)
            steps[i] += 1
            if d < RADIUS:
                within[i] += 1
                min_v[i] = min(min_v[i], v)
            slow = d < RADIUS and v < STILL_SPEED
            slow_in[i] += slow
            run[i] = run[i] + 1 if slow else 0
            best_run[i] = max(best_run[i], run[i])
            if term or trunc:
                alive[i] = False
                dropped[i] = bool(term)
    for e in envs:
        e.close()

    def ms(x):
        x = np.asarray(x, float)
        return [float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))]

    hit = min_d < RADIUS
    res = dict(policy=a.policy, episodes=n, launches=a.launches, horizon=a.horizon,
               hit10=ms(hit), hit5=ms(min_d < 0.05), stop=ms(best_run >= 1),
               held=ms(best_run >= a.hold_steps), hold_steps=a.hold_steps,
               min_dist=ms(min_d), ep_steps=ms(steps), dropped=ms(dropped), survived_full=ms(steps >= a.horizon),
               frac_time_within=ms(within / np.maximum(steps, 1)), steps_within=ms(within),
               slowest_in_circle_given_hit=ms(min_v[hit]) if hit.sum() > 1 else None,
               per_launch_hits=[int(hit[l * len(lattice):(l + 1) * len(lattice)].sum()) for l in range(a.launches)])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
