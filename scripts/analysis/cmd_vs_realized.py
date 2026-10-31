"""Commanded vs realized paddle motion, real robot logs vs the target sim.

Per control step t (50 ms), commanded displacement c_t = a_t * move_lims (0.26, 0.12 m) and realized
displacement d_t = p_t - p_{t-1} (paddle position after the step minus before, robot frame). Per axis
we fit an FIR response d_t = sum_k h_k c_{t-k}, k = 0..K-1, by least squares; its sum (DC gain) is
the steady-state paddle displacement per metre of commanded displacement per step, and its shape
shows the lag. Also reports the action statistics (|a|, saturation) and realized paddle speed.

Real logs: ``<run>/episode_hdf5/**/trajectory_data*.hdf5`` (``policy_action``, ``paddle_actual``).
Sim: ``--sim <policy_dir> ...`` rolls the policies out on ``configs/new_juggle/sysid_v2_hist4.yaml``
(or ``--sim-config``) and logs the same quantities from the true paddle state.

    python -m scripts.analysis.cmd_vs_realized --real <run_dir> ... --sim <policy_dir> ... --out <json>
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
MOVE_LIMS = np.array([0.26, 0.12])
DT = 0.05
K = 10


def load_real(run_dir):
    eps = []
    for f in sorted(glob.glob(os.path.join(run_dir, "episode_hdf5", "**", "trajectory_data*.hdf5"), recursive=True)):
        if "/estop/" in f:
            continue
        import h5py
        with h5py.File(f, "r") as h:
            if "policy_action" not in h:
                continue
            a = np.asarray(h["policy_action"][:, :2], float)
            p = np.asarray(h["paddle_actual"][:, :2], float)
            pre = np.asarray(h["paddle_cmd"][:, :2], float)
            post = np.asarray(h["paddle_cmd"][:, 6:8], float)
        if len(a) > 20:
            eps.append(dict(a=a, p=p, pre=pre, post=post, src=f))
    return eps


def sim_rollouts(policy_dir, episodes, horizon, seed_base, config, sim_overrides):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    from airhockey import AirHockeyEnv
    from scripts.ceiling.hit_motion import _load_agent

    torch.set_num_threads(1)
    cfg = yaml.safe_load(Path(config).read_text())["air_hockey"]
    cfg["simulator_params"].update(sim_overrides or {})
    probe = AirHockeyEnv(copy.deepcopy(cfg))
    reset_agent, act = _load_agent(Path(policy_dir), probe)
    probe.close()
    out = []
    for e in range(episodes):
        seed = seed_base + e
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        params = copy.deepcopy(cfg)
        params.update(seed=seed, max_timesteps=horizon + 1)
        params["simulator_params"]["seed"] = seed
        env = AirHockeyEnv(params)
        obs, _ = env.reset(seed=seed)
        sim = env.simulator
        reset_agent(obs)
        paddle = sim.paddles["paddle_ego"]
        base = lambda: np.array([-paddle.position[1], paddle.position[0]], float)
        last = np.zeros(2)
        ps, acts = [base()], []
        for _ in range(horizon):
            a = np.clip(act(obs, last), env.action_space.low, env.action_space.high)
            obs, r, term, trunc, info = env.step(a)
            last = a.copy()
            acts.append(a.copy()); ps.append(base())
            if term or trunc:
                break
        env.close()
        # align with the real logs: row t = action of step t and paddle position after it
        out.append(dict(a=np.asarray(acts), p=np.asarray(ps[1:]), p0=ps[0]))
    return out


def fir_gain(eps, axis, k=K):
    X, Y = [], []
    for e in eps:
        c = e["a"][:, axis] * MOVE_LIMS[axis]
        p = e["p"][:, axis]
        d = np.diff(p)  # d[t-1] = p_t - p_{t-1}, caused by actions up to t
        for t in range(k, len(p)):
            X.append(c[t - k + 1:t + 1][::-1])  # c_t, c_{t-1}, ...
            Y.append(d[t - 1])
    X, Y = np.asarray(X), np.asarray(Y)
    h, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ h
    r2 = 1 - resid.var() / Y.var()
    return h, float(r2)


def summarize(eps):
    a = np.concatenate([e["a"] for e in eps])
    p = [e["p"] for e in eps]
    v = np.concatenate([np.diff(x, axis=0) / DT for x in p])
    sp = np.linalg.norm(v, axis=1)
    s = dict(n_eps=len(eps), n_steps=int(len(a)),
             abs_a=np.abs(a).mean(0).tolist(), sat_frac=(np.abs(a) > 0.95).mean(0).tolist(),
             a_p90=np.percentile(np.abs(a), 90, axis=0).tolist(),
             cmd_speed=float(np.linalg.norm(a * MOVE_LIMS, axis=1).mean() / DT),
             speed_mean=float(sp.mean()), speed_p90=float(np.percentile(sp, 90)), speed_max=float(sp.max()),
             vabs=np.abs(v).mean(0).tolist())
    for ax, name in ((0, "x"), (1, "y")):
        h, r2 = fir_gain(eps, ax)
        s[f"fir_{name}"] = h.tolist()
        s[f"gain_{name}"] = float(h.sum())
        s[f"r2_{name}"] = r2
        s[f"t50_{name}"] = int(np.argmax(np.cumsum(h) >= 0.5 * h.sum()))
    # realized / commanded ratio of mean absolute motion
    s["ratio_abs"] = (np.array(s["vabs"]) * DT / (np.array(s["abs_a"]) * MOVE_LIMS)).tolist()
    return s


def fmt(name, s):
    return (f"{name:42s} eps {s['n_eps']:3d} steps {s['n_steps']:5d} | |a| x {s['abs_a'][0]:.2f} y {s['abs_a'][1]:.2f} "
            f"sat x {s['sat_frac'][0]*100:3.0f}% y {s['sat_frac'][1]*100:3.0f}% | cmd {s['cmd_speed']:.2f} m/s "
            f"| speed mean {s['speed_mean']:.2f} p90 {s['speed_p90']:.2f} max {s['speed_max']:.2f} "
            f"| gain x {s['gain_x']:.2f} (r2 {s['r2_x']:.2f}, t50 {s['t50_x']}) y {s['gain_y']:.2f} (r2 {s['r2_y']:.2f}, t50 {s['t50_y']})")


def _sim_job(job):
    return job[0], summarize(sim_rollouts(*job[1:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="*", default=[])
    ap.add_argument("--sim", nargs="*", default=[])
    ap.add_argument("--sim-config", default=str(REPO / "configs/new_juggle/sysid_v2_hist4.yaml"))
    ap.add_argument("--sim-override", default="{}", help="JSON dict merged into simulator_params")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = {}
    for r in args.real:
        name = "real:" + os.path.relpath(r, REPO / "data")
        res[name] = summarize(load_real(r))
        print(fmt(name, res[name]), flush=True)
    jobs = [("sim:" + os.path.relpath(d, REPO), d, args.episodes, args.horizon, 700000, args.sim_config,
             json.loads(args.sim_override)) for d in args.sim]
    with ProcessPoolExecutor(max(1, min(args.workers, len(jobs) or 1))) as ex:
        for name, s in ex.map(_sim_job, jobs):
            res[name] = s
            print(fmt(name, s), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(args=vars(args), results=res), indent=1))


if __name__ == "__main__":
    main()
