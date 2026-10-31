"""Open-loop replay of real-robot policy actions in the Box2D sim (paddle only).

For each real episode, the sim paddle starts at the real paddle's first logged position (at rest)
and receives the exact logged ``policy_action`` sequence; the puck is made a sensor so it cannot
push the paddle. Compares the sim paddle trajectory with the real one: position error, speed
ratio, and the gain / lag that best map the sim's per-step displacement onto the real one
(real_d[t] ~ g * sim_d[t - lag]).

    python -m scripts.analysis.replay_real_actions --real <run_dir> ... [--sim-override '{"enable_observation_delay": false}'] [--action-shift k]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from scripts.analysis.cmd_vs_realized import DT, load_real

REPO = Path(__file__).resolve().parents[2]


def replay_episode(ep, cfg, shift=0):
    from airhockey import AirHockeyEnv

    env = AirHockeyEnv(copy.deepcopy(cfg))
    env.reset(seed=0)
    sim = env.simulator
    off = np.array([sim.center_offset_constant, getattr(sim, "center_offset_constant_y", 0.0)])
    for puck in sim.pucks.values():
        for fx in puck.fixtures:
            fx.sensor = True
    paddle = sim.paddles["paddle_ego"]
    base0 = ep["p"][0] + off
    paddle.position = tuple(float(v) for v in sim._base_to_box2d_coords(base0))
    paddle.linearVelocity = (0.0, 0.0)
    for name in ("pose_hist", "dpose_hist"):
        dq = getattr(sim, name, None)
        if dq is not None:
            dq.clear()
    if hasattr(sim.pid_controller, "reset"):
        sim.pid_controller.reset()
    sim.last_action = np.zeros(2)
    ps = []
    acts = np.vstack([np.zeros((shift, 2)), ep["a"]])[:len(ep["a"])] if shift else ep["a"]
    for a in acts[1:]:
        sim.get_transition(np.asarray(a, float))
        sim.last_action = sim.convert_to_box2d_coords(np.asarray(a, float))
        ps.append(sim._box2d_to_base_coords(np.array(paddle.position, float)) - off)
    env.close()
    return np.vstack([ep["p"][:1], np.asarray(ps)])


def gain_lag(real_d, sim_d, max_lag=8):
    best = None
    for lag in range(0, max_lag + 1):
        r = real_d[lag:]
        s = sim_d[:len(sim_d) - lag] if lag else sim_d
        g = float((r * s).sum() / max((s * s).sum(), 1e-12))
        err = float(((r - g * s) ** 2).mean())
        r2 = 1 - err / float(r.var() + 1e-12)
        if best is None or r2 > best[2]:
            best = (lag, g, r2)
    return best


def _job(job):
    ep, cfg, shift = job
    return replay_episode(ep, cfg, shift)


def analyze(name, eps, cfg, workers, shift=0):
    with ProcessPoolExecutor(workers) as ex:
        sims = list(ex.map(_job, [(e, cfg, shift) for e in eps]))
    rows = []
    rd_all, sd_all = {0: [], 1: []}, {0: [], 1: []}
    for e, sp in zip(eps, sims):
        rp = e["p"]
        n = min(len(rp), len(sp))
        rp, sp = rp[:n], sp[:n]
        rd, sd = np.diff(rp, axis=0), np.diff(sp, axis=0)
        for ax in (0, 1):
            rd_all[ax].append(rd[:, ax]); sd_all[ax].append(sd[:, ax])
        rows.append(dict(rmse=np.sqrt(((rp - sp) ** 2).mean(0)).tolist(),
                         real_speed=float(np.linalg.norm(rd, axis=1).mean() / DT),
                         sim_speed=float(np.linalg.norm(sd, axis=1).mean() / DT)))
    out = dict(n_eps=len(eps),
               rmse_cm=(100 * np.mean([r["rmse"] for r in rows], axis=0)).tolist(),
               real_speed=float(np.mean([r["real_speed"] for r in rows])),
               sim_speed=float(np.mean([r["sim_speed"] for r in rows])))
    for ax, nm in ((0, "x"), (1, "y")):
        lag, g, r2 = gain_lag(np.concatenate(rd_all[ax]), np.concatenate(sd_all[ax]))
        out[f"lag_{nm}"], out[f"gain_{nm}"], out[f"r2_{nm}"] = lag, g, r2
    out["speed_ratio_real_over_sim"] = out["real_speed"] / out["sim_speed"]
    print(f"{name:40s} eps {out['n_eps']:3d} | speed real {out['real_speed']:.2f} sim {out['sim_speed']:.2f} "
          f"(real/sim {out['speed_ratio_real_over_sim']:.2f}) | rmse x {out['rmse_cm'][0]:.1f} y {out['rmse_cm'][1]:.1f} cm "
          f"| best map x: lag {out['lag_x']} gain {out['gain_x']:.2f} r2 {out['r2_x']:.2f}; "
          f"y: lag {out['lag_y']} gain {out['gain_y']:.2f} r2 {out['r2_y']:.2f}", flush=True)
    return out, sims


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="+", required=True)
    ap.add_argument("--sim-config", default=str(REPO / "configs/new_juggle/sysid_v2_hist4.yaml"))
    ap.add_argument("--sim-override", default="{}")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--action-shift", type=int, default=0, help="feed the sim a_{t-k}: k control steps of extra latency")
    ap.add_argument("--label", default="target")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.sim_config).read_text())["air_hockey"]
    cfg["simulator_params"].update(json.loads(args.sim_override))
    cfg["max_timesteps"] = 100000
    res = {}
    for r in args.real:
        name = os.path.relpath(r, REPO / "data").replace(
            "sysid_v2_real_workspace_original_hist4_ckpt1975000_newest_frame_maxr12_no_estop_count/", "")
        res[name], _ = analyze(f"[{args.label}] {name[:30]}", load_real(r), cfg, args.workers, args.action_shift)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(args=vars(args), results=res), indent=1))


if __name__ == "__main__":
    main()
