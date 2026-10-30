"""Survival + hitting-motion evaluation of juggling policies on the target sim.

Runs each policy on the unchanged target sim (``configs/new_juggle/sysid_v2_hist4.yaml``) and
reports, besides survival, how the paddle hits the puck. Positions are the true simulator state
in the robot frame the real logs use (``paddle_actual``): x in [x_min_lim = -0.83, -0.45]; the
top edge of the paddle workspace is x = -0.83 and "forward" means toward it.

Per hit (control step with a paddle-puck contact):
    dtop    distance of the paddle to the top edge (m)
    vfwd    forward paddle speed over the control step that ends in the contact (m/s)
    cock    backward travel in the 12 steps (0.6 s) before the hit: max x - x 12 steps before (m)
    swing   forward travel of the strike: max x in that window - x at the hit (m)
    pout    puck speed after the hit (m/s)
Per episode: fraction of steps with the paddle within 2 cm of the top edge.

Real-robot reference, original hist4 policies (69 hits, real eval logs of 2026-10-20): the paddle
is ~21 cm from the edge at contact (10th-90th pct 14-29 cm), 0.8 m/s forward, 15 cm swing, and
within 2 cm of the edge 4 % of the time.

    python -m scripts.ceiling.hit_motion --label <name> --episodes 10 --horizon 2000 <policy_dir> ...

Writes ``runs/ceiling/hit_motion/<label>/<policy>.json`` and prints one line per policy.
``--edge-velocity-fix`` applies the counterfactual top-edge velocity fix (``scripts/rma/edge_velocity_fix.py``).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
TARGET_CONFIG = REPO / "configs/new_juggle/sysid_v2_hist4.yaml"
OUT_ROOT = REPO / "runs/ceiling/hit_motion"
NEAR_EDGE = 0.02
COCK_WINDOW = 12


def _load_agent(run_dir, env):
    import torch
    from scripts.td3.eval_utils import augment_policy_observation, build_policy_env_view, load_policy_for_evaluation

    args = yaml.safe_load((run_dir / "args.yaml").read_text())
    use_last = bool(args.get("use_last_action_in_policy_state", False))
    if (run_dir / "rma_meta.json").exists():
        from scripts.rma.bundle import load_phase1
        from scripts.rma.evaluate import HistoryAgent

        bundle = load_phase1(str(run_dir))
        if bundle["mode"] != "history":
            raise ValueError(f"{run_dir}: needs a deployable history actor")
        agent = HistoryAgent(bundle["actor"])
        return agent.reset, lambda obs, last: agent.act(np.asarray(obs), last)
    view = build_policy_env_view(SimpleNamespace(single_observation_space=env.observation_space,
                                                 single_action_space=env.action_space), use_last)
    policy = load_policy_for_evaluation(str(run_dir / "model.pth"), view, 1,
                                        int(args["agent_hidden_layer_size"]), int(args["agent_num_hidden_layers"]))

    def act(obs, last):
        state = augment_policy_observation(torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0),
                                           torch.as_tensor(last, dtype=torch.float32).reshape(1, 2), use_last)
        return policy(state).cpu().numpy().reshape(2)

    return (lambda obs: None), act


def evaluate_policy(run_dir, episodes, horizon, seed_base, edge_velocity_fix=False):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import torch
    from airhockey import AirHockeyEnv
    from scripts.rma.edge_velocity_fix import install_edge_velocity_fix

    torch.set_num_threads(1)
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(TARGET_CONFIG.read_text())["air_hockey"]
    probe = AirHockeyEnv(copy.deepcopy(cfg))
    reset_agent, act = _load_agent(run_dir, probe)
    probe.close()
    xtop = float(cfg["simulator_params"]["x_min_lim"])
    episodes_out, hits = [], []
    for e in range(episodes):
        seed = seed_base + e
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        params = copy.deepcopy(cfg)
        params.update(seed=seed, max_timesteps=horizon + 1)
        params["simulator_params"]["seed"] = seed
        env = AirHockeyEnv(params)
        if edge_velocity_fix:
            install_edge_velocity_fix(env)
        obs, _ = env.reset(seed=seed)
        sim = env.simulator
        reset_agent(obs)
        paddle = sim.paddles["paddle_ego"]
        last = np.zeros(2)
        xs = [-paddle.position[1] - sim.center_offset_constant]
        ret, steps, failed = 0.0, 0, False
        for step in range(1, horizon + 1):
            a = np.clip(act(obs, last), env.action_space.low, env.action_space.high)
            obs, r, term, trunc, info = env.step(a)
            last = a.copy()
            ret += float(np.asarray(r).reshape(-1)[0])
            steps = step
            x = -paddle.position[1] - sim.center_offset_constant
            xs.append(x)
            if int(info.get("paddle_puck_collision_count", 0)) > 0:
                puck = next(iter(sim.pucks.values()))
                win = np.asarray(xs[max(0, len(xs) - 1 - COCK_WINDOW):])
                hits.append(dict(dtop=x - xtop, vfwd=-(xs[-1] - xs[-2]) / sim.time_per_step,
                                 cock=float(win.max() - win[0]), swing=float(win.max() - x),
                                 pout=float(np.linalg.norm(puck.linearVelocity))))
            if term or trunc:
                failed = True
                break
        env.close()
        xs = np.asarray(xs)
        episodes_out.append(dict(seed=seed, steps=steps, ret=ret, failed=failed,
                                 near_edge_frac=float(np.mean(xs - xtop < NEAR_EDGE)),
                                 mean_dtop=float(np.mean(xs - xtop))))
    return summarize(str(run_dir), episodes_out, hits, horizon)


def _mean_se(values):
    v = np.asarray(values, float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    return [float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")]


def summarize(run_dir, episodes, hits, horizon):
    h = {k: [x[k] for x in hits] for k in ("dtop", "vfwd", "cock", "swing", "pout")}
    return {
        "run_dir": run_dir, "horizon": horizon, "episodes": episodes, "n_hits": len(hits),
        "mean_steps": _mean_se([e["steps"] for e in episodes]),
        "reached_cap": float(np.mean([e["steps"] >= horizon and not e["failed"] for e in episodes])),
        "return_per_step": float(sum(e["ret"] for e in episodes) / sum(e["steps"] for e in episodes)),
        "near_edge_frac": _mean_se([e["near_edge_frac"] for e in episodes]),
        "hits": {k: _mean_se(v) for k, v in h.items()},
        "hit_dtop_median": float(np.median(h["dtop"])) if hits else float("nan"),
        "edge_hit_frac": float(np.mean(np.asarray(h["dtop"]) < NEAR_EDGE)) if hits else float("nan"),
        "pout_median": float(np.median(h["pout"])) if hits else float("nan"),
        "pout_frac_above_4": float(np.mean(np.asarray(h["pout"]) > 4.0)) if hits else float("nan"),
    }


def line(s):
    hh = s["hits"]
    return (f"steps {s['mean_steps'][0]:.0f} | cap {s['reached_cap']:.0%} | rew/step {s['return_per_step']:.3f} | "
            f"near-edge time {s['near_edge_frac'][0]:.1%} | hits {s['n_hits']}: edge {s['edge_hit_frac']:.0%}, "
            f"dtop med {s['hit_dtop_median'] * 100:.1f} cm, vfwd {hh['vfwd'][0]:.2f}, swing {hh['swing'][0] * 100:.1f} cm, "
            f"cock {hh['cock'][0] * 100:.1f} cm, pout {hh['pout'][0]:.2f} (med {s['pout_median']:.2f}, "
            f">4 m/s {s['pout_frac_above_4']:.0%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--label", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=2000)
    parser.add_argument("--seed-base", type=int, default=900000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--edge-velocity-fix", action="store_true")
    cli = parser.parse_args()
    out = OUT_ROOT / cli.label
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=min(cli.workers, len(cli.run_dirs))) as pool:
        futures = {r: pool.submit(evaluate_policy, r, cli.episodes, cli.horizon, cli.seed_base,
                              cli.edge_velocity_fix) for r in cli.run_dirs}
        for run_dir, fut in futures.items():
            s = fut.result()
            name = "__".join(Path(run_dir).resolve().relative_to(REPO).parts)
            (out / f"{name}.json").write_text(json.dumps(s, indent=2) + "\n")
            print(f"{run_dir}: {line(s)}", flush=True)


if __name__ == "__main__":
    main()
