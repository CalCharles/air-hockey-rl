#!/usr/bin/env python3
"""Paired sim2sim eval: source sim vs the same sim with the real-robot occlusion map.

    .venv/bin/python -m scripts.td3.extras.eval_occlusion_map_sim2sim \
        --runs runs/td3/sysid_v2_real_workspace/hist4_seed*/juggle_sysid_v2_hist4/checkpoint_1975000 \
        --source-config configs/new_juggle/sysid_v2_hist4.yaml \
        --target-config configs/new_juggle/sysid_v2_hist4_occlusion_map.yaml \
        --episodes 10 --out-dir runs/td3/sysid_v2_real_workspace/occlusion_map_sim2sim

Every policy (one checkpoint dir per seed: model.pth + args.yaml) runs the SAME ``--episodes`` reset seeds
(``--seed-base`` + i) in both envs, so source / target differences are paired. Return = summed env reward,
as in the training-time checkpoint evals. Per env the table reports the mean ± standard error ACROSS
POLICIES of each policy's mean return over its episodes.

Also logged per env: the fraction of steps whose newest puck observation is occluded and the occlusion
run lengths; in the target env, the map model's realised P(occluded | cell) against the map (calibration
check of the sim implementation).

Writes ``<out-dir>/results.json`` and ``<out-dir>/summary.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

PUCK_NEWEST_FLAG = 29  # obs[27:29] newest puck xy, obs[29] its occluded flag (history obs, 5 frames)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)["air_hockey"]


def run_episodes(policy, cfg, args_yaml, seeds, map_stats=None):
    env = AirHockeyEnv(cfg)
    sim = env.simulator
    use_last = bool(args_yaml["use_last_action_in_policy_state"])
    out = []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        last_action = torch.zeros((1, 2), dtype=torch.float32)
        ret, steps, flags, done = 0.0, 0, [], False
        while not done:
            o = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = policy(augment_policy_observation(o, last_action, use_last)).numpy().squeeze()
            obs, rew, term, trunc, _ = env.step(action)
            done = bool(term or trunc)
            ret += float(rew)
            steps += 1
            flags.append(bool(obs[PUCK_NEWEST_FLAG] > 0.5))
            last_action = torch.tensor(action, dtype=torch.float32).reshape(1, -1)
            if map_stats is not None and getattr(sim, "occlusion_map", None) is not None:
                cell = sim._occ_map_cell.get("puck_0")
                if cell is not None:
                    map_stats.setdefault(cell, [0, 0])
                    map_stats[cell][0] += 1
                    map_stats[cell][1] += int(sim._occ_map_occluded.get("puck_0", False))
        out.append(dict(seed=int(s), ret=ret, steps=steps, flags=flags))
    env.close()
    return out


def run_lengths(flag_lists):
    runs = []
    for flags in flag_lists:
        n = 0
        for f in flags + [False]:
            if f:
                n += 1
            elif n:
                runs.append(n)
                n = 0
    return runs


def mean_se(xs):
    xs = np.asarray(xs, float)
    se = xs.std(ddof=1) / np.sqrt(len(xs)) if len(xs) > 1 else float("nan")
    return float(xs.mean()), float(se)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="checkpoint dirs (model.pth + args.yaml), one per seed")
    ap.add_argument("--source-config", required=True)
    ap.add_argument("--target-config", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--out-dir", required=True)
    cli = ap.parse_args()

    seeds = [cli.seed_base + i for i in range(cli.episodes)]
    envs = {"source": load_cfg(cli.source_config), "target": load_cfg(cli.target_config)}
    results, map_stats = {}, {}
    for run in cli.runs:
        run = os.path.normpath(run)
        with open(os.path.join(run, "args.yaml")) as f:
            args_yaml = yaml.safe_load(f)
        probe = gym.vector.SyncVectorEnv([lambda: AirHockeyEnv(envs["source"])])
        view = build_policy_env_view(probe, bool(args_yaml["use_last_action_in_policy_state"]))
        probe.close()
        policy = load_policy_for_evaluation(
            model_path=os.path.join(run, "model.pth"),
            policy_env_view=view,
            action_scale=1,
            agent_hidden_layer_size=int(args_yaml["agent_hidden_layer_size"]),
            agent_num_hidden_layers=int(args_yaml["agent_num_hidden_layers"]),
        )
        name = Path(run).parts[-3] if len(Path(run).parts) >= 3 else run
        results[name] = {
            env_name: run_episodes(policy, cfg, args_yaml, seeds, map_stats if env_name == "target" else None)
            for env_name, cfg in envs.items()
        }
        s, t = (np.mean([e["ret"] for e in results[name][k]]) for k in ("source", "target"))
        print(f"{name}: source {s:.1f}  target {t:.1f}", flush=True)

    names = list(results)
    per = {k: [np.mean([e["ret"] for e in results[n][k]]) for n in names] for k in ("source", "target")}
    lens = {k: [np.mean([e["steps"] for e in results[n][k]]) for n in names] for k in ("source", "target")}
    delta = [t - s for s, t in zip(per["source"], per["target"])]
    occ = {}
    for k in ("source", "target"):
        flags = [e["flags"] for n in names for e in results[n][k]]
        allf = np.concatenate([np.asarray(f, bool) for f in flags])
        rl = np.asarray(run_lengths(flags))
        occ[k] = dict(frac=float(allf.mean()), runs=int(len(rl)), mean_run=float(rl.mean()) if len(rl) else 0.0,
                      max_run=int(rl.max()) if len(rl) else 0,
                      hist=[int((rl == i).sum()) for i in range(1, 8)] + [int((rl >= 8).sum())])

    lines = [
        f"# Occlusion-map sim2sim eval ({len(names)} policies × {cli.episodes} paired episodes per env)",
        "",
        f"- source: `{cli.source_config}`",
        f"- target: `{cli.target_config}`",
        f"- reset seeds {seeds[0]}…{seeds[-1]} (identical for every policy and both envs)",
        "",
        "## Return: mean ± standard error across policies (each policy = mean of its episodes)",
        "",
        "| | source sim | target sim (+ occlusion map) | target − source |",
        "|---|---|---|---|",
    ]
    ms, mt, md = mean_se(per["source"]), mean_se(per["target"]), mean_se(delta)
    lines.append(f"| **return** | **{ms[0]:.1f} ± {ms[1]:.1f}** | **{mt[0]:.1f} ± {mt[1]:.1f}** | {md[0]:+.1f} ± {md[1]:.1f} |")
    ls_, lt = mean_se(lens["source"]), mean_se(lens["target"])
    lines.append(f"| episode length | {ls_[0]:.1f} ± {ls_[1]:.1f} | {lt[0]:.1f} ± {lt[1]:.1f} | |")
    lines += ["", "## Per policy (mean ± SE over its episodes)", "", "| policy | source return | target return | Δ |", "|---|---|---|---|"]
    for n, s, t in zip(names, per["source"], per["target"]):
        se_s = mean_se([e["ret"] for e in results[n]["source"]])[1]
        se_t = mean_se([e["ret"] for e in results[n]["target"]])[1]
        lines.append(f"| {n} | {s:.1f} ± {se_s:.1f} | {t:.1f} ± {se_t:.1f} | {t - s:+.1f} |")
    lines += ["", "## Puck observation occlusion (newest puck frame in the obs)", "",
              "| | occluded steps | runs | mean run | max run | run lengths 1,2,…,7,≥8 |", "|---|---|---|---|---|---|"]
    for k in ("source", "target"):
        o = occ[k]
        lines.append(f"| {k} | {o['frac']*100:.1f} % | {o['runs']} | {o['mean_run']:.2f} | {o['max_run']} | {o['hist']} |")
    if map_stats:
        tgt_map = AirHockeyEnv(envs["target"]).simulator.occlusion_map["p"]
        err = [(c, n, k / n, tgt_map[c]) for c, (n, k) in map_stats.items() if n >= 20]
        if err:
            gap = np.array([abs(r - p) for _, _, r, p in err])
            w = np.array([n for _, n, _, _ in err], float)
            lines += ["", "## Map calibration (target env, map model alone)", "",
                      f"Cells with ≥ 20 visited steps: {len(err)}; step-weighted mean |realised − map| = "
                      f"{np.average(gap, weights=w)*100:.1f} pp (sampling noise expected at these counts)."]
    text = "\n".join(lines) + "\n"
    os.makedirs(cli.out_dir, exist_ok=True)
    Path(cli.out_dir, "summary.md").write_text(text)
    Path(cli.out_dir, "results.json").write_text(json.dumps(
        {n: {k: [{kk: vv for kk, vv in e.items() if kk != "flags"} for e in v] for k, v in r.items()}
         for n, r in results.items()}, indent=1))
    print(text)


if __name__ == "__main__":
    main()
