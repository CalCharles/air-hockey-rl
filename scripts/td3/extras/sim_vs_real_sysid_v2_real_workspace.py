#!/usr/bin/env python3
"""Sim vs real table for the `data/sysid_v2_real_workspace/hist4_seed*` juggle policies.

    .venv/bin/python -m scripts.td3.extras.sim_vs_real_sysid_v2_real_workspace sim     # roll every seed in sim
    .venv/bin/python -m scripts.td3.extras.sim_vs_real_sysid_v2_real_workspace table   # (re)build the table

`sim`: every seed's checkpoint is rolled for `--episodes` episodes in the sim config stored next to it (nominal sysid v2,
hist_len 4) with the same episode seeds for every policy (paired). Juggles / contacts come from the real pipeline's
counter (`helper.juggle_counter`) applied to the newest observed paddle / puck frame, as in sim2sim_online_finetune.

`table`: reads `sim_results.json` and every real run listed in `real_runs.json` ({"<seed>": "<run dir with
eval_per_episode.jsonl>"}), and writes `table.md`: one row per metric, Sim / Real columns, each cell mean ± SE
(ddof=1, /sqrt(n)) over the per-seed means; the kept / attempts row sums over seeds.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

POLICY_ROOT = "data/sysid_v2_real_workspace"
CKPT = "checkpoint_1975000"
OUT_DIR = "results/sim_vs_real_sysid_v2_real_workspace"
METRICS = [("episode_return", "Episodic return", ".1f"), ("episode_length", "Episode length", ".1f"),
           ("episode_juggles", "Juggles", ".2f"), ("episode_contacts", "Contacts", ".2f"),
           ("episode_success", "Success rate", ".2f"), ("episode_juggle_success", "Juggle success rate", ".2f")]


def seeds_on_disk():
    dirs = glob.glob(os.path.join(REPO, POLICY_ROOT, "hist4_seed*", CKPT))
    return sorted(int(re.search(r"seed(\d+)", d).group(1)) for d in dirs)


def rollout_seed(seed: int, episodes: int, episode_seed: int):
    from airhockey import AirHockeyEnv
    from scripts.rma.evaluate import PlainTD3Agent
    from scripts.td3.helper.juggle_counter import count_juggles_from_arrays

    ckpt = os.path.join(REPO, POLICY_ROOT, f"hist4_seed{seed}", CKPT)
    with open(os.path.join(ckpt, "args.yaml")) as f:
        args = yaml.safe_load(f)
    with open(os.path.join(ckpt, "config.yaml")) as f:
        cfg = copy.deepcopy(yaml.safe_load(f)["air_hockey"])
    cfg["seed"] = int(episode_seed)
    env = AirHockeyEnv(cfg)
    obs_dim, act_dim = int(np.prod(env.observation_space.shape)), int(np.prod(env.action_space.shape))
    agent = PlainTD3Agent(os.path.join(ckpt, "model.pth"), obs_dim, act_dim, args["use_last_action_in_policy_state"],
                          args["agent_hidden_layer_size"], args["agent_num_hidden_layers"])
    episodes_out = []
    for _ in range(episodes):
        obs, _ = env.reset()
        obs_seq, last_action, ret, steps, done, info = [np.asarray(obs)], np.zeros(act_dim, np.float32), 0.0, 0, False, {}
        while not done:
            action = agent.act(np.asarray(obs, np.float32), last_action)
            obs, rew, term, trunc, info = env.step(action)
            ret += float(np.asarray(rew).reshape(-1)[0])
            steps += 1
            done = bool(term or trunc)
            last_action = np.asarray(action, np.float32).reshape(-1)
            obs_seq.append(np.asarray(obs))
        o = np.asarray(obs_seq)
        counts = count_juggles_from_arrays(o[:, 12:14], o[:, 27:29], o[:, 29])
        episodes_out.append({"episode_return": ret, "episode_length": steps,
                             "episode_juggles": int(counts.n_juggles), "episode_contacts": int(counts.n_contacts),
                             "episode_juggle_success": bool(counts.juggle_success),
                             "episode_success": bool(info.get("success", False))})
    env.close()
    return episodes_out


def mean_se(x):
    x = np.asarray(x, dtype=np.float64)
    se = float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return float(np.mean(x)), se


def fmt(m, se, f):
    return f"{m:{f}} ± {se:{f}}" if np.isfinite(se) else f"{m:{f}}"


def load_real(out_dir):
    path = os.path.join(out_dir, "real_runs.json")
    runs = json.load(open(path)) if os.path.isfile(path) else {}
    real, attempts = {}, {}
    for seed, run_dir in runs.items():
        with open(os.path.join(REPO, run_dir, "eval_per_episode.jsonl")) as f:
            real[int(seed)] = [json.loads(line) for line in f if line.strip()]
        meta = json.load(open(os.path.join(REPO, run_dir, "eval_summary.json")))["run_meta"]
        attempts[int(seed)] = int(meta["n_attempts"])
    return real, attempts, runs


def build_table(out_dir):
    sim = {int(k): v for k, v in json.load(open(os.path.join(out_dir, "sim_results.json")))["seeds"].items()}
    real, real_attempts, runs = load_real(out_dir)
    # Sim never discards an episode, so every rollout is a kept attempt.
    columns = (("Sim", sim, {s: len(eps) for s, eps in sim.items()}), ("Real", real, real_attempts))
    lines = ["# Sim vs real — `data/sysid_v2_real_workspace` hist4 juggle policies (checkpoint 1975000)", "",
             "Every cell: mean ± SE (ddof=1, /√n) over the per-seed means, n = seeds in that column (a single "
             "seed shows its mean only). Sim: nominal sysid v2 hist4 training sim, same 10 episode seeds for "
             "every policy. Real: kept episodes of `async_td3_real_eval` (stopped / short attempts excluded). "
             "Kept / attempts: summed over seeds.", "",
             "| Metric | Sim | Real |", "|---|---|---|"]
    for key, name, f in METRICS:
        cells = []
        for _, data, _ in columns:
            per_seed = [np.mean([float(e[key]) for e in data[s]]) for s in sorted(data)]
            cells.append(fmt(*mean_se(per_seed), f) if per_seed else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    cells = []
    for _, data, attempts in columns:
        kept, tried = sum(len(data[s]) for s in data), sum(attempts[s] for s in data)
        cells.append(f"{kept}/{tried} ({100 * kept / tried:.0f}%)" if tried else "—")
    lines += ["| Kept / attempts | " + " | ".join(cells) + " |", ""]
    for col, data, _ in columns:
        lines.append(f"- {col}: {len(data)} seed(s) ({', '.join(str(s) for s in sorted(data))}); "
                     + ", ".join(f"seed {s} n={len(data[s])} eps" for s in sorted(data)))
    lines.append("")
    lines += ["## Real runs", ""] + [f"- seed {s}: `{runs[str(s)]}`" for s in sorted(map(int, runs))] + [""]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["sim", "table"])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episode-seed", type=int, default=12345)
    ap.add_argument("--out-dir", default=OUT_DIR)
    cli = ap.parse_args()
    out_dir = os.path.join(REPO, cli.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if cli.mode == "sim":
        results = {}
        for seed in seeds_on_disk():
            results[str(seed)] = rollout_seed(seed, cli.episodes, cli.episode_seed)
            print(f"[sim] seed {seed}: return {fmt(*mean_se([e['episode_return'] for e in results[str(seed)]]), '.1f')}",
                  flush=True)
        with open(os.path.join(out_dir, "sim_results.json"), "w") as f:
            json.dump({"checkpoint": CKPT, "episodes": cli.episodes, "episode_seed": cli.episode_seed,
                       "seeds": results}, f, indent=1)
    build_table(out_dir)


if __name__ == "__main__":
    main()
