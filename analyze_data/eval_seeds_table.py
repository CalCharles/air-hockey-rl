#!/usr/bin/env python3
"""Evaluate the latest checkpoint of every seed of one or more TD3 run groups in the Box2D sim and
print a per-seed table (mean ± standard error over episodes) plus a per-group row (mean ± standard
error across the seed means), with each trial's return / episode length in its own column.

    .venv/bin/python analyze_data/eval_seeds_table.py \
        --group hist2=runs/td3/sysid_v2_hist_len_2_train_step_2M \
        --group hist4=runs/td3/sysid_v2_hist_len_4_train_step_2M \
        --episodes 5 --out runs/td3/sysid_v2_hist_seed_eval

A group directory holds `seed*/<run>/` folders (the layout `run_experiments.py` writes per seed). In
each seed folder the run with the newest weights is used (a resumed run lands in `<run>r1/`). "Latest
checkpoint" = the run's top-level `model.pth`, which the trainer writes at the end of training (the
3M-step weights after the 2M -> 3M resume); `--checkpoint-step N` uses `checkpoint_N/model.pth` instead.

Each policy runs on its own run's `config.yaml` (so hist2 policies get the hist_len-2 sim and hist4
the hist_len-4 sim) with the env seed overridden by `--eval-seed`, so every policy sees the same
sequence of episode start states. Writes `<out>/seed_eval.json` and `<out>/seed_eval.md`.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from airhockey import AirHockeyEnv  # noqa: E402
from scripts.rma.evaluate import PlainTD3Agent  # noqa: E402

DEFAULT_GROUPS = [
    "hist2=runs/td3/sysid_v2_hist_len_2_train_step_2M",
    "hist4=runs/td3/sysid_v2_hist_len_4_train_step_2M",
]


def _yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def mean_se(values: List[float]) -> tuple[float, float]:
    """Mean and standard error of the mean (sample std / sqrt(n); 0 for n = 1)."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan")
    se = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return float(x.mean()), se


def fmt(values: List[float], digits: int = 1) -> str:
    m, se = mean_se(values)
    return f"{m:.{digits}f} ± {se:.{digits}f}"


def find_policy(seed_dir: str, checkpoint_step: int | None) -> tuple[str, str]:
    """(run_dir, model_path) of the run in `seed_dir` with the newest weights."""
    candidates = []
    for run_dir in sorted(glob.glob(os.path.join(seed_dir, "*", ""))):
        run_dir = run_dir.rstrip("/")
        if not os.path.exists(os.path.join(run_dir, "args.yaml")):
            continue
        if checkpoint_step is None:
            model = os.path.join(run_dir, "model.pth")
        else:
            model = os.path.join(run_dir, f"checkpoint_{checkpoint_step}", "model.pth")
        if os.path.exists(model):
            candidates.append((os.path.getmtime(model), run_dir, model))
    if not candidates:
        raise FileNotFoundError(f"no run with {'model.pth' if checkpoint_step is None else f'checkpoint_{checkpoint_step}'} under {seed_dir}")
    _, run_dir, model = max(candidates)
    return run_dir, model


def evaluate_policy(run_dir: str, model_path: str, episodes: int, eval_seed: int) -> Dict[str, Any]:
    args = _yaml(os.path.join(run_dir, "args.yaml"))
    air_hockey = copy.deepcopy(_yaml(os.path.join(run_dir, "config.yaml"))["air_hockey"])
    air_hockey["seed"] = int(eval_seed)
    env = AirHockeyEnv(air_hockey)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    agent = PlainTD3Agent(
        model_path,
        obs_dim,
        act_dim,
        bool(args.get("use_last_action_in_policy_state", True)),
        int(args.get("agent_hidden_layer_size", 64)),
        int(args.get("agent_num_hidden_layers", 2)),
    )

    returns, lengths, successes = [], [], []
    for _ in range(int(episodes)):
        obs, _ = env.reset()
        last_action = np.zeros(act_dim, dtype=np.float32)
        done, ret, steps, info = False, 0.0, 0, {}
        while not done:
            action = agent.act(np.asarray(obs, dtype=np.float32), last_action)
            obs, rew, term, trunc, info = env.step(action)
            ret += float(np.asarray(rew, dtype=np.float64).reshape(-1)[0])
            steps += 1
            done = bool(term or trunc)
            last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        returns.append(ret)
        lengths.append(steps)
        successes.append(float(bool(info.get("success", False))))
    env.close()
    return {
        "run_dir": run_dir,
        "model_path": model_path,
        "hist_len": air_hockey.get("simulator_params", {}).get("hist_len"),
        "returns": returns,
        "episode_lengths": lengths,
        "successes": successes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", default=None, metavar="LABEL=DIR",
                    help="run group: label and directory containing seed*/ folders (repeatable)")
    ap.add_argument("--episodes", type=int, default=5, help="episodes per seed")
    ap.add_argument("--eval-seed", type=int, default=0, help="env seed shared by every policy (paired episodes)")
    ap.add_argument("--checkpoint-step", type=int, default=None,
                    help="evaluate checkpoint_<N>/model.pth instead of the final top-level model.pth")
    ap.add_argument("--out", default="runs/td3/sysid_v2_hist_seed_eval")
    cli = ap.parse_args()

    groups = [g.split("=", 1) for g in (cli.group or DEFAULT_GROUPS)]
    results: Dict[str, Dict[str, Any]] = {}
    for label, group_dir in groups:
        seed_dirs = sorted(glob.glob(os.path.join(group_dir, "seed*")), key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
        if not seed_dirs:
            raise FileNotFoundError(f"no seed* folders under {group_dir}")
        results[label] = {}
        for seed_dir in seed_dirs:
            seed = os.path.basename(seed_dir)
            run_dir, model = find_policy(seed_dir, cli.checkpoint_step)
            print(f"[{label} {seed}] {os.path.relpath(model, REPO)}", flush=True)
            r = evaluate_policy(run_dir, model, cli.episodes, cli.eval_seed)
            print(f"    return {fmt(r['returns'])}  len {fmt(r['episode_lengths'])}", flush=True)
            results[label][seed] = r

    ckpt = "final model.pth" if cli.checkpoint_step is None else f"checkpoint_{cli.checkpoint_step}"
    ep_cols = range(1, cli.episodes + 1)
    lines = [
        f"Sim eval: {ckpt} per seed, {cli.episodes} episodes each, env seed {cli.eval_seed} (trial k is the same start state for every policy).",
        "Trial k: that episode's return (episode length in parentheses; ✗ = no success).",
        "Seed rows: Return / Episode length / Success = mean ± SE over the trials.",
        "Group-average rows: every column = mean ± SE across the per-seed values (per-seed means for the last three).",
        "",
        "| Group | Seed | " + " | ".join(f"Trial {k}" for k in ep_cols) + " | Return | Episode length | Success |",
        "|---|---|" + "---:|" * (cli.episodes + 3),
    ]
    for label, seeds in results.items():
        for seed, r in seeds.items():
            cells = [f"{ret:.1f} ({n}){'' if ok else ' ✗'}" for ret, n, ok in zip(r["returns"], r["episode_lengths"], r["successes"])]
            lines.append(f"| {label} | {seed} | " + " | ".join(cells)
                         + f" | {fmt(r['returns'])} | {fmt(r['episode_lengths'])} | {fmt(r['successes'], 2)} |")
    for label, seeds in results.items():
        per_seed = lambda k: [float(np.mean(r[k])) for r in seeds.values()]  # noqa: E731
        trials = [fmt([r["returns"][k] for r in seeds.values() if k < len(r["returns"])]) for k in range(cli.episodes)]
        lines.append(f"| **{label} avg** | all ({len(seeds)}) | " + " | ".join(trials)
                     + f" | **{fmt(per_seed('returns'))}** | {fmt(per_seed('episode_lengths'))} | {fmt(per_seed('successes'), 2)} |")
    lines += ["", "Checkpoints:"] + [f"- {label} {seed}: `{os.path.relpath(r['model_path'], REPO)}`"
                                     for label, seeds in results.items() for seed, r in seeds.items()]
    table = "\n".join(lines)
    print("\n" + table)

    os.makedirs(cli.out, exist_ok=True)
    with open(os.path.join(cli.out, "seed_eval.md"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(cli.out, "seed_eval.json"), "w") as f:
        json.dump({"episodes": cli.episodes, "eval_seed": cli.eval_seed, "checkpoint": ckpt, "results": results}, f, indent=2)
    print(f"\nwrote {cli.out}/seed_eval.md and seed_eval.json")


if __name__ == "__main__":
    main()
