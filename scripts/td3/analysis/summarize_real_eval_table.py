"""Real-robot version of the sim "mean return ± SE / juggles / ep length" table.

    python -m scripts.td3.analysis.summarize_real_eval_table \
        runs/async_td3/Oct_1_Reproduce_table [--blocks 5] [--kept-only]

Each subdirectory of the root is one policy (one row); every
``data_*/episode_summaries.jsonl`` under it is pooled, so a run that was
restarted after a crash still lands in the same row. Reads
``episode_summaries.jsonl`` rather than ``eval_per_episode.jsonl`` because it
holds *every* attempt, including the short / e-stopped episodes the eval
validator discards, so e-stops count as trials instead of being replaced.
Run the eval with ``--eval-episodes N --eval-max-attempts N`` to get exactly N
attempts.

``--blocks K`` splits each policy's trials (in time order) into K consecutive
blocks, shown in place of the sim table's ``env 0..4`` columns. SE is the
standard error over episodes, as in ``scripts/rma/eval_paired.py``. The table
is printed and written to ``<root>/real_eval_table.md``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def _is_estop(row) -> bool:
    return bool(
        float(row.get("episode_estop_flag") or 0.0) > 0.0
        or row.get("had_protective_stop")
        or row.get("had_controller_disconnect")
        or row.get("readiness_fail_estop")
    )


def _load_rows(policy_dir: str, kept_only: bool):
    rows = []
    for path in glob.glob(os.path.join(policy_dir, "**", "episode_summaries.jsonl"), recursive=True):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if kept_only:
        rows = [r for r in rows if r.get("kept", True)]
    return sorted(rows, key=lambda r: float(r.get("wall_time_s", 0.0)))


def _mean_se(values):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return float(v.mean()), se


def _fmt(x, nd=1):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def summarise_policy(rows, blocks: int):
    returns = [float(r["episode_return"]) for r in rows]
    estops = [_is_estop(r) for r in rows]
    mean, se = _mean_se(returns)
    clean_mean, clean_se = _mean_se([ret for ret, e in zip(returns, estops) if not e])
    block_means = []
    if blocks > 0:
        for chunk in np.array_split(np.asarray(returns, dtype=float), blocks):
            block_means.append(float(chunk.mean()) if chunk.size else float("nan"))
    return {
        "n": len(rows),
        "blocks": block_means,
        "mean": mean,
        "se": se,
        "juggles": float(np.mean([float(r.get("episode_juggles", 0) or 0) for r in rows])) if rows else float("nan"),
        "ep_len": float(np.mean([float(r["episode_length"]) for r in rows])) if rows else float("nan"),
        "n_estop": int(sum(estops)),
        "clean_mean": clean_mean,
        "clean_se": clean_se,
    }


def render_table(results, blocks: int) -> str:
    header = ["Policy", "Trials"] + [f"block {i}" for i in range(blocks)] + [
        "Mean return ± SE", "Juggles / ep", "Ep length", "E-stops", "Return w/o e-stop ± SE",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for name, s in results:
        estop_pct = 100.0 * s["n_estop"] / s["n"] if s["n"] else float("nan")
        cells = [name, str(s["n"])] + [_fmt(b) for b in s["blocks"]] + [
            f"**{_fmt(s['mean'])} ± {_fmt(s['se'])}**",
            _fmt(s["juggles"]),
            _fmt(s["ep_len"], 0),
            f"{s['n_estop']} ({_fmt(estop_pct, 0)}%)",
            f"{_fmt(s['clean_mean'])} ± {_fmt(s['clean_se'])}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", help="Folder with one subfolder per policy (each holding data_*/ runs).")
    parser.add_argument("--blocks", type=int, default=5,
                        help="Split each policy's trials into this many consecutive blocks (0 = no block columns).")
    parser.add_argument("--kept-only", action="store_true",
                        help="Only count episodes the eval validator kept (drops short / early-e-stop episodes).")
    args = parser.parse_args()

    results = []
    for policy_dir in sorted(p for p in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(p)):
        rows = _load_rows(policy_dir, args.kept_only)
        if rows:
            results.append((os.path.basename(policy_dir), summarise_policy(rows, args.blocks)))
    if not results:
        raise SystemExit(f"no episode_summaries.jsonl found under {args.root}")

    table = render_table(results, args.blocks)
    print(table)
    out_path = os.path.join(args.root, "real_eval_table.md")
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
