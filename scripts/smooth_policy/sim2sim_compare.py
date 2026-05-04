"""Aggregate sim2sim campaign results into a comparison table.

Walks a ``runs/td3/sim2sim/<src_to_tgt>/`` directory:

- ``zero_shot/metrics.json`` → one row.
- Each ``<method>/seed<N>/`` subdir under ``full_ft/``, ``residual/``,
  ``from_scratch/`` → extract TensorBoard scalars and compute the same
  headline metrics as ``notes/scratch/extract_expl_metrics.py``.

Writes ``comparison.md`` into the campaign dir and prints the table to
stdout.

Usage
-----
::

    python scripts/smooth_policy/sim2sim_compare.py \
        --campaign-dir runs/td3/sim2sim/sysid_hist4_to_heavy_puck \
        [--cutoff 100000]

See notes/scratch/sim2sim_infra_plan.md §6.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_CUTOFF = 100_000
METHOD_ORDER = ("full_ft", "residual", "from_scratch")


def _load_scalars(tb_dir: str) -> dict:
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        out[tag] = [(e.step, e.value) for e in events]
    return out


def _within(events, max_step):
    return [(s, v) for (s, v) in events if s <= max_step]


def _value_at_or_before(events, step):
    evs = _within(events, step)
    return evs[-1][1] if evs else None


def _max_value(events, max_step):
    evs = _within(events, max_step)
    return max(v for _, v in evs) if evs else None


def _tail_mean(events, max_step, n):
    evs = _within(events, max_step)
    if not evs:
        return None
    tail = evs[-n:]
    return sum(v for _, v in tail) / len(tail)


def _extract_tb_metrics(tb_dir: str, cutoff: int) -> dict:
    m = _load_scalars(tb_dir)
    ep_ret = m.get("charts/episodic_return", [])
    rolling = m.get("charts/rolling2k_avg_episode_return", [])
    max_ret = m.get("charts/max_episodic_return", [])
    pos = m.get("rewards/sampled_task_reward_positive_fraction", [])
    return {
        f"ret@{cutoff//1000}k": _value_at_or_before(rolling, cutoff),
        "tail10": _tail_mean(ep_ret, cutoff, 10),
        "tail50": _tail_mean(ep_ret, cutoff, 50),
        "max_ret": _max_value(max_ret if max_ret else ep_ret, cutoff),
        "pos_frac": _value_at_or_before(pos, cutoff),
    }


def _extract_zero_shot(metrics_path: Path) -> dict:
    with open(metrics_path, "r") as f:
        m = json.load(f)
    return {
        "ret@0": m.get("mean_return"),
        "tail10": m.get("tail10"),
        "tail50": None,   # zero-shot has no temporal training curve
        "max_ret": m.get("max_return"),
        "pos_frac": None,
        "_n_episodes": m.get("n_episodes"),
        "_std_return": m.get("std_return"),
    }


def _collect_method_seeds(method_dir: Path) -> list[Path]:
    if not method_dir.exists():
        return []
    return sorted(p for p in method_dir.iterdir() if p.is_dir())


def _mean_std(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    finite = [v for v in values if v is not None]
    if not finite:
        return None, None
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    return mean, var ** 0.5


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _fmt_mean_std(mean: Optional[float], std: Optional[float]) -> str:
    if mean is None:
        return "—"
    return f"{mean:.2f} ± {std:.2f}" if std is not None else f"{mean:.2f}"


def aggregate(campaign_dir: Path, cutoff: int) -> dict:
    result = {"campaign_dir": str(campaign_dir), "cutoff": cutoff, "rows": []}

    zero_shot_json = campaign_dir / "zero_shot" / "metrics.json"
    if zero_shot_json.exists():
        zs = _extract_zero_shot(zero_shot_json)
        result["rows"].append({
            "method": "zero_shot",
            "seed": "—",
            "metrics": zs,
            "summary": False,
        })

    for method in METHOD_ORDER:
        seed_dirs = _collect_method_seeds(campaign_dir / method)
        per_seed_metrics = []
        for seed_dir in seed_dirs:
            try:
                metrics = _extract_tb_metrics(str(seed_dir), cutoff)
            except Exception as exc:
                print(f"  WARN: failed to read {seed_dir}: {exc}", file=sys.stderr)
                continue
            per_seed_metrics.append(metrics)
            result["rows"].append({
                "method": method,
                "seed": seed_dir.name,
                "metrics": metrics,
                "summary": False,
            })
        if len(per_seed_metrics) >= 2:
            keys = set().union(*(m.keys() for m in per_seed_metrics))
            summary_metrics = {}
            for k in keys:
                values = [m.get(k) for m in per_seed_metrics]
                mean, std = _mean_std(values)
                summary_metrics[k] = (mean, std)
            result["rows"].append({
                "method": method,
                "seed": f"mean±std (n={len(per_seed_metrics)})",
                "metrics": summary_metrics,
                "summary": True,
            })

    return result


def render_markdown(result: dict) -> str:
    cutoff = result["cutoff"]
    ret_col = f"ret@{cutoff//1000}k"
    cols = [ret_col, "tail10", "tail50", "max_ret", "pos_frac"]
    header = "| method | seed | " + " | ".join(cols) + " |"
    sep = "|" + " --- |" * (len(cols) + 2)
    lines = [
        f"# sim2sim comparison — {result['campaign_dir']}",
        "",
        f"Cutoff: {cutoff:,} timesteps (training runs); zero-shot cell under "
        f"`ret@{cutoff//1000}k` holds mean return over eval episodes.",
        "",
        header,
        sep,
    ]
    for row in result["rows"]:
        method, seed = row["method"], row["seed"]
        metrics = row["metrics"]
        cells = []
        for c in cols:
            v = metrics.get(c) if c in metrics else metrics.get(ret_col)
            if method == "zero_shot" and c == ret_col:
                v = metrics.get("ret@0")
            if row["summary"]:
                mean, std = (v if isinstance(v, tuple) else (v, None))
                cells.append(_fmt_mean_std(mean, std))
            else:
                cells.append(_fmt(v))
        name_bold = f"**{method}**" if row["summary"] else method
        lines.append(f"| {name_bold} | {seed} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Aggregate sim2sim campaign results.")
    p.add_argument("--campaign-dir", required=True,
                   help="Path to runs/td3/sim2sim/<src_to_tgt>/")
    p.add_argument("--cutoff", type=int, default=DEFAULT_CUTOFF,
                   help=f"Training-timestep cutoff for metrics (default {DEFAULT_CUTOFF}).")
    p.add_argument("--out", default=None,
                   help="Path to write comparison.md (default: inside campaign dir).")
    args = p.parse_args(argv)

    campaign_dir = Path(args.campaign_dir)
    if not campaign_dir.exists():
        print(f"ERROR: campaign dir {campaign_dir} does not exist.", file=sys.stderr)
        return 1

    result = aggregate(campaign_dir, args.cutoff)
    md = render_markdown(result)
    print(md)

    out_path = Path(args.out) if args.out else campaign_dir / "comparison.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(md)
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
