"""Old-vs-new trainer throughput comparison (thin wrapper over run_experiments).

Runs every config in `--args-dir` once with the current checkout ("new") and
once from an old checkout ("old", a git worktree of the pre-optimisation
commit), one job per GPU, then prints a table with per-config speedups. For
plain batches of experiments use `scripts/td3/run_experiments.py` directly.

Usage:
    python -m scripts.td3.extras.throughput_bench \
        --args-dir configs/td3/throughput_bench \
        --out-root runs/td3/throughput_bench \
        --old-worktree runs/td3/throughput_opt/_work/baseline_wt \
        --gpus 0 2 3 [--versions old new] [--only touch_dr ...] [--summarise-only]

Create the old worktree with
    git worktree add runs/td3/throughput_opt/_work/baseline_wt bf9936e
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

from scripts.td3.run_experiments import REPO_ROOT, build_jobs, run_jobs, summarise_job


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--args-dir", default="configs/td3/throughput_bench")
    ap.add_argument("--out-root", default="runs/td3/throughput_bench")
    ap.add_argument("--old-worktree", default="runs/td3/throughput_opt/_work/baseline_wt")
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 3])
    ap.add_argument("--versions", nargs="+", default=["old", "new"])
    ap.add_argument("--only", nargs="*", default=None, help="config-name substring filters")
    ap.add_argument("--summarise-only", action="store_true")
    cli = ap.parse_args()

    configs = sorted(glob.glob(os.path.join(cli.args_dir, "*.yaml")))
    if cli.only:
        configs = [c for c in configs if any(f in os.path.basename(c) for f in cli.only)]
    out_root = os.path.abspath(cli.out_root)
    jobs_by_version: Dict[str, List[Dict]] = {}
    for version in cli.versions:
        cwd = REPO_ROOT if version == "new" else os.path.abspath(cli.old_worktree)
        jobs_by_version[version] = build_jobs(
            [os.path.abspath(c) for c in configs], "auto", os.path.join(out_root, version), cwd
        )

    if not cli.summarise_only:
        # Old (slow) jobs first so the tail of the schedule is short.
        ordered = [j for v in ("old", "new") for j in jobs_by_version.get(v, [])]
        run_jobs(ordered, cli.gpus, extra_args=[])

    rows = {v: {j["name"]: summarise_job(j) for j in jobs} for v, jobs in jobs_by_version.items()}
    lines = [
        "| Job | Version | Wall (s) | Pre-learning SPS | Training SPS | Mean ep len | Final eval return |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted({n for r in rows.values() for n in r}):
        for version in ("old", "new"):
            r = rows.get(version, {}).get(name)
            if r is None:
                continue
            lines.append(
                f"| {name} | {version} | {r['wall_s']:.0f} | {r['pre_sps']:.0f} | {r['train_sps']:.0f} | {r['ep_len']:.0f} | {r['final_eval']:.1f} |"
            )
        if name in rows.get("old", {}) and name in rows.get("new", {}):
            o, n = rows["old"][name], rows["new"][name]
            lines.append(
                f"| {name} | **speedup** | **{o['wall_s'] / max(n['wall_s'], 1e-9):.1f}x** | "
                f"{n['pre_sps'] / max(o['pre_sps'], 1e-9):.1f}x | {n['train_sps'] / max(o['train_sps'], 1e-9):.1f}x | | |"
            )
    table = "\n".join(lines)
    print(table)
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
