"""Run a set of TD3 training configs across GPUs — the canonical batch entrypoint.

Give it a list of TD3 args YAMLs (files, directories or globs), say whether
they are DR runs (`td3_training_dr`) or plain runs (`td3_training`), and a
GPU list. Jobs are launched **in order, at most one job per GPU**; when a
job finishes, the next one takes its GPU. Each job's stdout goes to
`<out-root>/<name>.stdout.log` and its run directory to `<out-root>/<name>/`.
When everything is done a `summary.md` (wall-clock, phase steps/s, mean
episode length, final eval) is written to `<out-root>`.

Examples
--------
    # canonical DR recipe on five tasks, GPUs 0/2/3
    python -m scripts.td3.run_experiments --mode dr \
        --configs configs/td3/tasks/*_dr.yaml \
        --gpus 0 2 3 --out-root runs/td3/full_dr

    # a directory of no-DR configs, GPU 1 only
    python -m scripts.td3.run_experiments --mode nodr \
        --configs configs/td3/tasks/*_sysid.yaml \
        --gpus 1 --out-root runs/td3/full_nodr

    # just regenerate the summary table of a finished batch
    python -m scripts.td3.run_experiments --mode dr --configs ... \
        --out-root runs/td3/full_dr --summarise-only

Anything after `--` is forwarded verbatim to every trainer invocation
(e.g. `-- --total-timesteps 100000`). `--device` and `--log-parent-dir` are
always set by this script.

`--mode auto` picks DR for YAMLs that set `eval_param_seed`, plain otherwise.
`--cwd` runs the jobs from another checkout (used by
`scripts/td3/extras/throughput_bench.py` for old-vs-new comparisons); the
`config:` path inside each YAML is resolved against *this* repo.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# --------------------------------------------------------------------------- jobs
def expand_configs(items: List[str]) -> List[str]:
    """Files, directories (all *.yaml inside, sorted) and globs -> sorted file list."""
    files: List[str] = []
    for item in items:
        if os.path.isdir(item):
            files.extend(sorted(glob.glob(os.path.join(item, "*.yaml"))))
        else:
            matches = sorted(glob.glob(item))
            if not matches:
                raise FileNotFoundError(f"No config matches {item!r}")
            files.extend(matches)
    seen = set()
    out = []
    for f in files:
        f = os.path.abspath(f)
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def trainer_module(mode: str, args_file: str) -> str:
    if mode == "auto":
        mode = "dr" if yaml.safe_load(open(args_file)).get("eval_param_seed") is not None else "nodr"
    return {"dr": "scripts.td3.td3_training_dr", "nodr": "scripts.td3.td3_training"}[mode]


def build_jobs(configs: List[str], mode: str, out_root: str, cwd: str, name_prefix: str = "") -> List[Dict]:
    jobs = []
    for args_file in configs:
        name = name_prefix + os.path.splitext(os.path.basename(args_file))[0]
        jobs.append(
            dict(
                name=name,
                args_file=args_file,
                module=trainer_module(mode, args_file),
                log_parent_dir=os.path.join(out_root, name),
                cwd=cwd,
            )
        )
    return jobs


def run_jobs(jobs: List[Dict], gpus: List[int], extra_args: List[str], poll_s: float = 5.0) -> None:
    """Launch `jobs` in order, one per GPU at a time; block until all finish."""
    python = sys.executable
    pending = list(jobs)
    running: Dict[int, tuple] = {}
    while pending or running:
        for gpu, (proc, job, t0, log) in list(running.items()):
            if proc.poll() is None:
                continue
            wall = time.time() - t0
            log.close()
            os.makedirs(job["log_parent_dir"], exist_ok=True)
            with open(os.path.join(job["log_parent_dir"], "run_meta.json"), "w") as f:
                json.dump({"wall_s": wall, "returncode": proc.returncode, "gpu": gpu, "args_file": job["args_file"]}, f)
            print(f"[done ] {job['name']} on gpu{gpu}: {wall:.0f}s rc={proc.returncode}", flush=True)
            del running[gpu]
        free = [g for g in gpus if g not in running]
        while pending and free:
            gpu = free.pop(0)
            job = pending.pop(0)
            args = yaml.safe_load(open(job["args_file"]))
            # Never pre-create log_parent_dir: the trainer appends r<N> to an
            # existing directory. Keep stdout next to it instead.
            os.makedirs(os.path.dirname(job["log_parent_dir"]), exist_ok=True)
            log = open(job["log_parent_dir"] + ".stdout.log", "w")
            cmd = [
                python, "-u", "-m", job["module"],
                "--args-file", job["args_file"],
                "--config", os.path.join(REPO_ROOT, args["config"]) if not os.path.isabs(args["config"]) else args["config"],
                "--device", f"cuda:{gpu}",
                "--log-parent-dir", job["log_parent_dir"],
                *extra_args,
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = job["cwd"] + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            proc = subprocess.Popen(cmd, cwd=job["cwd"], stdout=log, stderr=subprocess.STDOUT, env=env)
            running[gpu] = (proc, job, time.time(), log)
            print(f"[start] {job['name']} on gpu{gpu}", flush=True)
        time.sleep(poll_s)


# ------------------------------------------------------------------------ summary
def _tb_curve(run_dir: str):
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if "charts/episodic_return" not in tags:
        return None
    ev = ea.Scalars("charts/episodic_return")
    steps = np.array([e.step for e in ev])
    wall = np.array([e.wall_time for e in ev])
    lens = np.array([e.value for e in ea.Scalars("charts/episodic_length")]) if "charts/episodic_length" in tags else None
    tail_ret = None
    if "charts/avg_episodic_return" in tags:
        tail_ret = [(e.step, e.value) for e in ea.Scalars("charts/avg_episodic_return")]
    return steps, wall, lens, tail_ret


def _phase_sps(steps, wall, lo, hi) -> float:
    m = (steps >= lo) & (steps < hi)
    if m.sum() < 3:
        return float("nan")
    return float((steps[m][-1] - steps[m][0]) / max(wall[m][-1] - wall[m][0], 1e-9))


def summarise_job(job: Dict) -> Dict:
    run_dir = job["log_parent_dir"]
    meta = {}
    for meta_name in ("run_meta.json", "bench_meta.json"):  # bench_meta: pre-2026-09-03 runner
        meta_path = os.path.join(run_dir, meta_name)
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            break
    args = yaml.safe_load(open(job["args_file"]))
    ls, total = int(args["learning_starts"]), int(args["total_timesteps"])
    row = dict(name=job["name"], wall_s=meta.get("wall_s", float("nan")), rc=meta.get("returncode"),
               pre_sps=float("nan"), train_sps=float("nan"), ep_len=float("nan"), final_eval=float("nan"))
    curve = _tb_curve(run_dir) if os.path.isdir(run_dir) else None
    if curve is not None:
        steps, wall, lens, tail_ret = curve
        row["pre_sps"] = _phase_sps(steps, wall, 0, ls)
        row["train_sps"] = _phase_sps(steps, wall, ls, total + 1)
        if lens is not None and len(lens):
            row["ep_len"] = float(np.mean(lens[steps >= ls])) if (steps >= ls).any() else float(np.mean(lens))
        evals = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*", "multi_env_eval.json")))
        if os.path.exists(os.path.join(run_dir, "multi_env_eval.json")):
            evals.append(os.path.join(run_dir, "multi_env_eval.json"))
        if evals:
            row["final_eval"] = float(json.load(open(evals[-1]))["aggregate"]["mean_return_across_envs"])
        elif tail_ret:
            tail = [v for s, v in tail_ret if s >= 0.9 * total]
            if tail:
                row["final_eval"] = float(np.mean(tail))
    return row


def summary_table(rows: List[Dict]) -> str:
    lines = [
        "| Run | rc | Wall (s) | Pre-learning SPS | Training SPS | Mean ep len | Final eval return |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['rc']} | {r['wall_s']:.0f} | {r['pre_sps']:.0f} | {r['train_sps']:.0f} | {r['ep_len']:.0f} | {r['final_eval']:.1f} |"
        )
    lines.append("")
    lines.append("Final eval = last checkpoint's multi-env eval mean (DR runs) or the rolling training return over the last 10 % of steps (no-DR runs).")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra_args: List[str] = []
    if "--" in argv:
        split = argv.index("--")
        extra_args = argv[split + 1:]
        argv = argv[:split]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", required=True, help="TD3 args YAML files, directories or globs")
    ap.add_argument("--mode", choices=["dr", "nodr", "auto"], default="auto",
                    help="dr -> td3_training_dr, nodr -> td3_training, auto -> by eval_param_seed")
    ap.add_argument("--gpus", nargs="+", type=int, required=True, help="GPU ids; one job per GPU at a time")
    ap.add_argument("--out-root", required=True, help="parent directory for all run dirs")
    ap.add_argument("--cwd", default=REPO_ROOT, help="checkout to run the trainer from (default: this repo)")
    ap.add_argument("--summarise-only", action="store_true")
    cli = ap.parse_args(argv)

    configs = expand_configs(cli.configs)
    out_root = os.path.abspath(cli.out_root)
    os.makedirs(out_root, exist_ok=True)
    jobs = build_jobs(configs, cli.mode, out_root, os.path.abspath(cli.cwd))
    print(f"{len(jobs)} job(s) on GPUs {cli.gpus} -> {out_root}", flush=True)

    if not cli.summarise_only:
        run_jobs(jobs, cli.gpus, extra_args)

    table = summary_table([summarise_job(j) for j in jobs])
    print(table)
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
