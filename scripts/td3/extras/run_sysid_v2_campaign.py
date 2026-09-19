#!/usr/bin/env python3
"""Run the sysid-v2 policy campaign: every (task, method) cell of configs/td3/tasks_v2/manifest.json across GPUs.

    .venv/bin/python scripts/td3/extras/run_sysid_v2_campaign.py --gpus 0 1 2 3 --jobs-per-gpu 2 \
        --out-root runs/td3/sysid_v2_20260919            # everything; ~1 day on 4 GPUs
    ... --tasks juggle --methods sysid dr5_full          # a subset
    ... --resume                                         # skip finished cells, relaunch unfinished ones
    ... --summarise-only                                 # rebuild summary.md from what is on disk
    ... --final-eval-only                                # (re)run the paired final evaluation only
    ... -- --total-timesteps 30000 --learning-starts 2000 --checkpoint-interval 10000   # forwarded to every trainer (smoke test)

Jobs are launched in manifest order (juggle first, then the long tasks), at most `--jobs-per-gpu` per GPU
at a time. A cell's trainer is picked from the manifest (`scripts.td3.td3_training`, `td3_training_dr`,
`td3_training_her`, `scripts.rma.train_base_policy`); an RMA cell gets a second job (phase 2,
`scripts.rma.train_adaptation_module --phase1-dir <run dir>` -> `<run dir>/phase2`) that runs as soon as
its phase 1 finished. Each job's stdout goes to `<out-root>/<name>.stdout.log`, its run dir to
`<out-root>/<name>/` and `run_meta.json` (wall time, return code, GPU) next to it. `status.md` in the
out-root is rewritten after every job. When everything is done: `summary.md` (per-cell training table)
and `final_eval/` (scripts/td3/extras/eval_sysid_v2_campaign.py: every final policy on the same nominal
v2 sim and on the same fixed DR envs, identical episode seeds).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from scripts.td3.run_experiments import summarise_job  # noqa: E402

DEFAULT_MANIFEST = "configs/td3/tasks_v2/manifest.json"
TASK_ORDER = ["juggle", "puck_vel", "puck_goal", "puck_goal_vel", "touch", "reach", "reach_vel"]


class AdoptedProc:
    """A trainer launched by an earlier runner invocation and still alive: polled by pid, never signalled."""

    def __init__(self, pid: int, done_marker: str) -> None:
        self.pid, self.done_marker, self.returncode = int(pid), done_marker, None

    def poll(self) -> Optional[int]:
        if self.returncode is not None:
            return self.returncode
        alive = os.path.isdir(f"/proc/{self.pid}")
        if alive:
            try:
                with open(f"/proc/{self.pid}/status") as f:
                    alive = not any(line.startswith("State:") and "Z" in line.split()[1] for line in f)
            except OSError:
                alive = False
        if alive:
            return None
        self.returncode = 0 if os.path.isfile(self.done_marker) else 1
        return self.returncode


def find_live_pid(run_dir: str) -> Optional[tuple]:
    """(pid, gpu, start time) of a live trainer whose ``--log-parent-dir`` is exactly ``run_dir``."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        argv = [a.decode(errors="replace") for a in argv]
        if "--log-parent-dir" in argv and argv[argv.index("--log-parent-dir") + 1] == run_dir and "-m" in argv:
            gpu = None
            if "--device" in argv:
                dev = argv[argv.index("--device") + 1]
                gpu = int(dev.split(":")[1]) if ":" in dev else None
            return int(entry), gpu, os.stat(f"/proc/{entry}").st_ctime
    return None


class Job:
    def __init__(self, name: str, module: str, argv: List[str], run_dir: str, deps: List[str], kind: str, cell: Dict):
        self.name, self.module, self.argv, self.run_dir, self.deps, self.kind, self.cell = name, module, argv, run_dir, deps, kind, cell
        self.status = "pending"        # pending | running | done | failed | skipped
        self.rc: Optional[int] = None
        self.wall: float = float("nan")
        self.gpu: Optional[int] = None

    @property
    def meta_path(self) -> str:
        return os.path.join(self.run_dir, "run_meta.json")

    def finished_on_disk(self) -> bool:
        if self.kind == "phase2":
            return os.path.isfile(os.path.join(self.run_dir, "phase2_summary.json"))
        if not os.path.isfile(self.meta_path) or not os.path.isfile(os.path.join(self.run_dir, "model.pth")):
            return False
        try:
            return int(json.load(open(self.meta_path)).get("returncode", 1)) == 0
        except (OSError, ValueError):
            return False


def build_jobs(manifest: Dict, out_root: str, tasks: List[str], methods: List[str]) -> List[Job]:
    cells = [c for c in manifest["cells"] if c["task"] in tasks and c["method"] in methods]
    cells.sort(key=lambda c: (TASK_ORDER.index(c["task"]), manifest["methods"].index(c["method"])))
    jobs: List[Job] = []
    for c in cells:
        run_dir = os.path.join(out_root, c["name"])
        args_file = os.path.join(REPO, c["args_file"])
        sim_cfg = os.path.join(REPO, c["sim_config"])
        jobs.append(Job(c["name"], c["trainer"], ["--args-file", args_file, "--config", sim_cfg, "--log-parent-dir", run_dir],
                        run_dir, [], "train", c))
        if c.get("phase2_args_file"):
            p2_dir = os.path.join(run_dir, "phase2")
            jobs.append(Job(c["name"] + "_phase2", "scripts.rma.train_adaptation_module",
                            ["--args-file", os.path.join(REPO, c["phase2_args_file"]), "--phase1-dir", run_dir, "--log-parent-dir", p2_dir],
                            p2_dir, [c["name"]], "phase2", c))
    return jobs


def write_status(jobs: List[Job], out_root: str, t0: float) -> None:
    lines = [f"# sysid v2 campaign — status ({time.strftime('%Y-%m-%d %H:%M:%S')}, {(time.time() - t0) / 3600:.1f} h since launch)", "",
             "| job | status | gpu | wall (h) | rc |", "|---|---|---|---:|---|"]
    for j in jobs:
        wall = "" if j.wall != j.wall else f"{j.wall / 3600:.2f}"
        lines.append(f"| {j.name} | {j.status} | {'' if j.gpu is None else j.gpu} | {wall} | {'' if j.rc is None else j.rc} |")
    counts = {s: sum(j.status == s for j in jobs) for s in ("done", "running", "pending", "failed", "skipped")}
    lines += ["", ", ".join(f"{k}: {v}" for k, v in counts.items())]
    with open(os.path.join(out_root, "status.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


class _NoLog:
    def close(self) -> None:
        pass


def run_jobs(jobs: List[Job], gpus: List[int], jobs_per_gpu: int, extra: List[str], out_root: str, poll_s: float = 10.0,
             adopted: Optional[Dict[str, tuple]] = None) -> None:
    t0 = time.time()
    by_name = {j.name: j for j in jobs}
    running: Dict[str, tuple] = {}
    for name, (proc, t_start) in (adopted or {}).items():
        job = by_name[name]
        job.status = "running"
        running[name] = (proc, job, t_start, _NoLog())
    write_status(jobs, out_root, t0)
    while any(j.status in ("pending", "running") for j in jobs):
        # reap
        for name, (proc, job, t_start, log) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            job.wall, job.rc = time.time() - t_start, proc.returncode
            job.status = "done" if proc.returncode == 0 else "failed"
            os.makedirs(job.run_dir, exist_ok=True)
            with open(job.meta_path, "w") as f:
                json.dump({"wall_s": job.wall, "returncode": job.rc, "gpu": job.gpu, "args_file": job.argv[1], "module": job.module,
                           "extra_args": extra, "kind": job.kind, "adopted": isinstance(proc, AdoptedProc)}, f)
            print(f"[done ] {job.name} on gpu{job.gpu}: {job.wall / 3600:.2f} h rc={job.rc}", flush=True)
            del running[name]
            write_status(jobs, out_root, t0)
        # skip dependents of failures
        for j in jobs:
            if j.status == "pending" and any(by_name[d].status in ("failed", "skipped") for d in j.deps):
                j.status = "skipped"
                print(f"[skip ] {j.name}: dependency failed", flush=True)
        # launch: short phase-2 jobs first, then manifest order
        load = {g: sum(1 for (_, jb, _, _) in running.values() if jb.gpu == g) for g in gpus}
        for (_, jb, _, _) in running.values():       # adopted jobs on GPUs outside --gpus still count nowhere
            if jb.gpu not in load and jb.gpu is not None:
                load[jb.gpu] = load.get(jb.gpu, 0) + 1
        runnable = [j for j in jobs if j.status == "pending" and all(by_name[d].status == "done" for d in j.deps)]
        runnable.sort(key=lambda j: (0 if j.kind == "phase2" else 1))
        for job in runnable:
            free = [g for g in gpus if load[g] < jobs_per_gpu]
            if not free:
                break
            gpu = min(free, key=lambda g: load[g])
            load[gpu] += 1
            os.makedirs(os.path.dirname(job.run_dir), exist_ok=True)
            if os.path.exists(job.run_dir):          # unfinished leftover: keep it aside, the trainers refuse to reuse a dir
                shutil.move(job.run_dir, job.run_dir + "_failed_" + time.strftime("%Y%m%d-%H%M%S"))
            log = open(job.run_dir + ".stdout.log", "w")
            cmd = [sys.executable, "-u", "-m", job.module, *job.argv, "--device", f"cuda:{gpu}", *(extra if job.kind == "train" else [])]
            env = os.environ.copy()
            env["PYTHONPATH"] = REPO + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env)
            job.status, job.gpu = "running", gpu
            running[job.name] = (proc, job, time.time(), log)
            print(f"[start] {job.name} on gpu{gpu} ({job.module})", flush=True)
        write_status(jobs, out_root, t0)
        time.sleep(poll_s)
    write_status(jobs, out_root, t0)


def _phase2_row(job: Job) -> Dict:
    row = {"name": job.name, "rc": None, "wall_s": float("nan"), "final_eval": float("nan"), "latent_r2": float("nan")}
    try:
        row.update({k: v for k, v in json.load(open(job.meta_path)).items() if k in ("wall_s", "returncode")})
        row["rc"] = row.pop("returncode", None)
    except (OSError, ValueError):
        pass
    p = os.path.join(job.run_dir, "eval_final", "multi_env_eval.json")
    if os.path.isfile(p):
        d = json.load(open(p))
        row["final_eval"] = float(d["aggregate"]["mean_return_across_envs"])
    p = os.path.join(job.run_dir, "phase2_summary.json")
    if os.path.isfile(p):
        d = json.load(open(p))
        for k in ("final_holdout_r2", "holdout_r2", "latent_r2"):
            if k in d:
                row["latent_r2"] = float(d[k]) if isinstance(d[k], (int, float)) else float("nan")
                break
    return row


def write_summary(jobs: List[Job], out_root: str) -> str:
    lines = ["# sysid v2 campaign — training summary", "",
             "| cell | rc | wall (h) | training SPS | mean ep len | final eval return |", "|---|---|---:|---:|---:|---:|"]
    for j in jobs:
        if j.kind == "train":
            r = summarise_job({"name": j.name, "args_file": j.argv[1], "log_parent_dir": j.run_dir})
            lines.append(f"| {j.name} | {r['rc']} | {r['wall_s'] / 3600:.2f} | {r['train_sps']:.0f} | {r['ep_len']:.0f} | {r['final_eval']:.1f} |")
        else:
            r = _phase2_row(j)
            lines.append(f"| {j.name} (adapted) | {r['rc']} | {r['wall_s'] / 3600:.2f} | – | – | {r['final_eval']:.1f} |")
    lines += ["", "Final eval return = the trainer's own last evaluation: multi-env eval mean on that run's 5 fixed DR envs (DR / long-history / RMA runs; "
                  "RMA phase 2 = the adapted policy's eval_final), goal_eval mean return (HER sysid / low25 runs) or the rolling training return over "
                  "the last 10 % of steps (plain sysid / low25 runs). These are NOT comparable across methods — use final_eval/summary.md "
                  "(same envs and episode seeds for every policy)."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write(text)
    return text


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        extra, argv = argv[i + 1:], argv[:i]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--jobs-per-gpu", type=int, default=2)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--resume", action="store_true", help="skip cells whose run dir is complete")
    ap.add_argument("--summarise-only", action="store_true")
    ap.add_argument("--final-eval-only", action="store_true")
    ap.add_argument("--no-final-eval", action="store_true")
    ap.add_argument("--final-eval-args", nargs="*", default=[], help="extra args for eval_sysid_v2_campaign.py")
    cli = ap.parse_args(argv)

    manifest = json.load(open(os.path.join(REPO, cli.manifest)))
    tasks = cli.tasks or manifest["tasks"]
    methods = cli.methods or manifest["methods"]
    out_root = os.path.abspath(cli.out_root)
    os.makedirs(out_root, exist_ok=True)
    jobs = build_jobs(manifest, out_root, tasks, methods)
    with open(os.path.join(out_root, "campaign.json"), "w") as f:
        json.dump({"manifest": cli.manifest, "tasks": tasks, "methods": methods, "gpus": cli.gpus, "jobs_per_gpu": cli.jobs_per_gpu,
                   "extra_trainer_args": extra, "jobs": [{"name": j.name, "module": j.module, "run_dir": j.run_dir, "kind": j.kind, "deps": j.deps} for j in jobs],
                   "launched": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
    print(f"{len(jobs)} jobs ({sum(j.kind == 'train' for j in jobs)} training + {sum(j.kind == 'phase2' for j in jobs)} RMA phase 2) "
          f"on GPUs {cli.gpus} x {cli.jobs_per_gpu} -> {out_root}", flush=True)

    if not cli.summarise_only and not cli.final_eval_only:
        adopted: Dict[str, tuple] = {}
        if cli.resume:
            for j in jobs:
                if j.finished_on_disk():
                    j.status, j.rc = "done", 0
                    try:
                        j.wall = float(json.load(open(j.meta_path)).get("wall_s", float("nan")))
                    except (OSError, ValueError):
                        pass
                elif os.path.isdir(j.run_dir):
                    live = find_live_pid(j.run_dir)
                    if live is not None:
                        pid, gpu, t_start = live
                        marker = os.path.join(j.run_dir, "phase2_summary.json" if j.kind == "phase2" else "model.pth")
                        j.gpu = gpu
                        adopted[j.name] = (AdoptedProc(pid, marker), t_start)
                        print(f"resume: adopting live {j.name} (pid {pid}, gpu {gpu})", flush=True)
            print(f"resume: {sum(j.status == 'done' for j in jobs)} jobs already complete, {len(adopted)} adopted", flush=True)
        run_jobs(jobs, cli.gpus, cli.jobs_per_gpu, extra, out_root, adopted=adopted)
    print(write_summary(jobs, out_root), flush=True)
    if cli.summarise_only or cli.no_final_eval:
        return
    cmd = [sys.executable, "-u", "-m", "scripts.td3.extras.eval_sysid_v2_campaign", "--out-root", out_root, "--manifest", cli.manifest,
           "--tasks", *tasks, "--methods", *methods, *cli.final_eval_args]
    print("[final eval] " + " ".join(cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = REPO + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"], env["OMP_NUM_THREADS"] = "", "1"
    with open(os.path.join(out_root, "final_eval.stdout.log"), "w") as log:
        rc = subprocess.call(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f"[final eval] rc={rc} -> {os.path.join(out_root, 'final_eval', 'summary.md')}", flush=True)


if __name__ == "__main__":
    main()
