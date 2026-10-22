"""Train sustained juggling without changing canonical sysid-v2 sensing.

Three bounded trials in existing GPU allocations. Every simulator parameter,
including sensing, history sampling defaults, gains and workspace, is copied
unchanged from sysid_v2_hist2.yaml. Only the episode budget and the established
three-parameter physics DR are added. No clean-sensing curriculum is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

from scripts.td3.juggle_iterations import REPO, save_yaml, write_json

RECIPES = (("source_g0995", "latest_models/canonical/hist2_motion0_v2", .995),
           ("clean_warm_g0995", "runs/td3/juggle_iterations/iteration_02/long_discount", .995),
           ("source_g099", "latest_models/canonical/hist2_motion0_v2", .99))


def original_sensing_config():
    base = yaml.safe_load((REPO / "configs/new_juggle/sysid_v2_hist2.yaml").read_text())
    cfg = base["air_hockey"]
    dr = yaml.safe_load((REPO / "configs/new_juggle/tasks_v2/sim_dr3_juggle.yaml").read_text())["air_hockey"]
    cfg.update(max_timesteps=10000, domain_random=True,
               random_variables=copy.deepcopy(dr["random_variables"]),
               random_variable_ranges=copy.deepcopy(dr["random_variable_ranges"]))
    return base


def prepare():
    env_path = REPO / "configs/new_juggle/sustained/original_sensing.yaml"
    save_yaml(env_path, original_sensing_config())
    source = yaml.safe_load((REPO / "configs/td3/sustained/clean_upper.yaml").read_text())
    for name, model, gamma in RECIPES:
        args = copy.deepcopy(source)
        args.update(config=str(env_path.relative_to(REPO)), model_path=f"{model}/model.pth",
                    total_timesteps=1000000, gamma=gamma, run_name=name,
                    full_checkpoint_load="weights_only", survival_eval_scenarios="training",
                    survival_eval_steps=5000, survival_eval_episodes=8)
        save_yaml(REPO / f"configs/td3/sustained/original_{name}.yaml", args)


def assert_original_sensing(cfg):
    expected = original_sensing_config()["air_hockey"]
    if cfg != expected:
        raise ValueError("Original-sensing configuration changed; refusing to train/evaluate")


def worker(cli):
    root = Path(cli.root).resolve()
    run = root / cli.worker
    state_path = root / f"{cli.worker}_status.json"
    state = dict(status="baseline", started_unix=time.time(), allocation=os.environ.get("SLURM_JOB_ID"))
    write_json(state_path, state)
    try:
        args_file = root / f"{cli.worker}_args.yaml"
        args = yaml.safe_load(args_file.read_text())
        config = REPO / args["config"]
        assert_original_sensing(yaml.safe_load(config.read_text())["air_hockey"])
        # Same starts and original sensing for both initialization baselines.
        source_dir = str(Path(args["model_path"]).parent)
        with (root / f"{cli.worker}_baseline.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-m", "scripts.td3.juggle_survival",
                            "--run-dir", source_dir, "--config", str(config),
                            "--scenario-set", "training", "--episodes", "16", "--horizon", "5000",
                            "--seed-base", "20000", "--out-dir", str(root / f"{cli.worker}_baseline")],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["status"] = "training"
        write_json(state_path, state)
        with (root / f"{cli.worker}.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-m", "scripts.td3.td3_training",
                            "--args-file", str(args_file), "--log-parent-dir", str(run)],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["status"] = "validation"
        write_json(state_path, state)
        with (root / f"{cli.worker}_validation.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-m", "scripts.td3.juggle_survival",
                            "--run-dir", str(run), "--scenario-set", "training", "--episodes", "32",
                            "--horizon", "10000", "--seed-base", "100000",
                            "--out-dir", str(run / "validation")],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update(status="complete", finished_unix=time.time())
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        write_json(state_path, state)
        raise
    write_json(state_path, state)


def report(root):
    lines = ["# Original-sensing juggling", "",
             "Canonical sysid-v2 sensing and controller unchanged; three-parameter physics DR enabled.",
             "32 validation starts, seeds 100000–100031, 10,000-step cap.", "",
             "| Trial | Mean steps | Mean juggles | Genuine cap completion | GIF |",
             "|---|---:|---:|---:|---|"]
    scores = []
    for name, _, _ in RECIPES:
        path = root / name / "validation/results.json"
        if not path.exists():
            continue
        s = json.loads(path.read_text())["scenarios"]["training"]["summary"]
        lines.append(f"| {name} | {s['restricted_mean_steps']:.1f} | {s['mean_juggles']:.1f} | "
                     f"{s['sustained_juggling_fraction']:.1%} | [rollout]({name}/validation/training.gif) |")
        scores.append(((s["sustained_juggling_fraction"], s["mean_valid_juggling_steps"]), name))
    (root / "results.md").write_text("\n".join(lines) + "\n")
    return max(scores)[1] if scores else None


def supervise(cli):
    root = Path(cli.root).resolve()
    state = dict(status="running", started_unix=time.time(), allocations=cli.job_ids)
    write_json(root / "status.json", state)
    try:
        processes = []
        for (name, _, _), allocation in zip(RECIPES, cli.job_ids):
            args = yaml.safe_load((REPO / f"configs/td3/sustained/original_{name}.yaml").read_text())
            save_yaml(root / f"{name}_args.yaml", args)
            cmd = ["srun", f"--jobid={allocation}", "--overlap", "--exact", "--nodes=1", "--ntasks=1",
                   "--cpus-per-task=6", "--cpu-bind=none", "--gres=gpu:1", "--immediate=15",
                   f"--chdir={REPO}", "/usr/bin/env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
                   "MPLCONFIGDIR=/tmp/airhockey-mpl", sys.executable, "-u", "-m",
                   "scripts.td3.juggle_original_sensing", "--worker", name, "--root", str(root)]
            with (root / f"{name}_launcher.log").open("w") as log:
                processes.append(subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes):
            report(root)
            time.sleep(20)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("A trial failed; inspect worker status/logs")
        winner = report(root)
        state.update(status="complete", selected_trial=winner, finished_unix=time.time())
        branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
        if branch == "codex/sustained-juggling":
            stamp = time.strftime("%Y-%m-%d_%H-%M", time.gmtime())
            note = REPO / f"notes/scratch/experiments/{stamp}_original-sensing-juggle.md"
            if not note.exists():
                relative = os.path.relpath(root, note.parent)
                body = (root / "results.md").read_text().replace("](", f"]({relative}/")
                note.write_text(f"- Status: done; bounded simulation experiments, not indefinite survival.\n"
                                f"- Selected trial: {winner}; selection uses these validation starts.\n"
                                f"- Each trial: one training seed, 1M steps; three-parameter physics DR.\n\n" + body)
                subprocess.run(["git", "add", "--", str(note)], cwd=REPO, check=True)
                subprocess.run(["git", "commit", "--only", "-m", "Record original-sensing juggling results",
                                "--", str(note)], cwd=REPO, check=True)
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        write_json(root / "status.json", state)
        raise
    write_json(root / "status.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--worker", choices=[r[0] for r in RECIPES])
    parser.add_argument("--root", default="runs/td3/original_sensing")
    parser.add_argument("--job-ids", nargs=3, type=int, default=[106217, 106218, 106219])
    cli = parser.parse_args()
    if cli.prepare:
        prepare()
    elif cli.worker:
        worker(cli)
    elif cli.supervise:
        supervise(cli)
    else:
        if len(set(cli.job_ids)) != 3:
            parser.error("Three distinct existing allocations are required")
        root = Path(cli.root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (root / "launch.json").exists():
            raise FileExistsError("Campaign already launched")
        with (root / "supervisor.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-u", "-m", "scripts.td3.juggle_original_sensing",
                                     "--supervise", "--root", str(root), "--job-ids", *map(str, cli.job_ids)],
                                    cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(root / "launch.json", dict(pid=proc.pid, started_unix=time.time()))
        print(f"Original-sensing supervisor PID {proc.pid}; artifacts: {root}")


if __name__ == "__main__":
    main()
