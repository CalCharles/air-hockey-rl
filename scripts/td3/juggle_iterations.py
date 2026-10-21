"""Two adaptive iterations of clean-simulation juggling, with a GIF for every trial.

All orchestration runs in the main process, not subagents. Independent training
jobs run in parallel inside existing allocations. Iteration 2 waits for the
iteration-1 validation results, then continues the winner and tests replication
and a longer discount horizon. No control gains or limits are increased.
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

REPO = Path(__file__).resolve().parents[2]
NAMES = ("upper", "survival", "memory")


def save_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def prepare():
    base = yaml.safe_load((REPO / "configs/new_juggle/sysid_v2_hist2.yaml").read_text())
    cfg = base["air_hockey"]
    cfg.update(max_timesteps=10000, domain_random=False,
               random_variables=["paddle_density", "puck_damping", "gravity"],
               random_variable_ranges={"paddle_density": [2250., 3750.],
                                       "puck_damping": [.0825, .1375], "gravity": [-.9125, -.5475]})
    cfg["simulator_params"].update(
        history_sample_once_per_step=True, puck_noise=False,
        enable_random_occlusions=False, occlusion_map_path=None,
        enable_observation_delay=False, enable_action_delay=False,
        enable_puck_delay_interpolation=False, puck_obs_lag_seconds=0., puck_obs_lag_frame_period=0.,
    )
    source = yaml.safe_load((REPO / "configs/td3/sustained/g099.yaml").read_text())
    for name in NAMES:
        env = copy.deepcopy(base)
        if name != "upper":
            env["air_hockey"].update(enable_survival_bonus=True, survival_bonus_per_step=.25, puck_loss_penalty=5.)
        config_path = REPO / f"configs/new_juggle/sustained/clean_{name}.yaml"
        save_yaml(config_path, env)
        args = copy.deepcopy(source)
        args.update(config=str(config_path.relative_to(REPO)), total_timesteps=300000,
                    run_name=f"clean_{name}", seed=0, gamma=.99,
                    success_buffer_size=100000, watch_episode_interval=1000,
                    survival_eval_steps=5000, survival_eval_episodes=8,
                    survival_eval_scenarios="clean", actor_action_delta_penalty=.005)
        if name == "memory":
            args.update(rma_mode="history", rma_latent_dim=8, history_len=50,
                        step_features="latest_frame", history_embed_dim=32, history_conv_channels=32,
                        history_warm_start_plain=True, eval_param_seed=12345)
        save_yaml(REPO / f"configs/td3/sustained/clean_{name}.yaml", args)


def worker(cli):
    task = json.loads(Path(cli.worker).read_text())
    run = Path(task["run_dir"])
    run.parent.mkdir(parents=True, exist_ok=True)
    state_path = run.parent / f"{run.name}_status.json"
    state = dict(task=task, status="training", started_unix=time.time())
    write_json(state_path, state)
    try:
        with run.with_suffix(".log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-m", task["trainer"],
                            "--args-file", task["args_file"], "--log-parent-dir", str(run)],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["status"] = "validation"
        write_json(state_path, state)
        with run.with_suffix(".eval.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-m", "scripts.td3.juggle_survival",
                            "--run-dir", str(run), "--scenario-set", "clean", "--episodes", "16",
                            "--horizon", "5000", "--seed-base", "20000",
                            "--out-dir", str(run / "validation")],
                           cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update(status="complete", finished_unix=time.time())
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        write_json(state_path, state)
        raise
    write_json(state_path, state)


def task_score(task):
    path = Path(task["run_dir"]) / "validation/results.json"
    s = json.loads(path.read_text())["scenarios"]["clean"]["summary"]
    return s["sustained_juggling_fraction"], s["mean_valid_juggling_steps"], s["restricted_mean_steps"]


def report(root, iteration, tasks):
    lines = [f"# Juggling iteration {iteration}", "",
             "Clean Box2D; unchanged sysid-v2 controller and workspace. Validation seeds 20000–20015, 5,000-step cap.", "",
             "| Trial | Mean steps | Mean juggles | Genuine cap completion | Start GIF | End/failure GIF |",
             "|---|---:|---:|---:|---|---|"]
    for task in tasks:
        run = Path(task["run_dir"])
        summary_path = run / "validation/results.json"
        if not summary_path.exists():
            continue
        try:
            value = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        s = value["scenarios"]["clean"]["summary"]
        rel = os.path.relpath(run / "validation", root)
        lines.append(f"| {run.name} | {s['restricted_mean_steps']:.1f} | {s['mean_juggles']:.1f} | "
                     f"{s['sustained_juggling_fraction']:.1%} | [start]({rel}/clean_start.gif) | [end]({rel}/clean.gif) |")
    (root / f"iteration_{iteration:02d}.md").write_text("\n".join(lines) + "\n")


def record_iteration(root, iteration, tasks):
    """Commit only the new result note and an otherwise-clean index, if still on our branch."""
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    if branch != "codex/sustained-juggling":
        print(f"Skipping result commit: active branch changed to {branch}", flush=True)
        return
    stamp = time.strftime("%Y-%m-%d_%H-%M", time.gmtime())
    note = REPO / f"notes/scratch/experiments/{stamp}_clean-juggle-iteration-{iteration}.md"
    if note.exists():
        print(f"Skipping existing immutable result note {note}", flush=True)
        return
    relative = os.path.relpath(root, note.parent)
    body = (root / f"iteration_{iteration:02d}.md").read_text()
    body = body.replace("](", f"]({relative}/")
    winner = max(tasks, key=task_score)
    note.write_text(f"# Clean juggling iteration {iteration} results\n\n"
                    f"- **Date**: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}\n"
                    f"- **Status**: done (bounded iteration; not a claim of indefinite survival)\n"
                    f"- **Run dirs**: `{root}`\n"
                    f"- **Selected trial**: `{winner['name']}`; validation score `{task_score(winner)}`\n\n"
                    + "\n".join(body.splitlines()[1:]) + "\n\n"
                    "Each trial has one training seed; table means are over 16 validation episodes. "
                    "Rewards/memory differ as recorded in each run's args/config. Controller gains and limits are unchanged. "
                    "Selection uses genuine cap completion and repeated-juggle duration. Independent final tests follow selection.\n")
    index = REPO / "notes/scratch/experiments/INDEX.md"
    include_index = all(subprocess.run(command, cwd=REPO).returncode == 0 for command in (
        ["git", "diff", "--quiet", "--", str(index)],
        ["git", "diff", "--cached", "--quiet", "--", str(index)]))
    paths = [str(note)]
    if include_index:
        with index.open("a") as stream:
            stream.write(f"\n- {stamp.replace('_', ' ')} | Clean juggling iteration {iteration} | Selected {winner['name']}; bounded validation, see GIFs and results | [{note.name}]({note.name})\n")
        paths.append(str(index))
    subprocess.run(["git", "add", "--", *paths], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "--only", "-m", f"Record clean juggling iteration {iteration} results", "--", *paths],
                   cwd=REPO, check=True)


def run_stage(root, iteration, tasks, job_ids):
    processes = []
    for task, allocation in zip(tasks, job_ids):
        spec = root / f"iteration_{iteration:02d}_{Path(task['run_dir']).name}.json"
        write_json(spec, task)
        cmd = ["srun", f"--jobid={allocation}", "--overlap", "--exact", "--nodes=1", "--ntasks=1",
               "--cpus-per-task=6", "--cpu-bind=none", "--gres=gpu:1", "--immediate=15",
               f"--chdir={REPO}", "/usr/bin/env", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
               "MPLCONFIGDIR=/tmp/airhockey-mpl", sys.executable, "-u", "-m",
               "scripts.td3.juggle_iterations", "--worker", str(spec)]
        with spec.with_suffix(".launcher.log").open("w") as log:
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        processes.append(proc)
    while any(p.poll() is None for p in processes):
        report(root, iteration, tasks)
        time.sleep(20)
    report(root, iteration, tasks)
    if any(p.returncode != 0 for p in processes):
        raise RuntimeError(f"Iteration {iteration} failed; inspect worker status and launcher logs")


def supervise(cli):
    root = Path(cli.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    state = dict(status="running", iteration=1, allocations=cli.job_ids, started_unix=time.time(),
                 git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip())
    write_json(root / "status.json", state)
    try:
        tasks = []
        for name in NAMES:
            source = REPO / f"configs/td3/sustained/clean_{name}.yaml"
            args = yaml.safe_load(source.read_text())
            args_file = root / f"iteration_01/{name}_args.yaml"
            save_yaml(args_file, args)
            tasks.append(dict(name=name, trainer="scripts.rma.train_base_policy" if name == "memory" else "scripts.td3.td3_training",
                              run_dir=str(root / f"iteration_01/{name}"), args_file=str(args_file)))
        run_stage(root, 1, tasks, cli.job_ids)
        record_iteration(root, 1, tasks)
        best = max(tasks, key=task_score)
        state.update(iteration=2, iteration_1_winner=best, winner_score=task_score(best))
        write_json(root / "status.json", state)
        original = yaml.safe_load(Path(best["args_file"]).read_text())
        tasks2 = []
        for label, seed, gamma in (("continue", 0, .99), ("replicate", 1, .99), ("long_discount", 0, .995)):
            args = copy.deepcopy(original)
            args.update(total_timesteps=1000000, seed=seed, gamma=gamma)
            if label == "continue":
                args.update(model_path=str(Path(best["run_dir"]) / "training_state.pth"),
                            full_checkpoint_load="full_resume")
                if "history_warm_start_plain" in args:
                    args["history_warm_start_plain"] = False
            path = root / f"iteration_02/{label}_args.yaml"
            save_yaml(path, args)
            tasks2.append(dict(name=label, trainer=best["trainer"], args_file=str(path),
                               run_dir=str(root / f"iteration_02/{label}")))
        run_stage(root, 2, tasks2, cli.job_ids)
        record_iteration(root, 2, tasks2)
        winner = max(tasks2, key=task_score)
        state.update(status="final_evaluation", iteration_2_winner=winner, winner_score=task_score(winner))
        write_json(root / "status.json", state)
        for scenario in ("clean", "transfer"):
            with (root / f"final_{scenario}.log").open("w") as log:
                command = [sys.executable, "-u", "-m", "scripts.td3.juggle_survival",
                                "--run-dir", winner["run_dir"], "--scenario-set", scenario,
                                "--episodes", "32", "--horizon", "20000", "--seed-base", "200000",
                                "--out-dir", str(root / f"final_{scenario}")]
                if scenario == "transfer":
                    command.extend(["--config", "configs/new_juggle/sustained/juggle.yaml"])
                subprocess.run(command, cwd=REPO,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        clean_result = json.loads((root / "final_clean/results.json").read_text())
        clean_summary = clean_result["scenarios"]["clean"]["summary"]
        state["all_clean_test_episodes_valid_and_capped"] = clean_summary["sustained_juggling_fraction"] == 1.
        state["tested_horizon"] = 20000
        state.update(status="complete", finished_unix=time.time())
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
    mode.add_argument("--worker")
    parser.add_argument("--root", default="runs/td3/juggle_iterations")
    parser.add_argument("--job-ids", nargs="+", type=int, default=[106217, 106218, 106219])
    cli = parser.parse_args()
    if cli.prepare:
        prepare()
    elif cli.worker:
        worker(cli)
    elif cli.supervise:
        supervise(cli)
    else:
        if len(cli.job_ids) != 3 or len(set(cli.job_ids)) != 3:
            parser.error("three distinct existing allocations are required")
        root = Path(cli.root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (root / "launch.json").exists():
            raise FileExistsError("Iteration campaign already launched")
        with (root / "supervisor.log").open("w") as log:
            cmd = [sys.executable, "-u", "-m", "scripts.td3.juggle_iterations", "--supervise",
                   "--root", str(root), "--job-ids", *map(str, cli.job_ids)]
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(root / "launch.json", dict(pid=proc.pid, command=cmd))
        print(f"Adaptive experiment supervisor PID {proc.pid}; artifacts: {root}")


if __name__ == "__main__":
    main()
