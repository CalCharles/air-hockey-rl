"""Prepare Vista FB batch jobs without submitting; execute their manifests in Slurm.

Preparation uses only the standard library and is safe on a compute node.
The generated submit.sh must be invoked separately from a login node.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[2]


def prepare(folder, configs, seeds, parallel, account, partition, walltime, cpus, python):
    if parallel < 1 or cpus < parallel:
        raise ValueError("Require 1 <= parallel <= cpus")
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    # Directive values are intentionally narrow: no shell/newline injection.
    import re
    for value in (account, partition):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Invalid account or partition")
    if not re.fullmatch(r"\d{1,3}:[0-5]\d:[0-5]\d", walltime):
        raise ValueError("Walltime must be HH:MM:SS")
    configs = [Path(p).absolute() for p in configs]
    if len({p.stem for p in configs}) != len(configs):
        raise ValueError("Config filenames must have distinct stems")
    for config in configs:
        if not config.is_file():
            raise FileNotFoundError(config)
    python = Path(python).absolute()  # Do not resolve a venv's interpreter symlink.
    if not python.is_file():
        raise FileNotFoundError(python)
    folder = Path(folder).absolute()
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "logs").mkdir()
    snapshots = folder / "configs"
    snapshots.mkdir()
    runs = []
    for config in configs:
        snapshot = snapshots / config.name
        snapshot.write_bytes(config.read_bytes())
        for seed in seeds:
            name = f"{config.stem}_seed{seed}"
            output = folder / "runs" / name
            runs.append({"name": name, "output": str(output), "command": [str(python), "-u", "-m",
                "scripts.fb.fb_training", "--args-file", str(snapshot), "--seed", str(seed),
                "--output-dir", str(output), "--device", "cuda", "--torch-threads", "1",
                "--wandb-name", f"{folder.name}-{name}"]})
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    manifest = {"repo": str(ROOT), "prepared_revision": revision, "parallel": parallel, "runs": runs}
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    command = shlex.join([str(python), "-u", "-m", "scripts.fb.slurm_jobs", "run", str(manifest_path)])
    # Relative Slurm output paths are resolved against --chdir (the manifest folder).
    script = f'''#!/bin/bash
#SBATCH --job-name=fb-sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --time={walltime}
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
set -euo pipefail
cd {shlex.quote(str(ROOT))}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
{command}
'''
    (folder / "job.sbatch").write_text(script)
    submit = '''#!/bin/bash
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Refusing submission inside a Slurm allocation. Use a Vista login node." >&2
    exit 2
fi
case "$(hostname -s)" in
    login[0-9]*) ;;
    *) echo "Refusing submission from a non-login node." >&2; exit 2 ;;
esac
'''
    submit += f'exec sbatch --chdir={shlex.quote(str(folder))} {shlex.quote(str(folder / "job.sbatch"))}\n'
    (folder / "submit.sh").write_text(submit)
    return manifest_path


def run_manifest(path):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Manifest execution requires an allocated Slurm job")
    path = Path(path).absolute()
    manifest = json.loads(path.read_text())
    if not manifest["runs"] or manifest["parallel"] < 1:
        raise ValueError("Manifest requires runs and positive parallelism")
    repo = manifest["repo"]
    revision = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    if revision != manifest["prepared_revision"]:
        raise RuntimeError("Repository HEAD changed after preparation; prepare a new job bundle")
    # Fail before starting any process if this bundle has previously been executed.
    for run in manifest["runs"]:
        if Path(run["output"]).exists():
            raise FileExistsError(f"Run directory already exists: {run['output']}")
    results_path = path.parent / "results.jsonl"
    with results_path.open("x") as results:
        def worker(run):
            output = Path(run["output"])
            output.mkdir(parents=True, exist_ok=False)
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
            with (output / "train.log").open("x") as log:
                completed = subprocess.run(run["command"], cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
            return {"name": run["name"], "returncode": completed.returncode, "output": run["output"]}
        failed = False
        with ThreadPoolExecutor(max_workers=manifest["parallel"]) as executor:
            futures = {executor.submit(worker, run): run for run in manifest["runs"]}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"name": futures[future]["name"], "returncode": 1, "error": str(exc)}
                failed |= result["returncode"] != 0
                results.write(json.dumps(result) + "\n")
                results.flush()
                print(json.dumps(result), flush=True)
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prep = commands.add_parser("prepare", help="Write job files only; never calls sbatch")
    prep.add_argument("--folder", required=True)
    prep.add_argument("--configs", nargs="+", required=True)
    prep.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    prep.add_argument("--parallel", type=int, default=4)
    prep.add_argument("--account", default="ASC26008")
    prep.add_argument("--partition", default="gh")
    prep.add_argument("--walltime", default="24:00:00")
    prep.add_argument("--cpus", type=int, default=72)
    prep.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    run = commands.add_parser("run", help="Execute a prepared manifest inside a Slurm job")
    run.add_argument("manifest")
    args = vars(parser.parse_args())
    mode = args.pop("mode")
    if mode == "prepare":
        path = prepare(**args)
        print(f"Prepared {path}; no job submitted. Review job.sbatch and manifest.json.")
        print(f"From a Vista login node: bash {shlex.quote(str(path.parent / 'submit.sh'))}")
    else:
        sys.exit(run_manifest(args["manifest"]))


if __name__ == "__main__":
    main()
