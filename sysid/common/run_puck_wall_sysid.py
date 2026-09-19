#!/usr/bin/env python
"""One command from raw recordings to the puck free-flight and puck–wall fits.

Runs the stages in order and files everything under the two fits' folders:

    sysid/common/runs/<name>/               this run's index: README.md (headline numbers), pipeline_run.json
      segmentation_eval/                    stage 0 (optional): GIF + PNG + segments for --eval-sample random recordings (QA)
    sysid/puck_dynamics/data/<name>/        stage 1: free_fall/*.hdf5, manifest.{csv,json}, summary.md, recordings → input dir
    sysid/wall_collision/data/<name>/       stage 1: wall/*.hdf5, manifest.{csv,json}, summary.md, recordings → input dir
    sysid/puck_dynamics/results/<name>/     stage 2: gravity / puck_damping        (sysid/puck_dynamics/code/fit_puck.py)
    sysid/wall_collision/results/<name>/    stage 3: side / end wall restitution   (sysid/wall_collision/code/fit_walls.py, replays with stage 2's g / γ)

Stages are the standalone scripts (each has its own --help):
    sysid/common/segment_trajectories.py      auto-segmentation + frame calibration (+ GIFs)
    sysid/common/extract_sysid_sections.py    harvest quality-filtered free-fall / wall sections
    sysid/puck_dynamics/code/fit_puck.py      split by recording, (g, γ) grid, validation
    sysid/wall_collision/code/fit_walls.py    split by recording, Box2D replay restitution sweep, validation

Typical use on a new dataset:

    python sysid/common/run_puck_wall_sysid.py \\
        --input-dir /path/to/recordings --name <dataset_name> --eval-sample 10 --seed 0

Then read sysid/common/runs/<name>/README.md, look at the GIFs in segmentation_eval/ and the
calibration line (frame conventions differ between recording pipelines — see
notes/docs/environments/real-world/sysid-pipeline.md), and only then trust the numbers.
Extra options for a stage go through --segment-args / --extract-args / --puck-args / --wall-args
(quoted strings appended verbatim to that stage's command line). --root redirects the whole
tree (tests use a temporary directory).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_SYSID_ROOT = _REPO_ROOT / "sysid"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True, help="directory searched recursively for *.hdf5 recordings")
    p.add_argument("--name", required=True, help="run name: the <name> of the data / results folders under each fit")
    p.add_argument("--root", type=Path, default=_SYSID_ROOT, help="sysid tree to write into (default: the repo's sysid/)")
    p.add_argument("--eval-sample", type=int, default=10, help="recordings to render in segmentation_eval/ (0 skips the stage)")
    p.add_argument("--seed", type=int, default=0, help="seed for the eval sample and the train/val split")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--window-frames", type=int, default=20)
    p.add_argument("--sim-config", type=Path, default=_REPO_ROOT / "configs/new_juggle/sysid_best_params_hist2.yaml")
    p.add_argument("--segment-args", default="", help="extra CLI args for segment_trajectories.py")
    p.add_argument("--extract-args", default="", help="extra CLI args for extract_sysid_sections.py")
    p.add_argument("--puck-args", default="", help="extra CLI args for fit_puck.py (e.g. \"--figures-dir paper/figures/sysid\")")
    p.add_argument("--wall-args", default="", help="extra CLI args for fit_walls.py")
    p.add_argument("--skip-existing", action="store_true", help="skip a stage whose summary.md already exists (its previous pipeline_run.json entry is kept); also the way to regenerate README.md after rerunning a stage by hand")
    return p.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run_stage(name: str, cmd: list[str], log: dict, marker: Path | None, skip_existing: bool, previous: dict | None = None) -> bool:
    if skip_existing and marker is not None and marker.exists():
        print(f"[{name}] skipped (found {marker})")
        log[name] = dict(previous or {}, skipped=True, skip_command=shlex.join(cmd))
        return True
    print(f"[{name}] {shlex.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, text=True, capture_output=True)
    dt = time.time() - t0
    tail = "\n".join(l for l in proc.stdout.splitlines() if not l.startswith("pygame") and "Hello from" not in l)
    print(tail)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
    log[name] = {"command": shlex.join(cmd), "seconds": round(dt, 1), "returncode": proc.returncode,
                 "stdout_tail": tail[-3000:], "stderr_tail": proc.stderr[-3000:]}
    print(f"[{name}] done in {dt:.0f}s (exit {proc.returncode})\n")
    return proc.returncode == 0


def link_recordings(data_dir: Path, input_dir: Path):
    link = data_dir / "recordings"
    if not link.is_symlink() and not link.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        os.symlink(input_dir.resolve(), link)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    root = args.root
    run_dir = root / "common" / "runs" / args.name
    puck_data, wall_data = root / "puck_dynamics" / "data" / args.name, root / "wall_collision" / "data" / args.name
    puck_res, wall_res = root / "puck_dynamics" / "results" / args.name, root / "wall_collision" / "results" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    log: dict = {"started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": git_commit(),
                 "args": {k: str(v) for k, v in vars(args).items()},
                 "outputs": {k: _rel(v) for k, v in {"run": run_dir, "puck_data": puck_data, "wall_data": wall_data, "puck_results": puck_res, "wall_results": wall_res}.items()},
                 "stages": {}}
    prev: dict = {}
    if args.skip_existing and (run_dir / "pipeline_run.json").exists():
        try:
            prev = json.load(open(run_dir / "pipeline_run.json")).get("stages", {})
        except Exception:
            prev = {}
    ok = True

    if args.eval_sample > 0:
        cmd = [py, str(_HERE / "segment_trajectories.py"), "--input-dir", str(args.input_dir), "--sample", str(args.eval_sample),
               "--seed", str(args.seed), "--out", str(run_dir / "segmentation_eval")] + shlex.split(args.segment_args)
        ok &= run_stage("segmentation_eval", cmd, log["stages"], run_dir / "segmentation_eval" / "summary.md", args.skip_existing, prev.get("segmentation_eval"))

    cmd = [py, str(_HERE / "extract_sysid_sections.py"), "--input-dir", str(args.input_dir), "--out", str(puck_data), "--wall-out", str(wall_data)] + shlex.split(args.extract_args)
    ok &= run_stage("sections", cmd, log["stages"], puck_data / "summary.md", args.skip_existing, prev.get("sections"))
    if ok:
        link_recordings(puck_data, args.input_dir)
        link_recordings(wall_data, args.input_dir)

    if ok:
        cmd = [py, str(_SYSID_ROOT / "puck_dynamics/code/fit_puck.py"), "--sections-dir", str(puck_data), "--out", str(puck_res),
               "--seed", str(args.seed), "--val-fraction", str(args.val_fraction), "--window-frames", str(args.window_frames),
               "--sim-config", str(args.sim_config)] + shlex.split(args.puck_args)
        ok &= run_stage("puck_fit", cmd, log["stages"], puck_res / "summary.md", args.skip_existing, prev.get("puck_fit"))

    if ok:
        cmd = [py, str(_SYSID_ROOT / "wall_collision/code/fit_walls.py"), "--sections-dir", str(wall_data), "--puck-results", str(puck_res / "results.json"),
               "--out", str(wall_res), "--seed", str(args.seed), "--val-fraction", str(args.val_fraction),
               "--sim-config", str(args.sim_config)] + shlex.split(args.wall_args)
        ok &= run_stage("wall_fit", cmd, log["stages"], wall_res / "summary.md", args.skip_existing, prev.get("wall_fit"))

    log["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log["success"] = ok
    with open(run_dir / "pipeline_run.json", "w") as f:
        json.dump(log, f, indent=1)

    # ---- index README
    lines = [f"# Puck + wall system-ID run: {args.name}", "",
             f"Input `{args.input_dir}` · git `{log['git_commit']}` · started {log['started_utc']} · success: {ok}", "",
             "| stage | output | what to look at |", "|---|---|---|"]
    if args.eval_sample > 0:
        lines.append(f"| 0 segmentation eval | `{_rel(run_dir / 'segmentation_eval')}/` | `summary.md` counts; one `segmentation.gif` + `segmentation.png` per sampled recording; the calibration line in `sample_manifest.json` |")
    lines += [f"| 1 sections | `{_rel(puck_data)}/`, `{_rel(wall_data)}/` | each `summary.md` (kept / rejected counts; per-wall speed ratios), `manifest.csv`, `free_fall/` resp. `wall/`; `recordings` → the input directory |",
              f"| 2 puck free flight | `{_rel(puck_res)}/` | `summary.md` (fit quality abs + relative, percentile validation of the grid search), `fit_grid.png`, `validation_percentiles.png`, `sim_config_fitted.yaml`, key figures `puck_final_displacement_*` |",
              f"| 3 puck–wall collisions | `{_rel(wall_res)}/` | `summary.md` (fit quality abs + relative + angle, percentile validation of the sweeps, apparent wall lines), `fit_sweeps.png`, `validation_percentiles.png`, `sim_config_fitted.yaml`, key figures `wall_side_exit_*` |", ""]
    pct = lambda v: f"{100 * v:.0f} %"
    head = []
    if (puck_res / "results.json").exists():
        pk = json.load(open(puck_res / "results.json")); cal = pk.get("calibration") or {}
        pv = pk.get("validation", {}).get("fit_rel")
        head += [f"- Frame calibration: puck_x_sign {cal.get('puck_x_sign')}, paddle x = {cal.get('paddle_x_sign')}·pose_x {cal.get('paddle_x_offset'):+.3f} "
                 f"({cal.get('hits')}/{cal.get('n_impulses')} impulses explained)" if cal else "- Frame calibration: n/a",
                 f"- Puck: g = {pk['gravity_x']:+.3f} m/s², γ = {pk['damping']:.3f} 1/s — val rms {pk['val']['fit_rms_cm']:.2f} cm, "
                 f"final displacement {pct(pk['val']['fit_rel'])} of distance travelled (canonical {pk['canonical']['val']['fit_rms_cm']:.2f} cm, {pct(pk['canonical']['val']['fit_rel'])})"
                 + (f"; on val final displacement the selection beats {pct(pv['selected_beats'])} of the grid and reaches {pct(pv['selected_fraction_of_best'])} of the oracle "
                    f"(p90 grid point: {pct(pv['percentiles']['90']['fraction_of_best'])}, canonical beats {pct(pv['canonical_beats'])})" if pv else "")]
    if (wall_res / "results.json").exists():
        wr = json.load(open(wall_res / "results.json"))
        for w in wr.get("walls", []):
            ib, ic = w["best_index"], w["canonical_index"]; wv = w.get("validation", {}).get("speed_rel_err")
            head.append(f"- {w['wall_kind']} walls: {w['param']} = {w['best']:.3f} — val exit-speed err {w['val']['speed_err'][ib]:.3f} m/s = {pct(w['val']['speed_rel_err'][ib])} of real speed "
                        f"(canonical {w['canonical']:.3f}: {w['val']['speed_err'][ic]:.3f} m/s = {pct(w['val']['speed_rel_err'][ic])}); bounces train/val {w['n_train']}/{w['n_val']}"
                        + (f"; selection beats {pct(wv['selected_beats'])} of the sweep, p90 sweep point reaches {pct(wv['percentiles']['90']['fraction_of_best'])} of the oracle" if wv else ""))
        if wr.get("apparent_wall_lines"):
            head.append("- Apparent wall lines (apex p50 vs sim): " + ", ".join(f"{k} {v['apex_p50']:.3f}/{v['sim_contact_line']:.3f}" for k, v in sorted(wr["apparent_wall_lines"].items())))
        head.append(f"- Fitted parameters: `{json.dumps(wr['fitted_params'])}` → `{_rel(wall_res)}/sim_config_fitted.yaml` (not promoted into `configs/`).")
    if head:
        lines += ["## Headline", ""] + head + [""]
    lines += ["Docs: `notes/docs/environments/real-world/sysid-pipeline.md` (pipeline + replication checklist), "
              "`notes/docs/environments/real-world/sysid/puck-free-flight.md`, `notes/docs/environments/real-world/sysid/puck-wall-collision.md` (the two fits)."]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {run_dir / 'README.md'}  (success={ok})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
