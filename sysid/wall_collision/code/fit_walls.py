#!/usr/bin/env python
"""Puck–wall collision system identification: side / end wall restitution by
Box2D replay of real bounces with a train / validation split by recording.

Input: a wall sections directory written by
``sysid/common/extract_sysid_sections.py --wall-out`` (``wall/*.hdf5`` +
``manifest.json``), e.g. ``sysid/wall_collision/data/<name>/``, plus the puck
free-flight model the bounces are replayed with — either
``--puck-results <results.json>`` of a ``sysid/puck_dynamics/code/fit_puck.py``
run (its g / γ are used) or ``--puck-params G GAMMA``.

Output (``--out``, e.g. ``sysid/wall_collision/results/<name>/``):

    split.json               train / val recordings (+ the bounces of each)
    summary.md               fit quality (absolute + relative + angle), percentile validation, apparent wall lines, reproduce command
    results.json             everything numeric
    sim_config_fitted.yaml   --sim-config with side_wall_restitution / end_wall_restitution replaced (+ the puck g / γ used)
    fit_sweeps.png           per wall kind: absolute / relative / angle curves, sim-vs-real scatter, per-bounce errors
    validation_percentiles.png
    wall_side_exit_speed_{sweep,scatter}.{png,pdf}, wall_side_exit_angle_scatter.{png,pdf}
                             key figures (side walls; also copied to --figures-dir)

Every bounce is replayed from the real pre-impact state (fitted position /
velocity of the first clean pre-impact frame) with a static paddle until the
normal velocity reverses; side_wall_restitution (y± walls) and
end_wall_restitution (x± walls) are swept; five metrics (exit-speed and
normal-exit-speed error, absolute and relative to the real exit speed, exit
angle) → train minimum of --objective. Validation
(``sysid/common/fit_validation.py``): the selected value is ranked against the
validation error of every sweep value.

    python sysid/wall_collision/code/fit_walls.py \\
        --sections-dir sysid/wall_collision/data/mouse_dataset \\
        --puck-results sysid/puck_dynamics/results/mouse_dataset/results.json \\
        --out sysid/wall_collision/results/mouse_dataset
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sysid.common.trajectory_segmentation import SegmentationConfig, _json_default  # noqa: E402
from sysid.common.sysid_dataset import load_manifest, make_wall_bounces, split_by_source  # noqa: E402
from sysid.common.fit_validation import plot_percentiles, summarize_reports, PERCENTILE_LEGEND  # noqa: E402
from sysid.wall_collision.code.wall_restitution_fit import (  # noqa: E402
    END_WALLS, SIDE_WALLS, WALL_METRICS, bounce_state, load_clean_sim_config, plot_paper_figures, plot_sweeps, sweep_wall,
)

METRIC_DEFINITIONS = {
    "speed_err": "mean |exit speed sim − real| over reproduced bounces (m/s)",
    "normal_err": "mean |normal exit speed sim − real| (m/s)",
    "speed_rel_err": "mean |exit speed sim − real| / real exit speed",
    "normal_rel_err": "mean |normal exit speed sim − real| / real exit speed",
    "angle_err": "mean |exit direction sim − exit direction real| in degrees (post-impact velocity direction, sim frame)",
    "validation.selected_beats": "fraction of searched candidates whose validation error is worse than the selected parameters",
    "validation.fraction_of_best": "oracle validation error / candidate validation error (1 = as good as the best candidate on validation)",
    "validation.gain_vs_median": "(median − err) / (median − oracle): share of the median→oracle improvement realised",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sections-dir", type=Path, required=True, help="wall sections (wall/*.hdf5 + manifest.json)")
    p.add_argument("--out", type=Path, required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--puck-results", type=Path, help="results.json of a fit_puck.py run: its g / γ are used for the replays")
    src.add_argument("--puck-params", type=float, nargs=2, metavar=("GRAVITY_X", "DAMPING"), help="use these g / γ for the replays")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-config", type=Path, default=_REPO_ROOT / "configs/new_juggle/sysid_best_params_hist2.yaml",
                   help="base sim config: source of the canonical values and of everything not identified here")
    p.add_argument("--figures-dir", type=Path, default=None, help="also copy the key figures (PNG + PDF) here, e.g. paper/figures/sysid")
    p.add_argument("--restitution-range", type=float, nargs=3, default=(0.4, 1.0, 0.025), metavar=("LO", "HI", "STEP"))
    p.add_argument("--objective", choices=WALL_METRICS, default="speed_rel_err",
                   help="metric minimised on the train bounces (speed_rel_err = |exit speed sim − real| / real exit speed)")
    p.add_argument("--max-side-frames", type=int, default=10, help="clean frames on each side of the impact used for the pre / post fits")
    return p.parse_args()


def _f(m: str, x: float) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    if m == "angle_err":
        return f"{x:.1f}°"
    return f"{x:.3f}" if "rel" not in m else f"{100 * x:.1f} %"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def main():
    args = parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    seg_cfg = SegmentationConfig()
    manifest = load_manifest(args.sections_dir)
    rows = [r for r in manifest["sections"] if r["kind"] == "wall"]
    if args.puck_results is not None:
        pr = json.load(open(args.puck_results))
        g, gam = float(pr["gravity_x"]), float(pr["damping"])
        puck_src = f"--puck-results {shlex.quote(_rel(args.puck_results))}"
    else:
        g, gam = map(float, args.puck_params)
        puck_src = f"--puck-params {g} {gam}"
    cmd = (f"python sysid/wall_collision/code/fit_walls.py --sections-dir {shlex.quote(_rel(args.sections_dir))} {puck_src} --out {shlex.quote(_rel(out))} "
           f"--seed {args.seed} --val-fraction {args.val_fraction} --sim-config {shlex.quote(_rel(args.sim_config))} --objective {args.objective} "
           f"--restitution-range {' '.join(str(v) for v in args.restitution_range)} --max-side-frames {args.max_side_frames}")

    # ---- split by recording (same draw as the puck fit for the same seed / fraction / recordings)
    split = split_by_source(rows, args.val_fraction, args.seed, sources=manifest.get("sources"))
    with open(out / "split.json", "w") as f:
        json.dump({"sources": split["sources"], "train_files": [r["file"] for r in split["train"]], "val_files": [r["file"] for r in split["val"]]}, f, indent=1)
    base_sim = load_clean_sim_config(args.sim_config)["simulator_params"]

    # ---- bounce states + sweeps
    fit_cfg = SegmentationConfig(gravity_x=g, damping=gam)
    b_tr = make_wall_bounces(split["train"], args.sections_dir, seg_cfg)
    b_va = make_wall_bounces(split["val"], args.sections_dir, seg_cfg)
    st_tr = [(bounce_state(b, fit_cfg, args.max_side_frames), b.paddle_xy_a) for b in b_tr]
    st_va = [(bounce_state(b, fit_cfg, args.max_side_frames), b.paddle_xy_a) for b in b_va]
    values = np.arange(args.restitution_range[0], args.restitution_range[1] + 1e-9, args.restitution_range[2])
    overrides = {"gravity": g, "puck_damping": gam}
    print(f"split: {len(split['sources']['train'])} train / {len(split['sources']['val'])} val recordings; bounces train {len(st_tr)} / val {len(st_va)}; "
          f"replaying in Box2D with g={g:+.3f} γ={gam:.3f}")
    results = []
    for kind, walls in (("side", SIDE_WALLS), ("end", END_WALLS)):
        tr = [(s, p) for s, p in st_tr if s.wall in walls]
        va = [(s, p) for s, p in st_va if s.wall in walls]
        if not tr:
            print(f"[{kind}] no train bounces, skipped"); continue
        res = sweep_wall(kind, args.sim_config, overrides, tr, va, values, objective=args.objective)
        results.append(res)
        ib, ic = res.best_index, res.canonical_index
        print(f"[{kind}] {res.param}: selected {res.best:.3f} (objective {res.objective}), canonical {res.canonical:.3f}   [{res.n_reproduced_train}/{res.n_train} train bounces reproduced]")
        for m in WALL_METRICS:
            print(f"  {m:>15}: train {_f(m, res.train[m][ib])}  val {_f(m, res.val[m][ib])}   canonical train {_f(m, res.train[m][ic])}  val {_f(m, res.val[m][ic])}")
        for m, r in res.validation.items():
            print(f"  validation {m:>15}: selected beats {100 * r.selected_beats:.0f} % of sweep ({100 * r.selected_fraction_of_best:.0f} % of oracle); "
                  f"p50/p75/p90 reach {100 * r.percentiles[50]['fraction_of_best']:.0f}/{100 * r.percentiles[75]['fraction_of_best']:.0f}/{100 * r.percentiles[90]['fraction_of_best']:.0f} %; canonical beats {100 * r.canonical_beats:.0f} %")
    key_figs = []
    if results:
        plot_sweeps(results, out / "fit_sweeps.png")
        reps = [r.validation[m] for r in results for m in ("speed_err", "speed_rel_err", "normal_rel_err", "angle_err") if m in r.validation]
        if reps:
            plot_percentiles(reps, out / "validation_percentiles.png")
        key_figs = [str(f) for f in plot_paper_figures(results, out, kind="side")]

    # ---- apparent wall contact lines (measured puck-centre apex vs the sim's lines)
    apex = {}
    for s, _ in st_tr + st_va:
        apex.setdefault(s.wall, []).append(s.apex)
    sim_line = {"x": 0.5 * float(base_sim["length"]) - float(base_sim["puck_radius"]), "y": 0.5 * float(base_sim["width"]) - float(base_sim["puck_radius"])}
    wall_lines = {w: {"n": len(v), "apex_p50": float(np.median(v)), "apex_p90": float(np.percentile(v, 90)), "sim_contact_line": sim_line[w[0]]} for w, v in apex.items()}
    for w, d in sorted(wall_lines.items()):
        print(f"wall line {w}: measured puck-centre apex p50 {d['apex_p50']:.3f} / p90 {d['apex_p90']:.3f} m vs sim contact line {d['sim_contact_line']:.3f} (n={d['n']})")

    # ---- outputs
    fitted = {"gravity": round(g, 4), "puck_damping": round(gam, 4)} | {r.param: round(r.best, 4) for r in results}
    with open(args.sim_config) as f:
        cfg_yaml = yaml.safe_load(f)
    cfg_yaml["air_hockey"]["simulator_params"].update(fitted)
    with open(out / "sim_config_fitted.yaml", "w") as f:
        f.write(f"# Derived from {args.sim_config.name} by sysid/wall_collision/code/fit_walls.py (wall restitutions replaced; gravity / puck_damping = the puck model used for the replays)\n# {cmd}\n")
        yaml.safe_dump(cfg_yaml, f, sort_keys=False)
    res_json = {"sections_dir": str(args.sections_dir), "sim_config": str(args.sim_config), "command": cmd,
                "args": {k: str(v) for k, v in vars(args).items()}, "calibration": manifest.get("calibration"),
                "split_sources": split["sources"], "n_train": len(st_tr), "n_val": len(st_va),
                "puck_params_used": overrides, "metric_definitions": METRIC_DEFINITIONS,
                "walls": [r.to_json() for r in results], "apparent_wall_lines": wall_lines, "fitted_params": fitted, "key_figures": key_figs}
    with open(out / "results.json", "w") as f:
        json.dump(res_json, f, indent=1, default=_json_default)

    pct = lambda v: f"{100 * v:.0f} %"
    lines = ["# Puck–wall collisions — wall restitution by Box2D replay, train / validation", "",
             f"Sections `{_rel(args.sections_dir)}` · split seed {args.seed}, val fraction {args.val_fraction}: {len(split['sources']['train'])} train / {len(split['sources']['val'])} val recordings · "
             f"bounces train {len(st_tr)} / val {len(st_va)} (`split.json`) · puck model g={g:+.3f}, γ={gam:.3f} ({'from ' + _rel(args.puck_results) if args.puck_results else 'supplied'}) · "
             f"base config `{args.sim_config.name}` · sweep {args.restitution_range[0]}–{args.restitution_range[1]} step {args.restitution_range[2]} · selected on train `{args.objective}`", ""]
    for r in results:
        ib, ic = r.best_index, r.canonical_index; wv = r.validation.get("speed_rel_err")
        lines.append(f"**{r.wall_kind} walls: {r.param} = {r.best:.3f}** — val exit-speed err {r.val['speed_err'][ib]:.3f} m/s = {pct(r.val['speed_rel_err'][ib])} of real speed, "
                     f"exit angle {r.val['angle_err'][ib]:.1f}° (canonical {r.canonical:.3f}: {r.val['speed_err'][ic]:.3f} m/s = {pct(r.val['speed_rel_err'][ic])}, {r.val['angle_err'][ic]:.1f}°); "
                     f"bounces {r.n_train}/{r.n_val}" + (f"; selection beats {pct(wv.selected_beats)} of the sweep, p90 point reaches {pct(wv.percentiles[90]['fraction_of_best'])} of the oracle, canonical beats {pct(wv.canonical_beats)}" if wv else "") + ".")
    lines += ["", "## Fit quality", "",
              "| walls | param | fitted | train exit-speed err | val exit-speed err | train normal err | val normal err | train exit-angle err | val exit-angle err | canonical | val exit-speed err @canonical | val exit-angle err @canonical | bounces train/val (reproduced) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    wcell = lambda d, i: f"{d['speed_err'][i]:.3f} m/s ({100 * d['speed_rel_err'][i]:.1f} %)"
    ncell = lambda d, i: f"{d['normal_err'][i]:.3f} m/s ({100 * d['normal_rel_err'][i]:.1f} %)"
    for r in results:
        ib, ic = r.best_index, r.canonical_index
        lines.append(f"| {r.wall_kind} | {r.param} | **{r.best:.3f}** | {wcell(r.train, ib)} | {wcell(r.val, ib)} | {ncell(r.train, ib)} | {ncell(r.val, ib)} | {r.train['angle_err'][ib]:.1f}° | {r.val['angle_err'][ib]:.1f}° | {r.canonical:.3f} | {wcell(r.val, ic)} | {r.val['angle_err'][ic]:.1f}° | {r.n_train}/{r.n_val} ({r.n_reproduced_train}) |")
    lines += ["", "err = mean |exit speed sim − exit speed real| over bounces the sim reproduced (puck placed at the fitted real position / velocity of the first pre-impact window frame, "
              "stepped with a static paddle until the normal velocity reverses); the percentage is the same error divided by the real exit speed (mean of per-bounce ratios); "
              "exit-angle err = mean |direction of the sim exit velocity − direction of the real exit velocity|. "
              "Curves (absolute, relative, angle; train and val) and per-bounce scatter at the selected value: `fit_sweeps.png`. Key figures (side walls, individual PNG + PDF): "
              "`wall_side_exit_speed_sweep`, `wall_side_exit_speed_scatter`, `wall_side_exit_angle_scatter`.", ""]
    reps = [r.validation[m] for r in results for m in WALL_METRICS if m in r.validation]
    if reps:
        lines += ["## Did the sweep do anything? (validation over the restitution sweep)", ""] + summarize_reports(reps) + ["", PERCENTILE_LEGEND, ""]
    lines += ["## Apparent wall contact lines", "", "Measured puck-centre apex near impact vs the sim's `half-size − puck_radius`; a mismatch means the puck frame is offset / scaled relative to the sim table (shifts *when* a replayed bounce happens, not its exit speed).", "",
              "| wall | n | apex p50 (m) | apex p90 (m) | sim contact line (m) |", "|---|---|---|---|---|"]
    for w, d in sorted(wall_lines.items()):
        lines.append(f"| {w} | {d['n']} | {d['apex_p50']:.3f} | {d['apex_p90']:.3f} | {d['sim_contact_line']:.3f} |")
    if args.figures_dir is not None:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        for f in list(out.glob("wall_*.png")) + list(out.glob("wall_*.pdf")):
            shutil.copy2(f, args.figures_dir / f.name)
        lines += ["", f"Key figures copied to `{args.figures_dir}`."]
    lines += ["", "## Fitted parameters", "", "```yaml"] + [f"{k}: {v}" for k, v in fitted.items()] + ["```", "",
              f"Full config: `sim_config_fitted.yaml` (derived from `{args.sim_config.name}`; gravity / puck_damping are the replay model, not fitted here; not promoted into `configs/`).", "",
              "## Reproduce", "", "```bash", cmd, "```", "",
              "Docs: `notes/docs/environments/real-world/sysid/puck-wall-collision.md` (this fit), `notes/docs/environments/real-world/sysid-pipeline.md` (pipeline, validation method)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
