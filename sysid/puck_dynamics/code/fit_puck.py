#!/usr/bin/env python
"""Puck free-flight system identification: (gravity_x, puck_damping) by grid
search with a train / validation split by recording.

Input: a free-fall sections directory written by
``sysid/common/extract_sysid_sections.py`` (``free_fall/*.hdf5`` +
``manifest.json``), e.g. ``sysid/puck_dynamics/data/<name>/``.

Output (``--out``, e.g. ``sysid/puck_dynamics/results/<name>/``):

    split.json               train / val recordings (+ the clips of each)
    summary.md               fit quality (absolute + normalised), percentile validation, reproduce command
    results.json             everything numeric
    sim_config_fitted.yaml   --sim-config with gravity / puck_damping replaced
    fit_grid.png             (g, γ) grids on train / val for both metrics
    validation_percentiles.png
    puck_final_displacement_vs_percentile.{png,pdf}, puck_final_displacement_grid_{train,val}.{png,pdf}
                             key figures (also copied to --figures-dir)

Free-fall clips → fixed-length windows → (g, γ) grid scored on train AND val
with two metrics of the full in-window fit (rms in cm, final displacement
error / distance travelled) → train minimum of --objective. Validation
(``sysid/common/fit_validation.py``): the selected parameters are ranked
against the validation error of every grid point (beats-fraction, oracle,
p50 / p75 / p90 candidates and the share of the oracle they reach).

    python sysid/puck_dynamics/code/fit_puck.py \\
        --sections-dir sysid/puck_dynamics/data/mouse_dataset \\
        --out sysid/puck_dynamics/results/mouse_dataset
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
from sysid.common.sysid_dataset import load_manifest, make_free_fall_windows, split_by_source  # noqa: E402
from sysid.common.fit_validation import plot_percentiles, summarize_reports, PERCENTILE_LEGEND  # noqa: E402
from sysid.puck_dynamics.code.puck_dynamics_fit import (  # noqa: E402
    METRICS, METRIC_LABEL, METRIC_UNIT, grid_search, plot_grid, plot_paper_figures,
)

METRIC_DEFINITIONS = {
    "fit_rms_cm": "mean over windows of the rms position residual of the in-window LSQ fit (cm)",
    "fit_rel": "|model − measured| at the last sample of the in-window fit / path length of the measured puck trajectory in the window (mean of per-window ratios)",
    "validation.selected_beats": "fraction of searched candidates whose validation error is worse than the selected parameters",
    "validation.fraction_of_best": "oracle validation error / candidate validation error (1 = as good as the best candidate on validation)",
    "validation.gain_vs_median": "(median − err) / (median − oracle): share of the median→oracle improvement realised",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sections-dir", type=Path, required=True, help="free-fall sections (free_fall/*.hdf5 + manifest.json)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sim-config", type=Path, default=_REPO_ROOT / "configs/new_juggle/sysid_best_params_hist2.yaml",
                   help="base sim config: source of the canonical values and of everything not identified here")
    p.add_argument("--figures-dir", type=Path, default=None, help="also copy the key figures (PNG + PDF) here, e.g. paper/figures/sysid")
    p.add_argument("--window-frames", type=int, default=20, help="usable samples per free-fall datapoint (10 is too short to separate g from γ; 20 ≈ 1 s)")
    p.add_argument("--g-range", type=float, nargs=2, default=(-1.0, -0.4))
    p.add_argument("--gamma-range", type=float, nargs=2, default=(0.0, 0.4))
    p.add_argument("--objective", choices=METRICS, default="fit_rms_cm",
                   help="metric minimised on the train windows (fit_rms_cm = original grid-search criterion; fit_rel = final displacement / distance)")
    return p.parse_args()


def _f(m: str, x: float) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:.2f}" if m.endswith("_cm") else f"{100 * x:.1f} %"


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
    rows = [r for r in manifest["sections"] if r["kind"] == "free_fall"]
    cmd = (f"python sysid/puck_dynamics/code/fit_puck.py --sections-dir {shlex.quote(_rel(args.sections_dir))} --out {shlex.quote(_rel(out))} "
           f"--seed {args.seed} --val-fraction {args.val_fraction} --window-frames {args.window_frames} --sim-config {shlex.quote(_rel(args.sim_config))} "
           f"--objective {args.objective} --g-range {args.g_range[0]} {args.g_range[1]} --gamma-range {args.gamma_range[0]} {args.gamma_range[1]}")

    # ---- split by recording (drawn from every input recording so the wall fit holds out the same ones)
    split = split_by_source(rows, args.val_fraction, args.seed, sources=manifest.get("sources"))
    with open(out / "split.json", "w") as f:
        json.dump({"sources": split["sources"], "train_files": [r["file"] for r in split["train"]], "val_files": [r["file"] for r in split["val"]]}, f, indent=1)
    with open(args.sim_config) as f:
        cfg_yaml = yaml.safe_load(f)
    base_sim = cfg_yaml["air_hockey"]["simulator_params"]
    can_g, can_gam = float(base_sim["gravity"]), float(base_sim["puck_damping"])

    win_tr = make_free_fall_windows(split["train"], args.sections_dir, seg_cfg, args.window_frames)
    win_va = make_free_fall_windows(split["val"], args.sections_dir, seg_cfg, args.window_frames)
    print(f"split: {len(split['sources']['train'])} train / {len(split['sources']['val'])} val recordings; "
          f"free-fall clips {len(split['train'])} / {len(split['val'])}; windows of {args.window_frames} samples: train {len(win_tr)}, val {len(win_va)}")

    # ---- grid search on train, scored on val, validated over the whole grid
    print("scoring the (g, γ) grid on train and val …")
    fit = grid_search(win_tr, win_va, g_range=tuple(args.g_range), gamma_range=tuple(args.gamma_range),
                      objective=args.objective, canonical=(can_g, can_gam))
    plot_grid(fit, out / "fit_grid.png")
    if fit.validation:
        plot_percentiles([fit.validation[m] for m in METRICS], out / "validation_percentiles.png")
    key_figs = [str(f) for f in plot_paper_figures(fit, out)]
    print(f"puck: g={fit.gravity_x:+.3f} γ={fit.damping:.3f} (objective {fit.objective})   canonical g={can_g:+.3f} γ={can_gam:.3f}")
    for m in METRICS:
        print(f"  {METRIC_LABEL[m]:>22} [{METRIC_UNIT[m]}]: train {_f(m, fit.train[m])}  val {_f(m, fit.val[m])}   canonical train {_f(m, fit.canonical['train'][m])}  val {_f(m, fit.canonical['val'][m])}")
    for m, r in fit.validation.items():
        print(f"  validation {METRIC_LABEL[m]:>22}: selected beats {100 * r.selected_beats:.0f} % of grid ({100 * r.selected_fraction_of_best:.0f} % of oracle); "
              f"p50/p75/p90 reach {100 * r.percentiles[50]['fraction_of_best']:.0f}/{100 * r.percentiles[75]['fraction_of_best']:.0f}/{100 * r.percentiles[90]['fraction_of_best']:.0f} %; canonical beats {100 * r.canonical_beats:.0f} %")

    # ---- outputs
    fitted = {"gravity": round(fit.gravity_x, 4), "puck_damping": round(fit.damping, 4)}
    cfg_yaml["air_hockey"]["simulator_params"].update(fitted)
    with open(out / "sim_config_fitted.yaml", "w") as f:
        f.write(f"# Derived from {args.sim_config.name} by sysid/puck_dynamics/code/fit_puck.py (gravity, puck_damping replaced)\n# {cmd}\n")
        yaml.safe_dump(cfg_yaml, f, sort_keys=False)
    res = fit.to_json() | {
        "sections_dir": str(args.sections_dir), "sim_config": str(args.sim_config), "command": cmd,
        "args": {k: str(v) for k, v in vars(args).items()}, "calibration": manifest.get("calibration"),
        "split_sources": split["sources"], "n_clips": {"train": len(split["train"]), "val": len(split["val"])},
        "metric_definitions": METRIC_DEFINITIONS, "fitted_params": fitted, "key_figures": key_figs}
    with open(out / "results.json", "w") as f:
        json.dump(res, f, indent=1, default=_json_default)

    pct = lambda v: f"{100 * v:.0f} %"
    pv = fit.validation.get("fit_rel")
    lines = ["# Puck free flight — (gravity_x, puck_damping) by grid search, train / validation", "",
             f"Sections `{_rel(args.sections_dir)}` · split seed {args.seed}, val fraction {args.val_fraction}: {len(split['sources']['train'])} train / {len(split['sources']['val'])} val recordings · "
             f"free-fall clips {len(split['train'])} / {len(split['val'])} · {args.window_frames}-sample windows: {len(win_tr)} train, {len(win_va)} val (`split.json`) · "
             f"base config `{args.sim_config.name}` (canonical g={can_g:+.3f}, γ={can_gam:.3f})", "",
             f"**Result: g = {fit.gravity_x:+.3f} m/s², γ = {fit.damping:.3f} 1/s** — selected on train `{fit.objective}` over a {len(fit.grid_g)} × {len(fit.grid_gamma)} grid "
             f"g ∈ [{args.g_range[0]}, {args.g_range[1]}], γ ∈ [{args.gamma_range[0]}, {args.gamma_range[1]}] (0.02 steps, 0.005 fine grid)."
             + (f" On val final displacement the selection beats {pct(pv.selected_beats)} of the grid and reaches {pct(pv.selected_fraction_of_best)} of the oracle "
                f"(p90 grid point: {pct(pv.percentiles[90]['fraction_of_best'])}, canonical beats {pct(pv.canonical_beats)})." if pv else ""), "",
             "## Fit quality (full in-window fit)", "",
             "| params | train rms | train final displacement / distance | val rms | val final displacement / distance |", "|---|---|---|---|---|",
             f"| **fitted** g={fit.gravity_x:+.3f}, γ={fit.damping:.3f} | {_f('fit_rms_cm', fit.train['fit_rms_cm'])} cm | {_f('fit_rel', fit.train['fit_rel'])} | {_f('fit_rms_cm', fit.val['fit_rms_cm'])} cm | {_f('fit_rel', fit.val['fit_rel'])} |",
             f"| canonical g={can_g:+.3f}, γ={can_gam:.3f} | {_f('fit_rms_cm', fit.canonical['train']['fit_rms_cm'])} cm | {_f('fit_rel', fit.canonical['train']['fit_rel'])} | {_f('fit_rms_cm', fit.canonical['val']['fit_rms_cm'])} cm | {_f('fit_rel', fit.canonical['val']['fit_rel'])} |", "",
             "Both metrics come from the LSQ fit of the model a = g − γ v (linear in p0, v0) to all samples of each window: `rms` = rms position residual; "
             "`final displacement / distance` = distance between model and measured puck at the last sample of the window divided by the distance the puck travelled over the window "
             "(mean of per-window ratios).", "",
             "Grid on train and val, rms and final displacement: `fit_grid.png` (rows rms / final displacement; columns train / val; "
             "red star = selected, white circle = canonical, cyan triangle = that panel's own minimum). Key figures (individual PNG + PDF): "
             "`puck_final_displacement_vs_percentile`, `puck_final_displacement_grid_train`, `puck_final_displacement_grid_val`.", "",
             "## Did the grid search do anything? (validation landscape over the coarse grid)", ""]
    lines += summarize_reports([fit.validation[m] for m in METRICS]) if fit.validation else ["(no validation windows)"]
    lines += ["", PERCENTILE_LEGEND, ""]
    if args.figures_dir is not None:
        args.figures_dir.mkdir(parents=True, exist_ok=True)
        for f in list(out.glob("puck_*.png")) + list(out.glob("puck_*.pdf")):
            shutil.copy2(f, args.figures_dir / f.name)
        lines += [f"Key figures copied to `{args.figures_dir}`.", ""]
    lines += ["## Fitted parameters", "", "```yaml"] + [f"{k}: {v}" for k, v in fitted.items()] + ["```", "",
              f"Full config: `sim_config_fitted.yaml` (derived from `{args.sim_config.name}`; not promoted into `configs/`).", "",
              "## Reproduce", "", "```bash", cmd, "```", "",
              "Docs: `notes/docs/environments/real-world/sysid/puck-free-flight.md` (this fit), `notes/docs/environments/real-world/sysid-pipeline.md` (pipeline, validation method)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
