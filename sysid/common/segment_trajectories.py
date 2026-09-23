#!/usr/bin/env python
"""Automatically chunk real puck trajectories into free-fall / wall-collision /
paddle-collision (and opponent-hit / unknown / rest / occluded) segments.

Example — 10 random trajectories from the shared mouse dataset:

    python sysid/common/segment_trajectories.py \\
        --input-dir shared/mouse_state_data_all_new_len_gt130_take100_trim30/trimmed_hdf5 \\
        --sample 10 --seed 0 --out sysid/common/runs/mouse_dataset/segmentation_eval

Explicit files / a whole curated directory work too (``--inputs a.hdf5 b.hdf5``
or ``--input-dir <curated clips dir>``). Output per trajectory:
``<out>/<stem>/{segments.json, segmentation.gif, segmentation.png, segments/*.hdf5}``
plus ``<out>/summary.md`` and ``<out>/sample_manifest.json``.

Axis conventions (puck / pose mirrored into the sim frame) are auto-detected
per run — see ``sysid/common/trajectory_segmentation.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, fields
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sysid.common.trajectory_segmentation import (  # noqa: E402
    LABELS, SegmentationConfig, estimate_axis_transforms, is_split_schema_recording,
    list_split_schema_recordings, segment_trajectory, write_segment_hdf5s, write_segments_json,
)
from sysid.common.segment_rendering import render_gif, render_plot  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", type=Path, help="directory searched recursively for split-schema *.hdf5")
    src.add_argument("--inputs", type=Path, nargs="+", help="explicit HDF5 files")
    p.add_argument("--sample", type=int, default=None, help="randomly pick this many files")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--puck-x-sign", choices=["auto", "1", "-1"], default="auto",
                   help="mirror the logged puck x into the sim frame (auto: from free-flight acceleration sign)")
    p.add_argument("--paddle-x-sign", choices=["auto", "1", "-1"], default="auto",
                   help="sim paddle x = sign * pose_x + offset (auto: calibrated from puck-paddle interactions)")
    p.add_argument("--paddle-x-offset", type=str, default="auto", help="see --paddle-x-sign; metres or 'auto'")
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-camera", action="store_true", help="omit the camera panel from GIFs")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-hdf5", action="store_true", help="skip per-segment HDF5 slices")
    p.add_argument("--include-images", action="store_true", help="copy camera images into segment HDF5s")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--max-gif-frames", type=int, default=None)
    # Any SegmentationConfig field can be overridden: --cfg dv_threshold=0.4 fit_window=5
    p.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VALUE",
                   help="override SegmentationConfig fields, e.g. dv_threshold=0.4")
    return p.parse_args()


def build_config(overrides: list[str]) -> SegmentationConfig:
    cfg = SegmentationConfig()
    types = {f.name: f.type for f in fields(SegmentationConfig)}
    for kv in overrides:
        k, v = kv.split("=", 1)
        if k not in types:
            raise SystemExit(f"unknown config field {k!r}; valid: {sorted(types)}")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(float(v)) if isinstance(cur, (int, float)) else v)
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args.cfg)
    if args.inputs:
        files = [Path(f) for f in args.inputs]
        skipped = [p for p in files if not is_split_schema_recording(p)]
        if skipped:
            print(f"skipped {len(skipped)} non-split-schema input(s): "
                  + ", ".join(str(p) for p in skipped[:5]))
        files = [p for p in files if is_split_schema_recording(p)]
    else:
        files = list_split_schema_recordings(args.input_dir)
    if not files:
        raise SystemExit("no split-schema HDF5 recordings found "
                         "(need datasets puck / pose / cur_time; old train_vals dumps are skipped)")
    if args.sample is not None and args.sample < len(files):
        rng = random.Random(args.seed)
        files = sorted(rng.sample(files, args.sample))

    signs = estimate_axis_transforms(files, cfg)
    puck_sign = signs["puck_x_sign"] if args.puck_x_sign == "auto" else int(args.puck_x_sign)
    paddle_sign = signs["paddle_x_sign"] if args.paddle_x_sign == "auto" else int(args.paddle_x_sign)
    paddle_off = signs["paddle_x_offset"] if args.paddle_x_offset == "auto" else float(args.paddle_x_offset)
    print(f"frame calibration: puck_x_sign={puck_sign} (measured free-flight acc {signs['acc_measured']:+.3f} m/s^2 "
          f"over {signs['n_windows']} windows, cfg gravity_x {cfg.gravity_x:+.3f});  "
          f"paddle x_sim = {paddle_sign:+d} * pose_x {paddle_off:+.3f}  "
          f"(explains {signs['hits']}/{signs['n_impulses']} on-table impulses, "
          f"{signs['pass_throughs']} pass-through frames; decided={signs['decided']})")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "files": [str(f) for f in files], "sample": args.sample, "seed": args.seed,
        "axis_transform_estimate": signs, "puck_x_sign": puck_sign, "paddle_x_sign": paddle_sign,
        "paddle_x_offset": paddle_off,
        "config": asdict(cfg),
    }
    with open(args.out / "sample_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    rows = []
    totals = {lab: 0 for lab in LABELS}
    frame_totals = {lab: 0 for lab in LABELS}
    for i, path in enumerate(files):
        res = segment_trajectory(path, cfg, puck_x_sign=puck_sign, paddle_x_sign=paddle_sign,
                                 paddle_x_offset=paddle_off)
        d = args.out / path.stem
        d.mkdir(parents=True, exist_ok=True)
        write_segments_json(res, d / "segments.json")
        if not args.no_hdf5:
            write_segment_hdf5s(res, d / "segments", include_images=args.include_images)
        if not args.no_plot:
            render_plot(res, d / "segmentation.png")
        if not args.no_gif:
            render_gif(res, d / "segmentation.gif", fps=args.fps, camera=not args.no_camera,
                       max_frames=args.max_gif_frames)
        c, fc = res.counts(), res.frame_counts()
        for lab in LABELS:
            totals[lab] += c[lab]; frame_totals[lab] += fc[lab]
        ff = [s for s in res.segments if s.label == "free_fall"]
        longest = max((s.n_frames for s in ff), default=0)
        rms = [s.meta["fit_rms_m"] for s in ff if "fit_rms_m" in s.meta and s.n_frames >= 8]
        rows.append((path.stem, res.trajectory.n, c, longest, (100 * sum(rms) / len(rms)) if rms else float("nan")))
        print(f"[{i + 1}/{len(files)}] {path.name}: {res.trajectory.n} frames  " +
              "  ".join(f"{lab}={c[lab]}" for lab in LABELS if c[lab]))

    lines = ["# Automatic trajectory segmentation — summary", "",
             f"Files: {len(files)} (sample={args.sample}, seed={args.seed})  ",
             f"Frame calibration: puck_x_sign={puck_sign} (measured free-flight acc {signs['acc_measured']:+.3f} m/s²); "
             f"paddle x_sim = {paddle_sign:+d}·pose_x {paddle_off:+.3f} m "
             f"({signs['hits']}/{signs['n_impulses']} on-table impulses explained, {signs['pass_throughs']} pass-through frames)  ",
             f"Config: {json.dumps(asdict(cfg))}", "",
             "| trajectory | frames | " + " | ".join(LABELS) + " | longest free-fall (fr) | free-fall fit rms (cm, segs>=8fr) |",
             "|---|---|" + "---|" * len(LABELS) + "---|---|"]
    for stem, n, c, longest, rms in rows:
        lines.append(f"| {stem} | {n} | " + " | ".join(str(c[l]) for l in LABELS) + f" | {longest} | {rms:.2f} |")
    lines.append("| **total segments** | " + str(sum(r[1] for r in rows)) + " | " + " | ".join(str(totals[l]) for l in LABELS) + " | | |")
    lines.append("| **total frames** | | " + " | ".join(str(frame_totals[l]) for l in LABELS) + " | | |")
    lines += ["", "Per-trajectory outputs: `<stem>/segments.json`, `<stem>/segmentation.gif`, `<stem>/segmentation.png`, `<stem>/segments/*.hdf5`."]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out / 'summary.md'}")


if __name__ == "__main__":
    main()
