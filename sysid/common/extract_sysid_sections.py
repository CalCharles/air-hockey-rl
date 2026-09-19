#!/usr/bin/env python
"""Harvest sysid-ready sections from real recordings using the auto-segmenter.

Two kinds of sections are written, each as a split-schema HDF5 slice plus a
row in ``manifest.csv`` / ``manifest.json``:

* ``free_fall/``  — clean free-flight clips: >= ``--min-free-frames`` usable
  frames, puck faster than ``--min-speed``, no occlusion longer than the
  segmenter's bridge, and a damped-model fit rms <= ``--max-fit-rms`` cm.
* ``wall/``       — wall bounces whose pre- and post-impact velocities are
  well determined: the free-fall segments on both sides contribute at least
  ``--min-side-frames`` usable frames each, their fits are clean, the impact
  hides at most ``--max-gap`` frames, and the paddle is far away. The slice
  spans the two side windows; attrs give the impact frame within the slice,
  the wall side, the fitted velocities just before / after impact and the
  normal / tangential speed ratios.

All slices are in the sim frame (robot at x < 0, gravity towards x < 0 — see
``sysid/common/trajectory_segmentation.py``); the calibration used is
stored in every file's attrs and in the manifest.

    python sysid/common/extract_sysid_sections.py \\
        --input-dir shared/mouse_state_data_all_new_len_gt130_take100_trim30/trimmed_hdf5 \\
        --out sysid/puck_dynamics/data/mouse_dataset --wall-out sysid/wall_collision/data/mouse_dataset

``--out`` receives the free-fall clips (the puck-dynamics fit's data) and
``--wall-out`` the wall bounces (the wall-collision fit's data); each gets its
own ``manifest.{csv,json}`` + ``summary.md``. Without ``--wall-out`` both kinds
go into ``--out`` with one manifest. Every manifest also lists *all* input
recordings (``sources``) so the two fits draw the same train / val split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sysid.common.trajectory_segmentation import (  # noqa: E402
    SPLIT_DATASETS, SegmentationConfig, _json_default, estimate_axis_transforms,
    fit_damped, model_state, segment_trajectory,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="free-fall clips (+ manifest, summary); wall bounces too unless --wall-out is given")
    p.add_argument("--wall-out", type=Path, default=None, help="wall bounces (+ their own manifest, summary)")
    p.add_argument("--min-free-frames", type=int, default=10, help="usable frames for a free-fall clip")
    p.add_argument("--min-speed", type=float, default=0.15, help="m/s; mean fitted speed over the clip")
    p.add_argument("--max-fit-rms", type=float, default=2.5, help="cm; damped-model fit residual")
    p.add_argument("--min-side-frames", type=int, default=5, help="usable frames on each side of a wall bounce")
    p.add_argument("--max-side-frames", type=int, default=10, help="frames taken from each side for the velocity fit")
    p.add_argument("--max-side-rms", type=float, default=2.0, help="cm; fit residual of each side window")
    p.add_argument("--max-gap", type=int, default=2, help="occluded/stale frames hidden inside the impact")
    p.add_argument("--min-paddle-dist", type=float, default=0.25, help="m; paddle must be at least this far from the puck across the slice")
    p.add_argument("--min-pre-speed", type=float, default=0.3, help="m/s")
    p.add_argument("--cfg", nargs="*", default=[], metavar="KEY=VALUE")
    return p.parse_args()


def build_config(overrides):
    cfg = SegmentationConfig()
    for kv in overrides:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(float(v)))
    return cfg


def write_slice(src_path, dst_path, start, end, attrs):
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for k in SPLIT_DATASETS:
            if k in src:
                dst.create_dataset(k, data=src[k][start:end + 1])
        for k, v in attrs.items():
            dst.attrs[k] = json.dumps(v, default=_json_default) if isinstance(v, (dict, list)) else v


def side_fit(traj, frames, cfg, t0):
    """Fit the damped model on ``frames`` with time origin t0; return (fit, n, rms_cm)."""
    t = traj.t_rel[frames] - t0
    fit = fit_damped(t, traj.puck_xy[frames], cfg)
    return fit, len(frames), 100 * fit["rms"]


def main():
    args = parse_args()
    cfg = build_config(args.cfg)
    files = sorted(args.input_dir.rglob("*.hdf5"))
    if not files:
        raise SystemExit("no HDF5 files found")
    cal = estimate_axis_transforms(files, cfg)
    print(f"calibration: puck_x_sign={cal['puck_x_sign']} paddle x = {cal['paddle_x_sign']:+d}*pose_x {cal['paddle_x_offset']:+.3f} "
          f"({cal['hits']}/{cal['n_impulses']} impulses explained)")
    wall_out = args.wall_out if args.wall_out is not None else args.out
    (args.out / "free_fall").mkdir(parents=True, exist_ok=True)
    (wall_out / "wall").mkdir(parents=True, exist_ok=True)
    common = {"puck_x_sign": cal["puck_x_sign"], "paddle_x_sign": cal["paddle_x_sign"],
              "paddle_x_offset": cal["paddle_x_offset"]}

    rows = []
    n_ff_cand = n_wall_cand = 0
    reject = {"free_fall": {}, "wall": {}}

    def rej(kind, why):
        reject[kind][why] = reject[kind].get(why, 0) + 1

    for path in files:
        res = segment_trajectory(path, cfg, puck_x_sign=cal["puck_x_sign"],
                                 paddle_x_sign=cal["paddle_x_sign"], paddle_x_offset=cal["paddle_x_offset"])
        tr = res.trajectory
        segs = res.segments
        stem = path.stem

        # ---- free-fall clips
        for seg in segs:
            if seg.label != "free_fall":
                continue
            n_ff_cand += 1
            idx = np.flatnonzero(tr.usable[seg.start:seg.end + 1]) + seg.start
            if len(idx) < args.min_free_frames:
                rej("free_fall", "too_short"); continue
            fit, n, rms_cm = side_fit(tr, idx, cfg, tr.t_rel[idx[0]])
            if rms_cm > args.max_fit_rms:
                rej("free_fall", "fit_rms"); continue
            speeds = [np.linalg.norm(model_state(fit, tau, cfg)[1]) for tau in (0.0, tr.t_rel[idx[-1]] - tr.t_rel[idx[0]])]
            if np.mean(speeds) < args.min_speed:
                rej("free_fall", "slow"); continue
            d_pad = float(np.min(np.linalg.norm(tr.puck_xy[idx] - tr.paddle_xy[idx], axis=1)))
            v0 = model_state(fit, 0.0, cfg)[1]
            name = f"{stem}_free_fall_{seg.start}_{seg.end}.hdf5"
            attrs = dict(kind="free_fall", source=str(path), start=seg.start, end=seg.end, n_usable=int(n),
                         fit_rms_cm=float(rms_cm), v0_x=float(v0[0]), v0_y=float(v0[1]),
                         speed0=float(np.linalg.norm(v0)), min_paddle_dist=d_pad, **common)
            write_slice(path, args.out / "free_fall" / name, seg.start, seg.end, attrs)
            rows.append({"file": f"free_fall/{name}", **attrs})

        # ---- wall bounces with determinable pre/post velocities
        for i, seg in enumerate(segs):
            if seg.label != "wall_collision" or not seg.meta.get("events"):
                continue
            n_wall_cand += 1
            events = seg.meta["events"]
            if len(events) != 1:
                rej("wall", "multi_event"); continue
            ev = events[0]
            if ev["gap_frames"] > args.max_gap:
                rej("wall", "gap"); continue
            prev_seg = segs[i - 1] if i > 0 else None
            next_seg = segs[i + 1] if i + 1 < len(segs) else None
            if prev_seg is None or next_seg is None or prev_seg.label != "free_fall" or next_seg.label != "free_fall":
                rej("wall", "no_free_fall_neighbours"); continue
            a, b = ev["pre_end"], ev["split"]
            pre_idx = np.flatnonzero(tr.usable[prev_seg.start:a + 1]) + prev_seg.start
            post_idx = np.flatnonzero(tr.usable[b:next_seg.end + 1]) + b
            pre_idx = pre_idx[-args.max_side_frames:]
            post_idx = post_idx[:args.max_side_frames]
            if len(pre_idx) < args.min_side_frames or len(post_idx) < args.min_side_frames:
                rej("wall", "short_side"); continue
            t_imp = 0.5 * (tr.t_rel[a] + tr.t_rel[b])
            fpre, npre, rms_pre = side_fit(tr, pre_idx, cfg, t_imp)
            fpost, npost, rms_post = side_fit(tr, post_idx, cfg, t_imp)
            if rms_pre > args.max_side_rms or rms_post > args.max_side_rms:
                rej("wall", "side_rms"); continue
            lo, hi = int(pre_idx[0]), int(post_idx[-1])
            d_pad = float(np.min(np.linalg.norm(tr.puck_xy[lo:hi + 1] - tr.paddle_xy[lo:hi + 1], axis=1)))
            if d_pad < args.min_paddle_dist:
                rej("wall", "paddle_near"); continue
            p_pre, v_pre = model_state(fpre, 0.0, cfg)
            p_post, v_post = model_state(fpost, 0.0, cfg)
            speed_pre, speed_post = float(np.linalg.norm(v_pre)), float(np.linalg.norm(v_post))
            if speed_pre < args.min_pre_speed:
                rej("wall", "slow"); continue
            side = ev["wall"]
            axis = 0 if side.startswith("x") else 1
            sgn = 1.0 if side.endswith("+") else -1.0
            n_pre, n_post = sgn * v_pre[axis], sgn * v_post[axis]          # into the wall positive
            t_pre, t_post = v_pre[1 - axis], v_post[1 - axis]
            if n_pre <= 0 or n_post >= 0:
                rej("wall", "no_reversal_in_refit"); continue
            name = f"{stem}_wall_{side}_{lo}_{hi}.hdf5"
            attrs = dict(kind="wall", source=str(path), start=lo, end=hi, impact_pre_frame=a - lo, impact_post_frame=b - lo,
                         wall=side, gap_frames=ev["gap_frames"], n_pre=int(npre), n_post=int(npost),
                         rms_pre_cm=float(rms_pre), rms_post_cm=float(rms_post),
                         v_pre_x=float(v_pre[0]), v_pre_y=float(v_pre[1]), v_post_x=float(v_post[0]), v_post_y=float(v_post[1]),
                         speed_pre=speed_pre, speed_post=speed_post, speed_ratio=speed_post / speed_pre,
                         normal_in=float(n_pre), normal_out=float(-n_post), normal_ratio=float(-n_post / n_pre),
                         tangential_in=float(t_pre), tangential_out=float(t_post),
                         tangential_ratio=float(t_post / t_pre) if abs(t_pre) > 0.05 else float("nan"),
                         impact_x=float(0.5 * (p_pre[0] + p_post[0])), impact_y=float(0.5 * (p_pre[1] + p_post[1])),
                         min_paddle_dist=d_pad, **common)
            write_slice(path, wall_out / "wall" / name, lo, hi, attrs)
            rows.append({"file": f"wall/{name}", **attrs})

    # ---- manifest + summary (one per output directory)
    ff = [r for r in rows if r["kind"] == "free_fall"]
    wl = [r for r in rows if r["kind"] == "wall"]
    header = [f"Input: {args.input_dir} ({len(files)} files). Calibration: puck_x_sign={cal['puck_x_sign']}, "
              f"paddle x = {cal['paddle_x_sign']:+d}·pose_x {cal['paddle_x_offset']:+.3f}.", ""]
    ff_lines = [f"## Free-fall clips: {len(ff)} kept of {n_ff_cand} segments",
                f"rejections: {reject['free_fall']}",
                f"usable frames: total {sum(r['n_usable'] for r in ff)}, median {np.median([r['n_usable'] for r in ff]) if ff else float('nan'):.0f}, max {max((r['n_usable'] for r in ff), default=0)}",
                f"fit rms (cm): median {np.median([r['fit_rms_cm'] for r in ff]) if ff else float('nan'):.2f}, p90 {np.percentile([r['fit_rms_cm'] for r in ff], 90) if ff else float('nan'):.2f}",
                f"speed0 (m/s): median {np.median([r['speed0'] for r in ff]) if ff else float('nan'):.2f}, p90 {np.percentile([r['speed0'] for r in ff], 90) if ff else float('nan'):.2f}", "",
                "Files: `free_fall/*.hdf5` (sim frame), `manifest.csv` / `manifest.json` (one row per clip, all metrics)."]
    wl_lines = [f"## Wall bounces with pre/post velocities: {len(wl)} kept of {n_wall_cand} segments",
                f"rejections: {reject['wall']}", "",
                "| wall | n | normal_in median (m/s) | normal_ratio median [p25, p75] | speed_ratio median | tangential_ratio median |",
                "|---|---|---|---|---|---|"]
    for side in ("x+", "x-", "y+", "y-"):
        s = [r for r in wl if r["wall"] == side]
        if not s:
            continue
        nr = [r["normal_ratio"] for r in s]; tr_ = [r["tangential_ratio"] for r in s if np.isfinite(r["tangential_ratio"])]
        wl_lines.append(f"| {side} | {len(s)} | {np.median([r['normal_in'] for r in s]):.2f} | {np.median(nr):.2f} [{np.percentile(nr, 25):.2f}, {np.percentile(nr, 75):.2f}] | "
                        f"{np.median([r['speed_ratio'] for r in s]):.2f} | {np.median(tr_) if tr_ else float('nan'):.2f} |")
    wl_lines += ["", "Files: `wall/*.hdf5` (sim frame), `manifest.csv` / `manifest.json` (one row per bounce, all metrics).",
                 "Wall attrs: `impact_pre_frame` / `impact_post_frame` index the last clean frame before and first clean frame after impact within the slice."]

    def write_dir(out: Path, kept: list[dict], title: str, body: list[str]):
        keys = sorted({k for r in kept for k in r}, key=lambda k: (k != "file", k))
        with open(out / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(kept)
        with open(out / "manifest.json", "w") as f:
            json.dump({"calibration": cal, "config": asdict(cfg), "args": {k: str(v) for k, v in vars(args).items()},
                       "sources": [str(p) for p in files], "rejections": reject, "sections": kept}, f, indent=1, default=_json_default)
        (out / "summary.md").write_text("\n".join([title, ""] + header + body) + "\n")

    if wall_out == args.out:
        write_dir(args.out, rows, "# Extracted sysid sections", ff_lines + [""] + wl_lines)
    else:
        write_dir(args.out, ff, "# Extracted sysid sections — puck free flight", ff_lines)
        write_dir(wall_out, wl, "# Extracted sysid sections — puck–wall collisions", wl_lines)
    print("\n".join(header + ff_lines + [""] + wl_lines))


if __name__ == "__main__":
    main()
