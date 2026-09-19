#!/usr/bin/env python
"""Overlay real wall bounces with the Box2D replay under the identified restitution.

For verification and visualisation of a ``fit_walls.py`` run: picks representative validation
bounces (spread over the exit-speed-error distribution: 10 / 30 / 50 / 70 / 90 % for ``--n 5``),
replays each in the env built from the run's ``sim_config_fitted.yaml`` exactly as the fit does
(``replay_bounce``: puck at the fitted state of the first pre-impact frame, static paddle at its
real position, one step per real frame) and draws both on the table:

    real puck       the env's puck sprite + black trail (tracker samples; the sprite is absent on
                    occluded / stale frames)
    Box2D puck      blue ghost + trail (the simulator under the fitted wall restitution)
    fitted models   thin grey lines — the pre / post free-flight fits the exit speeds come from
    contact line    thin grey line — the sim's puck-centre contact line of that wall

Outputs in ``<results>/overlays/``: one GIF + last-frame PNG per example, ``mosaic.png``,
``overlay_summary.md`` / ``.csv`` with per-example exit speeds, angles and position errors.

    python sysid/wall_collision/code/render_overlays.py --results-dir sysid/wall_collision/results/mouse_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sysid.common.trajectory_segmentation import SegmentationConfig, fit_damped, model_state  # noqa: E402
from sysid.common.sysid_dataset import WallBounce, load_manifest, make_wall_bounces  # noqa: E402
from sysid.common.table_scene import (  # noqa: E402
    C_MODEL, C_REAL, C_SIM, TableScene, legend_strip, mosaic, pick_percentiles, write_gif, write_png,
)
from sysid.wall_collision.code.wall_restitution_fit import END_WALLS, SIDE_WALLS, bounce_state, build_env, replay_bounce  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True, help="a fit_walls.py output folder (results.json, sim_config_fitted.yaml)")
    p.add_argument("--sections-dir", type=Path, default=None, help="default: the one recorded in results.json")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--walls", choices=["side", "end", "all"], default="side", help="side = y± (identified), end = x± (not identified)")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--extra-steps", type=int, default=8, help="sim steps drawn after the real post-impact frame")
    p.add_argument("--out", type=Path, default=None, help="default <results-dir>/overlays")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--fps", type=int, default=10, help="GIF frame rate (real data are 20 Hz; 10 = half speed)")
    p.add_argument("--hold-last", type=int, default=8)
    return p.parse_args(argv)


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def replay(env, b: WallBounce, cfg: SegmentationConfig, max_side_frames: int, extra_steps: int) -> dict:
    st = bounce_state(b, cfg, max_side_frames)
    r = replay_bounce(env, st, b.paddle_xy_a, extra_steps=extra_steps)
    pre, post = b.pre_idx[-max_side_frames:], b.post_idx[:max_side_frames]
    r0 = int(b.usable_idx[pre[0]])                      # raw frame the sim starts from
    sim = np.vstack([st.p_start[None], r["poss"]])      # sim[f] = state at raw frame r0 + f
    n_frames = len(sim)
    raw = b.usable_idx - r0                             # raw frame of every real usable sample, relative to r0
    keep = (raw >= 0) & (raw < n_frames)
    real_frames, real_xy = raw[keep], b.xy[keep]
    # model trails (pre fit around t_a, post fit around t_b), on the real usable timestamps
    t_a, t_b = b.t[pre[-1]], b.t[post[0]]
    fpre, fpost = fit_damped(b.t[pre] - t_a, b.xy[pre], cfg), fit_damped(b.t[post] - t_b, b.xy[post], cfg)
    model_pre = np.asarray([model_state(fpre, float(b.t[i] - t_a), cfg)[0] for i in pre])
    model_post = np.asarray([model_state(fpost, float(b.t[i] - t_b), cfg)[0] for i in post])
    # per-frame position error where a real sample exists
    err = {int(f): 100 * float(np.linalg.norm(sim[f] - xy)) for f, xy in zip(real_frames, real_xy)}
    post_err = [err[int(f)] for f in real_frames if f >= st.steps_to_b and int(f) in err]
    v_exit = r["v_exit"]
    real_ang, sim_ang = float(np.arctan2(st.v_b[1], st.v_b[0])), float(np.arctan2(v_exit[1], v_exit[0]))
    return {"st": st, "sim": sim, "real_frames": real_frames, "real_xy": real_xy, "model_pre": model_pre, "model_post": model_post,
            "bounced": bool(r["bounced"]), "bounce_step": r["bounce_step"], "steps_to_b": st.steps_to_b, "hidden_frames": int(st.n_steps - 1), "err": err,
            "speed_in": float(np.linalg.norm(st.v_a)), "normal_in": st.normal_in,
            "real_exit": st.speed_out_real, "sim_exit": float(np.linalg.norm(v_exit)),
            "real_normal_out": st.normal_out_real, "sim_normal_out": float(-(v_exit @ st.normal)),
            "speed_rel_err": abs(float(np.linalg.norm(v_exit)) - st.speed_out_real) / max(st.speed_out_real, 1e-3),
            "angle_err_deg": float(np.degrees(abs(_wrap(sim_ang - real_ang)))),
            "post_pos_err_cm": float(np.mean(post_err)) if post_err else float("nan"),
            "rms_pre_cm": st.rms_pre_cm, "rms_post_cm": st.rms_post_cm}


def main(argv=None):
    a = parse_args(argv)
    res = json.load(open(a.results_dir / "results.json"))
    sections_dir = a.sections_dir or Path(res["sections_dir"])
    g, gam = float(res["puck_params_used"]["gravity"]), float(res["puck_params_used"]["puck_damping"])
    cfg = SegmentationConfig(gravity_x=g, damping=gam)
    max_side = int(res["args"]["max_side_frames"])
    fitted = {w["param"]: float(w["best"]) for w in res["walls"]}
    sim_cfg = a.results_dir / "sim_config_fitted.yaml"
    out = a.out or (a.results_dir / "overlays")
    for sub in ("gifs", "png"):                      # a re-render replaces the previous example set
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old_file in (out / sub).glob("example*"):
            old_file.unlink()

    manifest = load_manifest(sections_dir)
    sources = set(res["split_sources"]["train"] + res["split_sources"]["val"]) if a.split == "all" else set(res["split_sources"][a.split])
    walls = {"side": SIDE_WALLS, "end": END_WALLS, "all": SIDE_WALLS + END_WALLS}[a.walls]
    rows = [r for r in manifest["sections"] if r["kind"] == "wall" and r["source"] in sources and r["wall"] in walls]
    bounces = make_wall_bounces(rows, sections_dir, SegmentationConfig())
    env = build_env(sim_cfg)
    replays = [replay(env, b, cfg, max_side, a.extra_steps) for b in bounces]
    ok = [i for i, r in enumerate(replays) if r["bounced"]]
    errs = np.array([replays[i]["speed_rel_err"] for i in ok])
    picks = [ok[j] for j in pick_percentiles(errs, a.n)]
    print(f"{len(bounces)} {a.split} {a.walls}-wall bounces, {len(ok)} reproduced by the sim; exit-speed rel err p10/p50/p90 = "
          f"{100 * np.percentile(errs, 10):.0f}/{100 * np.percentile(errs, 50):.0f}/{100 * np.percentile(errs, 90):.0f} %; picked {len(picks)}")

    scene = TableScene(env)
    param_txt = ", ".join(f"{k.replace('_wall_restitution', '')}={v:.3f}" for k, v in fitted.items())
    legend = legend_strip([("real (tracker)", C_REAL), (f"Box2D  {param_txt}  g={g:+.3f} gamma={gam:.3f}", C_SIM), ("pre / post fits", C_MODEL)], a.width)
    summaries, table = [], []
    for i, k in enumerate(picks, 1):
        b, r = bounces[k], replays[k]
        pct = 100 * (np.searchsorted(np.sort(errs), r["speed_rel_err"]) / max(len(errs) - 1, 1))
        name = f"example{i}_{Path(b.clip).stem}"
        head = [(f"#{i}  {Path(b.clip).stem}  wall {b.wall}   in {r['speed_in']:.2f} m/s (normal {r['normal_in']:.2f})", C_REAL),
                (f"exit speed real {r['real_exit']:.2f} / Box2D {r['sim_exit']:.2f} m/s ({100 * r['speed_rel_err']:.0f} % err, p{pct:.0f} of {a.split})   "
                 f"normal out {r['real_normal_out']:.2f} / {r['sim_normal_out']:.2f}   angle err {r['angle_err_deg']:.1f} deg", C_SIM)]
        frames = []
        n_frames = len(r["sim"])
        for f in range(n_frames):
            img = scene.table()
            scene.draw_wall_line(img, b.wall)
            scene.draw_trail(img, r["model_pre"], C_MODEL, 1, dots=False)
            scene.draw_trail(img, r["model_post"], C_MODEL, 1, dots=False)
            scene.draw_sprite(img, b.paddle_xy_a, "paddle")
            shown = r["real_frames"] <= f
            scene.draw_trail(img, r["real_xy"][shown], C_REAL, 2)
            scene.draw_trail(img, r["sim"][: f + 1], C_SIM, 1, dots=False)
            here = np.flatnonzero(r["real_frames"] == f)
            if len(here):
                scene.draw_sprite(img, r["real_xy"][here[0]], "puck")
            scene.draw_ghost(img, r["sim"][f], C_SIM)
            e = r["err"].get(f)
            phase = "pre" if f < r["steps_to_b"] else ("post" if f > r["steps_to_b"] else "first post frame")
            pos_txt = f"Box2D - real {e:4.1f} cm" if e is not None else "real sample occluded"
            labels = head + [(f"frame {f:2d}/{n_frames - 1} {phase} | sim bounce step {r['bounce_step']} | impact hidden {r['hidden_frames']} fr | "
                              f"{pos_txt} | post mean {r['post_pos_err_cm']:.1f} cm", C_SIM)]
            frames.append(np.vstack([legend, scene.finish(img, a.width, labels)]))
        frames += [frames[-1]] * a.hold_last
        write_gif(frames, out / "gifs" / f"{name}.gif", a.fps)
        write_png(frames[-1], out / "png" / f"{name}.png")
        summaries.append(frames[-1])
        row = {"example": i, "clip": b.clip, "source": Path(b.source).name, "wall": b.wall, "percentile_of_split": round(pct),
               **{k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in r.items() if k_ in ("speed_in", "normal_in", "real_exit", "sim_exit", "speed_rel_err", "real_normal_out", "sim_normal_out", "angle_err_deg", "post_pos_err_cm", "bounce_step", "steps_to_b", "hidden_frames", "rms_pre_cm", "rms_post_cm")}}
        table.append(row)
        print(f"  #{i} {b.clip} {b.wall}: in {r['speed_in']:.2f} | exit real {r['real_exit']:.2f} sim {r['sim_exit']:.2f} m/s ({100 * r['speed_rel_err']:.0f} %) | angle err {r['angle_err_deg']:.1f} deg | post pos err {r['post_pos_err_cm']:.1f} cm")
    write_png(mosaic(summaries, ncols=1 if len(summaries) <= 3 else 2), out / "mosaic.png")
    with open(out / "overlay_summary.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(table[0])); wr.writeheader(); wr.writerows(table)
    lines = [f"# Puck–wall collisions — real vs Box2D overlays ({a.split} {a.walls}-wall bounces, {param_txt}, puck g = {g:+.3f}, γ = {gam:.3f})", "",
             f"{len(picks)} bounces picked at spread percentiles of the exit-speed relative error over the {len(ok)} {a.split} bounces the sim reproduced "
             f"(of {len(bounces)}; p10/p50/p90 = {100 * np.percentile(errs, 10):.0f}/{100 * np.percentile(errs, 50):.0f}/{100 * np.percentile(errs, 90):.0f} %). "
             f"Replay as in the fit: Box2D env from `{sim_cfg.name}`, puck at the fitted state of the first pre-impact frame, static paddle at its real "
             f"position, one step per real frame, {a.extra_steps} extra steps after the real post-impact frame. The real impact is hidden by the camera "
             "for one or more frames; the sim's contact is where the blue trail reverses.", "",
             "**Read the position errors with the wall-line offset in mind.** The fit compares exit *velocities*; positions after the bounce are shifted because "
             "the puck frame extends past the sim table at some walls (measured puck-centre apex vs sim contact line: "
             + ", ".join(f"{k} {v['apex_p50']:.3f} vs {v['sim_contact_line']:.3f} m" for k, v in sorted(res.get("apparent_wall_lines", {}).items()))
             + "), so the sim puck turns around ~1 frame earlier than the real one at y− and x+ and then leads it along the exit direction. "
             "The pre / post fits (grey) show the real velocity model the exit speed is read from; the sim's exit speed is compared with the post fit's.", "",
             "| # | clip | wall | in (m/s) | normal in | exit real | exit Box2D | rel err | percentile | angle err | normal out real / Box2D | post-impact pos err (cm) | sim bounce step / first real post frame | impact hidden (frames) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        lines.append(f"| {r['example']} | `{Path(r['clip']).stem}` | {r['wall']} | {r['speed_in']:.2f} | {r['normal_in']:.2f} | {r['real_exit']:.2f} | {r['sim_exit']:.2f} | {100 * r['speed_rel_err']:.0f} % | p{r['percentile_of_split']} | {r['angle_err_deg']:.1f}° | {r['real_normal_out']:.2f} / {r['sim_normal_out']:.2f} | {r['post_pos_err_cm']:.1f} | {r['bounce_step']} / {r['steps_to_b']} | {r['hidden_frames']} |")
    lines += ["", "Files: `gifs/example*.gif` (real sprite + black trail, Box2D blue ghost + trail, pre / post fits grey, wall contact line grey; half speed), `png/` last frames, `mosaic.png`, `overlay_summary.csv`.", ""]
    (out / "overlay_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
