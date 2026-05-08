"""2D zero-shot sweep over (puck_obs_sine_warp_amplitude × paddle_radius).

Locked: every other sim param matches `sysid_best_params_hist2.yaml` (the source
training env). Per the redesign plan, hist_len, all delays, all restitutions,
wall_cone, pid_kp etc. are held at source values — the only two perturbations
explored here are the new sine y-warp on puck observations and a mass-preserved
paddle-radius reduction.

Goal: find the smallest combined perturbation that drops `hist2_motion0_v2`
zero-shot mean below 60. Cheap pruning step before any from-scratch training.

Parallelization: this script runs ONE row of the grid (one paddle reduction
across all warp amplitudes). Launch 4 instances in parallel — one per GPU and
one paddle reduction — to cover the full 4×8 grid in ~12 min wall.

Usage (per GPU)::

    CUDA_VISIBLE_DEVICES=0 .venv/bin/python notes/scratch/sim2sim_warp_paddle_zs_sweep.py --paddle-pct 0
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python notes/scratch/sim2sim_warp_paddle_zs_sweep.py --paddle-pct 10
    CUDA_VISIBLE_DEVICES=2 .venv/bin/python notes/scratch/sim2sim_warp_paddle_zs_sweep.py --paddle-pct 20
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python notes/scratch/sim2sim_warp_paddle_zs_sweep.py --paddle-pct 30

Aggregate after all four finish::

    .venv/bin/python notes/scratch/sim2sim_warp_paddle_zs_sweep.py --aggregate
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.smooth_policy.sim2sim_eval import evaluate_zero_shot

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT = os.path.join(REPO, "latest_model/hist2_motion0_v2/model.pth")
BASE_CONFIG = os.path.join(
    REPO,
    "scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist2.yaml",
)
OUT_ROOT = os.path.join(REPO, "runs/td3/sim2sim/zs_warp_paddle_sweep")
N_EPISODES = 50
SEED = 0

SOURCE_PADDLE_RADIUS = 0.0508
PADDLE_PCTS = [0, 10, 20, 30]   # one row per GPU
WARP_AMPS = [0.0, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25]


def _paddle_radius_from_pct(pct: int) -> float:
    return round(SOURCE_PADDLE_RADIUS * (1.0 - pct / 100.0), 6)


def _cell_label(paddle_pct: int, warp_amp: float) -> str:
    warp_str = f"{warp_amp:.3f}".rstrip("0").rstrip(".") or "0"
    return f"p{paddle_pct:02d}_w{warp_str}"


def _load_base_cfg() -> dict:
    with open(BASE_CONFIG, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _write_perturbed_cfg(base_cfg: dict, sim_param_overrides: dict, dst_path: str) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["air_hockey"]["simulator_params"].update(sim_param_overrides)
    with open(dst_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_row(paddle_pct: int) -> list:
    base_cfg = _load_base_cfg()
    paddle_radius = _paddle_radius_from_pct(paddle_pct)
    rows = []
    for warp_amp in WARP_AMPS:
        label = _cell_label(paddle_pct, warp_amp)
        out_dir = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        override = {
            "paddle_radius": paddle_radius,
            "paddle_mass_reference_radius": SOURCE_PADDLE_RADIUS,  # mass-preserved vs source
            "puck_obs_sine_warp_amplitude": float(warp_amp),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{label}_", delete=False
        ) as tf:
            tmp_cfg_path = tf.name
        _write_perturbed_cfg(base_cfg, override, tmp_cfg_path)
        print(f"[{label}] paddle_pct={paddle_pct}  paddle_radius={paddle_radius}  warp={warp_amp}", flush=True)
        metrics = evaluate_zero_shot(
            checkpoint_path=CHECKPOINT,
            target_config_path=tmp_cfg_path,
            out_dir=out_dir,
            n_episodes=N_EPISODES,
            seed=SEED,
            save_gif=False,
        )
        os.unlink(tmp_cfg_path)
        rows.append({
            "label": label,
            "paddle_pct": paddle_pct,
            "paddle_radius": paddle_radius,
            "warp_amplitude": float(warp_amp),
            "mean_return": metrics["mean_return"],
            "std_return": metrics["std_return"],
            "median_return": metrics["median_return"],
            "max_return": metrics["max_return"],
            "n_zero_eps": int(sum(1 for r in metrics["per_episode_returns"] if r == 0.0)),
        })
    # Per-row JSON so the aggregator can pick up partial results without races.
    row_json = os.path.join(OUT_ROOT, f"row_p{paddle_pct:02d}.json")
    with open(row_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nrow done -> {row_json}\n", flush=True)
    return rows


def _aggregate() -> int:
    rows = []
    for pct in PADDLE_PCTS:
        path = os.path.join(OUT_ROOT, f"row_p{pct:02d}.json")
        if not os.path.exists(path):
            print(f"  missing {path} (row not finished yet?)", flush=True)
            continue
        with open(path) as f:
            rows.extend(json.load(f))
    if not rows:
        print("no rows to aggregate"); return 1

    # Build heatmap of mean_return as paddle_pct (rows) × warp_amp (cols).
    by = {(r["paddle_pct"], r["warp_amplitude"]): r for r in rows}
    md = [
        "# zs_warp_paddle_sweep — heatmap of source zero-shot mean return",
        "",
        f"- Source policy: `{os.path.relpath(CHECKPOINT, REPO)}`",
        f"- Base sim: `{os.path.relpath(BASE_CONFIG, REPO)}` (locked: delays, hist_len, restitutions, wall_cone, pid_kp)",
        f"- Episodes per cell: {N_EPISODES}",
        f"- Mass-preserved paddle (paddle_mass_reference_radius=0.0508)",
        "",
        "## Mean return (each cell shows mean ± std)",
        "",
    ]
    header = "| paddle\\\\warp | " + " | ".join(f"{w:g}" for w in WARP_AMPS) + " |"
    sep = "|" + "---|" * (1 + len(WARP_AMPS))
    md += [header, sep]
    for pct in PADDLE_PCTS:
        cells = [f"-{pct}%"]
        for w in WARP_AMPS:
            r = by.get((pct, w))
            if r is None:
                cells.append("·")
            else:
                m = r["mean_return"]; s = r["std_return"]
                marker = ""
                if m < 60: marker = " ✓"
                cells.append(f"{m:.1f}±{s:.0f}{marker}")
        md.append("| " + " | ".join(cells) + " |")
    md += ["", f"`✓` marks cells with mean < 60 (the mismatch goal).", ""]

    out = os.path.join(OUT_ROOT, "summary.md")
    with open(out, "w") as f:
        f.write("\n".join(md))
    out_json = os.path.join(OUT_ROOT, "summary.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print("\n".join(md))
    print(f"\n-> wrote {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--paddle-pct", type=int, choices=PADDLE_PCTS, default=None)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    if args.aggregate:
        return _aggregate()
    if args.paddle_pct is None:
        print("Pass --paddle-pct {0,10,20,30} or --aggregate"); return 2
    _run_row(args.paddle_pct)
    return 0


if __name__ == "__main__":
    sys.exit(main())
