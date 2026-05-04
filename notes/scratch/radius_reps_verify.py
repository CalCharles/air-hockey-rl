"""Verification rerun of the 2026-04-27 mass-preserved radius sweep.

Runs ~9 representatives spanning both curves (source + paddle 20/40/50/
60/70% + puck 30/50/70%) at n_episodes=50 (matching the doc) with 10
GIFs per setting for qualitative inspection. Output under
``runs/td3/sim2sim/perturbation_sweep_reps_verify/``.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.smooth_policy.sim2sim_eval import evaluate_zero_shot

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT = os.path.join(REPO, "latest_model/hist2_motion0/model.pth")
BASE_CONFIG = os.path.join(
    REPO,
    "scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_combined_v2.yaml",
)
OUT_ROOT = os.path.join(
    REPO, "runs/td3/sim2sim/perturbation_sweep_reps_verify"
)
N_EPISODES = 50
SEED = 0
N_GIFS = 10
FIXED_DELAY_SECONDS = 0.030  # +20% from source

SOURCE_PUCK_RADIUS = 0.03175
SOURCE_PADDLE_RADIUS = 0.0508


def _paddle(pct):
    r = round(SOURCE_PADDLE_RADIUS * (1.0 - pct / 100.0), 5)
    return {
        "label": f"paddle_pct{pct:02d}_r{int(round(r * 1e5)):05d}",
        "doc_mean_n50": None,
        "overrides": {
            "paddle_radius": r,
            "paddle_mass_reference_radius": SOURCE_PADDLE_RADIUS,
            "puck_radius": SOURCE_PUCK_RADIUS,
            "puck_mass_reference_radius": None,
        },
    }


def _puck(pct):
    r = round(SOURCE_PUCK_RADIUS * (1.0 - pct / 100.0), 5)
    return {
        "label": f"puck_pct{pct:02d}_r{int(round(r * 1e5)):05d}",
        "doc_mean_n50": None,
        "overrides": {
            "paddle_radius": SOURCE_PADDLE_RADIUS,
            "paddle_mass_reference_radius": None,
            "puck_radius": r,
            "puck_mass_reference_radius": SOURCE_PUCK_RADIUS,
        },
    }


# Source row matches the puck-sweep pct=0 override exactly (mass-ref
# flag set to source value; no-op numerically but identical code path).
SOURCE_REP = {
    "label": "source",
    "doc_mean_n50": 98.02,
    "overrides": {
        "paddle_radius": SOURCE_PADDLE_RADIUS,
        "paddle_mass_reference_radius": None,
        "puck_radius": SOURCE_PUCK_RADIUS,
        "puck_mass_reference_radius": SOURCE_PUCK_RADIUS,
    },
}

REPS = [
    SOURCE_REP,
    _paddle(20),
    _paddle(40),
    _paddle(50),
    _paddle(60),
    _paddle(70),
    _puck(30),
    _puck(50),
    _puck(70),
]

# Fill in doc means (from notes/scratch/sim2sim_delay_puck_perturbation_sweep.md)
DOC_MEANS = {
    "source": 98.02,
    "paddle_pct20_r04064": 84.50,
    "paddle_pct40_r03048": 75.14,
    "paddle_pct50_r02540": 63.64,
    "paddle_pct60_r02032": 49.26,
    "paddle_pct70_r01524": 43.84,
    "puck_pct30_r02222": 88.14,
    "puck_pct50_r01588": 78.52,
    "puck_pct70_r00953": 76.18,
}
for r in REPS:
    r["doc_mean_n50"] = DOC_MEANS.get(r["label"])


def _load_base():
    with open(BASE_CONFIG, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _write_perturbed(base_cfg, sim_overrides, dst):
    cfg = copy.deepcopy(base_cfg)
    cfg["air_hockey"]["simulator_params"].update(sim_overrides)
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> int:
    base_cfg = _load_base()
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    for rep in REPS:
        label = rep["label"]
        out_dir = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        ov = dict(rep["overrides"])
        ov["delay_seconds"] = FIXED_DELAY_SECONDS
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{label}_", delete=False
        ) as tf:
            tmp = tf.name
        _write_perturbed(base_cfg, ov, tmp)
        print(f"[{label}] :: {ov} (doc n=50 mean={rep['doc_mean_n50']})", flush=True)
        metrics = evaluate_zero_shot(
            checkpoint_path=CHECKPOINT,
            target_config_path=tmp,
            out_dir=out_dir,
            n_episodes=N_EPISODES,
            seed=SEED,
            save_gif=True,
            n_gifs=N_GIFS,
        )
        os.unlink(tmp)
        per_ep = metrics["per_episode_returns"]
        rows.append({
            "label": label,
            "overrides": ov,
            "doc_mean_n50": rep["doc_mean_n50"],
            "mean": metrics["mean_return"],
            "median": metrics["median_return"],
            "std": metrics["std_return"],
            "max": metrics["max_return"],
            "n_zero": int(sum(1 for r in per_ep if r == 0.0)),
            "n_ge100": int(sum(1 for r in per_ep if r >= 100)),
            "per_episode_returns": per_ep,
        })

    md = [
        "# Verification rerun — radius reps at n=50 with 10 GIFs each",
        "",
        f"- Checkpoint: `{os.path.relpath(CHECKPOINT, REPO)}`",
        f"- Base config: `{os.path.relpath(BASE_CONFIG, REPO)}`",
        f"- Fixed delay_seconds: {FIXED_DELAY_SECONDS}",
        f"- Episodes per setting: {N_EPISODES}, seed={SEED}",
        f"- GIFs per setting: {N_GIFS} (at seeds {SEED + N_EPISODES} +)",
        "",
        "| label | doc mean (n=50) | rerun mean | Δ | median | std | max | n_zero | n>=100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        doc = r["doc_mean_n50"]
        delta = r["mean"] - doc if doc is not None else None
        delta_str = f"{delta:+.2f}" if delta is not None else "—"
        doc_str = f"{doc:.2f}" if doc is not None else "—"
        md.append(
            f"| {r['label']} | {doc_str} | {r['mean']:.2f} | {delta_str} | "
            f"{r['median']:.2f} | {r['std']:.2f} | {r['max']:.2f} | "
            f"{r['n_zero']} | {r['n_ge100']} |"
        )
    with open(os.path.join(OUT_ROOT, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
