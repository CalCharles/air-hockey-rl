"""Re-run a small set of representative radius perturbations at n=10.

Picks 5 representatives from the 2026-04-27 mass-preserved sweep
(source, paddle -50/-60/-70%, puck -70%) and re-evaluates each at
n_episodes=10 with seed=0. Also records 2 GIFs per setting at fresh
seeds for qualitative inspection. Output under
``runs/td3/sim2sim/perturbation_sweep_reps_n10/``.
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
    REPO, "runs/td3/sim2sim/perturbation_sweep_reps_n10"
)
N_EPISODES = 10
SEED = 0
N_GIFS = 2
FIXED_DELAY_SECONDS = 0.030  # +20% from source

SOURCE_PUCK_RADIUS = 0.03175
SOURCE_PADDLE_RADIUS = 0.0508

REPS = [
    {
        "label": "source",
        "overrides": {
            "paddle_radius": SOURCE_PADDLE_RADIUS,
            "paddle_mass_reference_radius": SOURCE_PADDLE_RADIUS,
            "puck_radius": SOURCE_PUCK_RADIUS,
            "puck_mass_reference_radius": SOURCE_PUCK_RADIUS,
        },
    },
    {
        "label": "paddle_pct50_r02540",
        "overrides": {
            "paddle_radius": 0.02540,
            "paddle_mass_reference_radius": SOURCE_PADDLE_RADIUS,
            "puck_radius": SOURCE_PUCK_RADIUS,
            "puck_mass_reference_radius": None,
        },
    },
    {
        "label": "paddle_pct60_r02032",
        "overrides": {
            "paddle_radius": 0.02032,
            "paddle_mass_reference_radius": SOURCE_PADDLE_RADIUS,
            "puck_radius": SOURCE_PUCK_RADIUS,
            "puck_mass_reference_radius": None,
        },
    },
    {
        "label": "paddle_pct70_r01524",
        "overrides": {
            "paddle_radius": 0.01524,
            "paddle_mass_reference_radius": SOURCE_PADDLE_RADIUS,
            "puck_radius": SOURCE_PUCK_RADIUS,
            "puck_mass_reference_radius": None,
        },
    },
    {
        "label": "puck_pct70_r00953",
        "overrides": {
            "paddle_radius": SOURCE_PADDLE_RADIUS,
            "paddle_mass_reference_radius": None,
            "puck_radius": 0.00953,
            "puck_mass_reference_radius": SOURCE_PUCK_RADIUS,
        },
    },
]


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
        print(f"[{label}] :: {ov}", flush=True)
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
            "mean": metrics["mean_return"],
            "median": metrics["median_return"],
            "std": metrics["std_return"],
            "max": metrics["max_return"],
            "n_zero": int(sum(1 for r in per_ep if r == 0.0)),
            "n_ge100": int(sum(1 for r in per_ep if r >= 100)),
            "per_episode_returns": per_ep,
        })

    md = [
        "# Representative radius reps — n=10",
        "",
        f"- Checkpoint: `{os.path.relpath(CHECKPOINT, REPO)}`",
        f"- Base config: `{os.path.relpath(BASE_CONFIG, REPO)}`",
        f"- Fixed delay_seconds: {FIXED_DELAY_SECONDS}",
        f"- Episodes per setting: {N_EPISODES}, seed={SEED}",
        f"- GIFs per setting: {N_GIFS} (at seeds {SEED + N_EPISODES} +)",
        "",
        "| label | mean | median | std | max | n_zero | n>=100 | per-episode |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        per_ep_str = ", ".join(f"{x:.0f}" for x in r["per_episode_returns"])
        md.append(
            f"| {r['label']} | {r['mean']:.2f} | {r['median']:.2f} | "
            f"{r['std']:.2f} | {r['max']:.2f} | {r['n_zero']} | "
            f"{r['n_ge100']} | {per_ep_str} |"
        )
    with open(os.path.join(OUT_ROOT, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
