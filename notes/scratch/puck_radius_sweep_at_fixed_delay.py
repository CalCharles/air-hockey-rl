"""Fine-grained puck_radius sweep at fixed delay (+20% from source).

Holds every other knob at the sim2sim_combined_v2 settings (normal jitter,
range 0.35, paddle_radius restored to source, action_delay enabled). Sweeps
puck_radius from 0% to -70% perturbation in 10% steps to map the decay of
zero-shot return as the puck shrinks.

Output: per-setting metrics.json under
``runs/td3/sim2sim/perturbation_sweep/puck_radius_decay/<label>/`` plus a
``summary.md`` table at the campaign root.
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
# v2 already encodes: delay_seconds 0.035 (+40%), normal jitter range 0.35,
# paddle_radius restored, plus the carryover sim2sim_combined perturbations
# (pid_kp=7200, wall_cone=25, action_delay=True). We then override
# delay_seconds back to 0.030 (+20%) and sweep puck_radius.
BASE_CONFIG = os.path.join(
    REPO,
    "scripts/smooth_policy/amp_history/configs/new_juggle/sim2sim_combined_v2.yaml",
)
OUT_ROOT = os.path.join(REPO, "runs/td3/sim2sim/perturbation_sweep/puck_radius_decay")
N_EPISODES = 50
SEED = 0
N_GIFS = 0  # no GIFs in the sweep; eyeball v3 / v4 results for qualitative
SOURCE_PUCK_RADIUS = 0.03175
FIXED_DELAY_SECONDS = 0.030  # +20% from source

# (pct_shrink, puck_radius_meters)
SETTINGS = [
    (pct, round(SOURCE_PUCK_RADIUS * (1.0 - pct / 100.0), 5))
    for pct in (0, 10, 20, 30, 40, 50, 60, 70)
]


def _load_base_cfg():
    with open(BASE_CONFIG, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _write_perturbed_cfg(base_cfg: dict, sim_param_overrides: dict, dst_path: str) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["air_hockey"]["simulator_params"].update(sim_param_overrides)
    with open(dst_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main() -> int:
    base_cfg = _load_base_cfg()
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    for pct, puck_r in SETTINGS:
        label = f"pct{pct:02d}_pr{int(round(puck_r * 1e5)):05d}"
        out_dir = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        override = {"puck_radius": puck_r, "delay_seconds": FIXED_DELAY_SECONDS}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{label}_", delete=False
        ) as tf:
            tmp_cfg_path = tf.name
        _write_perturbed_cfg(base_cfg, override, tmp_cfg_path)
        print(f"[puck_radius] {label} :: pct={pct} puck_r={puck_r}m", flush=True)
        metrics = evaluate_zero_shot(
            checkpoint_path=CHECKPOINT,
            target_config_path=tmp_cfg_path,
            out_dir=out_dir,
            n_episodes=N_EPISODES,
            seed=SEED,
            save_gif=(N_GIFS > 0),
            n_gifs=N_GIFS,
        )
        os.unlink(tmp_cfg_path)
        per_ep = metrics["per_episode_returns"]
        rows.append({
            "pct_shrink": pct,
            "puck_radius_m": puck_r,
            "mean": metrics["mean_return"],
            "median": metrics["median_return"],
            "std": metrics["std_return"],
            "max": metrics["max_return"],
            "n_zero": int(sum(1 for r in per_ep if r == 0.0)),
            "n_ge100": int(sum(1 for r in per_ep if r >= 100)),
        })

    md = [
        "# hist2_motion0 — puck_radius decay at fixed delay +20%",
        "",
        f"- Checkpoint: `{os.path.relpath(CHECKPOINT, REPO)}`",
        f"- Base config: `{os.path.relpath(BASE_CONFIG, REPO)}` (overrides puck_radius, delay_seconds)",
        f"- Fixed delay_seconds: {FIXED_DELAY_SECONDS} (+20% from source 0.025)",
        "- Jitter: normal, delay_relative_range 0.35 (clipped)",
        f"- Episodes per setting: {N_EPISODES}, seed={SEED}",
        "",
        "| pct_shrink | puck_radius (m) | mean | median | std | max | n_zero | n>=100 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['pct_shrink']}% | {r['puck_radius_m']:.5f} | "
            f"{r['mean']:.2f} | {r['median']:.2f} | {r['std']:.2f} | "
            f"{r['max']:.2f} | {r['n_zero']} | {r['n_ge100']} |"
        )

    with open(os.path.join(OUT_ROOT, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
