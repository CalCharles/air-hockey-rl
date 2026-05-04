"""Per-perturbation sim2sim sweeps for the hist2_motion0 policy.

For each of three perturbation knobs (PID Kp, action delay, wall-bounce angle
cone) sweeps a range of values while holding *all* other sim params at the
source baseline (sysid_best_params_hist2.yaml). Reuses
``scripts/smooth_policy/sim2sim_eval.py``'s ``evaluate_zero_shot`` so the
metric definitions stay aligned with the rest of the sim2sim infra.

Output: per-setting metrics.json under
``runs/td3/sim2sim/hist2_motion0_to_sweeps/<knob>/<label>/`` plus a
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
CHECKPOINT = os.path.join(
    REPO, "runs/td3/hist_motion_collision/hist2_motion0/checkpoint_975000/model.pth"
)
BASE_CONFIG = os.path.join(
    REPO,
    "scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist2.yaml",
)
OUT_ROOT = os.path.join(REPO, "runs/td3/sim2sim/hist2_motion0_to_sweeps")
N_EPISODES = 25
SEED = 0


SWEEPS = [
    {
        "knob": "pid_kp",
        "settings": [
            ("kp_9000_pct0",  {"pid_kp": 9000}),
            ("kp_8100_pct10", {"pid_kp": 8100}),
            ("kp_7200_pct20", {"pid_kp": 7200}),
            ("kp_6300_pct30", {"pid_kp": 6300}),
            ("kp_5400_pct40", {"pid_kp": 5400}),
            ("kp_4500_pct50", {"pid_kp": 4500}),
        ],
    },
    {
        "knob": "action_delay",
        "settings": [
            ("delay_off_baseline",  {"enable_action_delay": False, "delay_seconds": 0.025}),
            ("delay_on_0p025",      {"enable_action_delay": True,  "delay_seconds": 0.025}),
            ("delay_on_0p030",      {"enable_action_delay": True,  "delay_seconds": 0.030}),
            ("delay_on_0p035",      {"enable_action_delay": True,  "delay_seconds": 0.035}),
            ("delay_on_0p040",      {"enable_action_delay": True,  "delay_seconds": 0.040}),
            ("delay_on_0p045",      {"enable_action_delay": True,  "delay_seconds": 0.045}),
        ],
    },
    {
        "knob": "wall_cone_deg",
        "settings": [
            ("wall_10deg_baseline", {"wall_direction_cone_deg": 10}),
            ("wall_20deg",          {"wall_direction_cone_deg": 20}),
            ("wall_30deg",          {"wall_direction_cone_deg": 30}),
            ("wall_40deg",          {"wall_direction_cone_deg": 40}),
            ("wall_50deg",          {"wall_direction_cone_deg": 50}),
            ("wall_60deg",          {"wall_direction_cone_deg": 60}),
        ],
    },
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
    for sweep in SWEEPS:
        knob = sweep["knob"]
        for label, override in sweep["settings"]:
            out_dir = os.path.join(OUT_ROOT, knob, label)
            os.makedirs(out_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", prefix=f"{label}_", delete=False
            ) as tf:
                tmp_cfg_path = tf.name
            _write_perturbed_cfg(base_cfg, override, tmp_cfg_path)
            print(f"[{knob}] {label} :: {override}", flush=True)
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
                "knob": knob,
                "label": label,
                "override": override,
                "mean_return": metrics["mean_return"],
                "std_return": metrics["std_return"],
                "median_return": metrics["median_return"],
                "max_return": metrics["max_return"],
                "n_zero_eps": int(sum(1 for r in metrics["per_episode_returns"] if r == 0.0)),
            })

    md_lines = [
        "# hist2_motion0 — single-knob sim2sim sweeps",
        "",
        f"- Checkpoint: `{os.path.relpath(CHECKPOINT, REPO)}`",
        f"- Base sim: `{os.path.relpath(BASE_CONFIG, REPO)}` (one knob varied at a time, all others held at baseline)",
        f"- Episodes per setting: {N_EPISODES}, seed={SEED}",
        "",
    ]
    for sweep in SWEEPS:
        knob = sweep["knob"]
        md_lines += [f"## {knob}", "", "| label | override | mean | median | std | max | n_zero |",
                     "|---|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            if r["knob"] != knob:
                continue
            md_lines.append(
                f"| {r['label']} | {json.dumps(r['override'])} "
                f"| {r['mean_return']:.2f} | {r['median_return']:.1f} | {r['std_return']:.2f} "
                f"| {r['max_return']:.0f} | {r['n_zero_eps']} |"
            )
        md_lines.append("")

    with open(os.path.join(OUT_ROOT, "summary.md"), "w") as f:
        f.write("\n".join(md_lines))
    with open(os.path.join(OUT_ROOT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== SUMMARY ===")
    print("\n".join(md_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
