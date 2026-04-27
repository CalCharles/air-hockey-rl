"""Independent puck_radius and paddle_radius sweeps, with mass preserved.

Runs two independent sweeps off the sim2sim_combined_v2 base (normal jitter
35%, action_delay=true, pid_kp=7200, wall_cone=25, delay fixed at 0.030 =
+20% from source):

  1. puck_radius sweep — holds paddle_radius at source 0.0508; varies
     puck_radius from 0% to -70% in 10% steps. Sets
     puck_mass_reference_radius=0.03175 so puck mass is conserved against
     the source value as the radius shrinks.

  2. paddle_radius sweep — holds puck_radius at source 0.03175; varies
     paddle_radius from 0% to -70% in 10% steps. Sets
     paddle_mass_reference_radius=0.0508 so paddle mass is conserved
     against the source value.

Outputs:
  runs/td3/sim2sim/perturbation_sweep/puck_radius_decay_mass_preserved/
  runs/td3/sim2sim/perturbation_sweep/paddle_radius_decay_mass_preserved/
each with a per-setting metrics.json and a summary.md / summary.json at
the campaign root.
"""

from __future__ import annotations

import copy
import json
import math
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
SWEEP_ROOT = os.path.join(REPO, "runs/td3/sim2sim/perturbation_sweep")
N_EPISODES = 50
SEED = 0
FIXED_DELAY_SECONDS = 0.030  # +20% from source

SOURCE_PUCK_RADIUS = 0.03175
SOURCE_PADDLE_RADIUS = 0.0508
PCT_LADDER = (0, 10, 20, 30, 40, 50, 60, 70)


def _load_base_cfg():
    with open(BASE_CONFIG, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _write_perturbed_cfg(base_cfg, sim_param_overrides, dst_path):
    cfg = copy.deepcopy(base_cfg)
    cfg["air_hockey"]["simulator_params"].update(sim_param_overrides)
    with open(dst_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_sweep(knob_name, source_radius, mass_ref_key, fixed_partner_overrides, out_root):
    """Sweep `knob_name` (one of paddle_radius, puck_radius) over the pct ladder.

    `mass_ref_key` is the corresponding *_mass_reference_radius key. The
    `fixed_partner_overrides` dict pins the *other* radius (and clears any
    mass-ref for that other knob to keep it isolated).
    """
    base_cfg = _load_base_cfg()
    os.makedirs(out_root, exist_ok=True)
    rows = []
    for pct in PCT_LADDER:
        radius = round(source_radius * (1.0 - pct / 100.0), 5)
        label = f"pct{pct:02d}_r{int(round(radius * 1e5)):05d}"
        out_dir = os.path.join(out_root, label)
        os.makedirs(out_dir, exist_ok=True)

        override = {
            knob_name: radius,
            mass_ref_key: source_radius,  # preserve mass against source radius
            "delay_seconds": FIXED_DELAY_SECONDS,
            **fixed_partner_overrides,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", prefix=f"{label}_", delete=False
        ) as tf:
            tmp = tf.name
        _write_perturbed_cfg(base_cfg, override, tmp)
        print(f"[{knob_name}] {label} :: {override}", flush=True)
        metrics = evaluate_zero_shot(
            checkpoint_path=CHECKPOINT,
            target_config_path=tmp,
            out_dir=out_dir,
            n_episodes=N_EPISODES,
            seed=SEED,
            save_gif=False,
        )
        os.unlink(tmp)
        per_ep = metrics["per_episode_returns"]

        # Record the box2d-derived effective density and mass for the swept
        # knob, to confirm mass preservation in the summary.
        sim = yaml.safe_load(open(BASE_CONFIG))["air_hockey"]["simulator_params"]
        sim.update(override)
        if knob_name == "paddle_radius":
            density_in = float(sim["paddle_density"])
            r_now = float(sim["paddle_radius"])
            density_eff = density_in * (source_radius / r_now) ** 2 if r_now > 0 else density_in
        else:
            density_in = float(sim["puck_density"])
            r_now = float(sim["puck_radius"])
            density_eff = density_in * (source_radius / r_now) ** 2 if r_now > 0 else density_in
        mass_eff = density_eff * math.pi * r_now ** 2

        rows.append({
            "pct_shrink": pct,
            f"{knob_name}_m": radius,
            "density_effective": density_eff,
            "mass_effective": mass_eff,
            "mean": metrics["mean_return"],
            "median": metrics["median_return"],
            "std": metrics["std_return"],
            "max": metrics["max_return"],
            "n_zero": int(sum(1 for r in per_ep if r == 0.0)),
            "n_ge100": int(sum(1 for r in per_ep if r >= 100)),
        })

    # Markdown summary
    md = [
        f"# hist2_motion0 — {knob_name} decay at fixed delay +20% (mass preserved)",
        "",
        f"- Checkpoint: `{os.path.relpath(CHECKPOINT, REPO)}`",
        f"- Base config: `{os.path.relpath(BASE_CONFIG, REPO)}`",
        f"- Fixed delay_seconds: {FIXED_DELAY_SECONDS} (+20% from source)",
        "- Jitter: normal, delay_relative_range 0.35 (clipped)",
        f"- Episodes per setting: {N_EPISODES}, seed={SEED}",
        f"- Mass-preserve flag: {mass_ref_key}={source_radius} (effective density auto-scaled)",
        f"- Held fixed: {fixed_partner_overrides}",
        "",
        f"| pct_shrink | {knob_name} (m) | dens_eff | mass_eff | mean | median | std | n_zero | n>=100 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['pct_shrink']}% | {r[f'{knob_name}_m']:.5f} | "
            f"{r['density_effective']:.1f} | {r['mass_effective']:.4f} | "
            f"{r['mean']:.2f} | {r['median']:.2f} | {r['std']:.2f} | "
            f"{r['n_zero']} | {r['n_ge100']} |"
        )
    with open(os.path.join(out_root, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("\n".join(md))
    return rows


def main() -> int:
    # Sweep 1: puck_radius. Pin paddle_radius at source. No paddle mass-ref
    # since paddle isn't moving; clear puck mass-ref happens via override.
    print("\n=== sweep 1/2: puck_radius (paddle pinned at source) ===\n", flush=True)
    _run_sweep(
        knob_name="puck_radius",
        source_radius=SOURCE_PUCK_RADIUS,
        mass_ref_key="puck_mass_reference_radius",
        fixed_partner_overrides={
            "paddle_radius": SOURCE_PADDLE_RADIUS,
            "paddle_mass_reference_radius": None,
        },
        out_root=os.path.join(SWEEP_ROOT, "puck_radius_decay_mass_preserved"),
    )

    # Sweep 2: paddle_radius. Pin puck_radius at source.
    print("\n=== sweep 2/2: paddle_radius (puck pinned at source) ===\n", flush=True)
    _run_sweep(
        knob_name="paddle_radius",
        source_radius=SOURCE_PADDLE_RADIUS,
        mass_ref_key="paddle_mass_reference_radius",
        fixed_partner_overrides={
            "puck_radius": SOURCE_PUCK_RADIUS,
            "puck_mass_reference_radius": None,
        },
        out_root=os.path.join(SWEEP_ROOT, "paddle_radius_decay_mass_preserved"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
