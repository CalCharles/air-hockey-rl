"""Evaluate a fixed policy-side filter against saved before-action body states.

No rollout, policy/regressor training, parameter search or checkpoint loading.
Uses the E010 grouped-seed benchmark and the filter's declared defaults.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts.analysis.puck_state_filter import PuckStateFilter


def evaluate(roots, out, holdout_seed=300024):
    rows = []
    for root in roots:
        for path in sorted(root.glob("training_seed*.npz")):
            seed = int(path.stem.split("seed")[-1])
            with np.load(path) as data:
                obs, truth = data["observation"], data["true_puck_xy"]
                if len(obs) < 3:
                    continue
                observer = PuckStateFilter()
                estimates = [observer.reset(obs[0])]
                estimates.extend(observer.update(observation) for observation in obs[1:])
                estimates = np.asarray(estimates)[2:]
                target = np.concatenate([truth[1:-1], (truth[1:-1] - truth[:-2]) / .05], axis=1)
                squared = (estimates - target) ** 2
                rows.append(dict(seed=seed, source=str(path), heldout=seed >= holdout_seed,
                                 steps=len(squared),
                                 position_rmse_cm=float(100 * np.sqrt(squared[:, :2].sum(1).mean())),
                                 prior_step_velocity_rmse_m_s=float(np.sqrt(squared[:, 2:].sum(1).mean()))))
    if not rows:
        raise ValueError("No saved trajectories")
    heldout = [row for row in rows if row["heldout"]]
    if not heldout:
        raise ValueError("No held-out seed group")
    result = dict(created=datetime.now(timezone.utc).isoformat(),
                  roots=[str(p) for p in roots], holdout_seed=holdout_seed,
                  parameters=dict(dt=.05, latency=.025, accel_std=4., measurement_std=.01,
                                  gravity=.73, damping=.11, walls=True),
                  training="none; one fixed setting; no actor/critic/regressor or parameter fit",
                  mean_episode_position_rmse_cm=float(np.mean([r["position_rmse_cm"] for r in heldout])),
                  mean_episode_velocity_rmse_m_s=float(np.mean([r["prior_step_velocity_rmse_m_s"] for r in heldout])),
                  episodes=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print("Fixed filter held-out position RMSE (cm):", result["mean_episode_position_rmse_cm"])
    print("Fixed filter held-out prior-step velocity proxy RMSE (m/s):", result["mean_episode_velocity_rmse_m_s"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--holdout-seed", type=int, default=300024)
    args = parser.parse_args()
    evaluate(args.roots, args.out, args.holdout_seed)
