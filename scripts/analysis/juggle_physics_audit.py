"""Recover evaluation reset physics, verifying saved initial observations.

This standalone diagnostic resets environments but takes no policy actions.
It never changes the randomization distribution or a training process.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch

from airhockey import AirHockeyEnv


def audit(result_path: Path, output: Path):
    payload = json.loads(result_path.read_text())
    rows = []
    for name, scenario in payload["scenarios"].items():
        for episode in scenario["episodes"]:
            seed = episode["seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            config = copy.deepcopy(scenario["config"])
            if config.get("simulator") != "box2d":
                raise ValueError("Reset reconstruction supports Box2D evaluation artifacts only")
            config.update(seed=seed, max_timesteps=payload["horizon"] + 1)
            config["simulator_params"]["seed"] = seed
            env = AirHockeyEnv(config)
            try:
                observation, _ = env.reset(seed=seed)
                with np.load(result_path.parent / episode["trajectory"]) as saved:
                    expected = saved["observation"][0]
                if not np.array_equal(np.asarray(observation, dtype=expected.dtype), expected):
                    raise ValueError(f"Initial observation mismatch: {name}, seed {seed}")
                row = dict(scenario=name, seed=seed, steps=episode["steps"],
                           reached_horizon=episode["reached_horizon"],
                           physical_juggles=episode.get("true_state_juggles"),
                           initial_observation_exact=True)
                for variable in config["random_variables"]:
                    value = float(getattr(env.simulator_params, variable))
                    low, high = config["random_variable_ranges"][variable]
                    if not low <= value <= high:
                        raise ValueError(f"Physics outside configured range: {variable}={value}")
                    row[variable] = value
                    row[variable + "_range_fraction"] = (value - low) / (high - low)
                rows.append(row)
            finally:
                env.close()
    if not rows:
        raise ValueError("No saved episodes")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} exact reset observations; wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.results, args.out)


if __name__ == "__main__":
    main()
