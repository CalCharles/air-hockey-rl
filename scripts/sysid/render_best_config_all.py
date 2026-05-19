#!/usr/bin/env python3
"""
Render side-by-side REAL vs SIM GIFs for ALL segments in system_id3/
using the best sysid config: kp=9000, kd=50, ki=0, density=3000.
"""

from __future__ import annotations

import copy
import glob
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.visualization.replay_real_in_sim import (
    load_sim_config,
    replay_episode,
)

from scripts.sysid._sysid_paths import DEFAULT_CONFIG, SYSID_DIR

OUT_DIR = SYSID_DIR / "visualizations_best_config"

BEST_KP = 9000
BEST_KD = 50
BEST_KI = 0
BEST_DENSITY = 3000


def find_all_segments() -> list[Path]:
    """Find all segment HDF5 files (skip full-trajectory files like circle_fast.hdf5)."""
    all_hdf5 = sorted(Path(SYSID_DIR).rglob("segment_*.hdf5"))
    return all_hdf5


def main():
    segments = find_all_segments()
    print(f"Found {len(segments)} segments to render")
    print(f"Config: kp={BEST_KP}, kd={BEST_KD}, ki={BEST_KI}, density={BEST_DENSITY}")
    OUT_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    for idx, seg_path in enumerate(segments):
        category = seg_path.parent.parent.name
        frame_dir = seg_path.parent.name
        gif_name = f"{category}_{frame_dir}.gif"
        out_path = OUT_DIR / gif_name

        print(f"\n[{idx+1}/{len(segments)}] {category}/{frame_dir} -> {gif_name}")

        # Patch config with best params (reload each time for clean state)
        sim_cfg = load_sim_config(str(DEFAULT_CONFIG), enable_noise=False)
        sim_cfg["simulator_params"]["pid_kp"] = BEST_KP
        sim_cfg["simulator_params"]["pid_kd"] = BEST_KD
        sim_cfg["simulator_params"]["pid_ki"] = BEST_KI
        sim_cfg["simulator_params"]["paddle_density"] = BEST_DENSITY

        # Write a temp config YAML so replay_episode can load it
        # Instead, call replay_episode with the original config path —
        # it will load & patch below. We need to use the CLI-level function.
        # Simpler: just call replay_episode with default config and override.
        metrics = replay_episode(
            episode_path=str(seg_path),
            config_path=str(DEFAULT_CONFIG),
            output_path=str(out_path),
            enable_noise=False,
            max_steps=None,
            fps=20,
            frame_width=160,
            start_frame=0,
            puck_vel_fit=False,
            puck_vel_half_window=5,
            sim_cfg_overrides={
                "pid_kp": BEST_KP,
                "pid_kd": BEST_KD,
                "pid_ki": BEST_KI,
                "paddle_density": BEST_DENSITY,
            },
            park_puck=True,
        )
        elapsed = time.time() - t0
        print(f"  paddle_mean={metrics['paddle']['mean_error_m']:.4f} m  "
              f"puck_mean={metrics['puck']['mean_error_m']:.4f} m  ({elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print(f"Done. {len(segments)} GIFs saved to: {OUT_DIR}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
