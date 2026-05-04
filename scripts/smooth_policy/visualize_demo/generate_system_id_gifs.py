#!/usr/bin/env python3
"""Generate timestep-annotated GIFs for the system-ID trajectory set."""

from pathlib import Path

from scripts.smooth_policy.visualize_demo.visualize_real_trajectory import (
    RealTrajectoryRenderer,
    create_trajectory_gif,
    extract_paddle_data,
)
from scripts.smooth_policy.visualize_demo.visualize_real_trajectory_split import (
    load_split_trajectory_data,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

SYSTEM_ID_HDF5 = [
    "real_runs/online_run/episode_hdf5/100-200/trajectory_data451.hdf5",
    "real_runs/online_run/episode_hdf5/100-200/trajectory_data461.hdf5",
    "real_runs/online_run/episode_hdf5/100-200/trajectory_data478.hdf5",
    "real_runs/online_run/episode_hdf5/100-200/trajectory_data458.hdf5",
    "real_runs/online_run/episode_hdf5/>200/trajectory_data476.hdf5",
]

OUTPUT_ROOT = REPO_ROOT / "real_runs" / "online_run" / "system_id_gifs"
OUTPUT_WIDTH = 320
FPS = 20


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    renderer = RealTrajectoryRenderer(
        table_length=1.9304,
        table_width=0.8636,
        paddle_radius=0.0508,
        puck_radius=0.03175,
        render_size=360,
        robot_x_offset=1.2,
        orientation="vertical",
        paddle_input_frame="table",
    )

    for rel_path in SYSTEM_ID_HDF5:
        hdf5_path = REPO_ROOT / rel_path
        if not hdf5_path.exists():
            print(f"SKIP (not found): {hdf5_path}")
            continue

        stem = hdf5_path.stem
        out_dir = OUTPUT_ROOT / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        gif_path = out_dir / "trajectory_visualization.gif"

        print(f"\n{'=' * 60}")
        print(f"Processing: {rel_path}")
        print(f"Output:     {gif_path}")

        train_vals = load_split_trajectory_data(hdf5_path)
        paddle_data = extract_paddle_data(train_vals, require_puck=False)

        create_trajectory_gif(
            paddle_data,
            renderer,
            gif_path,
            fps=FPS,
            output_width=OUTPUT_WIDTH,
        )
        print(f"Done: {gif_path}")

    print(f"\nAll GIFs saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
