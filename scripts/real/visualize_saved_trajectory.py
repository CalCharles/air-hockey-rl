#!/usr/bin/env python3
"""
Create GIF visualization(s) from real-robot trajectory HDF5 files.

This script reads `train_vals` from files saved by the real-world rollout entrypoints and renders
an approximate simulator-style top-down table view with:
- paddle position (`pose`)
- target/action position (`desired_pose`)
- puck position (`puck`)

If the trajectory is long, output is split into multiple GIFs to keep each file
size manageable.
"""

import argparse
from pathlib import Path

import h5py
import imageio
import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "opencv-python is required for visualization. Install training extras, e.g. `uv sync --extra train`."
    ) from exc


TABLE_LENGTH = 1.9304
TABLE_WIDTH = 0.8636
ROBOT_X_OFFSET = 1.2
RENDER_SIZE = 360

# train_vals layout from scripts/real/README.md
IDX_POSE_X = 5
IDX_POSE_Y = 6
IDX_DESIRED_X = 26
IDX_DESIRED_Y = 27
IDX_PUCK_X = 32
IDX_PUCK_Y = 33
IDX_PUCK_OCCLUDED = 34


def load_train_vals(hdf5_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as f:
        if "train_vals" not in f:
            raise ValueError(f"{hdf5_path} does not contain 'train_vals'")
        vals = f["train_vals"][:]
    if vals.ndim != 2 or vals.shape[1] < 35:
        raise ValueError(f"Unexpected train_vals shape: {vals.shape}")
    return vals


def table_to_pixel(table_x: float, table_y: float, ppm: float) -> tuple[int, int]:
    # Match render convention used in project: convert (x, y) -> (y, -x), then to pixel.
    render_x = table_y
    render_y = -table_x
    px = int((render_y + TABLE_LENGTH / 2.0) * ppm)
    py = int((render_x + TABLE_WIDTH / 2.0) * ppm)
    return px, py


def robot_to_pixel(robot_x: float, robot_y: float, ppm: float) -> tuple[int, int]:
    table_x = robot_x + ROBOT_X_OFFSET
    table_y = robot_y
    return table_to_pixel(table_x, table_y, ppm)


def load_table_background(frame_h: int, frame_w: int) -> np.ndarray:
    script_dir = Path(__file__).resolve().parent
    assets_dir = script_dir.parent.parent / "assets"
    table_path = assets_dir / "air_hockey_table.png"

    table_img = cv2.imread(str(table_path))
    if table_img is None:
        raise FileNotFoundError(f"Could not load table image from {table_path}")

    table_img = cv2.rotate(table_img, cv2.ROTATE_90_CLOCKWISE)
    table_img = cv2.resize(table_img, (frame_w, frame_h))
    return table_img


def render_frame(vals_row: np.ndarray, ppm: float, table_background: np.ndarray) -> np.ndarray:
    frame = table_background.copy()

    pose_px = robot_to_pixel(vals_row[IDX_POSE_X], vals_row[IDX_POSE_Y], ppm)
    desired_px = robot_to_pixel(vals_row[IDX_DESIRED_X], vals_row[IDX_DESIRED_Y], ppm)
    puck_px = table_to_pixel(vals_row[IDX_PUCK_X], vals_row[IDX_PUCK_Y], ppm)
    puck_occluded = vals_row[IDX_PUCK_OCCLUDED] > 0.5

    # Action/target (orange cross) + line from current paddle position
    cv2.line(frame, pose_px, desired_px, (0, 140, 255), 2)
    cross = 8
    cv2.line(frame, (desired_px[0] - cross, desired_px[1]), (desired_px[0] + cross, desired_px[1]), (0, 140, 255), 2)
    cv2.line(frame, (desired_px[0], desired_px[1] - cross), (desired_px[0], desired_px[1] + cross), (0, 140, 255), 2)

    # Paddle position (blue)
    cv2.circle(frame, pose_px, 11, (180, 75, 0), -1)
    cv2.circle(frame, pose_px, 11, (20, 20, 20), 2)

    # Puck position (green when visible, red when occluded)
    puck_color = (60, 180, 75) if not puck_occluded else (30, 30, 220)
    cv2.circle(frame, puck_px, 7, puck_color, -1)
    cv2.circle(frame, puck_px, 7, (20, 20, 20), 1)

    cv2.putText(frame, "Blue: paddle | Orange: target(action) | Green/Red: puck", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def save_gif_chunks(
    vals: np.ndarray,
    out_dir: Path,
    stem: str,
    fps: int,
    max_frames_per_gif: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    ppm = RENDER_SIZE / TABLE_WIDTH
    render_length = int(ppm * TABLE_LENGTH)
    frame_h = RENDER_SIZE
    frame_w = render_length
    table_background = load_table_background(frame_h, frame_w)

    n_frames = vals.shape[0]
    outputs: list[Path] = []

    for chunk_idx, start in enumerate(range(0, n_frames, max_frames_per_gif)):
        end = min(start + max_frames_per_gif, n_frames)
        chunk = vals[start:end]

        frames = []
        for row in chunk:
            frame = render_frame(row, ppm=ppm, table_background=table_background)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        out_path = out_dir / f"{stem}_part_{chunk_idx:03d}.gif"
        imageio.mimsave(out_path, frames, format="GIF", duration=1.0 / fps, loop=0)
        outputs.append(out_path)
    return outputs


def generate_gifs_from_hdf5(
    input_hdf5: str,
    output_dir: str = None,
    fps: int = 20,
    max_frames_per_gif: int = 250,
) -> list[Path]:
    input_path = Path(input_hdf5).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if max_frames_per_gif <= 0:
        raise ValueError("max_frames_per_gif must be > 0")
    if fps <= 0:
        raise ValueError("fps must be > 0")

    vals = load_train_vals(input_path)
    if output_dir is None:
        out_dir = Path(__file__).resolve().parent / "trajectory_gifs" / input_path.stem
    else:
        out_dir = Path(output_dir).expanduser().resolve()

    return save_gif_chunks(
        vals=vals,
        out_dir=out_dir,
        stem=input_path.stem,
        fps=fps,
        max_frames_per_gif=max_frames_per_gif,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one or more trajectory GIFs from a saved real-robot HDF5 trajectory."
    )
    parser.add_argument("input_hdf5", type=str, help="Path to trajectory_data*.hdf5")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for GIF files (default: scripts/real/trajectory_gifs/<input_stem>)",
    )
    parser.add_argument("--fps", type=int, default=20, help="GIF playback FPS.")
    parser.add_argument(
        "--max-frames-per-gif",
        type=int,
        default=250,
        help="Split output into multiple GIFs, this many frames per file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_hdf5).expanduser().resolve()
    outputs = generate_gifs_from_hdf5(
        input_hdf5=input_path,
        output_dir=args.output_dir,
        fps=args.fps,
        max_frames_per_gif=args.max_frames_per_gif,
    )
    vals = load_train_vals(input_path)
    if outputs:
        output_dir = outputs[0].parent
    elif args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / "trajectory_gifs" / input_path.stem
    else:
        output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"Loaded {vals.shape[0]} frames from {input_path}")
    print(f"Wrote {len(outputs)} GIF(s) to {output_dir}")
    for p in outputs:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"- {p} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
