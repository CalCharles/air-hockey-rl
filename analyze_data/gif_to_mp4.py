"""Convert .gif files to .mp4 (same name, same directory) using OpenCV + imageio.

Usage:
    python analyze_data/gif_to_mp4.py <directory> [--fps 20]

Converts every *.gif in <directory> to a same-named *.mp4 next to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def gif_to_mp4(gif_path: Path, mp4_path: Path, fps: int) -> None:
    reader = imageio.get_reader(str(gif_path))
    frames = [np.asarray(frame) for frame in reader]
    reader.close()
    if not frames:
        raise ValueError(f"No frames found in {gif_path}")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h))
    try:
        for frame in frames:
            if frame.ndim == 2:  # grayscale
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # RGBA
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if frame_bgr.shape[:2] != (h, w):
                frame_bgr = cv2.resize(frame_bgr, (w, h))
            writer.write(frame_bgr)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing .gif files")
    parser.add_argument("--fps", type=int, default=20, help="Output mp4 frame rate")
    args = parser.parse_args()

    gif_paths = sorted(args.directory.glob("*.gif"))
    if not gif_paths:
        print(f"No .gif files found in {args.directory}")
        return

    n_ok, n_fail = 0, 0
    for gif_path in gif_paths:
        mp4_path = gif_path.with_suffix(".mp4")
        try:
            gif_to_mp4(gif_path, mp4_path, args.fps)
            print(f"OK   {gif_path.name} -> {mp4_path.name}")
            n_ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {gif_path.name}: {exc}")
            n_fail += 1

    print(f"\nDone: {n_ok} converted, {n_fail} failed.")


if __name__ == "__main__":
    main()
