#!/usr/bin/env python3
"""Live preview of one camera. Defaults to index 1, the second /dev/video device.

    python analyze_data/view_camera.py
    python analyze_data/view_camera.py --camera-index 0

q / Esc to quit. Stop any other process holding the camera first.
"""

from __future__ import annotations

import argparse

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera-index", type=int, default=1)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {args.camera_index} (is another process using it?)")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"camera {args.camera_index}: {w}x{h}   q / Esc to quit")

    window = f"camera {args.camera_index}"
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                continue
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
