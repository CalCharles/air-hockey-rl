# Homography pipeline (real camera → table-aligned image → robot coordinates)

This document describes **how** the real-robot stack applies a perspective homography: calibration, stored matrices, per-frame warping, and conversion between warped pixels and robot meters. It does not cover failure modes or robustness.

## Role in the stack

The overhead camera sees the table at an angle. A **homography** (a 3×3 projective map) rectifies the image so that the playing surface is aligned with a fixed “table” coordinate frame. Detection and overlays run on the **warped** image; puck positions are then mapped to **robot-frame** \((x, y)\) in meters using a fixed offset and scale.

Primary implementation references:

| Piece | Location |
|-------|----------|
| Load `Mimg`, warp frames, camera loop | [`airhockey/sims/real/control_parameters.py`](../../../../airhockey/sims/real/control_parameters.py) |
| Puck detection and pixel → robot | [`airhockey/sims/real/image_detection.py`](../../../../airhockey/sims/real/image_detection.py) |
| Robot ↔ display pixel (overlays) | [`airhockey/sims/real/overlay_utils.py`](../../../../airhockey/sims/real/overlay_utils.py) |
| Env-level offsets | [`airhockey/sims/air_hockey_real.py`](../../../../airhockey/sims/air_hockey_real.py) (`offset_constants`, `visual_downscale_constant`) |

## Stored calibration artifacts

- **`assets/real/Mimg.npy`** — 3×3 homography applied to **upscaled** camera frames in the main pipeline (`homography_transform`). Loaded at import time in `control_parameters.py` (and referenced in `image_detection.py` for shared constants).
- **`assets/real/Mimg_tele.npy`** — Alternate homography used in **mimic** teleop (`mimic_control` in `control_parameters.py`): warp only, no 180° rotation step used in the main path.
- **`Mrob.npy`** — Produced alongside `Mimg` during some calibration flows; maps between rectified/table-aligned pixel conventions and robot millimeter corners (see calibration scripts below).

## How `Mimg` is defined (calibration)

A plane-to-plane homography is **fully determined by four point correspondences**. The codebase uses OpenCV:

- `cv2.getPerspectiveTransform(pts_src, pts_dst)` → 3×3 `M`
- `cv2.warpPerspective(image, M, dsize)` applies `M` to each pixel of `image`

### Robot-assisted calibration (preferred flow)

[`scripts/real/calibrate_robo_camera.py`](../../../../scripts/real/calibrate_robo_camera.py) (`calibrate_homography`):

1. The robot is moved to mark **four known positions** on the table; **four red pucks** are placed at those positions.
2. The camera image is captured; **four puck centers** are detected (`detect_four_red_pucks`), yielding pixel coordinates in the **640×480**-equivalent frame (then converted to OpenCV \((x, y)\) order).
3. Those points are optionally adjusted by small **tuning offsets**, then multiplied by **`upscale_constant` (3)** so they live in the same resolution as the warped pipeline (1920×1440 space).
4. **`pts1`** = scaled image points. **`robot_points_mm`** are the corresponding table corners in **millimeters** (script default corners, order fixed after sorting).
5. **`robot_reference_xy = robot_points_mm + offset_constants`** — the destination points in the **warp output** coordinate system use the same **`offset_constants`** (e.g. `(2250, 500)`) as the rest of the stack.
6. **`Mimg = cv2.getPerspectiveTransform(pts1, robot_reference_ordered)`** — maps the upscaled camera image into the rectified frame whose pixel origin is tied to that offset convention.
7. With `--save-homographies`, the script writes `Mimg.npy` and `Mrob.npy` to the **current working directory** (copy into `assets/real/` for runtime use).

### Manual script

[`scripts/real/generate_homography.py`](../../../../scripts/real/generate_homography.py) builds `Mimg` from **hand-chosen** `pts1` / `pts2` and offsets, then saves `Mimg.npy` / `Mrob.npy`. Uses the same OpenCV pattern as the live UR5 calibration; resolution and corner values may differ.

## Runtime: `homography_transform` (main camera path)

In [`control_parameters.py`](../../../../airhockey/sims/real/control_parameters.py), each frame is processed as follows:

1. **Rotate** the raw BGR frame 180° (`cv2.ROTATE_180`).
2. Optionally save a **downscaled** copy of the rotated frame for logging (`save_downscale_constant`, typically 2).
3. **Resize** to **1920×1440** (`640×480` × `upscale_constant` where `upscale_constant = 3`).
4. **`cv2.warpPerspective(image, Mimg, dsize=(1920, 1440))`** — output canvas size matches `original_size * upscale_constant`.
5. **Resize** the warped image by **`visual_downscale_constant` (2)** for display and detection — this is the **`showdst`** frame passed to puck detection and `cv2.imshow`.

Constants (`upscale_constant`, `original_size`, `visual_downscale_constant`, `Mimg`) are defined at module level in `control_parameters.py` and kept consistent with `image_detection.py` / `air_hockey_real.py` for offsets.

## Puck detection and pixel → robot \((x, y)\)

Puck finding (`find_red_hockey_puck` in [`image_detection.py`](../../../../airhockey/sims/real/image_detection.py)) receives **`showdst`**. It preprocesses the image (optional rotation for some call sites, half-size resize, ROI masking), runs color/morphology segmentation, and takes a **centroid** in **detector pixel coordinates**.

Those detector coordinates are converted to robot meters with **`_pixel_to_robot_xy`**:

- Detector coordinates are scaled by **4** to map back into the **homography output pixel grid** at full warped resolution (1920×1440), accounting for the **half-size** detector input relative to `showdst`.
- **`offset_constants`** (same vector as in calibration, e.g. `(2250, 500)`) are subtracted in pixel space.
- Values are multiplied by **`0.001`** to convert from millimeter-style pixel indexing to **meters** for \(x\).
- The robot \(y\) axis is **`y_robot = -(...)`** relative to the pixel \(y\) convention (matches `overlay_utils`).

So the chain is: **raw camera → rectify with `Mimg` → downscale for UI/detection → centroid in detector frame → scale/offset to meters**.

## Robot → pixels (overlays)

[`overlay_utils.py`](../../../../airhockey/sims/real/overlay_utils.py) implements the **inverse** of the affine step used after homography:

- **`robot_to_display_pixel`**: \((x_m, y_m)\) → multiply by 1000, flip \(y\), add `offset_constants`, divide by `visual_downscale_constant` for coordinates in the **showdst** frame.
- **`display_pixel_to_robot`**: the reverse map from a click or pixel back to meters.

Homography itself is **not** inverted in Python for puck position; the image is warped once per frame, then this **affine** map is used in the rectified/downscaled space.

## Single-point homography (teleop helper)

[`single_point_homography(matrix, point)`](../../../../airhockey/sims/real/control_parameters.py) applies the full **projective** map (with division by the homogeneous third component) to one \((x, y)\) pair. It is used where a point must be transformed **without** warping the whole image.

## Related reading

- High-level real stack map: [`overview.md`](overview.md)
- Safety and operation: [`../../repo/project-goal-and-safety.md`](../../repo/project-goal-and-safety.md) and the project README
