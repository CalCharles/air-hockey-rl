# Puck dropouts over the Wind Chill logo: radius gate, not colour threshold

**Date:** 2026-10-19 · **Status:** Done (detector fix applied; needs a real-robot rerun to confirm)

## Question
The occlusion map of the 10 hist4 juggle_sysid rollouts (2026-10-14, `real_runs/eval_sim_pretrained/policies_hist4_2026-09-19/juggle_sysid/data_20261014-201743`, figures in `results/occlusion_probability_and_traj_tracks/`) shows 50–94 % "occluded" cells around the centre logo. Is this the logo's colour fringing, and can camera settings fix it?

## Method
- Replayed every logged `train_img` (320×240, rotated raw) through the exact runtime path: upscale to 640×480 → rotate back → `homography_transform` (post-homography `showdst`) → `find_red_hockey_puck_antiglare` with the default antiglare ROI, feeding the detector its own history. Replay agrees with the logged occlusion flag on ~96 % of frames (150 vs 138 logged dropouts).
- Live PS3 Eye (gspca ov534, YUYV 640×480 @ 30 fps only) controls swept via V4L2 ioctls (no v4l-utils / sudo on the robot PC); logo false-red pixels and puck saturation/area measured in the detector frame.

## Findings
1. **In the dropped frames the red mask covers the puck cleanly.** The blob is rejected by `MAX_PUCK_RADIUS_PX = 10`: its enclosing radius is 10.0–11.6 px. Detector-frame median radius is 8.7 px everywhere except detector x 180–240 (table centre), where it's 10.0 px and 44 % of frames exceed 10. Reasons: the puck is closer to the camera there; the antiglare rescue ROI (detector x ≈ 145–225) adds a pink chroma halo (+36 px² median); logo fringe pixels attach to the blob. With the strict mask alone, 31 % of these frames would still fail the gate.
2. **Logo fringes alone are harmless.** After the 3×3 close/open they are 0 px at every camera setting tried; they only matter when they touch the puck.
3. **Camera settings can't fix it.** Sharpness is already 0 (its minimum). Turning AWB off doesn't freeze the current balance: the driver drops to default gains (table BGR 131/131/130 → 34/145/166). Lowering saturation cuts logo false-red and the puck by similar amounts (sat 32: logo −60 %, puck area −33 %). Nothing was changed on the camera.

## Change
`airhockey/sims/real/image_detection.py`: `MAX_PUCK_RADIUS_PX` and `SIMPLE_MAX_PUCK_RADIUS_PX` 10 → 12.

| Detector (1020 replayed frames) | detected | jumps > 6 cm |
|---|---|---|
| antiglare, max r 10 (old) | 870 (85.3 %) | 1 |
| antiglare, max r 11 | 975 (95.6 %) | 1 |
| **antiglare, max r 12 (new)** | **989 (97.0 %)** | 1 |
| antiglare, max r 12 + 5×5 open | 985 (96.6 %) | 1 |
| simple `red_puck`, max r 10 → 12 | 407 → 905 | 3 → 1 |

The 31 remaining misses are all true occlusions: the puck is under the arm or paddle at the robot end, or at the frame edge (traj 5 f144–148, traj 6 f118–136, traj 0 f44–51). Live check after the edit: 30/30 frames detected.

## Caveats
Logged frames are 320×240 and come from the queued camera; the live detector sees full-resolution frames. Re-run the occlusion map on the next real rollout to confirm.
