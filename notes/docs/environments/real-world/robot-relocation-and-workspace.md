# Moving the robot / changing the workspace — when do policies need retraining?

Written 2026-11-01 while evaluating `data/new_sim_best_policies/` on the real robot.
Motivation: the UR5 base is offset toward one side of the table (table center sits
at robot y = +0.0376), so the paddle workspace leaves a bigger gap to the +y wall
than to the −y wall, and moves toward that side look weaker.

## Frames (what the code actually does)

- **Policy observations are in the table frame.** Puck and paddle are converted as
  `table_xy = robot_xy + (center_offset_constant, center_offset_constant_y)`.
  On the real robot those offsets come from `TABLE_CENTER_OFFSET_X/Y` in
  `airhockey/sims/real/table_calibration.py` (currently 1.3359 / −0.0376, measured 2026-09-27).
- **Actions are relative displacements** (`a[0]·0.26 m`, `a[1]·0.12 m`), so a pure
  translation of the robot doesn't change what an action means.
- **Workspace limits are in the robot frame.** Real:
  `REAL_WORKSPACE_LIMS = (-0.83, -0.45, -0.33, 0.39)` in
  `airhockey/sims/air_hockey_real.py`, plus the far-corner chamfers
  (`corner_cut_biases` / `effective_x_max` in `airhockey/sims/real/coordinate_transform.py`).
  Sim: `x_min_lim / x_max_lim / y_min / y_max / top_abs / max_bias_p / max_bias_m`
  in `simulator_params`, applied in robot coordinates by `AirHockeyBox2D._clip_limits`
  after subtracting the same two offsets.
- **Camera calibration maps pixels into the robot frame** (April-tag / puck
  calibration, e.g. `scripts/real/calibrate_robo_camera_3_pucks_2cams.py`). The table
  offset is then added on top.

The `new_sim_best_policies` configs were trained with the real limits
`(-0.83, -0.45, -0.33, 0.39)` and the real offsets `(1.3359, -0.0376)`
(`max_bias_p -0.218`, `max_bias_m -0.266`). So the workspace they learned, in the
**table frame**, is:

| axis | table-frame range | gap to wall (paddle-center limit: ±0.381 m in y, 0.914 m in x) |
|---|---|---|
| y | [−0.368, +0.352] | −y side 1.3 cm, **+y side 2.9 cm** |
| x (robot end) | up to 0.886 | 2.9 cm |

(Table 0.8636 × 1.9304 m, paddle radius 0.0508 m.)

## Case 1 — physically re-center the robot, keep the workspace → no retrain

Redo all three of these after the move:

1. **Camera homographies.** They map into the robot frame, so they're invalid once the base moves.
2. **`TABLE_CENTER_OFFSET_X/Y`.** Re-measure with the paddle pushed against the walls
   (procedure in the `table_calibration.py` docstring). Update only that file.
3. **Robot-frame limits**, shifted so the **table-frame** workspace is unchanged:

   ```
   new_robot_lims = old_robot_lims + (old_offset − new_offset)    # per axis
   # e.g. y: [-0.33, 0.39] + (-0.0376 - new_Y)
   ```

   If `REAL_WORKSPACE_LIMS` is left as-is, the workspace moves with the robot. It
   then no longer matches training: one side gains room and the other may push the
   paddle into the rail. The corner chamfers are derived from the limits, so they
   follow automatically unless `max_bias_p/m` are set explicitly.

With all three done, the policy gets the same inputs and limits it was trained
with, so no retraining is needed. The arm runs in different joint configurations,
which changes the tracking dynamics a little. That's the change we're hoping
helps the +y side. The paddle PID sysid (`sysid/paddle_pid/`) was fit at the old
placement, so if behaviour shifts noticeably, re-check it.

## Case 2 — enlarge / change the workspace → retrain recommended

- The sim clips the paddle to the same limits, so the policies have never seen the
  paddle beyond y ≈ 0.352 (table frame). The new region is out of distribution.
- Policies can learn to rely on the clip: they command past the edge and let the
  clip stop the paddle at a known place. With a larger workspace, the same command
  sends the paddle further, toward the rail.
- A ~2 cm enlargement might work zero-shot and is worth one eval. But expect
  near-wall behaviour to change, and keep those numbers separate from the
  current eval set.
- Retraining is cheap. Put the new limits (and corner biases) in the sim config,
  computed with the **new** offsets so sim and real robot-frame limits match
  exactly, then retrain the seeds. Keep about 1 cm of margin to the rail
  rather than going all the way to 0.381.

## Recommended order

1. Move the robot, recalibrate, and keep the table-frame workspace identical
   (Case 1). Re-evaluate the current policies against the existing numbers. This
   isolates whether arm placement was the cause.
2. Only then widen the workspace, retrain in sim with the matching limits, and
   evaluate the retrained policies.
