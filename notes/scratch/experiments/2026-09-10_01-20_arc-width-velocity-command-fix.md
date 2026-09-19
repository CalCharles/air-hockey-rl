# Real-robot arcs swept a third of the table: the action is a velocity command, not a displacement

- **Date**: 2026-09-10 01:20 UTC
- **Status**: done (code landed; not yet re-run on hardware)
- **Run dirs (evidence)**: `data/robot_data_collection/curves_20260909_1953` (real, pre-fix),
  `data/robot_data_collection/paddle_motion_20260909_1808` (real line battery, used for the gain fit),
  `data/robot_data_collection/paddle_motion_sim_20260909_1952` (Box2D, pre-fix)
- **Code**: `scripts/robot_data_collection/trial_plan.py`,
  `scripts/robot_data_collection/collect_paddle_motion{,_sim}.py`
- **Configs**: `configs/robot_data_collection/paddle_motion_config.yaml` (unchanged)

## Question

The `--curves` half-arc battery renders as a full-width sweep in Box2D but traces roughly a
third of the table on the UR5. Where does the width go, and can the robot arc be widened while
keeping 20 timesteps per trial?

## What the data said

Pre-fix `arc_wide`, real vs sim, same plan and same commanded schedule:

| | commanded y span | realised y span | ratio |
|---|---|---|---|
| Box2D | 1.440 m | 0.713–0.718 m | 0.50 |
| real UR5 | 1.440 m | 0.208–0.249 m | **0.15** |

Two things rule out the usual suspects. `desired_pose_y` (the pose actually handed to `servoL`)
spans 0.24 m too — within 0.02 m of the realised pose — so the **command itself** was small, not
the tracking. And `hist_len: 1` makes `filter_update` the identity, so no smoothing is involved.

Per-step inspection of `traj_006_arc_wide_v0.80_trial1` shows why: the commanded target is exactly
`getTargetTCPPose() + action * move_lims` every step, re-anchored to the *current* pose. The
command never integrates. What the action sets is a constant **lead**, and a constant lead settles
at a constant **speed**:

```
v_realised  ≈  G · (action · move_lims)
```

Fitting `G` on the straight-line battery (steady-state median over the last 8 steps), both axes
agree to 2 %, so a single scalar suffices:

| | axis | sustained speed at action 1.0 | G |
|---|---|---|---|
| real UR5 (`servoL` lookahead 0.2, gain 700, dt ≈ 0.048 s) | y | 0.386 m/s | 3.22 1/s |
| real UR5 | x | 0.82 m/s | 3.15 1/s |
| Box2D (dt = 0.05 s) | — | — | ≈ 9.9 1/s |

`G · dt` is the fraction of a commanded step that is realised: **0.15 on the robot, 0.50 in sim.**

## Conclusion on the bug

`curve_action_schedule` wrote `action = per_step_displacement / move_lims`, which is only true when
the paddle reaches its target within one step. `DEFAULT_CURVE_TRACKING_GAIN = 2.0` was fitted in
Box2D, where it exactly cancels `G · dt = 0.5` — so the sim looked correct and hid the error. On
the robot the same schedule needed a gain of ~6.5, and the gain is clamped per trial so no action
reaches 1.0, so it could not have been raised anyway. **The sim was never evidence about the
robot here; the two backends differ by 3× in exactly the quantity the schedule assumed away.**

## Fix

Closed-loop pure-pursuit tracking (`--curve-tracking closed-loop`, now the default). Each step
projects the *measured* paddle pose onto the target ellipse, places a carrot 2 steps of travel
further along, and commands

```
action = clip( speed · unit(carrot − p) / (G · move_lims), −1, +1 )
```

The real collector closes the loop on `getTargetTCPPose()` — the same quantity `get_transition`
anchors its own command on. Because the carrot is re-derived from the measured pose, a
mis-estimated `G` changes the sweep *speed*, not its shape; and when the requested speed is past
the axis's authority the action saturates at 1.0 and the paddle covers as much of the full-width
arc as the timesteps allow.

Curve trials are now fixed at `--action-steps` (20), same as the lines. Speeds are 3 per shape,
spanning 2× and anchored at the top by what the arm can hold.

## Results

Box2D, closed-loop, 20 steps (`--directions none --repeats 1`), cross-track error measured against
the ideal ellipse:

| condition | realised Δy | max cross-track error |
|---|---|---|
| `arc_wide` 0.22 / 0.32 / 0.45 m/s | 0.185 / 0.295 / 0.441 m | 0.9 / 1.8 / 3.1 mm |
| `arc_medium` 0.24 / 0.35 / 0.50 m/s | 0.142 / 0.258 / 0.424 m | 0.4 / 0.9 / 1.7 mm |
| `arc_tight` 0.28 / 0.42 / 0.60 m/s | 0.115 / 0.252 / 0.443 m | 0.3 / 0.9 / 1.9 mm |

Predicted on the robot against the identified first-order arm (the same model over-delivered by
10–13 % in Box2D, so these are conservative): `arc_wide` reaches 26 / 38 / 49 % of the arc for a
realised y span of **0.16 / 0.26 / 0.35 m**, against 0.21–0.25 m for *every* pre-fix condition.

**20 steps cannot complete a corner-to-corner arc and that is now explicit, not hidden.** At
0.39 m/s lateral authority the arm has 0.96 s and the arc is 0.85 m long; completing it needs
~41 steps at the top speed. The plan printout carries a `full@` column with that number per
condition.

## Next

- Re-run on hardware and re-fit `G` from the session (`--curve-velocity-gain`); the fit used
  `paddle_motion_20260909_1808`, whose x-axis trials saturate against the workspace edge, so `G_x`
  rests on the 0.33-delta cells alone.
- The first-order model ignores the ~3 steps the arm spends ramping from rest. If measured reach
  falls short of prediction, that is the first place to look.
