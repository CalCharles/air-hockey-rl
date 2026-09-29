"""Where the air-hockey table sits in the UR5 base frame.

The env (table) frame has its origin at the table center, x along the table
toward the robot, y parallel to robot y. A position converts as

    table_xy = robot_xy + (TABLE_CENTER_OFFSET_X, TABLE_CENTER_OFFSET_Y)

so the table center is at robot (-TABLE_CENTER_OFFSET_X, -TABLE_CENTER_OFFSET_Y).
No rotation is modelled: the table's long axis is assumed parallel to robot x.

Measured 2026-09-27 with the 4" (r = 0.0508 m) paddle pushed against the walls,
table 76" x 34.25" inside the walls:
  * robot-end wall:  TCP x = -0.42154  ->  X = (0.9652 - 0.0508) - (-0.42154) = 1.3359
  * side walls:      TCP y = -0.35112 / +0.42623
                     ->  table center at robot y = +0.0376, so Y = -0.0376
Re-measure these (and update only this file) whenever the robot or table moves.
Previous setup: X = 1.2, Y = 0.
"""

TABLE_CENTER_OFFSET_X = 1.3359
# TCP x with the paddle touching the robot-end wall (same 2026-09-27 measurement).
ROBOT_END_WALL_TCP_X = -0.42154
TABLE_CENTER_OFFSET_Y = -0.0376

# Box2D's offset (airhockey_box2d / airhockey_base default). The sim frame does not
# depend on the real calibration; kept so sim-side constants stay bit-identical.
SIM_CENTER_OFFSET_X = 1.2

# Occluded-puck / paddle placeholder the policies saw in training: Box2D uses
# (-2 + center_offset_constant, 0) = (-0.8, 0) in the table frame. Real must use the
# same table-frame value regardless of where the table is relative to the robot.
OCCLUDED_PLACEHOLDER_TABLE_X = -2.0 + SIM_CENTER_OFFSET_X


def robot_to_table_xy(x, y, offset_x, offset_y):
    return float(x) + float(offset_x), float(y) + float(offset_y)


def table_to_robot_xy(x, y, offset_x, offset_y):
    return float(x) - float(offset_x), float(y) - float(offset_y)
