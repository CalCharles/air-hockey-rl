import numpy as np

# ---------------------------------------------------------------------------
# Far-edge corner cuts (chamfers)
# ---------------------------------------------------------------------------
# The arm can't reach the two far corners of its workspace, so the x_max edge is
# chamfered near y_min and y_max. The cut is described as a SHAPE — a slope plus
# how much y it ramps over — and the per-corner biases below are derived from it
# against whatever the current limits are. That keeps the cut congruent at both
# corners when x_max_lim / y_min / y_max move or go asymmetric; the older style
# of pinning absolute biases silently resized (or deleted) the cut instead.
#
# Shape is taken from the measured +y corner: slope 0.8 (dx per dy) ramping in
# over the last 0.0525 m of y, i.e. 0.042 m deep at the corner itself.
CORNER_CUT_SLOPE = 0.8
CORNER_CUT_Y_EXTENT = 0.1


def corner_cut_biases(
    x_max_lim,
    y_min,
    y_max,
    slope=CORNER_CUT_SLOPE,
    y_extent=CORNER_CUT_Y_EXTENT,
):
    """Derive the two corner-cut biases for ``edge_lims`` slots 2 and 3.

    Returns ``(bias_pos_y, bias_neg_y)`` — the x-intercepts of each corner's
    chamfer line, placed so the cut begins ``y_extent`` before the corner and is
    ``slope * y_extent`` deep at it, on both sides independently.
    """
    slope = float(slope)
    y_extent = max(0.0, float(y_extent))
    bias_pos_y = float(x_max_lim) + slope * (abs(float(y_max)) - y_extent)
    bias_neg_y = float(x_max_lim) + slope * (abs(float(y_min)) - y_extent)
    return bias_pos_y, bias_neg_y


def effective_x_max(y, lims, edge_lims):
    """x_max at a given y, with both far-corner cuts applied.

    Single source of truth for the chamfered edge so the clip path and the
    camera overlay can't drift apart. ``edge_lims`` is
    ``(top_abs, bot_abs, bias_pos_y, bias_neg_y)``: slot 2 shapes the +y corner,
    slot 3 the -y corner.
    """
    _, x_max_lim, _, _ = lims
    top_abs, _, bias_pos_y, bias_neg_y = edge_lims
    return min(
        x_max_lim,
        bias_pos_y - top_abs * y,
        bias_neg_y + top_abs * y,
    )


# limit rounding
def clip_limits(x,y,lims, edge_lims):
    x_min_lim, x_max_lim, y_min, y_max = lims

    y = np.clip(y, y_min, y_max, )
    # x_min = x_min_lim  + bot_abs * np.abs(y)
    x_min = x_min_lim
    x_max = effective_x_max(y, lims, edge_lims)
    x = np.clip(x, x_min, x_max, ) # Workspace limits
    return x,y

def get_clip_limits(x,y,lims, edge_lims):
    x_min_lim, x_max_lim, y_min, y_max = lims
    y = np.clip(y, y_min, y_max, )
    # x_min = x_min_lim  + bot_abs * np.abs(y)
    x_min = x_min_lim
    x_max = effective_x_max(y, lims, edge_lims)
    return x_min,x_max,y_min, y_max


def get_edge(x,y, w, h):
    # returns relative coordinate bounded by w,h
    if np.abs(x) <= w and np.abs(y) <= h:
        return np.array([x, y])
    s = (y)/(x) # slope
    if -h/2 <= s * w/2 <= h/2:
        if x > 0:
            # print("high x", s, np.array([w, s * w]))
            return np.array([w, s * w])
        elif x < 0:
            # print("low x", s, np.array([-w, -s * w]))
            return np.array([-w, -s * w])
    elif -w/2 <= h/(2*s) <= w/2:
        s_r = (x)/(y) # slope
        # print(y)
        if y > 0:
            # print("high y", s_r,y,h, np.array([h * s_r, h]))
            return np.array([h * s_r, h])
        elif y < 0:
            # print("low y", s_r, np.array([-h * s_r, -h]))
            return np.array([-h * s_r, -h])

def smoothen(history_pos, history_vel, relative, limits):
    # smoothens trajectories at the far end by lagging the inputs at the x endpoint
    pass

def compute_pol(x,y,true_pose, lims, move_lims, edge_lims):
    rmax_x, rmax_y = move_lims
    relx, rely = (x - true_pose[0]), (y-true_pose[1])
    rad = lambda x,y: np.sqrt(x ** 2 + y ** 2)
    dist = rad(relx, rely)
    polx, poly = min(dist, rmax_x) * relx / dist + true_pose[0], min(dist, rmax_y) * rely / dist + true_pose[1] # Project to circle
    polx, poly = clip_limits(polx, poly, lims, edge_lims)
    return polx, poly

def compute_rect(x,y,true_pose, lims, move_lims, edge_lims):
    rmax_x, rmax_y = move_lims
    relx, rely = (x - true_pose[0]), (y-true_pose[1])
    # print((y, rely, true_pose[1]))
    recx, recy = get_edge(relx, rely, rmax_x, rmax_y)
    recx, recy = recx + true_pose[0], recy + true_pose[1]
    x_min_lim, x_max_lim, y_min, y_max = lims
    recx, recy = clip_limits(recx, recy, lims, edge_lims)
    return recx, recy
