import numpy as np

# limit rounding
def clip_limits(x,y,lims, edge_lims):
    x_min_lim, x_max_lim, y_min, y_max = lims
    top_abs, bot_abs, max_bias_m, max_bias_p = edge_lims
    y = np.clip(y, y_min, y_max, )
    # x_min = x_min_lim  + bot_abs * np.abs(y)
    x_min = x_min_lim
    # x_max = x_max_lim - top_abs * np.abs(y)
    x_max = min(x_max_lim, max_bias_m - top_abs * y, max_bias_p + top_abs * y)
    # x_min, x_max = x_min_lim, x_max_lim
    # print(x_min, x_max,max_bias_m - top_abs * y, max_bias_p + top_abs * y, y)
    x = np.clip(x, x_min, x_max, ) # Workspace limits
    return x,y

def get_clip_limits(x,y,lims, edge_lims):
    x_min_lim, x_max_lim, y_min, y_max = lims
    top_abs, bot_abs, max_bias_m, max_bias_p = edge_lims
    y = np.clip(y, y_min, y_max, )
    # x_min = x_min_lim  + bot_abs * np.abs(y)
    x_min = x_min_lim
    # x_max = x_max_lim - top_abs * np.abs(y)
    x_max = min(x_max_lim, max_bias_m - top_abs * y, max_bias_p + top_abs * y)
    # x_min, x_max = x_min_lim, x_max_lim
    # print(x_min, x_max,max_bias_m - top_abs * y, max_bias_p + top_abs * y, y)
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
