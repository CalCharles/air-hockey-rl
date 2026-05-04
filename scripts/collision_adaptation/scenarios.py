"""
Crafted collision scenarios for visual inspection and adaptation validation.

Coordinate convention (base frame):
    x-axis: along table length.
        Negative x = "upper half" (against gravity; puck is juggled here).
        Positive x = player's side (gravity direction; paddle lives here, x ≈ 0 to +0.97).
    y-axis: across table width.

    base_coord_to_box2d(x, y) = (y, -x)

Gravity pulls pucks toward positive x at 0.65 m/s², so puck approach speed increases
during flight.  Scenarios are placed close to the paddle (gap 0.10–0.20 m) to control
the effective collision speed and keep GIFs short.

Speed tiers (from CollisionForceListener defaults):
    low  < 0.25 m/s
    mid  0.25–0.75 m/s
    high ≥ 0.75 m/s

All positions and velocities are in base frame.
"""

# Each entry:
#   name          : str  - used in filenames and scenarios.json keys
#   puck_pos      : (x, y) in base frame (m)
#   puck_vel      : (vx, vy) in base frame (m/s)
#   paddle_pos    : (x, y) in base frame (m) — player-side, positive x
#   n_steps       : int  - maximum steps to run
#   paddle_action : [ax, ay]  - action applied each step (normalised, -1 to 1)
SCENARIOS = [
    {
        # Puck very close, low initial speed → arrives at low/mid tier before gravity kicks in.
        "name": "head_on_slow",
        "puck_pos": (0.38, 0.0),
        "puck_vel": (0.10, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 80,
        "paddle_action": [0.0, 0.0],
    },
    {
        # Medium gap, mid initial speed → mid tier collision.
        "name": "head_on_mid",
        "puck_pos": (0.30, 0.0),
        "puck_vel": (0.45, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.0, 0.0],
    },
    {
        # Larger gap, high initial speed + gravity → high tier.
        "name": "head_on_fast",
        "puck_pos": (0.10, 0.0),
        "puck_vel": (0.90, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.0, 0.0],
    },
    {
        # Puck approaches at ~15° off the normal with y-offset → glancing collision.
        "name": "glancing_40deg",
        "puck_pos": (0.35, 0.06),
        "puck_vel": (0.40, -0.08),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.0, 0.0],
    },
    {
        # Active hit: paddle moves toward puck (action negative x).
        # Puck drifts +x, paddle chases from further right.
        "name": "active_hit",
        "puck_pos": (0.35, 0.03),
        "puck_vel": (0.25, 0.0),
        "paddle_pos": (0.65, 0.03),
        "n_steps": 80,
        "paddle_action": [-0.5, 0.0],   # drive paddle toward puck (−x direction)
    },

    # --- Additional scenarios: varied angles and higher speeds (paddle stationary) ---

    {
        # Very fast head-on: puck enters at ~1.5 m/s → high tier, big rebound.
        "name": "very_fast_head_on",
        "puck_pos": (0.10, 0.0),
        "puck_vel": (1.50, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 40,
        "paddle_action": [0.0, 0.0],
    },
    {
        # 30° off normal (shallow glancing), mid speed ~0.55 m/s.
        # Puck starts below paddle center and drifts up.
        "name": "angled_30deg_mid",
        "puck_pos": (0.33, -0.08),
        "puck_vel": (0.48, 0.28),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.0, 0.0],
    },
    {
        # 45° off normal, high speed ~0.9 m/s.
        "name": "angled_45deg_high",
        "puck_pos": (0.32, -0.12),
        "puck_vel": (0.636, 0.636),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.0, 0.0],
    },
    {
        # 60° off normal, mid speed ~0.6 m/s — steep approach, mostly sideways.
        # Paddle slightly offset in y to intercept.
        "name": "angled_60deg_mid",
        "puck_pos": (0.43, -0.06),
        "puck_vel": (0.30, 0.52),
        "paddle_pos": (0.50, 0.08),
        "n_steps": 60,
        "paddle_action": [0.0, 0.0],
    },
    {
        # ~68° off normal, high speed ~1.0 m/s — nearly perpendicular, high energy.
        "name": "angled_steep_high",
        "puck_pos": (0.43, -0.10),
        "puck_vel": (0.40, 1.00),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.0, 0.0],
    },

    # ==========================================================================
    # Paddle-moving scenarios
    # action[0] < 0 → paddle moves toward lower base_x (toward upper half / puck)
    # action[0] > 0 → paddle moves toward higher base_x (away from puck / retreating)
    # action[1]     → paddle moves in y direction (lateral)
    # Pre-collision paddle speeds (measured): toward ~0.13–0.28 m/s,
    #   retreat ~0.12–0.34 m/s, lateral ~0.10–0.19 m/s.
    # ==========================================================================

    # --- Paddle moving toward puck (convergent) ---

    {
        # Head-on puck, paddle converging slowly (action=-0.3).
        # Paddle pre-collision vx ≈ −0.13 m/s.
        "name": "pad_toward_slow",
        "puck_pos": (0.30, 0.0),
        "puck_vel": (0.45, 0.0),
        "paddle_pos": (0.65, 0.0),
        "n_steps": 80,
        "paddle_action": [-0.3, 0.0],
    },
    {
        # Head-on puck, paddle converging at medium speed (action=-0.6).
        # Paddle pre-collision vx ≈ −0.26 m/s.
        "name": "pad_toward_mid",
        "puck_pos": (0.20, 0.0),
        "puck_vel": (0.45, 0.0),
        "paddle_pos": (0.70, 0.0),
        "n_steps": 80,
        "paddle_action": [-0.6, 0.0],
    },
    {
        # Head-on puck, paddle at full convergence (action=-1.0).
        # Paddle pre-collision vx ≈ −0.28 m/s.
        "name": "pad_toward_fast",
        "puck_pos": (0.10, 0.0),
        "puck_vel": (0.45, 0.0),
        "paddle_pos": (0.75, 0.0),
        "n_steps": 80,
        "paddle_action": [-1.0, 0.0],
    },
    {
        # Fast puck + full convergence → highest combined approach speed.
        "name": "pad_toward_fast_fast_puck",
        "puck_pos": (0.10, 0.0),
        "puck_vel": (0.90, 0.0),
        "paddle_pos": (0.75, 0.0),
        "n_steps": 60,
        "paddle_action": [-1.0, 0.0],
    },
    {
        # 30° angled puck + paddle converging.  Paddle offset handled by meeting trajectory.
        "name": "angled_30deg_pad_toward",
        "puck_pos": (0.28, -0.08),
        "puck_vel": (0.48, 0.28),
        "paddle_pos": (0.62, 0.0),
        "n_steps": 80,
        "paddle_action": [-0.6, 0.0],
    },
    {
        # 45° angled puck + paddle converging. Paddle starts offset to intercept arc.
        "name": "angled_45deg_pad_toward",
        "puck_pos": (0.28, -0.10),
        "puck_vel": (0.636, 0.636),
        "paddle_pos": (0.58, 0.08),
        "n_steps": 60,
        "paddle_action": [-0.6, 0.0],
    },

    # --- Paddle retreating (divergent — fast puck catches up) ---

    {
        # Paddle retreats slowly (action=+0.3), puck fast enough to catch it.
        # Paddle pre-collision vx ≈ +0.12 m/s.
        "name": "pad_retreat_slow",
        "puck_pos": (0.20, 0.0),
        "puck_vel": (0.70, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.3, 0.0],
    },
    {
        # Paddle retreats at medium speed (action=+0.6), puck must be faster still.
        # Paddle pre-collision vx ≈ +0.14 m/s.
        "name": "pad_retreat_mid",
        "puck_pos": (0.10, 0.0),
        "puck_vel": (0.90, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.6, 0.0],
    },
    {
        # Fast retreat (action=+1.0), very fast puck (1.2 m/s) — barely catches.
        # Paddle pre-collision vx ≈ +0.19 m/s.
        "name": "pad_retreat_fast",
        "puck_pos": (0.05, 0.0),
        "puck_vel": (1.20, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 40,
        "paddle_action": [1.0, 0.0],
    },
    {
        # Very fast retreat (action=+1.0) + very fast puck (1.8 m/s) — high energy hit.
        "name": "pad_retreat_fast_fast_puck",
        "puck_pos": (0.05, 0.0),
        "puck_vel": (1.80, 0.0),
        "paddle_pos": (0.55, 0.0),
        "n_steps": 40,
        "paddle_action": [1.0, 0.0],
    },
    {
        # 30° angled puck + retreating paddle.
        "name": "angled_30deg_pad_retreat",
        "puck_pos": (0.20, -0.06),
        "puck_vel": (0.60, 0.25),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.4, 0.0],
    },

    # --- Paddle moving laterally (y-direction) during head-on collision ---

    {
        # Paddle sliding in +y while puck approaches head-on.
        # Paddle pre-collision vy ≈ +0.10 m/s.
        "name": "pad_lateral_pos_y",
        "puck_pos": (0.30, 0.0),
        "puck_vel": (0.60, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.0, 0.5],
    },
    {
        # Paddle sliding in −y while puck approaches head-on.
        "name": "pad_lateral_neg_y",
        "puck_pos": (0.30, 0.0),
        "puck_vel": (0.60, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.0, -0.5],
    },
    {
        # Fast lateral slide (action y=+1.0). Paddle pre-collision vy ≈ +0.19 m/s.
        "name": "pad_lateral_fast_pos_y",
        "puck_pos": (0.30, 0.0),
        "puck_vel": (0.60, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.0, 1.0],
    },
    {
        # Fast lateral slide in −y + high-speed puck.
        "name": "pad_lateral_fast_neg_y_fast_puck",
        "puck_pos": (0.15, 0.0),
        "puck_vel": (1.00, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 50,
        "paddle_action": [0.0, -1.0],
    },

    # --- Combined: convergent + lateral (diagonal paddle motion) ---

    {
        # Paddle converges AND slides in +y — diagonal paddle velocity.
        "name": "pad_diag_toward_pos_y",
        "puck_pos": (0.25, 0.0),
        "puck_vel": (0.50, 0.0),
        "paddle_pos": (0.65, 0.0),
        "n_steps": 80,
        "paddle_action": [-0.6, 0.5],
    },
    {
        # Paddle retreats AND slides in −y — opposite diagonal.
        "name": "pad_diag_retreat_neg_y",
        "puck_pos": (0.10, 0.0),
        "puck_vel": (0.90, 0.0),
        "paddle_pos": (0.50, 0.0),
        "n_steps": 60,
        "paddle_action": [0.5, -0.6],
    },
    {
        # 45° puck + convergent paddle + lateral — triple-dimension scenario.
        "name": "angled_45deg_pad_diag",
        "puck_pos": (0.30, -0.12),
        "puck_vel": (0.636, 0.636),
        "paddle_pos": (0.60, 0.0),
        "n_steps": 70,
        "paddle_action": [-0.5, 0.4],
    },

    # ==========================================================================
    # Extreme paddle-speed scenarios
    # These push the paddle to its highest achievable pre-collision speeds
    # using long-runway acceleration before the puck arrives.
    #
    # Measured pre-collision paddle speeds (base frame):
    #   extreme_toward_peaked    : vx = -0.435 m/s  (fastest toward)
    #   extreme_toward_plus_lat  : spd =  0.480 m/s  (highest total, toward + lateral)
    #   extreme_retreat_peaked   : vx = +0.414 m/s  (fast retreat)
    #   extreme_retreat_plus_lat : spd =  0.480 m/s  (highest total, retreat + lateral)
    #   extreme_toward_angled    : vx = -0.414 m/s  (fast toward + angled puck)
    # ==========================================================================

    {
        # Paddle from x=0.90 converging at full power (action=-1.0).
        # Collision at step ~5 when paddle has built to max toward speed.
        # Pre-collision pad_vx ≈ -0.435 m/s.
        "name": "extreme_toward_peaked",
        "puck_pos": (0.40, 0.0),
        "puck_vel": (1.00, 0.0),
        "paddle_pos": (0.90, 0.0),
        "n_steps": 40,
        "paddle_action": [-1.0, 0.0],
    },
    {
        # Paddle from x=0.90, full-power toward + full-power lateral (+y).
        # Diagonal paddle motion at highest achievable total speed ~0.480 m/s.
        # Pre-collision: pad_vx ≈ -0.436, pad_vy ≈ +0.201.
        "name": "extreme_toward_plus_lat",
        "puck_pos": (0.40, 0.0),
        "puck_vel": (1.00, 0.0),
        "paddle_pos": (0.90, 0.0),
        "n_steps": 40,
        "paddle_action": [-1.0, 1.0],
    },
    {
        # Paddle from x=0.25 retreating at full power (action=+1.0).
        # Fast puck (-0.05 → +x) catches retreating paddle after ~5 steps.
        # Pre-collision pad_vx ≈ +0.414 m/s.
        "name": "extreme_retreat_peaked",
        "puck_pos": (-0.05, 0.0),
        "puck_vel": (1.60, 0.0),
        "paddle_pos": (0.25, 0.0),
        "n_steps": 30,
        "paddle_action": [1.0, 0.0],
    },
    {
        # Retreat + full lateral combined: highest-total-speed retreat at ~0.480 m/s.
        # Pre-collision: pad_vx ≈ +0.436, pad_vy ≈ +0.201.
        "name": "extreme_retreat_plus_lat",
        "puck_pos": (-0.05, 0.0),
        "puck_vel": (1.60, 0.0),
        "paddle_pos": (0.25, 0.0),
        "n_steps": 30,
        "paddle_action": [1.0, 1.0],
    },
    {
        # Fast toward paddle meets angled puck: paddle at peak speed + non-normal impact.
        # Pre-collision pad_vx ≈ -0.414 m/s, puck at 45° from below.
        "name": "extreme_toward_angled",
        "puck_pos": (0.40, -0.10),
        "puck_vel": (1.00, 0.50),
        "paddle_pos": (0.90, 0.0),
        "n_steps": 40,
        "paddle_action": [-1.0, 0.0],
    },
]
