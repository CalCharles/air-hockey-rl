"""
Collect puck trajectories from the Box2D env and evaluate the velocity estimator
against ground-truth velocities.

For each scenario the puck is placed at a known position with a known velocity,
the env is stepped 10 times (no paddle action), and at each step we record:
  - noisy observed position (from the sim's normal pipeline: Gaussian noise + occlusion + delay)
  - ground-truth velocity   (from the Box2D body, no noise)
  - ground-truth position   (from the Box2D body, no noise)

Then fit_velocity_from_positions is run on the noisy positions and compared to GT.

Run from repo root:
    source .venv/bin/activate
    python3 notes/scratch/collect_and_evaluate_velocity_estimator.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from Box2D import b2Vec2

from airhockey import AirHockeyEnv
from airhockey.sims.real.velocity_estimator import fit_velocity_from_positions

# ── output dir ────────────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "velocity_estimator_viz", "sim_eval")
os.makedirs(OUT_DIR, exist_ok=True)

CONFIG_PATH = (
    "scripts/smooth_policy/amp_history/configs/new_juggle/"
    "sysid_best_params.yaml"
)

# Gravity in config is -0.65; the estimator expects the magnitude acting downward
# Box2D gravity = (0, -0.65) in Box2D coords.
# Converting to base (x_base = -y_b2d, y_base = x_b2d): puck ACCELERATES in +x_base.
# Estimator convention: positive g = deceleration. Acceleration = negative deceleration.
GRAVITY = (-0.65, 0.0)  # (gx, gy) deceleration in base coords

# Table safe bounds (puck_radius = 0.03175)
# In base coords: x_base = -by_box2d (LENGTH direction, ±0.9652m)
#                 y_base =  bx_box2d (WIDTH  direction, ±0.4318m)
PUCK_R   = 0.03175
X_BOUND  = 1.9304 / 2 - PUCK_R   # ≈ 0.933  (length direction)
Y_BOUND  = 0.8636 / 2 - PUCK_R   # ≈ 0.400  (width direction)

# ── coordinate helpers (airhockey_box2d.py lines 1211-1215) ──────────────────
def base_to_box2d(x, y):
    return float(y), float(-x)

def box2d_to_base(bx, by):
    return float(-by), float(bx)

# ── env creation ──────────────────────────────────────────────────────────────
def make_env():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["air_hockey"]
    return AirHockeyEnv(cfg)

# ── collision-free placement predictor ───────────────────────────────────────
def predict_final_pos(x0, y0, vx, vy, dt, n_steps):
    """Predict final puck position given gravity along +x in base coords."""
    T = dt * n_steps
    x_f = x0 + vx * T - 0.5 * GRAVITY[0] * T**2
    y_f = y0 + vy * T - 0.5 * GRAVITY[1] * T**2
    return x_f, y_f

# ── trajectory collection ────────────────────────────────────────────────────
def collect_trajectory(env, x0, y0, vx0, vy0, n_steps=10):
    """
    Reset env, override puck to (x0,y0) with velocity (vx0,vy0) in base coords,
    then step n_steps times. Returns None if a collision is detected.
    """
    obs, info = env.reset()
    sim = env.simulator
    dt = sim.time_per_step

    # Override puck body state
    puck = sim.pucks["puck_0"]
    bx, by = base_to_box2d(x0, y0)
    puck.position = b2Vec2(bx, by)
    bvx, bvy = base_to_box2d(vx0, vy0)
    puck.linearVelocity = b2Vec2(bvx, bvy)
    puck.angularVelocity = 0.0

    # Flush delay buffer and puck history so stale positions don't leak in
    if hasattr(sim, "puck_history"):
        sim.puck_history.clear()
    if hasattr(sim, "observation_puck_history"):
        sim.observation_puck_history = None
    if hasattr(sim, "_puck_delay_buffers"):
        sim._puck_delay_buffers = {}

    noisy_positions = []
    valid_flags     = []
    gt_velocities   = []
    gt_positions    = []

    for step_i in range(n_steps):
        state = sim.get_current_state()
        p = state["pucks"][0]

        # Noisy observed position (base coords, noise + occlusion + delay applied)
        noisy_positions.append(np.array(p["position"], dtype=float))
        valid_flags.append(not bool(p["occluded"]))

        # Ground-truth velocity (base coords, clean)
        gt_velocities.append(np.array(p["velocity"], dtype=float))

        # Ground-truth position from body
        bx_true, by_true = sim.pucks["puck_0"].position
        gt_positions.append(np.array(box2d_to_base(bx_true, by_true)))

        obs, rew, done, trunc, info = env.step(np.zeros(2))

        # Check paddle collision
        if info.get("paddle_puck_collision_count", 0) > 0:
            print(f"    [skip] paddle collision at step {step_i}")
            return None

        # Check wall collision by position
        cur_bx, cur_by = sim.pucks["puck_0"].position
        cx, cy = box2d_to_base(cur_bx, cur_by)
        if abs(cx) > X_BOUND or abs(cy) > Y_BOUND:
            print(f"    [skip] wall hit at step {step_i} pos=({cx:.3f},{cy:.3f})")
            return None

        if done:
            print(f"    [skip] episode ended at step {step_i}")
            return None

    return {
        "noisy_positions": np.array(noisy_positions),
        "valid_flags":     np.array(valid_flags),
        "gt_velocities":   np.array(gt_velocities),
        "gt_positions":    np.array(gt_positions),
        "times":           np.arange(n_steps) * dt,
        "dt":              dt,
        "x0": x0, "y0": y0, "vx0": vx0, "vy0": vy0,
    }

# ── scenarios ─────────────────────────────────────────────────────────────────
# (x0, y0, vx, vy, label)  — all in base coords
# x_base = LENGTH direction (±0.933m safe), gravity accelerates in +x
# y_base = WIDTH  direction (±0.400m safe), no gravity
# Puck placed near center; velocities chosen so 10 steps stay in bounds.
SCENARIOS = [
    (0.0,  0.0,  0.20,  0.00, "slow along length"),
    (0.0,  0.0,  0.60,  0.00, "medium along length"),
    (0.0,  0.0,  1.20,  0.00, "fast along length"),
    (0.0,  0.0, -0.60,  0.00, "medium against gravity"),
    (0.0,  0.0,  0.50,  0.25, "diagonal length+width"),
    (0.0,  0.0, -0.50,  0.25, "diagonal against gravity+width"),
    (0.0,  0.0,  0.00,  0.30, "pure width direction"),
    (0.0,  0.0,  0.80, -0.20, "fast length + small width"),
]

# ── plotting ──────────────────────────────────────────────────────────────────
def plot_scenario(traj, result, scenario_label, save_path):
    positions = traj["noisy_positions"]
    times     = traj["times"]
    valid     = traj["valid_flags"]
    gt_vel    = traj["gt_velocities"]
    gt_pos    = traj["gt_positions"]
    v_est     = result["v_at_times"]

    fig, ax = plt.subplots(figsize=(7, 6))

    # GT trajectory (dashed black)
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], "k--", linewidth=1.2, label="GT trajectory", zorder=2)

    # Noisy observed positions
    ax.scatter(positions[valid,  0], positions[valid,  1],
               color="steelblue", s=55, zorder=5, label="observed (valid)")
    if (~valid).any():
        ax.scatter(positions[~valid, 0], positions[~valid, 1],
                   color="lightgray", s=55, zorder=4, edgecolors="gray",
                   label="observed (occluded)")

    # Fitted trajectory line (reconstruct from estimator)
    dt_arr = times - times[0]
    gx, gy = GRAVITY
    vx0_fit  = v_est[-1, 0] + gx * dt_arr[-1]
    vy0_fit  = v_est[-1, 1] + gy * dt_arr[-1]
    x0_fit   = gt_pos[0, 0]
    y0_fit   = gt_pos[0, 1]
    t_line   = np.linspace(0, dt_arr[-1], 200)
    x_line   = x0_fit + vx0_fit * t_line - 0.5 * gx * t_line**2
    y_line   = y0_fit + vy0_fit * t_line - 0.5 * gy * t_line**2
    ax.plot(x_line, y_line, color="tomato", linewidth=1.5, zorder=3, label="fitted trajectory")

    # Arrow scale: 5% of trajectory span
    span = max(np.ptp(gt_pos[:, 0]), np.ptp(gt_pos[:, 1]), 0.05)
    arrow_scale = span * 0.08

    for i in range(len(times)):
        p = positions[i]
        # Estimated velocity arrows (blue)
        ve = v_est[i]
        if np.linalg.norm(ve) > 1e-6:
            ax.annotate("", xy=(p[0] + ve[0]*arrow_scale, p[1] + ve[1]*arrow_scale),
                        xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.0), zorder=7)

        # GT velocity arrows (orange), drawn from GT position
        gp = gt_pos[i]
        gv = gt_vel[i]
        if np.linalg.norm(gv) > 1e-6:
            ax.annotate("", xy=(gp[0] + gv[0]*arrow_scale, gp[1] + gv[1]*arrow_scale),
                        xytext=(gp[0], gp[1]),
                        arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.0), zorder=7)

    # Large arrows at collision moment (t[-1])
    p_end = positions[-1]
    ve_end = result["v_at_end"]
    ax.annotate("", xy=(p_end[0] + ve_end[0]*arrow_scale*2, p_end[1] + ve_end[1]*arrow_scale*2),
                xytext=(p_end[0], p_end[1]),
                arrowprops=dict(arrowstyle="-|>", color="royalblue", lw=2.0), zorder=8)
    gp_end = gt_pos[-1]
    gv_end = gt_vel[-1]
    ax.annotate("", xy=(gp_end[0] + gv_end[0]*arrow_scale*2, gp_end[1] + gv_end[1]*arrow_scale*2),
                xytext=(gp_end[0], gp_end[1]),
                arrowprops=dict(arrowstyle="-|>", color="darkorange", lw=2.0), zorder=8)

    # Per-timestep error table as text
    lines = ["t    GT_vx  GT_vy  est_vx est_vy err(m/s)"]
    for i, t in enumerate(times):
        gv = gt_vel[i]; ve = v_est[i]
        err = np.linalg.norm(gv - ve)
        lines.append(f"{t:.2f}  {gv[0]:+.3f} {gv[1]:+.3f}  {ve[0]:+.3f} {ve[1]:+.3f}  {err:.4f}")
    ax.text(0.01, 0.01, "\n".join(lines), transform=ax.transAxes,
            fontsize=6, family="monospace", verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    err_end = np.linalg.norm(gt_vel[-1] - result["v_at_end"])
    ax.set_title(
        f"{scenario_label}\n"
        f"GT v_end=[{gt_vel[-1,0]:+.3f}, {gt_vel[-1,1]:+.3f}]  "
        f"est=[{ve_end[0]:+.3f}, {ve_end[1]:+.3f}]  "
        f"err={err_end:.4f} m/s  SNR={result['snr']:.1f}",
        fontsize=9,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Legend with proxy artists for arrows
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", marker=">", label="estimated velocity"),
        Line2D([0], [0], color="darkorange", marker=">", label="GT velocity"),
        Line2D([0], [0], color="k", linestyle="--", label="GT trajectory"),
        Line2D([0], [0], color="tomato", label="fitted trajectory"),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    env = make_env()
    dt = env.simulator.time_per_step
    print(f"Env created. dt={dt:.4f}s  ({1/dt:.0f} Hz)")

    collected = []
    for x0, y0, vx, vy, label in SCENARIOS:
        # Sanity-check: will the puck stay in bounds for 10 steps?
        xf, yf = predict_final_pos(x0, y0, vx, vy, dt, n_steps=10)
        if abs(xf) > X_BOUND * 0.9 or abs(yf) > Y_BOUND * 0.9:
            print(f"[skip-pred] '{label}': predicted exit ({xf:.3f},{yf:.3f})")
            continue

        print(f"Collecting: '{label}'  v0=({vx},{vy}) m/s ...")
        traj = collect_trajectory(env, x0, y0, vx, vy, n_steps=10)
        if traj is None:
            continue

        result = fit_velocity_from_positions(
            traj["noisy_positions"], traj["times"], traj["valid_flags"], gravity=GRAVITY
        )
        if result is None:
            print(f"  [skip] not enough valid frames")
            continue

        gt_end = traj["gt_velocities"][-1]
        err_end = np.linalg.norm(gt_end - result["v_at_end"])
        print(f"  n_valid={result['n_valid']}  SNR={result['snr']:.1f}  "
              f"err_at_end={err_end:.4f} m/s")

        fname = label.replace(" ", "_") + ".png"
        plot_scenario(traj, result, label, os.path.join(OUT_DIR, fname))
        collected.append((label, traj, result))

    # Combined overview
    n = len(collected)
    if n == 0:
        print("No trajectories collected.")
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = np.array(axes).flatten()

    for ax_i, (label, traj, result) in enumerate(collected):
        ax = axes[ax_i]
        positions = traj["noisy_positions"]
        times     = traj["times"]
        valid     = traj["valid_flags"]
        gt_vel    = traj["gt_velocities"]
        gt_pos    = traj["gt_positions"]
        v_est     = result["v_at_times"]

        ax.plot(gt_pos[:, 0], gt_pos[:, 1], "k--", lw=1.0)
        ax.scatter(positions[valid,  0], positions[valid,  1],
                   color="steelblue", s=30, zorder=5)
        if (~valid).any():
            ax.scatter(positions[~valid, 0], positions[~valid, 1],
                       color="lightgray", s=30, zorder=4, edgecolors="gray")

        span = max(np.ptp(gt_pos[:, 0]), np.ptp(gt_pos[:, 1]), 0.05)
        arrow_scale = span * 0.08
        for i in range(len(times)):
            ve = v_est[i]; gv = gt_vel[i]
            p  = positions[i]; gp = gt_pos[i]
            if np.linalg.norm(ve) > 1e-6:
                ax.annotate("", xy=(p[0]+ve[0]*arrow_scale, p[1]+ve[1]*arrow_scale),
                            xytext=(p[0], p[1]),
                            arrowprops=dict(arrowstyle="->", color="steelblue", lw=0.8))
            if np.linalg.norm(gv) > 1e-6:
                ax.annotate("", xy=(gp[0]+gv[0]*arrow_scale, gp[1]+gv[1]*arrow_scale),
                            xytext=(gp[0], gp[1]),
                            arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.8))

        err_end = np.linalg.norm(gt_vel[-1] - result["v_at_end"])
        ax.set_title(f"{label}\nerr={err_end:.4f} m/s  SNR={result['snr']:.1f}",
                     fontsize=8)
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    for ax_i in range(n, len(axes)):
        axes[ax_i].set_visible(False)

    plt.suptitle(
        "Velocity estimator vs ground truth (Box2D sim)\n"
        "Blue arrows = estimated | Orange arrows = GT | Dashed = GT trajectory",
        fontsize=10,
    )
    plt.tight_layout()
    overview_path = os.path.join(OUT_DIR, "overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nOverview saved: {overview_path}")
    print("Done.")


if __name__ == "__main__":
    main()
