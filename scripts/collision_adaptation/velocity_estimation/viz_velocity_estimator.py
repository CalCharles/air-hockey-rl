"""
Visualize velocity estimation from noisy position trajectories.

For each example: scatter raw positions (valid=blue, occluded=gray),
overlay the fitted trajectory line, show gravity-corrected positions,
and draw velocity arrows at each timestep.

Run from repo root:
    source .venv/bin/activate
    python3 notes/scratch/viz_velocity_estimator.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from airhockey.sims.real.velocity_estimator import fit_velocity_from_positions

OUT_DIR = os.path.join(os.path.dirname(__file__), "velocity_estimator_viz")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_trajectory(ax, positions, times, valid_mask, result, g, title):
    """
    Single-trajectory diagnostic plot.

    Shows:
    - Raw observed positions (blue=valid, gray=occluded)
    - Gravity-corrected positions (green crosses) — what the fit actually sees
    - Fitted linear trajectory line (red)
    - Velocity arrows at each timestep
    - Velocity arrow at collision moment (larger, black)
    """
    t0 = times[0]
    dt = times - t0

    # Gravity-corrected y positions (what the linear fit operates on)
    y_corr = positions[:, 1] + 0.5 * g * dt ** 2
    pos_corr = np.column_stack([positions[:, 0], y_corr])

    # Reconstruct fitted line: evaluate fitted positions at each timestep
    # The fit gives us velocity at each time, and we know v_at_times.
    # Reconstruct x0, y0 from v_at_end and times[-1]:
    v = result["v_at_times"]
    # x(t) = x0 + vx*t  => x0 = x_end - vx * dt[-1]
    x0_fit = positions[-1, 0] - v[-1, 0] * dt[-1]
    # y_corr(t) = y0 + vy0 * t  => y0 = y_corr_end - vy0 * dt[-1]
    # vy0 = vy at t0 = vy_end + g * dt[-1]
    vy0 = v[-1, 1] + g * dt[-1]
    y0_fit = pos_corr[-1, 1] - vy0 * dt[-1]

    # Dense line for fitted trajectory (in original coordinates)
    t_line = np.linspace(0, dt[-1], 200)
    x_line = x0_fit + v[-1, 0] * t_line
    y_corr_line = y0_fit + vy0 * t_line
    y_line = y_corr_line - 0.5 * g * t_line ** 2  # back to original

    # --- Plot ---
    # Raw observed positions
    valid = valid_mask.astype(bool)
    ax.scatter(positions[valid, 0], positions[valid, 1],
               color="steelblue", s=60, zorder=5, label="observed (valid)")
    if (~valid).any():
        ax.scatter(positions[~valid, 0], positions[~valid, 1],
                   color="lightgray", s=60, zorder=4, edgecolors="gray",
                   label="observed (occluded)")

    # Gravity-corrected positions
    ax.scatter(pos_corr[:, 0], pos_corr[:, 1],
               marker="+", color="green", s=80, zorder=6,
               linewidths=1.5, label="gravity-corrected")

    # Fitted trajectory line (original coords)
    ax.plot(x_line, y_line, color="tomato", linewidth=1.5,
            zorder=3, label="fitted trajectory")

    # Velocity arrows at each timestep (scaled for readability)
    arrow_scale = 0.05  # seconds — arrow tip = position + v * arrow_scale
    for i in range(len(times)):
        ax.annotate(
            "", xy=(positions[i, 0] + v[i, 0] * arrow_scale,
                    positions[i, 1] + v[i, 1] * arrow_scale),
            xytext=(positions[i, 0], positions[i, 1]),
            arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.0),
            zorder=7,
        )

    # Larger arrow at collision moment (last point)
    collision_scale = 0.12
    ax.annotate(
        "", xy=(positions[-1, 0] + v[-1, 0] * collision_scale,
                positions[-1, 1] + v[-1, 1] * collision_scale),
        xytext=(positions[-1, 0], positions[-1, 1]),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0),
        zorder=8,
    )

    # Label timesteps
    for i in range(len(times)):
        ax.text(positions[i, 0] + 0.003, positions[i, 1] + 0.003,
                f"t{i}", fontsize=7, color="gray")

    v_end = result["v_at_end"]
    speed = np.linalg.norm(v_end)
    ax.set_title(
        f"{title}\n"
        f"v_end=[{v_end[0]:.3f}, {v_end[1]:.3f}] m/s  |v|={speed:.3f}  "
        f"SNR={result['snr']:.1f}  n_valid={result['n_valid']}",
        fontsize=9,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(fontsize=7, loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)


def make_example(seed, true_vx, true_vy0, g, noise_std, occluded_idx=None, n=10, dt=0.05):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    positions = np.column_stack([
        true_vx * t + rng.normal(0, noise_std, n),
        true_vy0 * t - 0.5 * g * t**2 + rng.normal(0, noise_std, n),
    ])
    valid_mask = np.ones(n, dtype=bool)
    if occluded_idx:
        for idx in occluded_idx:
            valid_mask[idx] = False
    result = fit_velocity_from_positions(positions, t, valid_mask, gravity=(0.0, g))
    return positions, t, valid_mask, result


examples = [
    # (seed, vx,  vy0,  g,   noise, occluded,       title)
    (0,  0.50, 1.00, 0.7, 0.005, [],          "Clean — g=0.7, fast puck"),
    (1,  0.50, 1.00, 0.7, 0.005, [2, 5, 8],   "3 occluded frames"),
    (2,  0.30, 0.80, 0.0, 0.003, [],          "g=0 (flat table)"),
    (3,  0.05, 0.05, 0.7, 0.020, [],          "Low speed + high noise (low SNR)"),
    (4,  0.80, 0.20, 0.7, 0.008, [0, 1, 7, 8], "4 occluded (including ends)"),
    (5, -0.40, 0.60, 0.7, 0.006, [],          "Negative vx (puck moving left)"),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for ax, (seed, vx, vy0, g, noise, occ, title) in zip(axes, examples):
    positions, t, valid_mask, result = make_example(seed, vx, vy0, g, noise, occ)
    if result is None:
        ax.set_title(f"{title}\nNot enough valid frames")
        continue
    expected_vy_end = vy0 - g * t[-1]
    full_title = f"{title}\nexpected v_end=[{vx:.2f}, {expected_vy_end:.3f}]"
    plot_trajectory(ax, positions, t, valid_mask, result, g, full_title)

plt.suptitle(
    "Velocity estimator: gravity-corrected linear regression\n"
    "Blue arrows = v at each timestep | Black arrow = v at collision moment | "
    "Green + = gravity-corrected positions",
    fontsize=10,
)
plt.tight_layout()
out_path = os.path.join(OUT_DIR, "all_examples.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Also save individual plots for closer inspection
for i, (seed, vx, vy0, g, noise, occ, title) in enumerate(examples):
    positions, t, valid_mask, result = make_example(seed, vx, vy0, g, noise, occ)
    if result is None:
        continue
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    expected_vy_end = vy0 - g * t[-1]
    full_title = f"{title}\nexpected v_end=[{vx:.2f}, {expected_vy_end:.3f}]"
    plot_trajectory(ax2, positions, t, valid_mask, result, g, full_title)
    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f"example_{i+1}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {fname}")

print("Done.")
