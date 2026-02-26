"""Distribution of y_displacement and x_fall_height for approach intervals whose peak has vx==0."""

import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

fall_heights = []
y_displacements = []

files = sorted(glob.glob(os.path.join(LOG_DIR, "inflection_*.json")))
for f in files:
    with open(f) as fh:
        data = json.load(fh)

    # Build set of peak timesteps where vx == 0
    zero_vx_peaks = set()
    for p in data.get("peaks", []):
        if isinstance(p, dict) and "velocity" in p:
            if p["velocity"]["vx"] == 0.0:
                zero_vx_peaks.add(p["timestep"])

    # Filter approach intervals to those starting at a zero-vx peak
    for a in data.get("approach_intervals", []):
        peak_t = a["interval"]["start"]
        if peak_t not in zero_vx_peaks:
            continue
        fh_val = a.get("x_fall_height")
        yd_val = a.get("y_displacement")
        if fh_val is None or yd_val is None:
            continue
        fall_heights.append(fh_val)
        y_displacements.append(yd_val)

fall_heights = np.array(fall_heights)
y_displacements = np.array(y_displacements)

print(f"Approach intervals with vx==0 at peak: {len(fall_heights)}")
print(f"\nx_fall_height:   min={fall_heights.min():.4f}  max={fall_heights.max():.4f}  "
      f"mean={fall_heights.mean():.4f}  median={np.median(fall_heights):.4f}  "
      f"std={fall_heights.std():.4f}")
print(f"y_displacement:  min={y_displacements.min():.4f}  max={y_displacements.max():.4f}  "
      f"mean={y_displacements.mean():.4f}  median={np.median(y_displacements):.4f}  "
      f"std={y_displacements.std():.4f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Approach intervals where peak vx == 0 (parabolic fit)", fontsize=14)

ax = axes[0, 0]
ax.hist(fall_heights, bins=40, edgecolor="black", alpha=0.7)
ax.set_xlabel("x_fall_height (m)")
ax.set_ylabel("Count")
ax.set_title(f"Fall Height (n={len(fall_heights)})")
ax.axvline(np.median(fall_heights), color="red", linestyle="--",
           label=f"median={np.median(fall_heights):.3f}")
ax.legend()

ax = axes[0, 1]
ax.hist(y_displacements, bins=40, edgecolor="black", alpha=0.7, color="orange")
ax.set_xlabel("y_displacement (m)")
ax.set_ylabel("Count")
ax.set_title(f"Y Displacement (n={len(y_displacements)})")
ax.axvline(np.median(y_displacements), color="red", linestyle="--",
           label=f"median={np.median(y_displacements):.3f}")
ax.legend()

ax = axes[1, 0]
ax.scatter(fall_heights, y_displacements, alpha=0.3, s=10)
ax.set_xlabel("x_fall_height (m)")
ax.set_ylabel("y_displacement (m)")
ax.set_title("Fall Height vs Y Displacement")

ax = axes[1, 1]
ax.hist2d(fall_heights, y_displacements, bins=30, cmap="YlOrRd")
ax.set_xlabel("x_fall_height (m)")
ax.set_ylabel("y_displacement (m)")
ax.set_title("2D Histogram")
plt.colorbar(ax.collections[0], ax=ax, label="Count")

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "zero_vx_approach_distributions.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
plt.close()
