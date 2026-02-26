"""Analyze distributions of x_fall_height and y_displacement across all approach intervals."""

import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

fall_heights = []
y_displacements = []
durations = []

files = sorted(glob.glob(os.path.join(LOG_DIR, "inflection_*.json")))
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    for a in data.get("approach_intervals", []):
        fh_val = a.get("x_fall_height")
        yd_val = a.get("y_displacement")
        if fh_val is None or yd_val is None:
            continue
        fall_heights.append(fh_val)
        y_displacements.append(yd_val)
        durations.append(a.get("duration", 0))

fall_heights = np.array(fall_heights)
y_displacements = np.array(y_displacements)
durations = np.array(durations)

print(f"Total approach intervals: {len(fall_heights)}")
print(f"\nx_fall_height:   min={fall_heights.min():.4f}  max={fall_heights.max():.4f}  "
      f"mean={fall_heights.mean():.4f}  median={np.median(fall_heights):.4f}")
print(f"y_displacement:  min={y_displacements.min():.4f}  max={y_displacements.max():.4f}  "
      f"mean={y_displacements.mean():.4f}  median={np.median(y_displacements):.4f}")
print(f"duration:        min={durations.min()}  max={durations.max()}  "
      f"mean={durations.mean():.1f}  median={np.median(durations):.1f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Fall height histogram
ax = axes[0, 0]
ax.hist(fall_heights, bins=50, edgecolor="black", alpha=0.7)
ax.set_xlabel("x_fall_height (m)")
ax.set_ylabel("Count")
ax.set_title(f"Fall Height Distribution (n={len(fall_heights)})")
ax.axvline(np.median(fall_heights), color="red", linestyle="--", label=f"median={np.median(fall_heights):.3f}")
ax.legend()

# Y displacement histogram
ax = axes[0, 1]
ax.hist(y_displacements, bins=50, edgecolor="black", alpha=0.7, color="orange")
ax.set_xlabel("y_displacement (m)")
ax.set_ylabel("Count")
ax.set_title(f"Y Displacement Distribution (n={len(y_displacements)})")
ax.axvline(np.median(y_displacements), color="red", linestyle="--", label=f"median={np.median(y_displacements):.3f}")
ax.legend()

# Scatter: fall height vs y displacement
ax = axes[1, 0]
ax.scatter(fall_heights, y_displacements, alpha=0.3, s=10)
ax.set_xlabel("x_fall_height (m)")
ax.set_ylabel("y_displacement (m)")
ax.set_title("Fall Height vs Y Displacement")

# Duration histogram
ax = axes[1, 1]
ax.hist(durations, bins=50, edgecolor="black", alpha=0.7, color="green")
ax.set_xlabel("Duration (timesteps)")
ax.set_ylabel("Count")
ax.set_title(f"Interval Duration Distribution (n={len(durations)})")
ax.axvline(np.median(durations), color="red", linestyle="--", label=f"median={np.median(durations):.0f}")
ax.legend()

plt.tight_layout()
out_path = os.path.join(LOG_DIR, "approach_distributions.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
plt.close()
