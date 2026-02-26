"""Analyze distribution of vx (and vy) at peaks across all inflection logs."""

import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

vx_vals = []
vy_vals = []

files = sorted(glob.glob(os.path.join(LOG_DIR, "inflection_*.json")))
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    for p in data.get("peaks", []):
        if isinstance(p, dict) and "velocity" in p:
            vx_vals.append(p["velocity"]["vx"])
            vy_vals.append(p["velocity"]["vy"])

vx_vals = np.array(vx_vals)
vy_vals = np.array(vy_vals)

vx_zero = int(np.sum(vx_vals == 0.0))
vx_nonzero = len(vx_vals) - vx_zero

print(f"Total peaks with velocity: {len(vx_vals)}")
print(f"  vx == 0: {vx_zero} ({100*vx_zero/len(vx_vals):.1f}%)")
print(f"  vx != 0: {vx_nonzero} ({100*vx_nonzero/len(vx_vals):.1f}%)")
print(f"\nvx:  min={vx_vals.min():.6f}  max={vx_vals.max():.6f}  "
      f"mean={vx_vals.mean():.6f}  std={vx_vals.std():.6f}")
print(f"vy:  min={vy_vals.min():.6f}  max={vy_vals.max():.6f}  "
      f"mean={vy_vals.mean():.6f}  std={vy_vals.std():.6f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.hist(vx_vals, bins=50, edgecolor="black", alpha=0.7)
ax.set_xlabel("vx at peak (m/s)")
ax.set_ylabel("Count")
ax.set_title(f"Peak vx Distribution (n={len(vx_vals)})")
ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="vx=0")
ax.axvline(vx_vals.mean(), color="orange", linestyle="--", label=f"mean={vx_vals.mean():.4f}")
ax.legend()

ax = axes[1]
ax.hist(vy_vals, bins=50, edgecolor="black", alpha=0.7, color="orange")
ax.set_xlabel("vy at peak (m/s)")
ax.set_ylabel("Count")
ax.set_title(f"Peak vy Distribution (n={len(vy_vals)})")
ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="vy=0")
ax.axvline(vy_vals.mean(), color="blue", linestyle="--", label=f"mean={vy_vals.mean():.4f}")
ax.legend()

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "peak_velocity_distributions.png")
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
plt.close()
