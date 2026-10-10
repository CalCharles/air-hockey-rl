"""Per-trial table + mean ± standard error for one real eval run.

    python -m scripts.td3.analysis.eval_per_trial <eval data_root_dir or data_* dir>

Reads every ``eval_per_episode.jsonl`` (kept episodes) under the given dir,
in time order, prints one row per trial and mean ± SE (std ddof=1 / sqrt(n)).
"""
import glob
import json
import os
import sys

import numpy as np

FIELDS = ["episode_return", "episode_juggles", "episode_contacts", "episode_length"]

root = sys.argv[1]
paths = sorted(glob.glob(os.path.join(root, "**", "eval_per_episode.jsonl"), recursive=True))
rows = [json.loads(line) for p in paths for line in open(p) if line.strip()]
rows.sort(key=lambda r: r.get("wall_time_s", 0.0))
if not rows:
    sys.exit(f"no eval_per_episode.jsonl rows under {root}")

print(f"{'trial':>5} " + " ".join(f"{f.replace('episode_', ''):>9}" for f in FIELDS) + "  end_reason")
for i, r in enumerate(rows, 1):
    print(f"{i:>5} " + " ".join(f"{float(r.get(f) or 0):>9.2f}" for f in FIELDS) + f"  {r.get('episode_end_reason', '')}")
print(f"\nn = {len(rows)} trials")
for f in FIELDS:
    v = np.array([float(r.get(f) or 0) for r in rows])
    se = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
    print(f"{f:>18}: {v.mean():.2f} ± {se:.2f} (SE)")
