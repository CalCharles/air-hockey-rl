"""Average actor weights over a run's checkpoints (SWA) into a new policy dir.

    python swa.py OUT_ROOT RUN_DIR MIN_STEP [MAX_STEP] [--target]
--target averages actor_target.pth instead of model.pth.
"""
import re
import shutil
import sys
from pathlib import Path

import torch

out_root, run, lo = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
hi = int(sys.argv[4]) if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else 10**12
fname = "actor_target.pth" if "--target" in sys.argv else "model.pth"
dirs = []
for c in run.glob("checkpoint_*"):
    m = re.fullmatch(r"checkpoint_(\d+)", c.name)
    if m and lo <= int(m[1]) <= hi and (c / fname).exists():
        dirs.append((int(m[1]), c))
dirs = [c for _, c in sorted(dirs)]
if hi >= 10**12 and (run / fname).exists():
    dirs.append(run)
states = [torch.load(d / fname, map_location="cpu", weights_only=False) for d in dirs]
states = [s["actor"] if "actor" in s and isinstance(s["actor"], dict) else s for s in states]
avg = {k: (sum(s[k].float() for s in states) / len(states)).to(states[0][k].dtype)
       if states[0][k].is_floating_point() else states[0][k] for k in states[0]}
tag = f"{run.parent.name}__{run.name}__swa{lo // 1000}k" + ("" if hi >= 10**12 else f"-{hi // 1000}k") + ("_tgt" if fname != "model.pth" else "")
out = out_root / tag
out.mkdir(parents=True, exist_ok=True)
for f in ("args.yaml", "config.yaml", "rma_meta.json"):
    shutil.copy(run / f, out / f)
torch.save(avg, out / "model.pth")
(out / "sources.txt").write_text("\n".join(str(d / fname) for d in dirs) + "\n")
print(out, len(dirs))
