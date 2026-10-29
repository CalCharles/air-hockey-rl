#!/bin/bash
# Usage: scripts/ceiling/select_and_final.sh <run_dir> <first_ckpt> <last_ckpt> <label>
# Scores <run_dir> and its checkpoints first..last (100k spacing) on selection_long, picks the best
# (reached-cap fraction, then mean steps), then runs the final protocol on it.
set -e
cd "$(dirname "$0")/../.."
run=$1; first=$2; last=$3; label=$4
cands="$run"; for k in $(seq $first 100000 $last); do [ -d $run/checkpoint_$k ] && cands="$cands $run/checkpoint_$k"; done
.venv/bin/python -m scripts.ceiling.eval_target --protocol selection_long --label sellong__$label --chunks 4 --workers 44 --gif-steps 0 $cands 2>&1 | grep -v -i warn | grep "mean steps"
best=$(.venv/bin/python - "$label" <<'PY'
import json, glob, sys
rows = []
for f in glob.glob(f"runs/ceiling/evals/selection_long/sellong__{sys.argv[1]}/*/results.json"):
    d = json.load(open(f)); s = d["scenarios"]["target"]["summary"]
    rows.append((s["horizon_survival_fraction"], s["restricted_mean_steps"], d["run_dir"]))
print(max(rows)[2])
PY
)
echo "selected: $best"
.venv/bin/python -m scripts.ceiling.eval_target --protocol final --label final__$label --chunks 20 --workers 40 $best 2>&1 | grep -v -i warn | grep "mean steps"
