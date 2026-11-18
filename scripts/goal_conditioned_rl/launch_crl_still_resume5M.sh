#!/usr/bin/env bash
# Continue the 10 stationary-goal CRL runs (paper / c0, s0-s4) from their converged final.pt to 5M env steps,
# convergence stop off. Weights + optimizer state + step are restored (train_crl.py --resume); the replay was not
# saved and is refilled by the loaded policy. Each run logs into its own run dir at its own step, so the
# TensorBoard curves continue the existing ones. final.pt is first copied to final_converged.pt; the resumed run
# saves ckpt_<step>.pt every 250k and final_5M.pt at the end. Log: <run dir>/train_resume.log.
# Waits until GPUs 0 and 1 have BOTH been <= MAX_UTIL % on every 10 s sample for QUIET_MIN minutes.
set -uo pipefail
cd "$(dirname "$0")/../.."
MAX_UTIL=${MAX_UTIL:-5}
QUIET_MIN=${QUIET_MIN:-10}
TOTAL=${TOTAL:-5000000}
need=$(( QUIET_MIN * 6 ))
declare -A quiet=([0]=0 [1]=0)
echo "$(date '+%F %T') waiting for GPUs 0 and 1 at <= ${MAX_UTIL}% for ${QUIET_MIN} min"
while true; do
  ok=1
  for g in 0 1; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ "$u" -le "$MAX_UTIL" ]; then quiet[$g]=$(( ${quiet[$g]} + 1 )); else
      [ "${quiet[$g]}" -ge 6 ] && echo "$(date '+%F %T') GPU $g busy (${u}%) after $(( ${quiet[$g]} / 6 )) quiet min"
      quiet[$g]=0; fi
    [ "${quiet[$g]}" -ge $need ] || ok=0
  done
  [ $ok -eq 1 ] && break
  sleep 10
done
echo "$(date '+%F %T') GPUs 0 and 1 quiet for ${QUIET_MIN} min; launching"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
launch() {  # variant seed gpu
  local dir="runs/crl/crl_$1_still_s$2"
  [ -f "$dir/final_converged.pt" ] || cp "$dir/final.pt" "$dir/final_converged.pt"
  nohup .venv/bin/python -m scripts.goal_conditioned_rl.train_crl --config "configs/crl/juggle_crl_$1_still.yaml" \
    --seed "$2" --device "cuda:$3" --run-dir "$dir" --resume "$dir/final_converged.pt" --no-convergence \
    --total-steps "$TOTAL" --final-name final_5M.pt > "$dir/train_resume.log" 2>&1 &
  echo "$1 seed $2 -> cuda:$3 pid $!"
}
# c0 is the slow one: split its remaining steps about evenly (GPU 0: 4.75M, GPU 1: 5.5M).
launch c0 0 0; launch c0 4 0; launch paper 0 0; launch paper 1 0; launch paper 2 0
launch c0 1 1; launch c0 2 1; launch c0 3 1; launch paper 3 1; launch paper 4 1
echo LAUNCHED
