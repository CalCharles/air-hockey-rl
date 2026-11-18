#!/usr/bin/env bash
# 5 seeds x {paper, c0} stationary-goal CRL (configs/crl/juggle_crl_{paper,c0}_still.yaml) on the first two
# free GPUs. A GPU counts as free once its utilization has been <= MAX_UTIL % on every sample (every 10 s) for
# QUIET_MIN minutes in a row; any busy sample resets that GPU's clock. We never pile onto someone else's job.
# Logs: runs/crl/crl_<v>_still_s<seed>/train.log ; this script's own log says which GPUs were picked.
set -uo pipefail
cd "$(dirname "$0")/../.."
MAX_UTIL=${MAX_UTIL:-5}
QUIET_MIN=${QUIET_MIN:-10}
GPUS=${GPUS:-"0 1 2 3"}
need=$(( QUIET_MIN * 6 ))
declare -A quiet
for g in $GPUS; do quiet[$g]=0; done
echo "$(date '+%F %T') waiting for 2 of GPUs {$GPUS} at <= ${MAX_UTIL}% for ${QUIET_MIN} min"
while true; do
  free=()
  for g in $GPUS; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$g" | tr -d ' ')
    if [ "$u" -le "$MAX_UTIL" ]; then
      quiet[$g]=$(( ${quiet[$g]} + 1 ))
    else
      [ "${quiet[$g]}" -ge 6 ] && echo "$(date '+%F %T') GPU $g busy again (${u}%) after $(( ${quiet[$g]} / 6 )) quiet min"
      quiet[$g]=0
    fi
    [ "${quiet[$g]}" -ge $need ] && free+=("$g")
  done
  [ ${#free[@]} -ge 2 ] && break
  sleep 10
done
A=${free[0]}; B=${free[1]}
echo "$(date '+%F %T') GPUs $A and $B quiet for ${QUIET_MIN} min; launching"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
launch() {  # variant seed gpu
  local dir="runs/crl/crl_$1_still_s$2"
  mkdir -p "$dir"
  nohup .venv/bin/python -m scripts.goal_conditioned_rl.train_crl --config "configs/crl/juggle_crl_$1_still.yaml" \
    --seed "$2" --device "cuda:$3" --run-dir "$dir" > "$dir/train.log" 2>&1 &
  echo "$1 seed $2 -> cuda:$3 pid $!"
}
launch paper 0 "$A"; launch paper 1 "$A"; launch c0 0 "$A"; launch c0 1 "$A"; launch c0 2 "$A"
launch paper 2 "$B"; launch paper 3 "$B"; launch paper 4 "$B"; launch c0 3 "$B"; launch c0 4 "$B"
echo LAUNCHED
