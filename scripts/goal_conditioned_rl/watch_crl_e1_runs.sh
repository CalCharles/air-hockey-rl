#!/usr/bin/env bash
# Exit (with a one-line reason) on the first event worth acting on for the runs/crl/crl_*_still_e1_s* runs
# (TD3-inputs ablation, E1):
#   CRASH     a log shows a Python error, or a process is gone without the final "done" line
#   NAN       a logged value is nan/inf
#   STALL     a log has not been written for STALL_MIN minutes (default 30)
#   PAPER_DONE the paper run finished (reported once; pass --after-paper to skip it)
#   ALL_DONE  every run finished
cd "$(dirname "$0")/../.." || exit 1
STALL_MIN=${STALL_MIN:-30}
skip_paper=${1:-}
while true; do
  done_all=1; done_paper=1
  for d in runs/crl/crl_*_still_e1_s*; do
    log="$d/train.log"; name=$(basename "$d")
    if grep -q '"done": true' "$log"; then continue; fi
    done_all=0; [[ $name == crl_paper_* ]] && done_paper=0
    if grep -qE 'Traceback|Error|Killed' "$log"; then echo "CRASH $name: $(grep -m1 -E 'Error' "$log")"; exit 0; fi
    if ! pgrep -f -- "--run-dir $d" > /dev/null; then echo "CRASH $name: process gone without done"; exit 0; fi
    if [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt $(( STALL_MIN * 60 )) ]; then
      echo "STALL $name: no log for $STALL_MIN min"; exit 0; fi
    if grep -qiE ': nan|: inf|: -inf' "$log"; then echo "NAN $name"; exit 0; fi
  done
  [ $done_all = 1 ] && { echo "ALL_DONE"; exit 0; }
  [ $done_paper = 1 ] && [ "$skip_paper" != "--after-paper" ] && { echo "PAPER_DONE"; exit 0; }
  sleep 120
done
