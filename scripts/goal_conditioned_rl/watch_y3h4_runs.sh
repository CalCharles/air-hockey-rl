#!/usr/bin/env bash
# Exit (with a one-line reason) on the first event worth acting on for the runs/crl/y3h4_s* runs:
#   CRASH     a log shows a Python error, or a process is gone without the final "done" line
#   NAN       a training line (updates > 0) has a non-finite accuracy
#   STALL     a log has not been written for STALL_MIN minutes (default 30)
#   ALL_DONE  every run finished
cd "$(dirname "$0")/../.." || exit 1
STALL_MIN=${STALL_MIN:-30}
while true; do
  done_all=1
  for d in runs/crl/y3h4_s*; do
    log="$d/train.log"; name=$(basename "$d")
    if grep -q '"done": true' "$log"; then continue; fi
    done_all=0
    if grep -qE 'Traceback|Error|Killed' "$log"; then echo "CRASH $name: $(grep -m1 -E 'Error' "$log")"; exit 0; fi
    if ! pgrep -f -- "--run-dir $d" > /dev/null; then echo "CRASH $name: process gone without done"; exit 0; fi
    if [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt $(( STALL_MIN * 60 )) ]; then
      echo "STALL $name: no log for $STALL_MIN min"; exit 0; fi
    if grep -E '"updates": [1-9]' "$log" | grep -qiE '"acc": (nan|inf)'; then echo "NAN $name"; exit 0; fi
  done
  [ $done_all = 1 ] && { echo "ALL_DONE"; exit 0; }
  sleep 120
done
