#!/usr/bin/env bash
# Exit (with a one-line reason) on the first event worth acting on for the runs/crl/crl_* overnight runs:
#   CRASH     a run's log shows a Python error, or its process is gone without the final "done" line
#   NAN       a logged training metric is nan/inf
#   STALL     a run's log has not been written for 20 min
#   PAPER_DONE all paper seeds finished (only reported once: pass --after-paper to skip it)
#   ALL_DONE  every run finished
cd "$(dirname "$0")/../.." || exit 1
skip_paper=${1:-}
while true; do
  done_all=1; done_paper=1
  for d in runs/crl/crl_*_s*; do
    log="$d/train.log"; name=$(basename "$d")
    if grep -q '"done": true' "$log"; then continue; fi
    done_all=0; [[ $name == crl_paper_* ]] && done_paper=0
    if grep -qE 'Traceback|Error|Killed' "$log"; then echo "CRASH $name: $(grep -m1 -E 'Error' "$log")"; exit 0; fi
    if ! pgrep -f -- "--run-dir $d" > /dev/null; then echo "CRASH $name: process gone without done"; exit 0; fi
    if [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt 1200 ]; then echo "STALL $name: no log for 20 min"; exit 0; fi
  done
  if grep -l -iE '"(nan|inf)"|: nan|: inf' runs/crl/crl_*_s*/train.log > /dev/null 2>&1; then
    echo "NAN in $(grep -l -iE ': nan|: inf' runs/crl/crl_*_s*/train.log | tr '\n' ' ')"; exit 0
  fi
  [ $done_all = 1 ] && { echo "ALL_DONE"; exit 0; }
  [ $done_paper = 1 ] && [ "$skip_paper" != "--after-paper" ] && { echo "PAPER_DONE"; exit 0; }
  sleep 60
done
