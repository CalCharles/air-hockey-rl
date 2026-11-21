#!/usr/bin/env bash
# Exit (one-line reason) on the first event worth acting on for a train_jg0 run directory (default runs/crl/jg0_best_s0):
#   CRASH  a Python error in stdout.log, or the process is gone without the "done" line
#   NAN    skipped_updates > 0 (jg0 skips non-finite updates) or a nan critic loss
#   STALL  train.log not written for STALL_MIN minutes (default 30; a round takes ~3 min)
#   DONE   the run finished
cd "$(dirname "$0")/../.." || exit 1
RUN=${1:-runs/crl/jg0_best_s0}; STALL_MIN=${STALL_MIN:-30}
while true; do
  log="$RUN/train.log"
  if grep -q '"done": true' "$log" 2>/dev/null; then echo "DONE $(tail -1 "$log")"; exit 0; fi
  if grep -qE 'Traceback|Error' "$RUN/stdout.log"; then echo "CRASH: $(grep -m1 -E 'Error' "$RUN/stdout.log")"; exit 0; fi
  if ! pgrep -f -- "train_jg0.*--run-dir $RUN" > /dev/null; then echo "CRASH: process gone without done"; exit 0; fi
  if [ $(( $(date +%s) - $(stat -c %Y "$log") )) -gt $(( STALL_MIN * 60 )) ]; then echo "STALL: no log for $STALL_MIN min"; exit 0; fi
  if grep -qE '"skipped_updates": [1-9]|"critic_loss": NaN' "$log"; then echo "NAN: $(grep -m1 -E '"skipped_updates": [1-9]|NaN' "$log" | cut -c1-200)"; exit 0; fi
  sleep 300
done
