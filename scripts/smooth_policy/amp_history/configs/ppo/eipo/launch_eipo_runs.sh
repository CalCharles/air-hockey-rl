#!/bin/bash
# Launch EIPO training runs across 3 disc stationarity modes, each with
# multiple parameter variants. Runs are spread across 4 GPUs round-robin.
#
# Directory structure:
#   runs/eipo_runs/
#     target/
#       tau0.005_alphalr0.01/    — default recommended
#       tau0.001_alphalr0.01/    — slower EMA for more stability
#       tau0.005_alphalr0.005/   — slower alpha adaptation
#     live/
#       alphalr0.01/             — baseline
#       alphalr0.005/            — slower alpha
#       disc_lr1e-5/             — lower disc lr to reduce non-stationarity
#     snapshot/
#       alphalr0.01/             — default
#       alphalr0.005/            — slower alpha
#       alphainit0.5/            — start with less task weight
#
# Usage:
#   bash scripts/smooth_policy/amp_history/configs/ppo/eipo/launch_eipo_runs.sh
#
# To attach:  tmux attach -t eipo
# To check:   tmux ls

set -e
cd "$(git rev-parse --show-toplevel)"

SESSION="eipo"
SCRIPT="scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py"
BASE_CFG="scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml"
RUNS_DIR="runs/eipo_runs"

# Ensure output dirs exist for tee
mkdir -p "$RUNS_DIR"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "init"

GPU=0
WIN=0

launch() {
    local name="$1"
    local dir="$2"
    shift 2
    local extra_args="$@"

    mkdir -p "$dir"

    if [ $WIN -eq 0 ]; then
        tmux rename-window -t "$SESSION" "$name"
    else
        tmux new-window -t "$SESSION" -n "$name"
    fi

    tmux send-keys -t "$SESSION:$name" \
        "source /home/air-hockey/air-hockey-rl/.venv/bin/activate && \
        cd /home/air-hockey/air-hockey-rl && \
        python $SCRIPT --args-file $BASE_CFG \
        --device cuda:$GPU \
        --log-parent-dir $dir \
        --run-name $name \
        $extra_args \
        2>&1 | tee $dir/train.log" Enter

    GPU=$(( (GPU + 1) % 4 ))
    WIN=$((WIN + 1))
}

# =============================================================================
# TARGET MODE — recommended, 3 variants
# =============================================================================

# Target: default recommended settings
launch "target_default" "$RUNS_DIR/target/tau0.005_alphalr0.01" \
    --disc-stationarity-mode target \
    --disc-ema-tau 0.005 \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 1.0

# Target: slower EMA (more stable reward signal)
launch "target_slow_ema" "$RUNS_DIR/target/tau0.001_alphalr0.01" \
    --disc-stationarity-mode target \
    --disc-ema-tau 0.001 \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 1.0

# Target: slower alpha adaptation
launch "target_slow_alpha" "$RUNS_DIR/target/tau0.005_alphalr0.005" \
    --disc-stationarity-mode target \
    --disc-ema-tau 0.005 \
    --eipo-alpha-lr 0.005 \
    --eipo-alpha-init 1.0

# =============================================================================
# LIVE MODE — baseline, 3 variants
# =============================================================================

# Live: default
launch "live_default" "$RUNS_DIR/live/alphalr0.01" \
    --disc-stationarity-mode live \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 1.0

# Live: slower alpha
launch "live_slow_alpha" "$RUNS_DIR/live/alphalr0.005" \
    --disc-stationarity-mode live \
    --eipo-alpha-lr 0.005 \
    --eipo-alpha-init 1.0

# Live: lower disc learning rate to reduce non-stationarity naturally
launch "live_low_disc_lr" "$RUNS_DIR/live/disc_lr1e-5" \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-5 \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 1.0

# =============================================================================
# SNAPSHOT MODE — 3 variants
# =============================================================================

# Snapshot: default
launch "snap_default" "$RUNS_DIR/snapshot/alphalr0.01" \
    --disc-stationarity-mode snapshot \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 1.0

# Snapshot: slower alpha
launch "snap_slow_alpha" "$RUNS_DIR/snapshot/alphalr0.005" \
    --disc-stationarity-mode snapshot \
    --eipo-alpha-lr 0.005 \
    --eipo-alpha-init 1.0

# Snapshot: lower initial alpha (less task weight initially, let disc guide early)
launch "snap_low_init" "$RUNS_DIR/snapshot/alphainit0.5" \
    --disc-stationarity-mode snapshot \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 0.5

tmux select-window -t "$SESSION:target_default"

echo ""
echo "=== EIPO training: 9 runs launched in tmux session '$SESSION' ==="
echo ""
echo "  TARGET MODE (GPU round-robin):"
echo "    target_default    — tau=0.005, alpha_lr=0.01"
echo "    target_slow_ema   — tau=0.001, alpha_lr=0.01"
echo "    target_slow_alpha — tau=0.005, alpha_lr=0.005"
echo ""
echo "  LIVE MODE:"
echo "    live_default      — alpha_lr=0.01"
echo "    live_slow_alpha   — alpha_lr=0.005"
echo "    live_low_disc_lr  — disc_lr=1e-5, alpha_lr=0.01"
echo ""
echo "  SNAPSHOT MODE:"
echo "    snap_default      — alpha_lr=0.01"
echo "    snap_slow_alpha   — alpha_lr=0.005"
echo "    snap_low_init     — alpha_init=0.5, alpha_lr=0.01"
echo ""
echo "  Output: $RUNS_DIR/<mode>/<variant>/"
echo ""
echo "  tmux attach -t $SESSION"
echo "  tmux list-windows -t $SESSION"
echo ""
