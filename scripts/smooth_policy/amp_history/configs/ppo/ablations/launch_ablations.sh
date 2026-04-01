#!/bin/bash
# EIPO vs non-EIPO ablation study — 35 runs across 7 blocks.
# See .claude/eipo_ablation_plan.md for experimental design and key comparisons.
#
# Runs are distributed across 4 GPUs as sequential queues — each GPU window
# runs its assigned jobs one after another, so only 4 jobs run at any time.
#
# GPU 0 (9 runs): amp_base, amp_disc_lr_5e5, eipo_live_disc_lr_5e5,
#                 amp_w_35_65, amp_w_9_1, eipo_live_alr_2,
#                 eipo_target_tau_001, amp_disc_updates_5, amp_high_reg
# GPU 1 (9 runs): eipo_live_base, amp_disc_lr_5e4, eipo_live_disc_lr_5e4,
#                 amp_w_65_35, eipo_live_alr_001, eipo_live_ainit_01,
#                 eipo_target_tau_01, eipo_live_disc_updates_1, eipo_live_low_reg
# GPU 2 (9 runs): eipo_target_base, amp_disc_lr_1e3, eipo_live_disc_lr_1e3,
#                 amp_w_667_333, eipo_live_alr_005, eipo_live_ainit_5,
#                 eipo_target_tau_05, eipo_live_disc_updates_5, eipo_live_high_reg
# GPU 3 (8 runs): amp_disc_lr_1e5, eipo_live_disc_lr_1e5, amp_w_2_8, amp_w_8_2,
#                 eipo_live_alr_05, eipo_target_tau_0001, amp_disc_updates_1, amp_low_reg
#
# Usage:
#   bash scripts/smooth_policy/amp_history/configs/ppo/ablations/launch_ablations.sh
#
# Attach:  tmux attach -t ablations
# Check:   tmux list-windows -t ablations

set -e
cd "$(git rev-parse --show-toplevel)"

SESSION="ablations"
SCRIPT="scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py"
AMP_CFG="scripts/smooth_policy/amp_history/configs/ppo/ablations/amp_ablation_base.yaml"
EIPO_CFG="scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml"
RUNS_AMP="runs/ablations/amp"
RUNS_EIPO="runs/ablations/eipo"
VENV="source /home/air-hockey/air-hockey-rl/.venv/bin/activate"
REPO="cd /home/air-hockey/air-hockey-rl"

mkdir -p "$RUNS_AMP" "$RUNS_EIPO"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "gpu0"

# Helper: build a single python run command (no newlines, suitable for chaining)
run() {
    local name="$1"
    local dir="$2"
    local cfg="$3"
    local gpu="$4"
    shift 4
    mkdir -p "$dir"
    echo "echo '>>> Starting $name' && python $SCRIPT --args-file $cfg --device cuda:$gpu --log-parent-dir $dir --run-name $name $@ 2>&1 | tee $dir/train.log"
}

# =============================================================================
# Build per-GPU command chains
# =============================================================================

# --- GPU 0 ---
GPU0_CMDS=(
    "$(run amp_base             $RUNS_AMP/amp_base             $AMP_CFG  0)"
    "$(run amp_disc_lr_5e5      $RUNS_AMP/amp_disc_lr_5e5      $AMP_CFG  0  --disc-learning-rate 5e-5)"
    "$(run eipo_live_disc_lr_5e5 $RUNS_EIPO/eipo_live_disc_lr_5e5 $EIPO_CFG 0  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 5e-5 --eipo-alpha-lr 0.01 --checkpoint-freq 500)"
    "$(run amp_w_35_65          $RUNS_AMP/amp_w_35_65          $AMP_CFG  0  --task-reward-weight 0.35 --disc-reward-weight 0.65)"
    "$(run amp_w_9_1            $RUNS_AMP/amp_w_9_1            $AMP_CFG  0  --task-reward-weight 0.9 --disc-reward-weight 0.1)"
    "$(run eipo_live_alr_2      $RUNS_EIPO/eipo_live_alr_2     $EIPO_CFG 0  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.2 --eipo-alpha-init 1.0 --checkpoint-freq 500)"
    "$(run eipo_target_tau_001  $RUNS_EIPO/eipo_target_tau_001 $EIPO_CFG 0  --use-eipo true --disc-stationarity-mode target --disc-ema-tau 0.001 --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --checkpoint-freq 500)"
    "$(run amp_disc_updates_5   $RUNS_AMP/amp_disc_updates_5   $AMP_CFG  0  --num-discriminator-updates 5)"
    "$(run amp_high_reg         $RUNS_AMP/amp_high_reg         $AMP_CFG  0  --disc-logit-reg 0.1 --disc-grad-penalty 10.0)"
)

# --- GPU 1 ---
GPU1_CMDS=(
    "$(run eipo_live_base       $RUNS_EIPO/eipo_live_base      $EIPO_CFG 1  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --eipo-alpha-init 1.0 --checkpoint-freq 500)"
    "$(run amp_disc_lr_5e4      $RUNS_AMP/amp_disc_lr_5e4      $AMP_CFG  1  --disc-learning-rate 5e-4)"
    "$(run eipo_live_disc_lr_5e4 $RUNS_EIPO/eipo_live_disc_lr_5e4 $EIPO_CFG 1  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 5e-4 --eipo-alpha-lr 0.01 --checkpoint-freq 500)"
    "$(run amp_w_65_35          $RUNS_AMP/amp_w_65_35          $AMP_CFG  1  --task-reward-weight 0.65 --disc-reward-weight 0.35)"
    "$(run eipo_live_alr_001    $RUNS_EIPO/eipo_live_alr_001   $EIPO_CFG 1  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.001 --eipo-alpha-init 1.0 --checkpoint-freq 500)"
    "$(run eipo_live_ainit_01   $RUNS_EIPO/eipo_live_ainit_01  $EIPO_CFG 1  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --eipo-alpha-init 0.1 --checkpoint-freq 500)"
    "$(run eipo_target_tau_01   $RUNS_EIPO/eipo_target_tau_01  $EIPO_CFG 1  --use-eipo true --disc-stationarity-mode target --disc-ema-tau 0.01 --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --checkpoint-freq 500)"
    "$(run eipo_live_disc_updates_1 $RUNS_EIPO/eipo_live_disc_updates_1 $EIPO_CFG 1  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --num-discriminator-updates 1 --checkpoint-freq 500)"
    "$(run eipo_live_low_reg    $RUNS_EIPO/eipo_live_low_reg   $EIPO_CFG 1  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --disc-logit-reg 0.001 --disc-grad-penalty 1.0 --checkpoint-freq 500)"
)

# --- GPU 2 ---
GPU2_CMDS=(
    "$(run eipo_target_base     $RUNS_EIPO/eipo_target_base    $EIPO_CFG 2  --use-eipo true --disc-stationarity-mode target --disc-ema-tau 0.005 --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --eipo-alpha-init 1.0 --checkpoint-freq 500)"
    "$(run amp_disc_lr_1e3      $RUNS_AMP/amp_disc_lr_1e3      $AMP_CFG  2  --disc-learning-rate 1e-3)"
    "$(run eipo_live_disc_lr_1e3 $RUNS_EIPO/eipo_live_disc_lr_1e3 $EIPO_CFG 2  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-3 --eipo-alpha-lr 0.01 --checkpoint-freq 500)"
    "$(run amp_w_667_333        $RUNS_AMP/amp_w_667_333        $AMP_CFG  2  --task-reward-weight 0.667 --disc-reward-weight 0.333)"
    "$(run eipo_live_alr_005    $RUNS_EIPO/eipo_live_alr_005   $EIPO_CFG 2  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.005 --eipo-alpha-init 1.0 --checkpoint-freq 500)"
    "$(run eipo_live_ainit_5    $RUNS_EIPO/eipo_live_ainit_5   $EIPO_CFG 2  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --eipo-alpha-init 5.0 --checkpoint-freq 500)"
    "$(run eipo_target_tau_05   $RUNS_EIPO/eipo_target_tau_05  $EIPO_CFG 2  --use-eipo true --disc-stationarity-mode target --disc-ema-tau 0.05 --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --checkpoint-freq 500)"
    "$(run eipo_live_disc_updates_5 $RUNS_EIPO/eipo_live_disc_updates_5 $EIPO_CFG 2  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --num-discriminator-updates 5 --checkpoint-freq 500)"
    "$(run eipo_live_high_reg   $RUNS_EIPO/eipo_live_high_reg  $EIPO_CFG 2  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --disc-logit-reg 0.1 --disc-grad-penalty 10.0 --checkpoint-freq 500)"
)

# --- GPU 3 ---
GPU3_CMDS=(
    "$(run amp_disc_lr_1e5      $RUNS_AMP/amp_disc_lr_1e5      $AMP_CFG  3  --disc-learning-rate 1e-5)"
    "$(run eipo_live_disc_lr_1e5 $RUNS_EIPO/eipo_live_disc_lr_1e5 $EIPO_CFG 3  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-5 --eipo-alpha-lr 0.01 --checkpoint-freq 500)"
    "$(run amp_w_2_8            $RUNS_AMP/amp_w_2_8            $AMP_CFG  3  --task-reward-weight 0.2 --disc-reward-weight 0.8)"
    "$(run amp_w_8_2            $RUNS_AMP/amp_w_8_2            $AMP_CFG  3  --task-reward-weight 0.8 --disc-reward-weight 0.2)"
    "$(run eipo_live_alr_05     $RUNS_EIPO/eipo_live_alr_05    $EIPO_CFG 3  --use-eipo true --disc-stationarity-mode live --disc-learning-rate 1e-4 --eipo-alpha-lr 0.05 --eipo-alpha-init 1.0 --checkpoint-freq 500)"
    "$(run eipo_target_tau_0001 $RUNS_EIPO/eipo_target_tau_0001 $EIPO_CFG 3  --use-eipo true --disc-stationarity-mode target --disc-ema-tau 0.0001 --disc-learning-rate 1e-4 --eipo-alpha-lr 0.01 --checkpoint-freq 500)"
    "$(run amp_disc_updates_1   $RUNS_AMP/amp_disc_updates_1   $AMP_CFG  3  --num-discriminator-updates 1)"
    "$(run amp_low_reg          $RUNS_AMP/amp_low_reg          $AMP_CFG  3  --disc-logit-reg 0.001 --disc-grad-penalty 1.0)"
)

# =============================================================================
# Launch one tmux window per GPU, chaining all its runs with &&
# =============================================================================

launch_gpu_window() {
    local win_name="$1"
    shift
    local cmds=("$@")

    # Join all commands with ' && '
    local chain
    chain=$(printf ' && %s' "${cmds[@]}")
    chain="${chain:4}"  # strip leading ' && '

    tmux send-keys -t "$SESSION:$win_name" \
        "$VENV && $REPO && $chain && echo '=== GPU $win_name: all runs complete ===' " Enter
}

# Rename the initial window and create the rest
tmux rename-window -t "$SESSION" "gpu0"
tmux new-window -t "$SESSION" -n "gpu1"
tmux new-window -t "$SESSION" -n "gpu2"
tmux new-window -t "$SESSION" -n "gpu3"

launch_gpu_window "gpu0" "${GPU0_CMDS[@]}"
launch_gpu_window "gpu1" "${GPU1_CMDS[@]}"
launch_gpu_window "gpu2" "${GPU2_CMDS[@]}"
launch_gpu_window "gpu3" "${GPU3_CMDS[@]}"

tmux select-window -t "$SESSION:gpu0"

echo ""
echo "=== Ablation study: 35 runs queued across 4 GPU windows in tmux session '$SESSION' ==="
echo ""
echo "  gpu0 (9 runs): amp_base, amp_disc_lr_5e5, eipo_live_disc_lr_5e5,"
echo "                 amp_w_35_65, amp_w_9_1, eipo_live_alr_2,"
echo "                 eipo_target_tau_001, amp_disc_updates_5, amp_high_reg"
echo ""
echo "  gpu1 (9 runs): eipo_live_base, amp_disc_lr_5e4, eipo_live_disc_lr_5e4,"
echo "                 amp_w_65_35, eipo_live_alr_001, eipo_live_ainit_01,"
echo "                 eipo_target_tau_01, eipo_live_disc_updates_1, eipo_live_low_reg"
echo ""
echo "  gpu2 (9 runs): eipo_target_base, amp_disc_lr_1e3, eipo_live_disc_lr_1e3,"
echo "                 amp_w_667_333, eipo_live_alr_005, eipo_live_ainit_5,"
echo "                 eipo_target_tau_05, eipo_live_disc_updates_5, eipo_live_high_reg"
echo ""
echo "  gpu3 (8 runs): amp_disc_lr_1e5, eipo_live_disc_lr_1e5, amp_w_2_8, amp_w_8_2,"
echo "                 eipo_live_alr_05, eipo_target_tau_0001, amp_disc_updates_1, amp_low_reg"
echo ""
echo "  Output: runs/ablations/{amp,eipo}/<run_name>/"
echo "  tmux attach -t $SESSION"
echo ""
