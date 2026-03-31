#!/bin/bash
# EIPO vs non-EIPO ablation study — 35 runs across 7 blocks.
# See .claude/eipo_ablation_plan.md for experimental design and key comparisons.
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

mkdir -p "$RUNS_AMP" "$RUNS_EIPO"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "init"

GPU=0
WIN=0

launch() {
    local name="$1"
    local dir="$2"
    local cfg="$3"
    shift 3
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
        python $SCRIPT --args-file $cfg \
        --device cuda:$GPU \
        --log-parent-dir $dir \
        --run-name $name \
        $extra_args \
        2>&1 | tee $dir/train.log" Enter

    GPU=$(( (GPU + 1) % 4 ))
    WIN=$((WIN + 1))
}

# =============================================================================
# BLOCK A — Core comparison (3 runs)
# Q: Does EIPO help? Does disc smoothing add on top?
# Key: amp_base vs eipo_live_base (balancing effect)
#      eipo_live_base vs eipo_target_base (smoothing effect)
# =============================================================================

launch "amp_base" "$RUNS_AMP/amp_base" "$AMP_CFG"

launch "eipo_live_base" "$RUNS_EIPO/eipo_live_base" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 1.0 \
    --checkpoint-freq 500

launch "eipo_target_base" "$RUNS_EIPO/eipo_target_base" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode target \
    --disc-ema-tau 0.005 \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 1.0 \
    --checkpoint-freq 500

# =============================================================================
# BLOCK B — Discriminator LR sweep (8 runs)
# Q: Is EIPO more robust to disc_lr than fixed-weight AMP?
# =============================================================================

launch "amp_disc_lr_1e5" "$RUNS_AMP/amp_disc_lr_1e5" "$AMP_CFG" \
    --disc-learning-rate 1e-5

launch "amp_disc_lr_5e5" "$RUNS_AMP/amp_disc_lr_5e5" "$AMP_CFG" \
    --disc-learning-rate 5e-5

launch "amp_disc_lr_5e4" "$RUNS_AMP/amp_disc_lr_5e4" "$AMP_CFG" \
    --disc-learning-rate 5e-4

launch "amp_disc_lr_1e3" "$RUNS_AMP/amp_disc_lr_1e3" "$AMP_CFG" \
    --disc-learning-rate 1e-3

launch "eipo_live_disc_lr_1e5" "$RUNS_EIPO/eipo_live_disc_lr_1e5" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-5 \
    --eipo-alpha-lr 0.01 \
    --checkpoint-freq 500

launch "eipo_live_disc_lr_5e5" "$RUNS_EIPO/eipo_live_disc_lr_5e5" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 5e-5 \
    --eipo-alpha-lr 0.01 \
    --checkpoint-freq 500

launch "eipo_live_disc_lr_5e4" "$RUNS_EIPO/eipo_live_disc_lr_5e4" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 5e-4 \
    --eipo-alpha-lr 0.01 \
    --checkpoint-freq 500

launch "eipo_live_disc_lr_1e3" "$RUNS_EIPO/eipo_live_disc_lr_1e3" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-3 \
    --eipo-alpha-lr 0.01 \
    --checkpoint-freq 500

# =============================================================================
# BLOCK C — Fixed weight sweep + matched-weight comparison (6 runs)
# Q: Can manual weight tuning replicate EIPO? amp_w_667_333 is the
#    non-EIPO equivalent of EIPO alpha_init=1.0 (task:disc = 2:1 effective ratio).
# =============================================================================

launch "amp_w_2_8" "$RUNS_AMP/amp_w_2_8" "$AMP_CFG" \
    --task-reward-weight 0.2 \
    --disc-reward-weight 0.8

launch "amp_w_35_65" "$RUNS_AMP/amp_w_35_65" "$AMP_CFG" \
    --task-reward-weight 0.35 \
    --disc-reward-weight 0.65

launch "amp_w_65_35" "$RUNS_AMP/amp_w_65_35" "$AMP_CFG" \
    --task-reward-weight 0.65 \
    --disc-reward-weight 0.35

launch "amp_w_667_333" "$RUNS_AMP/amp_w_667_333" "$AMP_CFG" \
    --task-reward-weight 0.667 \
    --disc-reward-weight 0.333

launch "amp_w_8_2" "$RUNS_AMP/amp_w_8_2" "$AMP_CFG" \
    --task-reward-weight 0.8 \
    --disc-reward-weight 0.2

launch "amp_w_9_1" "$RUNS_AMP/amp_w_9_1" "$AMP_CFG" \
    --task-reward-weight 0.9 \
    --disc-reward-weight 0.1

# =============================================================================
# BLOCK D — EIPO alpha sensitivity (6 runs)
# Q: How sensitive is EIPO to its own hyperparameters?
#    If badly-tuned EIPO loses to best non-EIPO -> fragile benefit.
# =============================================================================

launch "eipo_live_alr_001" "$RUNS_EIPO/eipo_live_alr_001" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.001 \
    --eipo-alpha-init 1.0 \
    --checkpoint-freq 500

launch "eipo_live_alr_005" "$RUNS_EIPO/eipo_live_alr_005" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.005 \
    --eipo-alpha-init 1.0 \
    --checkpoint-freq 500

launch "eipo_live_alr_05" "$RUNS_EIPO/eipo_live_alr_05" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.05 \
    --eipo-alpha-init 1.0 \
    --checkpoint-freq 500

launch "eipo_live_alr_2" "$RUNS_EIPO/eipo_live_alr_2" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.2 \
    --eipo-alpha-init 1.0 \
    --checkpoint-freq 500

launch "eipo_live_ainit_01" "$RUNS_EIPO/eipo_live_ainit_01" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 0.1 \
    --checkpoint-freq 500

launch "eipo_live_ainit_5" "$RUNS_EIPO/eipo_live_ainit_5" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 5.0 \
    --checkpoint-freq 500

# =============================================================================
# BLOCK E — EMA tau sweep, target mode (4 runs)
# Q: How much does disc smoothing strength matter?
#    Does tau=0.05 (fast) ≈ eipo_live_base?
# =============================================================================

launch "eipo_target_tau_0001" "$RUNS_EIPO/eipo_target_tau_0001" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode target \
    --disc-ema-tau 0.0001 \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --checkpoint-freq 500

launch "eipo_target_tau_001" "$RUNS_EIPO/eipo_target_tau_001" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode target \
    --disc-ema-tau 0.001 \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --checkpoint-freq 500

launch "eipo_target_tau_01" "$RUNS_EIPO/eipo_target_tau_01" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode target \
    --disc-ema-tau 0.01 \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --checkpoint-freq 500

launch "eipo_target_tau_05" "$RUNS_EIPO/eipo_target_tau_05" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode target \
    --disc-ema-tau 0.05 \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --checkpoint-freq 500

# =============================================================================
# BLOCK F — Discriminator update count (4 runs)
# Q: Does EIPO better tolerate high disc update rates (more non-stationarity)?
# =============================================================================

launch "amp_disc_updates_1" "$RUNS_AMP/amp_disc_updates_1" "$AMP_CFG" \
    --num-discriminator-updates 1

launch "amp_disc_updates_5" "$RUNS_AMP/amp_disc_updates_5" "$AMP_CFG" \
    --num-discriminator-updates 5

launch "eipo_live_disc_updates_1" "$RUNS_EIPO/eipo_live_disc_updates_1" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --num-discriminator-updates 1 \
    --checkpoint-freq 500

launch "eipo_live_disc_updates_5" "$RUNS_EIPO/eipo_live_disc_updates_5" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --num-discriminator-updates 5 \
    --checkpoint-freq 500

# =============================================================================
# BLOCK G — Discriminator regularization (4 runs)
# Q: Does EIPO's advantage persist with different disc regularization levels?
# disc_logit_reg: L2 on output logits (keeps discriminator conservative)
# disc_grad_penalty: WGAN-like gradient penalty (training stability)
# =============================================================================

launch "amp_low_reg" "$RUNS_AMP/amp_low_reg" "$AMP_CFG" \
    --disc-logit-reg 0.001 \
    --disc-grad-penalty 1.0

launch "amp_high_reg" "$RUNS_AMP/amp_high_reg" "$AMP_CFG" \
    --disc-logit-reg 0.1 \
    --disc-grad-penalty 10.0

launch "eipo_live_low_reg" "$RUNS_EIPO/eipo_live_low_reg" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --disc-logit-reg 0.001 \
    --disc-grad-penalty 1.0 \
    --checkpoint-freq 500

launch "eipo_live_high_reg" "$RUNS_EIPO/eipo_live_high_reg" "$EIPO_CFG" \
    --use-eipo true \
    --disc-stationarity-mode live \
    --disc-learning-rate 1e-4 \
    --eipo-alpha-lr 0.01 \
    --disc-logit-reg 0.1 \
    --disc-grad-penalty 10.0 \
    --checkpoint-freq 500

# =============================================================================

tmux select-window -t "$SESSION:amp_base"

echo ""
echo "=== Ablation study: 35 runs in tmux session '$SESSION' ==="
echo ""
echo "  BLOCK A (3) — core:    amp_base | eipo_live_base | eipo_target_base"
echo "  BLOCK B (8) — disc_lr: amp_disc_lr_{1e5,5e5,5e4,1e3}"
echo "                         eipo_live_disc_lr_{1e5,5e5,5e4,1e3}"
echo "  BLOCK C (6) — weights: amp_w_{2_8,35_65,65_35,667_333,8_2,9_1}"
echo "  BLOCK D (6) — alpha:   eipo_live_alr_{001,005,05,2}"
echo "                         eipo_live_ainit_{01,5}"
echo "  BLOCK E (4) — tau:     eipo_target_tau_{0001,001,01,05}"
echo "  BLOCK F (4) — updates: amp_disc_updates_{1,5}"
echo "                         eipo_live_disc_updates_{1,5}"
echo "  BLOCK G (4) — reg:     amp_{low,high}_reg"
echo "                         eipo_live_{low,high}_reg"
echo ""
echo "  Output: runs/ablations/{amp,eipo}/<run_name>/"
echo "  tmux attach -t $SESSION"
echo ""
