DEVICES=(0 1 2)          # change if you want different GPU IDs
HIDDEN_SIZES=(192 256 512)

for i in "${!HIDDEN_SIZES[@]}"; do
  dev="${DEVICES[$i]}"
  hs="${HIDDEN_SIZES[$i]}"

  CUDA_VISIBLE_DEVICES="$dev" \
  python scripts/smooth_policy/amp_history/amp_training/amp_training_lsgan.py \
    --args-file scripts/smooth_policy/amp_history/configs/pid/no_amp_default.yaml \
    --log_parent_dir runs/temporal/temporal_and_action/hs${hs} \
    --run-name "lsgan_ta025_am025_hs${hs}_gpu${dev}" \
    --temporal-alignment-reward-scale 0.25 \
    --action-magnitude-reward-scale 0.25 \
    --agent-hidden-size "$hs" \
    --device cuda:0 &
done

wait