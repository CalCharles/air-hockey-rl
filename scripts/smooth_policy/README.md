# Smooth Policy Training Guide

This document provides a cleaner and more structured version of the training notes for `scripts/smooth_policy`.

## Overview

The most recent training scripts are located in:

- `scripts/smooth_policy/amp_history/amp_training`
  - `lsgan` corresponds to PPO training.
  - `lsgan_sac` corresponds to SAC training.

Although these scripts are labeled "amp", AMP can be disabled to run standard (non-AMP) training.

## Run Configuration Structure

Each run uses:

1. **Config file**: defines environment/task details.
2. **Args file**: defines training behavior, model/runtime settings, and related options.

You can either specify the config file inside the args file or pass it directly from the command line.

### Reward Scaling Note (Puck Juggle)

- Base reward scaling is relevant for puck juggle because the original reward scale is high.
- The scale is a multiplier applied to the task-specific reward contribution.
- A scale of `0.3` is sometimes used to improve AMP balancing (currently hard-coded in some usage paths).
- For standard use, `1.0` is also acceptable.

## Legacy Training Files

The following files were used in earlier experiments:

- `default_training`
- `caps_finetuning`
- `finetune_reward_scaling`

Their logic is mostly correct, but they are no longer the primary workflow. Because additional config options were introduced over time, old configuration combinations may not transfer cleanly.

- `caps_finetuning`: implementation of the CAPS approach (not recommended).
- `finetune_reward_scaling`: gradually reduces puck juggle reward scale so velocity/jerk rewards become more influential.
- These were introduced before PID controller integration.

## PID Controller Notes

PID usage is configured in the config file. Action scale must differ substantially between PID and non-PID setups.

Current design limitation:

- Action scale is set in the args file, but should ideally be coupled with environment settings.

Default values used:

- `0.02` for the original (non-PID) controller
- `0.25` for the PID controller

Additional notes:

- These values refer to the maximum action norm.
- `0.25` is generally sufficient for PID-based runs.
- PID hyperparameters can be set in the config file.

Practical guidance:

- If using PID, keep action scale near `0.25` (or another reasonable value).
- If not using PID, use `0.02`.

## SAC Notes

- SAC includes an `autotune` parameter that adjusts entropy weighting based on current entropy estimates.
- With AMP enabled, stability can degrade somewhat; testing has not yet been exhaustive.
- Without AMP, this setup is generally expected to be stable.
- The implementation uses the transformed Bellman operator for Q-values to mitigate explosion.
- This helps keep entropy balancing reasonable by preventing actor-loss terms from being dominated by very large Q-values.

## Recommended Files

For most use cases, use files under `amp_training` with the examples below.

### Environment Configs

- PID config: `scripts/smooth_policy/amp_history/configs/pid/pid_default_config.yaml`
- Non-PID config: `scripts/smooth_policy/amp_history/configs/default_config.yaml`

### PPO Args

- No AMP: `scripts/smooth_policy/amp_history/configs/pid/no_amp_default.yaml`
- With AMP: `scripts/smooth_policy/amp_history/configs/pid/amp_default.yaml`

### SAC Args

- No AMP: `scripts/smooth_policy/amp_history/configs/sac/sac_puck_juggle.yaml`
- With AMP: `scripts/smooth_policy/amp_history/configs/sac/sac_amp.yaml`

## Example Commands

### PPO with additional temporal alignment and action-magnitude reward on `cuda:2`

```bash
python scripts/smooth_policy/amp_history/amp_training/amp_training_lsgan.py --args-file scripts/smooth_policy/amp_history/configs/pid/no_amp_default.yaml --log_parent_dir runs/temporal/temporal_and_action/run --temporal_alignment_reward_scale=0.25 --action_magnitude_reward_scale=0.25  --device=cuda:2
```

### SAC with more aggressive value-update frequency (improved sample efficiency)

```bash
python scripts/smooth_policy/amp_history/amp_training/sac/amp_training_lsgan_sac.py \
    --args_file scripts/smooth_policy/amp_history/configs/sac/sac_puck_juggle.yaml --q-frequency 10
```

## Using AMP

### 1) Preprocess demonstration data

This step reads and stores demonstration data:

`scripts/smooth_policy/amp_history/amp_training/amp_data/prepare_position_dataset.py`

Example (including intermediate actions):

```bash
python scripts/smooth_policy/amp_history/amp_training/amp_data/prepare_position_dataset.py --include-actions --output-path scripts/smooth_policy/amp_history/amp_training/amp_data/dataset_with_action_history.pt
```

### 2) Run training with an AMP configuration and demo data

#### PPO with discriminator

```bash
python scripts/smooth_policy/amp_history/amp_training/amp_training_lsgan.py --args-file scripts/smooth_policy/amp_history/configs/pid/amp_default.yaml --demo_data_path scripts/smooth_policy/amp_history/amp_training/amp_data/dataset_with_action_history.pt --use_action_discriminator --log_parent_dir runs/amp_with_actions/ppo/test_with_history/run
```

#### SAC with discriminator

```bash
python scripts/smooth_policy/amp_history/amp_training/sac/amp_training_lsgan_sac.py --args-file scripts/smooth_policy/amp_history/configs/sac/sac_amp.yaml --demo_data_path scripts/smooth_policy/amp_history/amp_training/amp_data/dataset_with_action_history.pt --use_action_discriminator --log_parent_dir runs/amp_with_actions/sac/test_with_history/run
```
