# Smooth Policy Training Guide

Structured notes for [`scripts/smooth_policy`](.).

## What is actually in use

| Track | Entrypoint | Notes |
|-------|------------|--------|
| **TD3** (most recent development) | [`amp_history/amp_training/td3/td3_training.py`](amp_history/amp_training/td3/td3_training.py) | **No AMP** — no discriminator or demo dataset; dual-head critics, exploration helpers under [`td3/helper/`](amp_history/amp_training/td3/helper/), async real tooling under [`td3/extras/`](amp_history/amp_training/td3/extras/). |
| **PPO + AMP** (current AMP training) | [`amp_history/amp_training/amp_training.py`](amp_history/amp_training/amp_training.py) | PPO with optional least-squares AMP discriminator; can disable AMP for plain PPO. |
| **Discriminator data** | [`amp_history/amp_training/amp_data/`](amp_history/amp_training/amp_data/) | **Data processing only** — builds `.pt` datasets from HDF5 trajectories for the PPO+AMP discriminator (`--demo_data_path`, action/puck options). Not used by TD3. |

### Legacy folders (little or no active use)

The following still exist for older experiments; **prefer TD3 or `amp_training.py` for new work**:

- [`amp_training/sac/`](amp_history/amp_training/sac/) — SAC ± AMP (`amp_training_sac.py`)
- [`amp_training/rma/`](amp_history/amp_training/rma/) — RMA / adaptation variants
- [`amp_training/self_supervised/`](amp_history/amp_training/self_supervised/) — SSL-style AMP training

Root-level scripts under [`amp_history/amp_training/`](amp_history/amp_training/) such as `default_training`, `caps_finetuning`, and `finetune_reward_scaling` (if present) are also legacy — see **Legacy Training Files** below.

## Run configuration structure

Each run uses:

1. **Config file**: environment/task (often PID vs non-PID, reward, sim params).
2. **Args file**: training algorithm, model sizes, device, logging, AMP flags, paths to demo data (PPO+AMP only).

You can reference the config from the args YAML or pass paths on the command line.

## Async TD3 real transition holds

The async real collector has two hold layers during sensitive transitions:

- Collector hold in [`amp_history/amp_training/td3/extras/async_td3_real.py`](amp_history/amp_training/td3/extras/async_td3_real.py)
  - zeros policy actions for a few steps
  - can disable exploration noise during the hold
  - can reset or preserve the policy's last-action state
- Simulator hold in [`airhockey/sims/air_hockey_real.py`](../../airhockey/sims/air_hockey_real.py)
  - anchors the commanded TCP target to the current pose
  - blocks `servoL` while the hold is active

Intended collector hold reasons are:

- `startup_reset_to_policy`
- `reset_fsm_to_policy`
- `hard_reset_reset_fsm_to_policy`
- `hard_reset_to_policy`
- `robot_recovered_to_ready`
- `actor_sync_update`

Intended simulator-side recovery smoothing is:

- `estop_clear`
- `safety_rearm` after a genuine recovery from:
  - protective stop
  - controller disconnect
  - failed safety check

Important implementation note:

- Internal `transition_hold:*` blocks should not create a fresh `safety_rearm`.
- `safety_rearm` is meant for real command-path recovery, not for ordinary reset-to-policy smoothing.

Main async TD3 parameters to tune:

- `transition_hold_steps_post_reset`
  - hold length for reset-to-policy handoffs
- `transition_hold_steps_post_estop_enter`
  - optional immediate hold on entering protective stop
- `transition_hold_steps_post_estop_clear`
  - hold after protective stop / readiness recovery
- `transition_hold_steps_post_actor_sync`
  - hold after loading a newly published actor
- `transition_hold_steps_post_safety_rearm`
  - simulator-side rearm hold after genuine command recovery
- `transition_disable_exploration_noise`
  - disables exploration noise while collector hold is active
- `transition_last_action_mode`
  - only relevant when `use_last_action_in_policy_state=True`
  - `zero`, `executed`, or `keep`

Practical guidance:

- If reset-to-policy handoff feels abrupt, adjust `transition_hold_steps_post_reset`.
- If recovery after a real e-stop or controller issue feels abrupt, adjust `transition_hold_steps_post_estop_clear` and `transition_hold_steps_post_safety_rearm`.
- If you only want to inspect the mechanism, enable `debug_control`, `debug_control_every`, and `transition_hold_debug` in the real simulator config.

### Reward scaling note (puck juggle)

- Base reward scaling matters because the raw task reward can be large.
- The scale is a multiplier on the task-specific reward contribution.
- A scale of `0.3` is sometimes used to improve AMP balancing (hard-coded in some paths).
- For non-AMP or TD3 runs, `1.0` is often fine; tune per experiment.

## Legacy training files

Older top-level scripts under `amp_history/amp_training/` (when present):

- `default_training`
- `caps_finetuning`
- `finetune_reward_scaling`

Their logic may still run but they are **not** the primary workflow; config drift means old combinations may not load cleanly.

- `caps_finetuning`: CAPS-style approach (not recommended for new work).
- `finetune_reward_scaling`: ramps down puck juggle reward scale so velocity/jerk terms weigh more — predates PID integration in many configs.

## PID controller notes

PID usage is set in the **environment config**. Action scale must differ between PID and non-PID setups.

Current design limitation:

- Action scale is often set in the **args** file but should stay consistent with the env config.

Default norms (max action magnitude) seen in practice:

- `0.02` — original non-PID-style control
- `0.25` — PID-style runs

If using PID, keep action scale near `0.25` (or another value you validate). If not using PID, use something like `0.02`. PID hyperparameters live in the env config.

## Recommended configs and commands

### Environment configs (PPO / shared)

- PID: [`amp_history/configs/pid/pid_default_config.yaml`](amp_history/configs/pid/pid_default_config.yaml)
- Non-PID: [`amp_history/configs/default_config.yaml`](amp_history/configs/default_config.yaml)

### PPO (`amp_training.py`)

- No AMP: [`amp_history/configs/pid/no_amp_default.yaml`](amp_history/configs/pid/no_amp_default.yaml)
- With AMP: [`amp_history/configs/pid/amp_default.yaml`](amp_history/configs/pid/amp_default.yaml)

### TD3 (`td3/td3_training.py`)

- Examples: [`amp_history/configs/td3/`](amp_history/configs/td3/) (e.g. [`td3_standard.yaml`](amp_history/configs/td3/td3_standard.yaml))
- Real / online-style: [`amp_history/configs/td3_real_world/`](amp_history/configs/td3_real_world/)

### Example: PPO with extra shaping on `cuda:2`

```bash
python scripts/smooth_policy/amp_history/amp_training/amp_training.py \
  --args-file scripts/smooth_policy/amp_history/configs/pid/no_amp_default.yaml \
  --log_parent_dir runs/temporal/temporal_and_action/run \
  --temporal_alignment_reward_scale=0.25 \
  --action_magnitude_reward_scale=0.25 \
  --device=cuda:2
```

### Example: TD3 (no AMP, no demo data)

```bash
python scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py \
  --args-file scripts/smooth_policy/amp_history/configs/td3/td3_standard.yaml \
  --log_parent_dir runs/td3/standard/run
```

## PPO + AMP: data and training

### 1) Build discriminator dataset (`amp_data`)

[`amp_data/prepare_position_dataset.py`](amp_history/amp_training/amp_data/prepare_position_dataset.py) reads HDF5 trajectories and writes a `.pt` tensor dataset (windowed paddle states; optional actions / puck windows) for the discriminator. [`prepare_position_dataset_split.py`](amp_history/amp_training/amp_data/prepare_position_dataset_split.py) supports split outputs.

Example (with intermediate actions):

```bash
python scripts/smooth_policy/amp_history/amp_training/amp_data/prepare_position_dataset.py \
  --include-actions \
  --output-path scripts/smooth_policy/amp_history/amp_training/amp_data/dataset_with_action_history.pt
```

### 2) Train PPO with discriminator

```bash
python scripts/smooth_policy/amp_history/amp_training/amp_training.py \
  --args-file scripts/smooth_policy/amp_history/configs/pid/amp_default.yaml \
  --demo_data_path scripts/smooth_policy/amp_history/amp_training/amp_data/dataset_with_action_history.pt \
  --use_action_discriminator \
  --log_parent_dir runs/amp_with_actions/ppo/test_with_history/run
```

## Legacy: SAC + AMP

The SAC stack under [`amp_training/sac/`](amp_history/amp_training/sac/) is **not** the focus of current development. If you still run it:

- SAC uses an `autotune` entropy target; with AMP enabled, stability was less thoroughly exercised than PPO+AMP.
- Implementation uses a transformed Bellman operator on Q-values to limit blow-up.

### SAC example configs (legacy)

- No AMP: [`amp_history/configs/sac/sac_puck_juggle.yaml`](amp_history/configs/sac/sac_puck_juggle.yaml)
- With AMP: [`amp_history/configs/sac/sac_amp.yaml`](amp_history/configs/sac/sac_amp.yaml)

### SAC + discriminator example (legacy)

```bash
python scripts/smooth_policy/amp_history/amp_training/sac/amp_training_sac.py \
  --args-file scripts/smooth_policy/amp_history/configs/sac/sac_amp.yaml \
  --demo_data_path scripts/smooth_policy/amp_history/amp_training/amp_data/dataset_with_action_history.pt \
  --use_action_discriminator \
  --log_parent_dir runs/amp_with_actions/sac/test_with_history/run
```

### SAC tuning example (legacy)

```bash
python scripts/smooth_policy/amp_history/amp_training/sac/amp_training_sac.py \
  --args_file scripts/smooth_policy/amp_history/configs/sac/sac_puck_juggle.yaml \
  --q-frequency 10
```
