# TD3 Training

Training, sweep, and aggregation tooling for the TD3 air-hockey policy.

## Files

| File | Purpose |
|------|---------|
| `td3_training.py` | Core training script |
| `helper/generate_sweep.py` | Generate a bash launch script from a sweep YAML |
| `helper/aggregate_sweep.py` | Aggregate finished sweep runs into a CSV / wandb table |
| `helper/run_wandb_sweep.py` | Create a wandb sweep and run a wandb agent |

Sweep YAML configs live in `../../../configs/td3/`.

---

## 1. Running a sweep with bash (no wandb account required)

### Step 1 — Write a sweep YAML

```yaml
# configs/td3/my_sweep.yaml
base_args_file: scripts/smooth_policy/amp_history/configs/td3/td3_no_alignment.yaml
log_parent_dir: /data2/calebc/air_hockey/my_sweep
run_name_prefix: td3_sweep

gpus: [0, 1, 2, 3]
max_parallel: 8          # max concurrent background jobs; 0 = unlimited

mode: grid               # grid | random | individual
# num_samples: 20        # only used when mode: random

extra_args:              # static overrides applied to every run
  total_timesteps: 1000000
  seed: 0

params:
  task_reward_weight:
    values: [0.5, 1.0, 2.0]
  motion_reward_weight:
    linspace: {start: 0.0, stop: 1.0, num: 4}
  jerk_reward_weight:
    logspace: {start: -1, stop: 0, num: 3}   # 10^-1 .. 10^0
```

**Param spec options:**

| Key | Description |
|-----|-------------|
| `values: [v1, v2, ...]` | Explicit list |
| `linspace: {start, stop, num}` | Evenly spaced |
| `logspace: {start, stop, num}` | Log-spaced (`base` defaults to 10) |
| `range: {start, stop, step}` | Step range |

**Sweep modes:**

| Mode | Behaviour |
|------|-----------|
| `grid` | Cartesian product of all param values |
| `random` | Random sample of `num_samples` combos |
| `individual` | One run per (param, value) pair; other params stay at base-config defaults |

### Step 2 — Generate the launch script

```bash
uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/generate_sweep.py \
    --sweep-file configs/td3/my_sweep.yaml \
    --output-file run_my_sweep.sh \
    --eval-output-file eval_my_sweep.sh   # optional: also write an eval script
```

Preview runs without writing:
```bash
uv run ... --sweep-file configs/td3/my_sweep.yaml --output-file /dev/null --dry-run
```

### Step 3 — Launch

```bash
bash run_my_sweep.sh                   # launch all runs
DRY_RUN=1 bash run_my_sweep.sh         # preview without launching
ps -fu "$USER" | grep td3_training     # check status
```

### Step 4 — Evaluate (optional)

```bash
bash eval_my_sweep.sh                  # runs collect_policy_data.py on every run dir
```

Each run's output is saved to `<run_dir>/rollout/` (GIFs, `per_timestep.csv`, `metadata.yaml`).

---

## 2. Aggregating results

After runs finish (and optionally after evaluation), aggregate all results into a CSV:

```bash
# Via sweep YAML (auto-reads log_parent_dir, columns = swept params)
uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/aggregate_sweep.py \
    --sweep-file configs/td3/my_sweep.yaml \
    --output-csv my_sweep_results.csv

# Directly at the log directory
uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/aggregate_sweep.py \
    --sweep-dir /data2/calebc/air_hockey/my_sweep \
    --output-csv my_sweep_results.csv

# Quick TSV preview to stdout (no --output-csv)
uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/aggregate_sweep.py \
    --sweep-file configs/td3/my_sweep.yaml
```

**Output columns (always present):**

| Column | Source |
|--------|--------|
| `run_dir`, `run_name` | Directory / args.yaml |
| `<param_keys>` | args.yaml (restricted to swept params when `--sweep-file` is used) |
| `final_rolling_return` | Last value of `charts/rolling2k_avg_episode_return` in TensorBoard |
| `max_rolling_return` | Maximum of the same curve |
| `final_episode_return` | Last raw episode return logged to TensorBoard |
| `max_episode_return` | Maximum raw episode return |
| `tb_steps` | Training step of the last TensorBoard event |
| `eval_*` | From `rollout/metadata.yaml` and `rollout/per_timestep.csv` (if eval was run) |

### Upload to wandb as a summary table

```bash
uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/aggregate_sweep.py \
    --sweep-file configs/td3/my_sweep.yaml \
    --output-csv my_sweep_results.csv \
    --wandb-project my-project \
    --wandb-entity my-team
```

This creates a single wandb run of type `sweep_aggregate` containing a `sweep_results` table.

---

## 3. Running a sweep via wandb agent

This creates a proper wandb sweep (visible in the wandb UI) and runs trials through the
wandb agent. Requires a wandb account and `wandb_project` in the sweep YAML.

### Add wandb fields to your sweep YAML

```yaml
# Add to the top of your sweep YAML:
wandb_project: my-project-name
wandb_entity:  my-team          # optional
```

### Create the sweep and start an agent

```bash
# Create sweep + start agent immediately (blocks until --count trials finish)
uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/run_wandb_sweep.py \
    --sweep-file configs/td3/my_sweep.yaml \
    --device cuda:0 \
    --count 20

# Create the sweep only, print the wandb agent command, then exit
uv run scripts/smooth_policy/amp_history/amp_training/td3/helper/run_wandb_sweep.py \
    --sweep-file configs/td3/my_sweep.yaml \
    --create-only
```

### Run multiple agents in parallel (one per GPU)

After `--create-only` prints the sweep ID, start one agent per GPU:

```bash
SWEEP_ID=abc123

uv run .../run_wandb_sweep.py --sweep-file ... --sweep-id $SWEEP_ID --device cuda:0 --count 10 &
uv run .../run_wandb_sweep.py --sweep-file ... --sweep-id $SWEEP_ID --device cuda:1 --count 10 &
uv run .../run_wandb_sweep.py --sweep-file ... --sweep-id $SWEEP_ID --device cuda:2 --count 10 &
uv run .../run_wandb_sweep.py --sweep-file ... --sweep-id $SWEEP_ID --device cuda:3 --count 10 &
wait
```

### How each trial works

1. The agent calls `wandb.init()` to create a run and receive the trial's hyperparameters.
2. The wrapper immediately calls `wandb.finish()` to release its handle.
3. `td3_training.py` is launched as a subprocess with the trial params as CLI flags.
4. The subprocess resumes the same wandb run (`WANDB_RUN_ID` + `WANDB_RESUME=allow`)
   and logs all training metrics to it.

All metrics (episode returns, reward components, etc.) appear under the single run in the
wandb sweep UI.

### Notes on sweep modes

| generate_sweep mode | wandb agent behaviour |
|--------------------|-----------------------|
| `grid` | All combinations, one per trial |
| `random` | Random sampling; use `--count` to limit |
| `individual` | Treated as `grid` — all combinations run. Use `--count` to limit, or prefer the bash approach for true individual-param sweeps. |

---

## Typical workflow (bash sweep + wandb aggregation)

```bash
# 1. Generate scripts
uv run .../generate_sweep.py \
    --sweep-file configs/td3/my_sweep.yaml \
    --output-file run_sweep.sh \
    --eval-output-file eval_sweep.sh

# 2. Launch training
bash run_sweep.sh

# 3. (After training) Evaluate all runs
bash eval_sweep.sh

# 4. Aggregate and upload
uv run .../aggregate_sweep.py \
    --sweep-file configs/td3/my_sweep.yaml \
    --output-csv results.csv \
    --wandb-project my-project
```
