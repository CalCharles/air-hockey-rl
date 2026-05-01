# Training Commands & Output Locations

Summary of the two training studies set up on the `ppo-discriminator-stuff` branch:
the **EIPO stationarity-mode sweep** and the **EIPO vs AMP ablation study**.

---

## Where things live

- **Training script (shared):** [scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py](scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py)
- **Config / args YAMLs:**
  - EIPO sweep base: [scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml](scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml)
  - Ablations AMP baseline: [scripts/smooth_policy/amp_history/configs/ppo/ablations/amp_ablation_base.yaml](scripts/smooth_policy/amp_history/configs/ppo/ablations/amp_ablation_base.yaml)
  - Ablations EIPO variants share the EIPO `eipo_target.yaml` above
- **Launcher scripts:**
  - EIPO: [scripts/smooth_policy/amp_history/configs/ppo/eipo/launch_eipo_runs.sh](scripts/smooth_policy/amp_history/configs/ppo/eipo/launch_eipo_runs.sh)
  - Ablations: [scripts/smooth_policy/amp_history/configs/ppo/ablations/launch_ablations.sh](scripts/smooth_policy/amp_history/configs/ppo/ablations/launch_ablations.sh)
- **Per-run resolved settings** (written at run start): `<run_dir>/config.yaml` (training config) and `<run_dir>/args.yaml` (CLI args)

> **Note:** commit `c5050f3` moved this script from `amp_training/` into
> `amp_training/ppo/`. The ablations launcher uses the new path; the EIPO
> launcher ([launch_eipo_runs.sh:30](scripts/smooth_policy/amp_history/configs/ppo/eipo/launch_eipo_runs.sh#L30))
> still points at the old location and will fail as-is. Fix by changing
> `SCRIPT="scripts/smooth_policy/amp_history/amp_training/train_lsgan_eipo.py"`
> to `.../amp_training/ppo/train_lsgan_eipo.py` before running.

---

## 1. EIPO stationarity sweep (9 runs)

Compares `target` / `live` / `snapshot` discriminator-stationarity modes, each
with a few parameter variants. All 9 runs launched in parallel across 4 GPUs
(round-robin) in one tmux session.

### Launch

```bash
bash scripts/smooth_policy/amp_history/configs/ppo/eipo/launch_eipo_runs.sh
tmux attach -t eipo
tmux list-windows -t eipo
```

Base config: [scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml](scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml)

### Output location

```
runs/eipo_runs/
  target/
    tau0.005_alphalr0.01/       ← recommended default
    tau0.001_alphalr0.01/       ← slower EMA
    tau0.005_alphalr0.005/      ← slower alpha
  live/
    alphalr0.01/                ← baseline
    alphalr0.005/               ← slower alpha
    disc_lr1e-5/                ← low disc lr
  snapshot/
    alphalr0.01/                ← default
    alphalr0.005/               ← slower alpha
    alphainit0.5/               ← lower initial alpha
```

Each run directory contains:
- `model_mixed.pth` — π_{E+I}, the policy to deploy
- `model_task.pth` — π_E, task-only reference
- `config.yaml`, `args.yaml` — exact settings used
- `train.log` — full stdout/stderr (via `tee`)
- TensorBoard event files (load the run dir directly in TB)
- `checkpoint_<iter>/` — periodic checkpoints
- `eval_mixed/`, `eval_task/` — eval rollouts

See [scripts/smooth_policy/amp_history/configs/ppo/eipo/README.md](scripts/smooth_policy/amp_history/configs/ppo/eipo/README.md)
for the full writeup of stationarity modes and what each key tensorboard metric means.

### Running a single variant manually

```bash
python scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py \
    --args-file scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml \
    --device cuda:0 \
    --log-parent-dir runs/eipo_runs/target/tau0.005_alphalr0.01 \
    --run-name target_default \
    --disc-stationarity-mode target \
    --disc-ema-tau 0.005 \
    --eipo-alpha-lr 0.01 \
    --eipo-alpha-init 1.0
```

---

## 2. EIPO vs AMP ablation study (35 runs)

Larger sweep: 35 runs across 7 thematic blocks (disc LR, task/disc weight,
alpha LR, alpha init, tau, disc updates, regularization), comparing plain AMP
to EIPO variants. Runs are **queued sequentially per GPU** — 4 run in parallel
at a time, the rest wait in line on their GPU's window.

### Launch

```bash
bash scripts/smooth_policy/amp_history/configs/ppo/ablations/launch_ablations.sh
tmux attach -t ablations
tmux list-windows -t ablations        # 4 windows: gpu0, gpu1, gpu2, gpu3
```

Base configs:
- AMP baselines: [scripts/smooth_policy/amp_history/configs/ppo/ablations/amp_ablation_base.yaml](scripts/smooth_policy/amp_history/configs/ppo/ablations/amp_ablation_base.yaml)
- EIPO variants: [scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml](scripts/smooth_policy/amp_history/configs/ppo/eipo/eipo_target.yaml)

### GPU → run queues

| GPU | # runs | Runs |
|-----|--------|------|
| 0 | 9 | amp_base, amp_disc_lr_5e5, eipo_live_disc_lr_5e5, amp_w_35_65, amp_w_9_1, eipo_live_alr_2, eipo_target_tau_001, amp_disc_updates_5, amp_high_reg |
| 1 | 9 | eipo_live_base, amp_disc_lr_5e4, eipo_live_disc_lr_5e4, amp_w_65_35, eipo_live_alr_001, eipo_live_ainit_01, eipo_target_tau_01, eipo_live_disc_updates_1, eipo_live_low_reg |
| 2 | 9 | eipo_target_base, amp_disc_lr_1e3, eipo_live_disc_lr_1e3, amp_w_667_333, eipo_live_alr_005, eipo_live_ainit_5, eipo_target_tau_05, eipo_live_disc_updates_5, eipo_live_high_reg |
| 3 | 8 | amp_disc_lr_1e5, eipo_live_disc_lr_1e5, amp_w_2_8, amp_w_8_2, eipo_live_alr_05, eipo_target_tau_0001, amp_disc_updates_1, amp_low_reg |

### Output location

```
runs/ablations/
  amp/<run_name>/        ← plain-AMP runs
  eipo/<run_name>/       ← EIPO runs
```

Each run dir has the same contents as the EIPO sweep runs above
(`model_mixed.pth`, `config.yaml`, `args.yaml`, `train.log`, TB events, checkpoints).
Ablation EIPO runs use `--checkpoint-freq 500`.

---

## Default output fallback

If `--log-parent-dir` is not passed, the script falls back to
`runs/default_training/{task_name}/{run_name}_{timestamp}/`
(see [train_lsgan_eipo.py:671-682](scripts/smooth_policy/amp_history/amp_training/ppo/train_lsgan_eipo.py#L671-L682)).
If the target dir already exists, an `r1`, `r2`, … suffix is appended so prior
runs are never overwritten.

---

## Monitoring

```bash
# Live training output
tail -f runs/eipo_runs/<mode>/<variant>/train.log
tail -f runs/ablations/{amp,eipo}/<run_name>/train.log

# TensorBoard across all runs in a study
tensorboard --logdir runs/eipo_runs
tensorboard --logdir runs/ablations
```

Key EIPO metrics (also documented in the EIPO README):
`eipo/alpha`, `eipo/max_stage`, `eipo/task_return_mixed`,
`eipo/task_return_task_only`, `eipo/task_gap`, `eipo/policy_divergence`,
`eipo/disc_reward_drift`.
