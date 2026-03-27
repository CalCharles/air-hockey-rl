# EIPO Runs — Configuration Guide

EIPO (Extrinsic-Intrinsic Policy Optimization) balances task reward and discriminator (style) reward
using a Lagrangian multiplier `alpha` and two co-trained policies. See `train_lsgan_eipo.py` for implementation.

---

## The Core Problem: Discriminator Non-Stationarity

EIPO assumes the intrinsic reward is a fixed function. But because the discriminator trains alongside
the policy in a GAN loop, its reward signal shifts every iteration — creating a moving target for `alpha`.

The `disc_stationarity_mode` setting controls how this is handled.

---

## Stationarity Modes

### `live` — Baseline (no stationarity guarantee)

Rewards come directly from the live discriminator, which trains normally every iteration.

- Simplest setup, no extra copies
- Reward signal can drift significantly between iterations
- Use as a **control** to see whether stationarity modes actually help
- Can reduce drift naturally by lowering `disc_learning_rate` (e.g. `1e-5`)

```yaml
disc_stationarity_mode: "live"
```

---

### `target` — Recommended

Maintains a **slow-moving EMA copy** of the discriminator alongside the live one.

- Live disc trains normally on rollout data
- Policy rewards are computed from the **target** (EMA) disc — stable, slowly shifting
- After each disc update: `target = tau * live + (1 - tau) * target`
- Lower `tau` → slower target, more stable but lags further behind

```yaml
disc_stationarity_mode: "target"
disc_ema_tau: 0.005   # 0.001 for even slower / more stable
```

Logged as `eipo/disc_reward_drift` — how far the live disc has moved from the target.

---

### `snapshot` — Strongest within-stage stationarity

Freezes a **copy of the live disc at each EIPO stage switch**.

- Reward is completely fixed within a stage (min-stage or max-stage)
- When the stage switches, a new snapshot is captured from the live disc
- Results in discrete reward jumps between stages
- Works best when stages last many iterations

```yaml
disc_stationarity_mode: "snapshot"
```

---

### `frozen` — Matches paper assumption

Discriminator is **fully frozen** — no training after EIPO starts.

- Requires pre-training the disc first (e.g. a completed AMP run)
- Load via `discriminator_path`
- Perfect stationarity, but rewards become uninformative if the policy improves significantly past
  what the pre-trained disc saw

```yaml
disc_stationarity_mode: "frozen"
discriminator_path: "path/to/pretrained_disc.pth"
```

---

## Key Varying Parameters

| Parameter | What it controls |
|---|---|
| `disc_stationarity_mode` | How the disc reward signal is stabilized |
| `disc_ema_tau` | EMA decay rate (target mode only) — lower = slower/more stable |
| `eipo_alpha_lr` | Step size for the Lagrangian `alpha` update |
| `eipo_alpha_init` | Initial value of `alpha` — balances task vs disc weight at start |
| `disc_learning_rate` | How fast the live disc trains — lower reduces drift in live mode |

### How `alpha` works

`alpha` controls how much task reward matters relative to disc reward:

- **π_{E+I} (mixed policy):** trained on `(1 + alpha) * task_reward + disc_reward`
- **π_E (task-only policy):** trained on `alpha * task_reward`

`alpha` updates only when leaving max-stage, based on the task performance gap between the two policies:
- Mixed is **worse** at the task than task-only → `alpha` increases (more task emphasis)
- Mixed is **matching** task-only → `alpha` decreases (disc gets more influence)

---

## Run Variants

All 9 runs launched via `launch_eipo_runs.sh`, spread across 4 GPUs:

```
runs/eipo_runs/
  target/
    tau0.005_alphalr0.01/    — default recommended
    tau0.001_alphalr0.01/    — slower EMA, more stable reward
    tau0.005_alphalr0.005/   — slower alpha adaptation
  live/
    alphalr0.01/             — baseline
    alphalr0.005/            — slower alpha
    disc_lr1e-5/             — low disc lr to reduce natural drift
  snapshot/
    alphalr0.01/             — default
    alphalr0.005/            — slower alpha
    alphainit0.5/            — start with less task weight (alpha=0.5)
```

### What to compare

- **target vs live vs snapshot** at same `alpha_lr`: does stationarity mode matter?
- **tau0.005 vs tau0.001** (target): how slow should the EMA be?
- **alphalr0.01 vs alphalr0.005** (across modes): sensitivity to alpha step size
- **alphainit0.5 vs alphainit1.0** (snapshot): does starting alpha matter?

---

## Tensorboard Metrics

| Key | Meaning |
|---|---|
| `eipo/alpha` | Lagrangian multiplier value |
| `eipo/max_stage` | Current stage: 1=max, 0=min |
| `eipo/task_return_mixed` | Task reward of π_{E+I} |
| `eipo/task_return_task_only` | Task reward of π_E |
| `eipo/task_gap` | Difference (mixed − task-only); alpha tries to drive this to 0 |
| `eipo/policy_divergence` | KL-like divergence between the two policies |
| `eipo/disc_reward_drift` | (target/snapshot) reward shift from live vs stable disc |

---

## Launching

```bash
# Launch all 9 runs in a tmux session
bash scripts/smooth_policy/amp_history/configs/ppo/eipo/launch_eipo_runs.sh

# Attach to monitor
tmux attach -t eipo
tmux list-windows -t eipo
```

Output checkpoints saved under `runs/eipo_runs/<mode>/<variant>/`:
- `model_mixed.pth` — π_{E+I}, the policy to deploy
- `model_task.pth` — π_E, the task-only reference
