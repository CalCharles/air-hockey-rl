# RMA baseline (Rapid Motor Adaptation) with TD3 on the canonical domain randomization

**Code**: `scripts/rma/` (self-contained; index in [`scripts/rma/README.md`](../../../scripts/rma/README.md)) ·
**Configs**: [`configs/rma/rma_juggle_dr_phase1.yaml`](../../../configs/rma/rma_juggle_dr_phase1.yaml), [`configs/rma/rma_juggle_dr_phase2.yaml`](../../../configs/rma/rma_juggle_dr_phase2.yaml) ·
**Tests**: `pytest scripts/rma/tests -q` ·
**Results**: [`2026-09-11_01-05_rma-td3-baseline.md`](../../scratch/experiments/2026-09-11_01-05_rma-td3-baseline.md) (latent fit, first policy comparison) · [`2026-09-18_21-00_rma-fair-baselines-long-history-control.md`](../../scratch/experiments/2026-09-18_21-00_rma-fair-baselines-long-history-control.md) (**fair comparison**: long-history control + 3 plain-DR seeds, paired eval)

RMA (Kumar, Fu, Pathak, Malik, *RMA: Rapid Motor Adaptation for Legged Robots*,
RSS 2021) is the standard "explicit adaptation" sim2real scheme and the natural
baseline for the paper's *aggressive physics-parameter randomization* scheme,
which trains adaptation only implicitly (the policy has to infer the current
dynamics from its observation history). Both are trained on the **same**
randomized simulator (`sim_paramrand_pm25.yaml`: paddle_density / puck_damping /
gravity drawn uniformly within ±25 % of sysid on every reset, observation
delay / puck noise / occlusion on) with the **same** TD3 recipe, so the
comparison isolates the RMA mechanism.

## The method, as implemented

```
phase 1 (sim, privileged)          phase 2 (sim, supervised)              deployment
e_t ──► mu ──► z_t ─┐              (x,a)_{t-50:t-1} ──► phi ──► ẑ_t ─┐    (x,a) history ──► phi ──► ẑ_t ─┐
x_t, a_{t-1} ───────┴► pi ──► a_t   regress ẑ_t → z_t = mu(e_t)         │    x_t, a_{t-1} ────────────────┴► pi ──► a_t
Q(x_t, e_t, a_t) (critic, sim-only)  on rollouts of pi(x, a_{t-1}, ẑ_t)  ┘    (no fine-tuning, no e_t)
```

| Element | Paper | This implementation |
|---|---|---|
| Privileged factors e_t | mass, friction, motor strength, terrain … (17-dim) | the 3 randomized physics parameters, normalised to [-1, 1] over their DR ranges (`EnvParamNormalizer`; the sysid centre is 0, OOD physics falls outside ±1) |
| Encoder μ | MLP 256-128 → 8-dim z | same (`EnvFactorEncoder`, ELU, linear output) |
| Base policy input | x_t, a_{t-1}, z_t | same: 30-dim history obs, last action, z (`RMAActor`; trunk = the canonical `DeterministicAgent` residual MLP, 64 wide × 2 blocks) |
| Phase-1 RL | PPO; π and μ trained jointly end-to-end | **TD3** (project recipe, see below); π and μ trained jointly end-to-end through the actor loss; critic Q(x_t, e_t, a_t) sees the raw factors (asymmetric, sim-only) |
| Adaptation module φ | 50-step (x, a) history → per-step MLP → 32-dim → 3 × Conv1d (32 ch, k = 8/5/5, s = 4/1/1) → linear → z | same (`AdaptationModule`; conv output 3 × 32 = 96 for H = 50). Per-step x = the newest frame of the history obs (paddle x, y, valid; puck x, y, valid) + the 2-dim action (`step_features: latest_frame`; `full_obs` uses all 30 dims) |
| Phase-2 training | supervised MSE to z_t on **on-policy** rollouts of π(x, a₋₁, ẑ_t), iterated | same (`train_adaptation_module.py`: collect → regress → repeat; 20 × 20k steps; rolling 400k-step dataset; 10 % of episodes held out) |
| Deployment | π(x_t, a_{t-1}, ẑ_t); no fine-tuning | same (`AdaptedActor`, `AdaptedRMAAgent`) |
| History padding | – | the first H steps are padded by repeating the first state with a zero action, identically online and in the dataset |

Everything RMA-specific lives in `scripts/rma/`; the phase-1 trainer is the
canonical loop of `scripts/td3/td3_training.py` (TD3 with transformed Bellman
targets, PER, single flat 1M replay buffer, primitive exploration, CUDA-graph
updates, CPU rollout, async checkpoint eval) with three local changes: the env
wrapper appends e_t to the observation, the actor is `RMAActor` (μ is part of
its parameters, so the same optimizer / Polyak target / CPU replica / checkpoint
train and carry it), and the critic's observation is `[x_t, e_t]`.

## The fair control: long-history TD3 (`rma_mode: history`)

The plain-DR policy of the paper sees only the 5-frame (30-dim) observation,
whereas the deployed RMA policy additionally gets a 50-step state/action window
through φ. A comparison against plain DR alone therefore confounds "RMA's
supervised latent" with "a longer history". The control is the same trainer
with `rma_mode: history` ([`configs/rma/history_juggle_dr.yaml`](../../../configs/rma/history_juggle_dr.yaml)):
the actor is `HistoryActor` = **exactly the deployable RMA architecture**
(50-step window of the same per-step features → the same conv encoder → 8-dim
z → the same trunk) trained end-to-end by the TD3 actor loss with **no
privileged input**; the critic sees `[x_t, window]`. `HistoryEnvVector`
appends the window (padded identically to the adaptation module's history) to
every observation so the canonical replay / update machinery carries it.
RMA's only remaining difference to this control is the supervised, privileged
latent. `scripts/rma/eval_paired.py` evaluates any set of RMA phase-2 runs,
history runs and plain-DR runs on the same envs and episode seeds (ID + OOD)
and writes `paired_eval.md`.

## Is TD3 instead of PPO acceptable?

**Yes, with two caveats that the pipeline measures rather than hides.**

Why it is fine:

1. **The RL algorithm is not part of the RMA contribution.** RMA is the
   decomposition into a privileged latent trained with the policy and a
   supervised adaptation module trained on on-policy data; the paper describes
   phase 1 only as "model-free RL". The one requirement is that z be a
   *policy-relevant* compression of e, i.e. that μ receive the policy's
   gradient. In TD3 that is the deterministic policy gradient through the
   actor loss, which plays exactly the role of PPO's surrogate gradient.
2. **Off-policy replay is well-posed.** e_t is stored with every transition
   and z_t is recomputed from the *current* μ at update time, so there are no
   stale latents in the buffer; a transition's latent is a deterministic
   function of stored data.
3. **It is the fair comparison.** The paper trains every sim2real scheme with
   TD3 ("the pipeline does not depend on the choice of algorithm; we use TD3
   throughout"). An RMA-with-PPO baseline would confound the adaptation
   mechanism with an algorithm change, and PPO on this task would need its own
   recipe (the TD3 recipe depends on the h-transform, PER and the exploration
   primitives; a PPO run would be a different, untuned learner).

Caveats:

- **Deterministic actor → less diverse phase-2 data.** PPO's stochastic policy
  gives on-policy data diversity for free; TD3's rollouts are deterministic.
  `rollout_action_noise` (default 0 = exactly the deployed policy, the paper's
  setting) can widen coverage. The metric to watch is the gap between the
  validation MSE (held-out episodes of the *same* rollout policy) and the
  on-policy MSE measured on each iteration's fresh rollouts *before* the
  update (`collect/online_latent_mse`) — a distribution-shift detector.
- **The critic sees the raw factors.** The asymmetric critic is the usual
  implementation of privileged actor-critic methods and is sim-only, but it
  means part of any phase-1 gain over plain DR can come from an un-aliased
  critic rather than from z. The paired evaluation separates the two: the
  *nominal* agent (z = μ(0), the base policy run at the sysid-centre latent, i.e.
  RMA without adaptation) vs *privileged* (oracle z) vs *adapted* (ẑ from φ).

## What is reported

**Latent fitting (adaptation module).** Per iteration, on held-out episodes:
latent MSE overall, per dim, and by steps-since-reset bucket (t < 10, 10–25,
25–50, ≥ 50: how fast φ locks in), per-dim R² of ẑ against z; a linear
probe (ridge, fitted on train) ẑ → e with per-parameter validation R² (can the
physics parameters be read off the estimated latent?), the same probe from the
true z (ceiling), the pre-update on-policy MSE, and the return of the rollouts
that generated the data. `phase2_metrics.jsonl`, TensorBoard, and table B of
`scripts/rma/summarize.py`.

**Policy performance.** The privileged phase-1 policy is evaluated at every
checkpoint on the fixed 5-env DR eval set (`eval_param_seed: 12345`, identical
to `td3_training_dr`; `multi_env_eval.json` in the canonical schema so
`run_experiments.py --summarise-only` and the existing plots read it). Phase 2
evaluates *adapted / privileged / nominal / td3_dr* (the on-disk plain-DR
policy `runs/td3/tasks_20260904/dr/juggle_dr`) on the same envs **and the same
episode seeds** (paired), on the in-distribution set and on the 5-env
out-of-distribution set drawn from `random_variable_ranges_OOD` (paddle
density 2–2.5×, damping 2–2.5×, gravity 2–2.5× sysid). `phase2_summary.json`;
tables A and C of `summarize.py`.

## Headline results (juggle, ±25 % DR, 2M steps; paired eval, 20 episodes × 5 envs per split, same episode seeds for every agent)

| policy (kind, mean over seeds ± std) | n seeds | in-distribution | out-of-distribution (physics 2–2.5×) |
|---|---:|---:|---:|
| long-history TD3 control (50-step window, end-to-end, no privileged input) | 2 | **192.5 ± 0.0** | **130.6 ± 10.7** |
| RMA adapted (deployable φ + π) | 2 | 149.5 ± 0.5 | 120.1 ± 0.2 |
| RMA nominal (z = μ(0)) | 2 | 150.2 ± 6.3 | 103.4 ± 9.3 |
| RMA privileged (oracle z, sim only) | 2 | 143.6 ± 5.0 | 85.6 ± 36.3 |
| plain-DR TD3 (5-frame obs, the paper's scheme) | 3 | 133.2 ± 11.0 | 90.2 ± 3.1 |

- **RMA's gain over plain DR is mostly the longer history.** Given the same
  50-step window and encoder, TD3 trained end-to-end reaches 192 (of a
  250 ceiling) vs RMA's 150; RMA's supervised 8-dim latent is the bottleneck
  because the phase-1 base policy learns to nearly ignore z under ±25 % DR
  (action sensitivity to z falls from 30–50 % of |a| at 250k steps to ~10 %
  at 2M), so adapted ≈ nominal ≈ privileged.
- RMA adapted is the most consistent OOD RMA variant; the oracle latent
  collapses in one seed when fed physics far outside the training box.
- Adaptation module: validation R² 0.62 / 0.43 after 400k on-policy steps
  (0.71 at 800k); recovers paddle mass (probe R² 0.95), gravity partially
  (0.58), puck damping barely (0.19); converges within ~25 steps; on-policy
  MSE = validation MSE (no train/deploy gap from the deterministic actor).
- Tables via `scripts/rma/summarize.py` / `scripts/rma/eval_paired.py`,
  figure via `scripts/rma/plot_results.py`, latent usage via
  `scripts/rma/analyze_latent_usage.py`. Real-robot numbers: none yet.

## Running

```bash
.venv/bin/python -m scripts.rma.train_base_policy --args-file configs/rma/rma_juggle_dr_phase1.yaml \
    --device cuda:0 --log-parent-dir runs/rma/juggle_dr_phase1_seed0          # ~1-2 h
.venv/bin/python -m scripts.rma.train_adaptation_module --args-file configs/rma/rma_juggle_dr_phase2.yaml \
    --phase1-dir runs/rma/juggle_dr_phase1_seed0 --log-parent-dir runs/rma/juggle_dr_phase1_seed0/phase2
.venv/bin/python -m scripts.rma.summarize --phase1-dirs runs/rma/juggle_dr_phase1_seed0 \
    --baseline-dirs runs/td3/tasks_20260904/dr/juggle_dr --phase2-dirs runs/rma/juggle_dr_phase1_seed0/phase2
```

Phase 1 is also a `run_experiments.py` mode (`--mode rma`, or `auto` via the
`rma_latent_dim` key — checked before `her_k`). Other tasks: point `config:` at any
`configs/new_juggle/tasks/sim_dr_<task>.yaml` (the DR block is what RMA needs).

### Goal-conditioned tasks (HER), 2026-09-19

When the sim config sets `return_goal_obs: true` (the two puck-goal tasks) the same
trainer runs the HER recipe of `scripts/td3/td3_training_her.py` in both modes:

- the inner env is `GoalEnvVector` (x_t = `[observation, desired_goal]`, 32 dims) and
  the wrapper appends e_t (privileged) or the 50-step window (history) after it;
  `step_features: latest_frame` keeps reading the paddle / puck frame from the first
  30 dims, so the window layout is unchanged;
- every finished episode is hindsight-relabelled (`her_k`, `her_strategy`,
  `her_done_on_success`, `her_goal_filter` — the same keys as the HER trainer, so its
  recipes load as-is) before it enters replay; the relabeler rewrites the goal slice
  only, the e_t / window suffix stays;
- the stored next observation is the final one on every episode end (as in the HER
  trainer), and the critic sees `[x_t, goal, e_t | window]`;
- checkpoint / final evaluation, phase 2 rollouts and `eval_paired.py` run the goal env
  through `scripts.rma.evaluate.FlatGoalEnv` (flat goal observations, success = goal
  reached), so the reported success rate is the task's own.

Configs: `configs/td3/tasks_v2/puck_goal{,_vel}_{drlong,rma}_{3p,full}.yaml` (+ the
phase-2 files under `configs/rma/tasks_v2/`); first use in the sysid-v2 campaign
([`2026-09-19_01-50`](../../scratch/experiments/2026-09-19_01-50_sysid-v2-compiled-params-policy-campaign.md)).

## Real-robot deployment (not wired)

The deployable policy is `AdaptedActor` = φ + π with a 50-step
`StepHistoryBuffer` (`scripts/rma/history.py`) reset at every episode start.
`scripts/td3/extras/async_td3_real_eval.py --agent` dispatch
(`scripts/td3/helper/real_eval_agents.py`) is the place to add an `rma` agent
that wraps `AdaptedRMAAgent` from `scripts/rma/evaluate.py`; it needs the
per-step latest-frame features and the action fed back after every step, which
the eval agent already does in sim.
