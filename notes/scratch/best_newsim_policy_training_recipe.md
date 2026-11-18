# How the best new-sim policy is trained (`newsim_scratch_cap7000_swa`)

Compiled 2026-11-05 from the saved training args
(`latest_models/ceiling/newsim_scratch_cap7000_swa/seedN/args_actor_training.yaml`), the code, and the experiment note
[`2026-11-03_08-19`](experiments/2026-11-03_08-19_new-sim-from-scratch-convergence-cap7000.md), which is the source of
truth for the results. The training spec is `configs/ceiling/scratch_newsim.yaml`.

## Summary
- **What it is:** TD3 trained from scratch on the new sim. The actor sees a 50-step history; the critic also sees the
  true sim state (asymmetric actor-critic).
- **Training details:**
  - Training-only reward shaping (the `term03` recipe).
  - Episode cap of 7,000 steps.
  - Trained until convergence (5–13M steps).
  - The final actor is a stochastic weight average (SWA) of the checkpoints from the last 40 % of training.
- **Result:** 94.0 ± 1.5 % of 100 eval episodes reach 10k steps (mean ± std err over 5 seeds), vs 82.2 ± 3.7 % for the
  raw final checkpoint.
- **Not used:** CQL, critic ensembles, residuals, warm starts.

## Environment
- **Sim:** the new sim, `configs/new_juggle/sysid_v2_hist4_realplant.yaml` (sysid v2 physics, PID kp 9038 / ki 1929
  / kd 0, 4-step PID-target moving average, 25 ms puck observation lag, no observation delay).
  - It adds **2 control steps of action latency**, applied outside the sim (`action_latency_steps: 2`).
  - The generated copy is `configs/ceiling/generated/scratch_newsim/cap7000_sim.yaml`. It differs only in
    `max_timesteps: 7000`.
- **Task:** `puck_juggle_upper_half_reward`. Episodes end when the puck hits the bottom, passes the paddle, or reaches
  the enemy goal.
- **Domain randomization:** resampled at every reset, ±25 % around sysid.
  - paddle_density 2250–3750
  - puck_damping 0.0825–0.1375
  - gravity −0.9125 to −0.5475
- **Sensor noise:** puck noise std 1 cm; random occlusions at a 5 % rate.
- **Reward shaping** (training only, from the true sim state, in `scripts/rma/motion_shaping.py`; evaluation uses
  the unshaped reward):
  - `edge`: −0.5 per step, ramping in within 5 cm of the top edge of the paddle workspace.
  - `weak`: −3 per contact if the paddle's forward speed is under 0.3 m/s.
  - `pout`: up to +2 per contact for puck exit speeds from 1.5 to 3.5 m/s.
  - `term`: the episode ends if contact happens within 3 cm of the top edge.
  - `actor_pre_tanh_penalty`: 0 (the "nopretanh" part of the recipe name).

## Architecture
All networks use the same `ResidualMLPTrunk` (from `scripts/td3/agent.py`):
- Each residual block is 4 × [Linear → LayerNorm → SiLU] with a skip connection.
- "2 layers × 64" means 2 such blocks, 64 wide.

**Actor** (`HistoryActor`; trained as `PrivilegedHistoryActor`, which only skips the privileged slice of the
observation):
- **Inputs:**
  - x_t: the standard 30-dim history observation (5 frames of paddle and puck [x, y, valid]).
  - a_{t−1}: the previous action (2-dim).
  - z: an 8-dim latent from the history encoder.
- **History encoder** (the RMA adaptation-module architecture, trained end to end by the RL loss):
  - Input: a 50-step window of per-step features. Each feature is the latest paddle [x, y, valid], the latest puck
    [x, y, valid], and the action (8 dims).
  - A 2-layer MLP embeds each step into 32 dims (ELU).
  - Three 1-D convs, 32 channels, kernels 8/5/5, strides 4/1/1.
  - A linear layer maps the flattened output to z (8 dims).
- **Trunk:** input 30 + 2 + 8 = 40 dims, then the 2 × 64 residual blocks.
- **Output:** a linear head, then tanh, giving the 2-dim action in [−1, 1].
- **Deployable:** it never reads the true state, so its weights load into a plain `HistoryActor`.

**Critics:**
- 2 critics (`num_critics: 2`), standard TD3 clipped double-Q. There is no ensemble and no REDQ subset.
- `TD3QNetwork`: the 2 × 64 residual trunk, then a scalar head initialized with std 0.01.
- Input: x_t (30), the flattened 50 × 8 window (400), the **true sim state** (8: noise-free, undelayed puck and paddle
  position and velocity), and the action (2).
- Q is learned in a transformed space: the Bellman target is passed through h(·) (eps 1e-3), and h⁻¹ maps back to
  returns.

## Losses
- **Critic target:** y = r + γ(1 − done) · min(Q1_target, Q2_target)(s′, ã′), computed in return space and then
  passed through h(·).
  - ã′ is the target action plus TD3 smoothing noise (std 0.2, clipped to ±0.5).
  - **No CQL** (`cql_alpha: 0`). No weight decay.
- **Actor loss:** −(1 − γ) · Q1(s, π(s)), using critic 1 only and converted back to return space. There are no extra
  penalties (action-delta 0, pre-tanh 0).
- **Target networks:** Polyak averaging with τ = 0.005, after every 10th critic update. The actor target is updated at
  the same time.

## Hyperparameters

| | Value |
|---|---|
| γ | 0.995 |
| Learning rates | actor 3e-4, critic 1e-3 (Adam) |
| Batch size | 512 |
| Update schedule | every 32 env steps: **8 critic updates, then 2 actor updates** |
| Updates per env step | 1/4 critic and 1/16 actor; about 2.0M critic / 0.5M actor updates for a mean 8.2M-step run |
| Learning starts | 20,000 steps (random actions before that) |
| Replay | one 1M-transition buffer with prioritized replay (α 0.6, β 0.4 → 1 over 200k steps); each batch is 70 % prioritized, 30 % uniform |
| Exploration | Gaussian noise with std 0.1, plus 5-step motion primitives (15 % → 5 % chance over 200k steps) |
| Envs | 1 (CPU rollout, CUDA-graph updates) |
| Episode cap | 7,000 steps (cap 2000 gave 63.4 ± 18.6 %) |
| Checkpoints | every 100k steps |

## Training length, SWA, and evaluation
- **Converged** means `q_loss`, `actor_loss`, and the average return each change by less than 10 % between the two
  halves of the last 1M steps, and the return is above 50 (the run has taken off).
  - This was checked at 5M steps, then every 2M.
  - Seeds stopped at 5M / 7M / 13M / 9M / 7M. Seed 2 sat at the floor until about 6M.
- **SWA:** a uniform average of the actor checkpoints in [0.6M, M], where M is the stopping step.
- **Critics for the SWA actors:**
  - Fitted afterwards, with the actor frozen.
  - Warm-started from the last raw checkpoint and trained for 2M more steps on data from the SWA actor (in
    `runs/ceiling/swa_critic_fit/`).
  - Their min(Q1, Q2) is about 50 % below the Monte Carlo return. The raw checkpoints show the same, so this comes from
    the recipe, not the fit.
- **Evaluation:** `scripts.analysis.new_sim_eval`, 100 episodes with a 10k cap and seeds 1000000+, never used for
  selection.

| Seed | Stopped at | Reach 10k (SWA) | Reach 10k (raw checkpoint) |
|---|---|---|---|
| 0 | 5M | 96 ± 2 % | 68 ± 5 % |
| 1 | 7M | 88 ± 3 % | 83 ± 4 % |
| 2 | 13M | 94 ± 2 % | 89 ± 3 % |
| 3 | 9M | 96 ± 2 % | 87 ± 3 % |
| 4 | 7M | 96 ± 2 % | 84 ± 4 % |
| **Mean** | | **94.0 ± 1.5 %** | 82.2 ± 3.7 % |

## Things to know when reusing it
- **The critic can't be used on the robot:** it needs the true sim state. Real-robot fine-tuning would need a new critic
  (or a non-privileged one trained in sim).
- **Residual fine-tuning:** the residual code (`scripts/td3/residual_agent.py`) only wraps the plain 2 × 64 MLP actor.
  Wrapping this history actor needs code changes.
- **Update rate:** this recipe makes about 60× more critic updates per env step than the residual recipe (1/4 vs about
  1/250).
- **Actor loss scaling:** the sim actor loss is scaled by (1 − γ) = 0.005; the real-robot learner uses plain −Q. Any
  penalty weight therefore doesn't carry over between the two unchanged.
- **Training entrypoint:** `python -m scripts.rma.train_base_policy --args-file <generated cap7000_sN.yaml>`. The per-seed
  args are in `configs/ceiling/generated/scratch_newsim/`.
