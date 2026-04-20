# Residual RL for online sim2sim / sim2real fine-tuning — design & continuation plan

Handoff doc. If prompted with "continue working on residual RL" (or similar),
start here.

## Status

**Design stage.** Nothing is implemented yet. This doc captures:
- the canonical method we chose (and why),
- the concrete wiring into the existing TD3 code,
- sensible default hyperparameters,
- the first experiment to run.

No code has been written. First step is implementation (§3), not more design.

## 1. Method (canonical residual RL)

Both seminal papers propose the same action-space decomposition:

```
π(s) = clip( π_base(s) + π_residual(s), a_min, a_max )
```

- **π_base**: fixed, pre-trained (our sim-trained TD3 actor).
- **π_residual**: trained from scratch on the new env via standard RL.
- **Init**: residual output ≈ 0 at t=0, so initial behavior equals base policy
  (no regression when fine-tuning starts).
- **Critic**: trained from scratch on the new env. Pretrained Q is stale for
  the new dynamics, and RPL explicitly recommends not warm-starting it.

References:
- Silver, Allen, Tenenbaum, Kaelbling. *Residual Policy Learning.* arXiv:1812.06298 (2018).
- Johannink, Bahl, Nair, Luo et al. *Residual Reinforcement Learning for Robot Control.* ICRA 2019 / arXiv:1812.03201.

Both appeared December 2018 and use the same formulation. Silver et al. is the
more commonly cited "original" method; Johannink et al. is the robotics-
specific variant (hand-designed P-controller base + learned residual via
TD3/SAC). Our setup matches Johannink et al. structurally, except our base is
an RL-trained policy rather than a hand-designed controller.

## 2. Why this variant (not alternatives)

- **Action-space residual (chosen)**: the original, algorithm-agnostic form.
  Works with our existing TD3 unchanged except for the actor wrapper.
- Observation/latent residual: not canonical, not what the papers do.
- Joint fine-tuning of base + residual: RPL mentions it as an option, but the
  sim2real variant in both papers keeps base **frozen**. Frozen base is also
  much safer for real-robot deployment — worst-case behavior is bounded by
  `|π_residual|_∞`.

## 3. Implementation plan

### 3.1 New `ResidualActor` wrapper

New file, e.g. `scripts/smooth_policy/residual_agent.py`:

```python
import torch
import torch.nn as nn
from scripts.smooth_policy.deterministic_agent import DeterministicAgent

class ResidualActor(nn.Module):
    """π(s) = clip(base(s) + residual(s), -1, 1). Base frozen."""

    def __init__(self, base_actor: DeterministicAgent,
                 residual_actor: DeterministicAgent,
                 residual_scale: float = 0.25,
                 action_low: float = -1.0, action_high: float = 1.0):
        super().__init__()
        self.base = base_actor.requires_grad_(False).eval()
        self.residual = residual_actor
        # Override the residual's action_scale buffer so its output lies in
        # [-residual_scale, +residual_scale]. Avoids an external multiply.
        self.residual.action_scale.fill_(residual_scale)
        # Zero the residual's output head so residual(s) == 0 at init.
        nn.init.zeros_(self.residual.actor_mean_head.weight)
        nn.init.zeros_(self.residual.actor_mean_head.bias)
        self.register_buffer("action_low",  torch.tensor(action_low))
        self.register_buffer("action_high", torch.tensor(action_high))

    def get_action(self, x):
        with torch.no_grad():
            a_base = self.base.get_action(x)
        a_res = self.residual.get_action(x)
        return torch.clamp(a_base + a_res, self.action_low, self.action_high)

    def get_action_mean(self, x):
        # Required by any caller that uses raw means (none in TD3 rollout).
        raise NotImplementedError("Residual actor exposes get_action only.")

    def forward(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            return self.get_action(x)
```

Note: the `DeterministicAgent` output head is `actor_mean_head` (see
`scripts/smooth_policy/deterministic_agent.py:53`), not `fc_last`.

### 3.2 Wiring into `td3_training.py`

Touch points in
`scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py`:

| Location | Change |
|---|---|
| line ~798-829 (checkpoint load) | Add a `residual` load mode. Load checkpoint actor weights into `base_actor`; instantiate a fresh `residual_actor` (same arch); wrap as `ResidualActor`. **Do not** load qf1/qf2 from checkpoint. |
| actor optimizer | `Adam(actor.residual.parameters(), lr=...)` — base params are already frozen via `requires_grad_(False)`, but still pass only residual params to the optimizer to avoid AdamW weight-decay weirdness. |
| line ~1162 (rollout) | No change. `deterministic_actor_action(actor, …)` already calls `actor.get_action(…)`, which the wrapper implements. |
| line ~1164 (exploration noise) | No change. Noise is added to the combined action, same as standard TD3. This matches Johannink et al. |
| line ~1747 (policy loss) | No change. `deterministic_actor_action(actor, sampled_policy_observations)` produces the combined action; gradient flows only through `residual` since `base.get_action` is under `torch.no_grad()`. |
| target actor | Wrap similarly: `ResidualActor(base_actor, residual_actor_target, …)` with the **same** (shared) frozen base. Soft-update only the `residual` submodule at tau step. |
| critic | No changes. Critic sees combined action; train from scratch. |

### 3.3 Config changes

New file `scripts/smooth_policy/amp_history/configs/td3/td3_residual.yaml`,
based on `td3_recommended.yaml` with:

- `model_path: <path to sim-trained checkpoint>`  # frozen base
- `load_mode: residual`                           # new mode to add
- `residual_rl: true`
- `residual_scale: 0.25`                          # main hyperparameter
- `learning_starts: 2000`                         # we already behave
- `exploration_noise: 0.05`                       # don't shred the base
- `buffer_size: 20000`                            # online FT is short
- `total_timesteps: 100_000`                      # FT budget, not full training
- Disable exploration primitives:
  `exploration_primitive_chance_pre_learning_starts: 0`,
  `policy_takeover_weight: 0`.
- `config:` still points at the **target env's** sim config (different dynamics
  than the base was trained on — that's the whole point).

### 3.4 Real-world rollout

`scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py:122`
calls `actor.get_action(policy_obs)`. Drop in a `ResidualActor` at load and it
works unchanged.

## 4. Default hyperparameters (first try)

| Param | Value | Source |
|---|---|---|
| `residual_scale` | 0.25 | Conservative start. Johannink et al. used full-scale residuals, but for sim2real stability begin small and anneal up if policy is under-expressive. |
| residual head init | zeros | RPL §3, Johannink §IV.B. |
| Base frozen | yes | Both papers' sim2real variant. |
| Critic init | from scratch | RPL §3. |
| Actor lr | 3e-4 (same as base training) | standard TD3 |
| Exploration noise | 0.05 (half of base training) | preserve base behavior early |
| Learning starts | 2k steps | base already explores the task well |
| Total timesteps | 100k | online FT budget |

## 5. First experiment: sim2sim validation

Before touching the real robot, validate the whole pipeline sim2sim.

**Base**: TD3 checkpoint trained on `sysid_best_params_hist4.yaml` (current
recommended default).

**Target sim**: perturbed physics. Suggested perturbation (one at a time,
then combined):
- gravity: -0.661 → -0.80 (heavier effective puck)
- puck_damping: 0.178 → 0.30
- paddle_density: 3000 → 1500

**Baselines to compare against**:
1. **Zero-shot**: run base policy on perturbed sim, no fine-tuning. Establishes
   the sim2sim gap.
2. **Full fine-tune**: resume TD3 on perturbed sim with `load_mode=fine_tune`
   (existing code path). This is the non-residual baseline.
3. **Residual RL**: this plan.
4. **From scratch**: TD3 on perturbed sim with no warm-start.

**Metrics**: return@100k on perturbed sim, tail10/tail50, sample efficiency
(timesteps to reach zero-shot base return on the original sim).

**Expected outcome** (from the literature): residual ≥ full fine-tune on
sample efficiency, both ≫ from-scratch, both ≥ zero-shot by a wide margin.

## 6. Open questions (decide during implementation, not now)

- Share base between online actor and target actor (saves memory and
  compute) or duplicate? Share — base is frozen, target-smoothing only
  applies to the learnable part.
- Should `residual_scale` be learned (per-dim gate) rather than a constant?
  RPL stayed with a constant. Start with constant; only revisit if residual
  is consistently saturating.
- Where to put the policy-smoothness / motion-reward penalties — on combined
  action or residual alone? **Combined action**, because that's what the
  robot actually executes. The existing motion reward is computed from env
  state, so no change needed.

## 7. What to do when resuming this work

1. Re-read §3 (implementation plan).
2. Implement `ResidualActor` (§3.1).
3. Add `load_mode: residual` to `td3_training.py` (§3.2).
4. Create `td3_residual.yaml` (§3.3).
5. Run the sim2sim validation from §5. First run: base + one perturbation
   (e.g. gravity → -0.80), residual vs. full-fine-tune vs. zero-shot.
6. Only after sim2sim works, plan the real-robot run.

When implementation is non-trivial and stable, promote the design portion
(§1, §2, §3) to `notes/docs/training/residual-rl.md` and leave this scratch
doc as the experiment log + next steps.
