# Residual RL for `async_td3_real_modular` — implementation plan

Mirror the sim2sim residual recipe (`recency_top50`) inside the real-world async
TD3 pipeline. Source of truth for the recipe:
[`notes/docs/training/residual-rl-recipe.md`](../docs/training/residual-rl-recipe.md).
The sim implementation is the `full_checkpoint_load == "residual"` branch in
`scripts/smooth_policy/amp_history/amp_training/td3/td3_training.py`.

## Goal

A real-world run that:

1. Loads a frozen base actor from a sim checkpoint.
2. Trains a fresh, zero-init residual head + fresh critic on real data, with the
   recency_top50 data-balance recipe.
3. Saves checkpoints whose format drops in to the existing real-world rollout /
   eval scripts unchanged.

The orchestrator (`async_td3_real_modular.py`) should not need substantive
changes — the residual machinery lives behind `_init_sync_learner_state`, the
actor-sync hop, and `Args`, exactly the way the sim2sim path lives behind
`td3_training.py`'s `checkpoint_load_mode == "residual"` branch.

## Design (1:1 with sim2sim)

- `Args.full_checkpoint_load = "residual"`, plus `residual_*` knobs.
- `ResidualActor(base, residual_head)` from
  `scripts/smooth_policy/residual_agent.py` — drop-in `DeterministicAgent` API,
  frozen base + zero-init head + clamp.
- Optimizer over `actor.residual.parameters()` only.
- Critic + critic-target rebuilt from scratch.

Async real has two actor instances — learner's (gradient steps) and collector's
(rollout, periodically reloaded from the learner). Residual mode means **both**
are `ResidualActor`s wrapping the **same** base weights with **synced** residual
heads. The critic only lives on the learner side, so that part is identical to
sim2sim.

## Checkpoint format compatibility (sim vs real)

Confirmed before writing this plan:

| Artifact | Sim (`td3_training.py`) | Real (`async_td3_real.py`) | Compatible? |
|----------|------------------------|----------------------------|-------------|
| `model.pth` | `actor.state_dict()` | `actor.state_dict()` | **Yes — byte-identical**. `ResidualActor.state_dict()` from sim loads into a real-side `ResidualActor` with the same wrapper. |
| `actor_target.pth` / `qf{1,2}{,_target}.pth` | each `.state_dict()` | each `.state_dict()` | Yes — same. |
| `model_ema.pth` (optional) | `actor_ema.state_dict()` | (not yet written) | Will be same once added. |
| `training_state.pth` vital keys | `actor`, `actor_target`, `qf{1,2}{,_target}`, `success_replay_buffer`, `failure_replay_buffer`, `rng_states` | same | Yes. |
| `training_state.pth` non-vital keys | `q_optimizer`, `actor_optimizer`, `train_metrics`, plus sim-specific runtime fields (`obs`, `temporal_paddle_history`, `velocity_magnitudes`, …) | `q_optimizer`, `actor_optimizer`, `train_metrics`, plus real-specific (`collector_total_steps`, `run_elapsed_total_s`, `rolling50_*`) | **Drift.** Real's `_load_training_state_checkpoint` (`async_td3_real.py:809`) currently enforces `_NON_VITAL_TRAINING_STATE_KEYS` as all-or-nothing — a sim `training_state.pth` has *some* of those keys (q_optimizer, actor_optimizer, train_metrics) but not the real-only ones, so the validator raises `partial non-vital fields`. **Relaxing this** (see below) makes the formats fully cross-loadable. |

**Conclusion**: the format is the same where it matters for residual (`model.pth`
is byte-compatible). Two follow-ups, one for residual specifically and one
general:

- **Residual mode (specific)**: load the source via `torch.load(model_path)` +
  `extract_deterministic_state_dict`, mirroring `td3_training.py:853`. Residual
  ignores everything past the actor, so going through the full loader and
  stripping fields would be wasted work.
- **`_load_training_state_checkpoint` (general)**: drop the all-or-nothing
  non-vital check and default missing keys per-key. The downstream readers in
  `main` and `_init_sync_learner_state` are already per-key gated
  (`if "collector_total_steps" in training_state_checkpoint:` at L2995,
  `if "q_optimizer" in resume_checkpoint:` at L2592), so the validator is the
  only thing blocking partial loads. Replacing it with a log line listing which
  non-vital keys defaulted preserves diagnostic visibility without rejecting
  legitimate cross-source checkpoints. Vital-keys check stays strict. This is a
  general-purpose improvement that also unblocks future sim → real
  `weights_only` / `fine_tune` flows. (Aligns with the
  `feedback_loader_defaults` memory: per-key gates with safe defaults > strict
  shape validators.)

  Default mapping for the non-vital keys:

  | key | default |
  |-----|---------|
  | `q_optimizer`, `actor_optimizer` | absent (existing branch already handles) |
  | `learner_q_updates`, `learner_actor_updates` | `0` |
  | `train_metrics` | `{}` |
  | `collector_total_steps` | `0` |
  | `run_elapsed_total_s` | `0.0` |
  | `rolling50_*_values`, `rolling50_estop_episode_flags` | `[]` |

  No call-site changes needed — the readers already coerce these via
  `_coerce_float_list` / `if key in dict` gates.

## Change list (file by file)

### 1. `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real.py`

This is where the shared infra lives — the modular entrypoint just imports.

a. **`Args` (~L463)** — add residual knobs, default to "off":

   - `full_checkpoint_load: Literal["full_resume","weights_only","fine_tune","residual"] = "full_resume"` (mirrors `td3_training.py:544`).
   - `residual_scale: float = 0.15` (recipe default).
   - `residual_weight_decay: float = 0.0`.
   - `residual_scale_end: float | None = None`.
   - `residual_ema_decay: float | None = None`.
   - `residual_action_l2: float = 0.0`.

b. **`_init_sync_learner_state` (~L2509)** — branch on `args.full_checkpoint_load`:

   - In `"residual"` mode, build the standard `actor` (which becomes the base),
     then load only the actor weights from `args.model_path` via
     `torch.load(args.model_path)` + `extract_deterministic_state_dict`
     (existing helper at L128); **skip** the rest of the resume path (no
     replay/optimizer/runtime restore) — i.e. the function should treat
     `resume_checkpoint=None` for runtime fields when in residual mode.
   - Build `residual_online` / `residual_target` `DeterministicAgent`s with
     `action_scale=args.residual_scale`, zero-init heads via
     `zero_init_residual_head` (already exported from `residual_agent.py`).
   - Wrap as `actor = ResidualActor(base, residual_online, action_low, action_high)`,
     `actor_target = ResidualActor(actor.base, residual_target, …)` —
     share the same frozen base instance, matching `td3_training.py:885`.
   - `actor_optimizer = Adam(actor.residual.parameters(), lr=args.policy_lr, weight_decay=args.residual_weight_decay)`.
   - Critic + critic-target stay fresh (already the case — built before any resume).
   - Optional EMA: if `residual_ema_decay` is set, build `actor_ema` as a third
     `ResidualActor` with a frozen-grad EMA copy of the residual head, and stash
     it on `LearnerRuntimeState` (new field).
   - Print the same "Residual mode: …" banner as `td3_training.py:908`.

c. **`LearnerRuntimeState` (~L2473)** — add `actor_ema: DeterministicAgent | None = None`.
   Optional EMA support; default `None` preserves existing behavior.

d. **`_run_sync_learner_iteration` (~L2627)** — two small additions:

   - After actor loss is computed but before backward, if
     `args.full_checkpoint_load == "residual"` and `args.residual_action_l2 > 0`,
     add `λ * mean(residual_action²)` to actor_loss
     (mirrors `td3_training.py:1928`).
   - After `actor_optimizer.step()`, if `state.actor_ema is not None`, apply EMA
     update on `state.actor_ema.residual` parameters from `state.actor.residual`
     (mirrors `td3_training.py:1937`).
   - **Skip** residual_scale annealing for v1 (real runs aren't time-bounded the
     way sim2sim is — see Open Questions §1).

e. **`main` (~L2933)** — when `args.full_checkpoint_load == "residual"`:

   - Skip `_load_training_state_checkpoint` (residual ignores everything past
     the actor; load via `torch.load(model_path)` +
     `extract_deterministic_state_dict` inside the residual init branch).
   - Skip `load_replay_from_checkpoint` (recipe relies on warm-start replay,
     not a stale source replay — see Open Questions §3).
   - Hard-fail if `model_path is None`.
   - Hard-fail if `args.eval_mode` is set (mirror sim safety check).

f. **`_load_training_state_checkpoint` (~L809)** — relax non-vital validation
   (general-purpose, not residual-specific):

   - Keep the vital-keys strict check.
   - Replace the `partial non-vital fields` raise with a per-key default fill.
     If a non-vital key is missing, set the dict entry to its safe default
     (table above) and log a single line listing which keys were defaulted.
   - No call-site changes — `main` and `_init_sync_learner_state` already
     per-key gate these fields.
   - This also unblocks future sim → real `weights_only` / `fine_tune` resumes.

f. **`_save_async_checkpoint` (~L300)** — no change. `actor.state_dict()` already
   serializes the full `ResidualActor` (base + residual + buffers); this matches
   how sim2sim writes `model.pth`. Verified by inspection.

g. **`_build_collector_actor` helper** — extract the actor-construction block
   (`async_td3_real.py:1320s`, `async_td3_real_modular.py:649`) into a shared
   helper. In residual mode it returns a `ResidualActor` wrapping a zero-init
   residual head; the existing `state_dict()` copy from `learner_state.actor`
   then populates both base and residual weights identically.

### 2. `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_modular.py`

The collector instantiates an actor at L649. One change:

- Replace the inline `actor = DeterministicAgent(...)` + `state_dict` copy block
  with a call to the new `_build_collector_actor(args, train_args, learner_state, ...)`
  helper. The helper handles the residual-vs-non-residual branch.
- The per-actor-sync reload at L897 stays as-is (it's a `state_dict()` copy —
  works for `ResidualActor` because `nn.Module.state_dict` handles nested
  submodules transparently).

### 3. New config: `scripts/smooth_policy/amp_history/configs/td3_real_world/td3_residual.yaml`

Modeled on `td3_sim2sim_residual.yaml` but with real-world flavoring from
`td3_online.yaml`:

- Recipe defaults: `success_top_fraction: 0.5`, `q_weight_decay: 0.001`,
  `residual_scale: 0.15`, `q_lr: 0.0003`, `q_updates: 4`, `per_*` PER on.
- `recent_episode_window_size: 250` (real default, matches `td3_online.yaml`).
- `full_checkpoint_load: "residual"`, `model_path: <source sim ckpt>`,
  `config: configs/real_configs/rollout_config.yaml`.
- Real-world-only knobs from `td3_online.yaml`:
  - Warm-start dirs (`warm_start_hdf5_dirs`, `warm_start_hdf5_recursive`).
  - `replay_source_priority: "warmstart_only"` — residual MUST NOT inherit a
    stale checkpoint replay.
  - `load_replay_from_checkpoint: false`.
  - `checkpoint_root_dir`, simulator-space exploration magnitudes (real-tuned).
- `enable_periodic_checkpointing: true` with
  `checkpoint_every_successful_online_episodes` chosen to land ~10 checkpoints
  over the budget — same per-checkpoint-eval workflow the recipe doc requires.
- Drop `total_timesteps` (irrelevant for the async loop) — runs are bounded by
  wall clock / smoke-test seconds / operator stop.

### 4. Tests

Mirror the existing test pattern at
`scripts/smooth_policy/amp_history/amp_training/td3/tests/test_async_td3_real_args_mapping.py`:

- `test_residual_mode_init_builds_residual_actor`: stub `torch.load`, call
  `_init_sync_learner_state` with `full_checkpoint_load="residual"`, assert
  `state.actor` is a `ResidualActor`,
  `state.actor.residual.actor_mean_head.weight.abs().sum() == 0`, optimizer
  param_group params == residual params, critic is fresh (initialized weights,
  not loaded).
- `test_residual_collector_actor_wraps_learner_actor`: end-to-end build of the
  collector actor in residual mode, assert it is a `ResidualActor` and a
  `state_dict()` round-trip from the learner reproduces the residual head
  weights.
- `test_residual_args_yaml_round_trip`: load `td3_residual.yaml` via
  `_build_args_file_defaults` and assert the residual knobs reach `Args`.

### 5. Doc

Add a short section to `notes/docs/training/residual-rl-recipe.md` titled
"Real-world residual" that:

- Points at `td3_residual.yaml` and the modular entrypoint.
- Notes the one functional delta vs sim2sim: warm-start replay (HDF5 dirs)
  replaces the synthetic "fill from scratch" pattern.
- Repeats the per-checkpoint-eval requirement (existing eval scripts work
  because `model.pth` already saves the wrapped `ResidualActor`).

## Open questions / risks

1. **`residual_scale_end` annealing** wants a known total-step budget. Real runs
   aren't time-bounded the way sim2sim is. Either drive it off a config field
   (`residual_anneal_total_steps`) that you set explicitly, or keep
   `residual_scale_end=None` (no anneal) for v1 and revisit. v1: no anneal.
2. **Source checkpoint format** — verified above: `model.pth` is byte-compatible
   sim ↔ real. The residual branch loads the source via `torch.load(model_path)`
   + `extract_deterministic_state_dict` and does NOT go through
   `_load_training_state_checkpoint` (which would reject sim sources due to
   non-vital-key drift).
3. **Replay source** — `replay_source_priority: "warmstart_only"` is the only
   safe default for residual real (you don't want to seed the new critic with a
   base-policy replay from an obsolete dynamics distribution).
4. **Rollout / eval scripts** — `_save_async_checkpoint` writes `model.pth` as
   the wrapped `ResidualActor` state_dict. Downstream rollout
   (`scripts/real/rollout_new.py`, eval drivers) needs to rebuild the same
   `ResidualActor` shell to load it. Audit those scripts for residual checkpoint
   loading after the training-side change lands. Likely needs a small
   "load `model.pth` as `ResidualActor` if base+residual keys are present"
   branch, mirroring whatever the sim2sim eval driver does today.

## Order of implementation

1. Relax `_load_training_state_checkpoint` non-vital validation (independent of
   residual; ship first and verify existing real → real resumes still pass).
2. Add `Args` fields + arg validation in `main`.
3. Residual branch in `_init_sync_learner_state`.
4. `_build_collector_actor` helper + use it from the modular entrypoint.
5. Tests (1)–(3) above, plus a test that a partial-non-vital
   `training_state.pth` loads without raising and produces the documented
   defaults.
6. New `td3_residual.yaml`.
7. EMA + `residual_action_l2` in `_run_sync_learner_iteration`.
8. Doc update.
9. Audit real-world rollout / eval scripts for residual checkpoint loading.
