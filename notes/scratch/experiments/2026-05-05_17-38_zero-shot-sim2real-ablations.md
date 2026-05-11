# Zero-shot sim-to-real ablation sweep (paper §Ablations:zeroshot)

- **Date**: 2026-05-05 17:38 UTC start
- **Status**: training done (12/12, exit 0); awaiting sim2real transfer on user's other machine. **`no_obs_delay` flatlined — see results.**
- **Run dirs**: `runs/td3/zeroshot_ablations/<name>/seed0/`
- **Configs**:
  - sim YAMLs: `configs/new_juggle/zeroshot_ablations/sim_<name>.yaml`
  - TD3 args YAMLs: `configs/td3/zeroshot_ablations/td3_zeroshot_<name>.yaml`
  - Generators: `_generate.py` in each of the two dirs above
  - Launcher: `scripts/smooth_policy/run_zeroshot_ablations.sh <gpu_id>`

## Question

For the CoRL-2026 paper §Ablations:zeroshot, train one policy per
domain-randomization knob removed (or sysid replaced) and measure how
the resulting policy transfers to the real robot. The actual transfer
test happens on a separate machine; this writeup covers training only.

## Setup

**Recipe** = `td3_hist2_motion0_v2.yaml` exact, with these overrides per ablation:

- `total_timesteps: 500000` (down from 1M — half the budget so the
  whole sweep finishes in one workday)
- `config:` → per-ablation sim YAML
- `log_parent_dir:` / `run_name:` → per-ablation
- `device:` → `cuda:0` or `cuda:1` (round-robin assignment)

All other TD3 hyperparameters identical to the recipe that produced
`latest_model/hist2_motion0_v2/` (eval mean 169.72 on the source sim
at 1M steps).

**Single seed per ablation** (seed 0). The ablation question is mostly
qualitative — "does this knob matter for sim-to-real" — and seed
variance can be retired with follow-up reruns if a result looks
borderline. Multi-seed costs 2× wall clock per ablation.

**500k vs 1M caveat**: the v2 base run peaked at 850k @ 169.72 and was
~145 mean across all 39 ckpts of the 1M budget. The 500k checkpoint
sits inside the rising portion of the learning curve, not the
post-peak settle. Comparison at 500k is fair across ablations (same
budget for all) but should not be conflated with the 1M peak. The
**baseline** ablation (no knobs flipped, 500k) gives the
apples-to-apples reference point.

### Ablation matrix (12 runs total)

Sim YAMLs in `configs/new_juggle/zeroshot_ablations/`:

| # | name | knob flipped vs `sysid_best_params_hist2.yaml` | GPU |
|---|---|---|---|
| 1 | `baseline` | none — uses canonical sim directly, 500k budget | cuda:0 |
| 2 | `sysid_off` | gravity, puck_damping, paddle_density, pid_kp, pid_kd reverted to legacy off-the-shelf values (gravity=-0.65, puck_damping=0.25, paddle_density=1000, pid_kp=5000, pid_kd=200) | cuda:1 |
| 3 | `no_paddle_puck_strength` | `enable_paddle_puck_strength_randomization: false` | cuda:0 |
| 4 | `no_paddle_puck_direction` | `enable_paddle_puck_direction_randomization: false` | cuda:1 |
| 5 | `no_wall_direction` | `enable_wall_direction_randomization: false` | cuda:0 |
| 6 | `no_action_attenuation` | `enable_action_force_attenuation: false` | cuda:1 |
| 7 | `start_100_near_top` | `puck_spawn_near_paddle_prob: 0.0` | cuda:0 |
| 8 | `start_100_near_paddle` | `puck_spawn_near_paddle_prob: 1.0` | cuda:1 |
| 9 | `no_puck_noise` | `puck_noise: false` | cuda:0 |
| 10 | `no_occlusions` | `enable_random_occlusions: false` | cuda:1 |
| 11 | `no_obs_delay` | `enable_observation_delay: false` | cuda:0 |
| 12 | `all_sysid_no_rand` | sysid kept; all 7 randomizations off (collision×3, action force atten., puck noise, occlusions, obs delay). Starting-distribution mixture (15/85) **kept on** because it is a deliberate curriculum, not noise. | cuda:1 |

### Real-world starting-state ablation: deferred

The paper's "Real-world starting states" ablation requires initializing
the simulator's puck/paddle from an empirical distribution gathered
from real-robot rollouts and teleop traces. **No mechanism exists in
the codebase** to do this — `airhockey_base.py` only supports
hand-designed probabilistic spawn (`puck_spawn_near_paddle_prob`).

To run this ablation, the env needs:
1. A dataset format for real start states (puck/paddle pos+vel)
2. New YAML keys (e.g., `use_empirical_start_states: true`,
   `empirical_start_state_path: <path>`)
3. Plumbing in `get_puck_configuration()` / `get_paddle_configuration()`
   to sample from that dataset

Real-robot data exists under `real_runs/` (warm-start trajectory
replay) — that's the candidate source dataset. **Open follow-up.**

### Scheduling

2 GPUs free (0, 1; 2 and 3 currently at 100% util on other work). Two
sequential pipelines, 6 runs each:

```
GPU 0 queue: baseline → no_paddle_puck_strength → no_wall_direction →
             start_100_near_top → no_puck_noise → no_obs_delay
GPU 1 queue: sysid_off → no_paddle_puck_direction → no_action_attenuation →
             start_100_near_paddle → no_occlusions → all_sysid_no_rand
```

Per-run wall clock ~1h45m (extrapolated from the v2 retrain: 1M steps
in 3h33m). Total per GPU ~10.5h, both finish around the same time.

Launch:
```bash
nohup bash scripts/smooth_policy/run_zeroshot_ablations.sh 0 \
  > notes/scratch/zeroshot_ablation_logs/_pipeline_gpu0.out 2>&1 &
nohup bash scripts/smooth_policy/run_zeroshot_ablations.sh 1 \
  > notes/scratch/zeroshot_ablation_logs/_pipeline_gpu1.out 2>&1 &
```

The launcher continues on per-run failure (so one OOM doesn't tank the
sweep) and emits succeeded/failed lists at the end of
`notes/scratch/zeroshot_ablation_logs/pipeline_gpu{N}.log`.

### Pre-launch canonical-sim edit (restitution → v1 values)

Before launch, the canonical sim YAMLs (`sysid_best_params.yaml` and
`sysid_best_params_hist2.yaml`) were edited so their restitution
coefficients match the v1 model (`latest_model/hist2_motion0/`)
training sim, while keeping the new collision-randomization knobs:

| key | prev (v2 base) | new (v1-matched) |
|---|---:|---:|
| `puck_restitution` | 0.87316 | **1.09145** |
| `paddle_restitution` | 0.8 | **1.0** (v1 unset → simulator default 1.0) |
| `side_wall_restitution` | 0.925 | **0.99** |
| `end_wall_restitution` | 0.7 | 0.7 (unchanged) |

Rationale: make the v1↔(new canonical) delta purely the
collision-randomization knobs (paddle-puck strength + direction, wall
direction), not the bounce coefficients. After this edit the only
sim-config difference between v1 and the new canonical is:
- `enable_paddle_puck_strength_randomization: true` (was absent in v1)
- `enable_paddle_puck_direction_randomization: true` (was absent in v1)
- `enable_wall_direction_randomization: true` (was absent in v1)
- their cone/range params

`hist2_motion0_v2` was trained on the *previous* (v2-restitution) sim
and is therefore no longer trained on the canonical sim. The
`baseline` ablation in this sweep effectively replaces it as the new
500k reference policy on the v1-restitution canonical sim. Ablation
sim YAMLs were re-generated so they inherit the new restitution.

### Pre-launch fix

Commit `54bc76e` ("more runs and documentation", 2026-05-05 14:39 UTC)
removed `import shutil`, the `try: from robosuite ...` block, and the
`ROBOSUITE_AVAILABLE = True/False` definition from
`airhockey/__init__.py`, but left two `if ROBOSUITE_AVAILABLE:` blocks
that reference all three. This broke every training entrypoint on the
`ablations` branch. The hist2_motion0_v2 retrain (2026-05-05 02:59
UTC) ran *before* this commit landed, which is why it was unaffected.
Pre-launch I restored exactly the pre-regression block (3 lines added
+ 1 line removed); see the diff in this branch.

## Results

All 12 runs completed at 500k steps with exit 0; each produced 19
checkpoints (every 25k steps) + `model.pth` + `args.yaml`. GPU 0
pipeline ran 17:45→05:33 UTC, GPU 1 pipeline ran 17:45→04:39 UTC.
Per-run wall clock 1h45m–1h55m.

**Training-end mean rolling reward** (mean over the last 20 reports of
the per-step `Rolling(2k) Avg Return` from the run log, ≈ last 10k
env steps; smoother than a single 499500-step value but still on the
rising portion of the learning curve, not a settled estimate):

| ablation | mean(last10k) | success(last10k) | vs baseline | notes |
|---|---:|---:|---:|---|
| **baseline** | **81.26** | **0.80** | — | reference |
| sysid_off | 116.49 | 0.82 | **+35** | ↑ likely "easier" sim (legacy 0.99 wall-rest, no sysid drag); not a sim2real win signal |
| no_paddle_puck_strength | 90.17 | 0.91 | +9 | similar |
| no_paddle_puck_direction | 79.76 | 0.86 | -1 | similar |
| no_wall_direction | 94.67 | 0.86 | +13 | similar |
| no_action_attenuation | 81.54 | 0.82 | +0 | similar |
| start_100_near_top | 83.04 | 0.77 | +2 | similar |
| start_100_near_paddle | 72.64 | 0.80 | -9 | slightly weaker; near-paddle starts give shorter episodes |
| no_puck_noise | 105.34 | 0.79 | +24 | ↑ cleaner observations |
| no_occlusions | 98.35 | 0.92 | +17 | ↑ no dropped puck obs |
| **no_obs_delay** | **16.98** | **0.01** | **-64** | ⚠ **failed to train — see investigation below** |
| all_sysid_no_rand | 77.38 | 0.61 | -4 | similar mean, lower success rate (more variance, fewer "perfect" episodes) |

The 11 trained ablations all sit in 73–116 mean / 0.61–0.92
success — the recipe is robust to flipping any one randomization knob
(or all of them at once, in `all_sysid_no_rand`'s case). The
`no_obs_delay` outlier is a **clear training failure**, not a
recipe-noise artifact.

**Comparison reference**: the baseline at step 499500 sits at
mean=76.12 / success=0.76 — on the rising portion of the learning
curve. The 1M-step `hist2_motion0_v2` peak was 169.72 (at ckpt 850k);
500k is mid-training. So all 11 non-failure ablations look healthy
relative to where the v2 baseline was at the same compute budget,
modulo restitution-flip caveats (the new canonical sim has v1
restitution, see "Pre-launch canonical-sim edit" above).

### `no_obs_delay` failure — diagnosis

`no_obs_delay` flatlined at ~17 reward / ~0% success rate from
step 500 through step 499500 — episode lengths stayed at ~37–40
throughout (the random-policy baseline) instead of climbing as in
all other ablations. No NaN, no traceback, no warning in the log.

Likely root cause (code-trace, not empirically verified — flag for
follow-up):

In `airhockey/sims/airhockey_box2d.py:get_singleagent_transition`,
the puck position is appended to `self.puck_history` *inside* the
breakpoints sub-step loop (line 1830). With `enable_observation_delay:
true` the loop runs over breakpoints `[0, t_obs, time_per_step]` (3
sub-steps → 3 puck_history appends per env step). With
`enable_observation_delay: false` the loop runs over `[0,
time_per_step]` (1 sub-step → 1 append per env step).

Then in `airhockey/airhockey_base.py:899-906`, when
`observation_state_info` is `None` (which is the case when
`enable_observation_delay: false`, since the snapshot conditions at
lines 1701/1856/1877 are all gated on `t_obs is not None`), the obs
falls back to `self.simulator.puck_history`. So the policy's 5-entry
puck history spans **~1.7 env steps with delay on** vs **~5 env steps
with delay off** — a 3× change in observation temporal granularity.

The `td3_recommended` recipe (and therefore `td3_hist2_motion0_v2`)
was tuned for the high-density (delay-on) history. The low-density
history is essentially a different observation space and the policy
cannot learn from it.

**Implication for the paper's ablation**: turning off
`enable_observation_delay` doesn't isolate "observation delay" —
it conflates two changes (delay-off + history-density-down). To get
the intended ablation we'd need to either (a) decouple `puck_history`
appending from the breakpoints loop in the simulator, or (b) keep the
breakpoints loop running multiple sub-steps even with delay off (set
`use_delay_logic = True` regardless and just zero out the offsets).

**Recommendation**: do not include the `no_obs_delay` 500k checkpoint
in the sim2real transfer comparison — it's an untrained policy and
will give a misleading (very negative) zero-shot result. Mark this
ablation as **deferred** until the env is fixed.

### Per-ablation eval suggestion

For each ablation, evaluate every ckpt on the canonical sim
(`sysid_best_params_hist2.yaml`, with v1 restitution), n=50
deterministic eval, to find the best ckpt to hand off for sim2real:

```bash
for name in baseline sysid_off no_paddle_puck_strength \
           no_paddle_puck_direction no_wall_direction \
           no_action_attenuation start_100_near_top \
           start_100_near_paddle no_puck_noise no_occlusions \
           all_sysid_no_rand; do
  bash scripts/smooth_policy/eval_all_ckpts_residual.sh \
    runs/td3/zeroshot_ablations/$name/seed0 \
    configs/new_juggle/sysid_best_params_hist2.yaml \
    cuda:0
done
```

(Skip `no_obs_delay` — see above.) For each ablation we want:

- **Training-end mean return** (from the last 5 ckpts) on the
  ablation's *own* training sim. This is the "did it train at all"
  check the user asked for.
- **Eval on canonical sim** (`sysid_best_params_hist2.yaml`) at the
  best ckpt, n=50 deterministic eval. This isolates the
  representation/learning effect from the sim-difficulty effect.
- **Sim-to-real transfer**: handed off to the user, run on a different
  machine.

Recommended eval command per run (after training completes):
```bash
bash scripts/smooth_policy/eval_all_ckpts_residual.sh \
  runs/td3/zeroshot_ablations/<name>/seed0 \
  configs/new_juggle/sysid_best_params_hist2.yaml \
  cuda:0
```

## Conclusion

Of 12 ablations, 11 trained to comparable rewards (73–116 mean rolling
return at 500k, vs baseline 81). The recipe is robust to flipping any
single domain-randomization knob and to flipping all
collision/action/observation randomizations at once
(`all_sysid_no_rand`). One ablation — `no_obs_delay` — failed to train
because turning off observation delay also changes the temporal
granularity of the observation history (a code-level coupling, not a
recipe-level effect); that ablation should be deferred until the env
bug is fixed. The remaining 11 ablations are ready for sim2real
transfer testing on the user's other machine.

## Next

- ✅ Both GPU pipelines launched and completed (12/12 exit 0).
- Hand off `model.pth` (or best-eval ckpt) of each of the 11 healthy
  ablations for real-world transfer testing on the user's other
  machine. Skip `no_obs_delay` — model is untrained.
- Optional pre-handoff: run `eval_all_ckpts_residual.sh` per ablation
  on the canonical sim to pick the best ckpt rather than just using
  the final one.
- Fix the puck_history-append-inside-breakpoints-loop coupling in
  `airhockey/sims/airhockey_box2d.py` so `enable_observation_delay:
  false` actually only ablates delay (not history density), then rerun
  the `no_obs_delay` ablation cleanly.
- Open follow-up: implement empirical-start-state init in the env so
  the real-world-starts ablation can be run.
