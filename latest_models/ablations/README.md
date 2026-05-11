# Ablation deployment models

Sixteen TD3 policies trained for the CoRL-2026 zero-shot sim2real ablation
study (`paper/main.tex` §Ablations:zeroshot). Each subdirectory contains:

- **`training_state.pth`** — full TD3 training-state checkpoint (actor + targets +
  Q networks + optimizer + replay + RNG + saved `args`). Suitable for real-world
  rollout via `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_eval.py`
  or further fine-tuning via `async_td3_real_modular.py`.
- **`README.md`** — 1-2 sentence high-level description + standardized detailed
  config block (always-on defaults, knobs ON, knobs OFF, source paths, training
  metric at the picked checkpoint).

Source-checkpoint convention:
- 15 ablations trained to a 700k budget — picked **`checkpoint_675000`** (highest
  available; the trainer's off-by-one means there's no `checkpoint_700000`).
- `paramrand_pm25` trained to 2M — picked **`checkpoint_1000000`** (rolling-5 mean
  peaked there at 132.7 with a single-ckpt high of 145.5; the trajectory plateaus
  around 1M and does not improve meaningfully through 2M).

| Folder | Type | Source step | Summary |
|---|---|---:|---|
| `baseline` | baseline | 675,000 | Canonical TD3 recipe with every randomization knob at its default sysid+DR setting. |
| `sysid_off` | single-knob | 675,000 | Reverts the 5 sysid-tuned physics parameters back to legacy off-the-shelf values to test how much real-world system identification matters for transfer. |
| `no_paddle_puck_strength` | single-knob | 675,000 | Disables paddle-puck collision STRENGTH randomization (impulse magnitude × U[0. |
| `no_paddle_puck_direction` | single-knob | 675,000 | Disables paddle-puck collision DIRECTION randomization (±10° impulse-direction cone). |
| `no_wall_direction` | single-knob | 675,000 | Disables WALL collision direction randomization (±10° puck-wall bounce-angle cone). |
| `no_action_attenuation` | single-knob | 675,000 | Disables stochastic action-force attenuation (the canonical setup randomly drops commanded paddle force to 25-75% with 30% probability per step). |
| `start_100_near_top` | single-knob | 675,000 | Starts every episode with the puck near the top of the table (puck_spawn_near_paddle_prob=0). |
| `start_100_near_paddle` | single-knob | 675,000 | Starts every episode with the puck near the paddle (puck_spawn_near_paddle_prob=1. |
| `no_puck_noise` | single-knob | 675,000 | Disables additive Gaussian puck-position observation noise (std=0. |
| `no_occlusions` | single-knob | 675,000 | Disables random puck-observation occlusions (the canonical setup drops puck observations at 2. |
| `no_obs_delay_randomization` | single-knob | 675,000 | Disables per-step JITTER on the 25 ms observation delay (delay mechanism stays on, fixed at 25 ms). |
| `all_sysid_no_rand_v2` | isolation | 675,000 | Sysid params kept at best-fit values, but all engineered randomization OFF (paddle-puck strength + direction, wall direction, action attenuation, puck noise, occlusions, delay-jitter). |
| `only_obs_noise_occlusion` | isolation | 675,000 | Isolation study: only `puck_noise` + `enable_random_occlusions` ON; all collision randomization (×3), action attenuation, and delay-jitter OFF. |
| `only_action_attenuation` | isolation | 675,000 | Isolation study: only `enable_action_force_attenuation` ON; all collision randomization (×3), puck noise, occlusions, and delay-jitter OFF. |
| `only_action_attenuation_obs_noise_occlusion` | isolation | 675,000 | Isolation study: action attenuation + puck noise + occlusions ON; collision randomization (×3) and delay-jitter OFF. |
| `paramrand_pm25` | paramrand | 1,000,000 | Physics-parameter domain randomization: paddle_density / puck_damping / gravity are drawn uniform within ±25% of their sysid values per episode reset. |

---

## Reproduction

Each model's `README.md` lists the exact `Sim config` and `TD3 args` paths used
to train it. To reproduce a run:

```bash
.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \
  --args-file <td3 args yaml from the README>
```

(For `paramrand_pm25`, the entrypoint is `td3_training_dr` instead of `td3_training`.)

The original 700k continuation runs each `full_resume`d from a 500k baseline run
in `runs/td3/zeroshot_ablations/...` — see the corresponding `model_path:` field
in their TD3 args YAML.

## Source experiments (background context)

- `notes/scratch/experiments/2026-05-05_17-38_zero-shot-sim2real-ablations.md` —
  the 500k base sweep (12 single-knob ablations; `no_obs_delay` failed to train
  due to env-coupling bug, was replaced in the 700k extension by
  `no_obs_delay_randomization`).
- `notes/scratch/experiments/2026-05-09_18-50_zeroshot-ablations-700k.md` — +200k
  continuation sweep producing the 12 `*_extend` runs (means 88-122 at 675k).
- `notes/scratch/experiments/2026-05-10_*_isolation_*.md` — 3 fresh isolation runs
  at 700k (only_obs_noise_occlusion 94, only_action_attenuation 134,
  only_action_attenuation_obs_noise_occlusion 88).
- `notes/scratch/experiments/2026-05-10_*_paramrand_2M.md` — physics-parameter DR
  2M run (peak 145.5 at 1M, back-half plateau ~118).

(Some experiment writeups may not yet exist if this folder was built before the
matching writeup landed.)
