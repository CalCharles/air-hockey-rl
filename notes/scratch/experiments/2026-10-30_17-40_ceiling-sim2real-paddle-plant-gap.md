# Why the ceiling policies fail on the real robot: the sim paddle is ~2× too fast

- **Date**: 2026-10-30 17:40 CDT start
- **Status**: done. Retraining / fine-tuning results and the hist4 sim-vs-real check: [`2026-10-31_02-00_new-sim-retraining-and-finetuning.md`](2026-10-31_02-00_new-sim-retraining-and-finetuning.md).
- **Question (user)**: The ceiling policies (`strike_shaped`, `strike_term03`, `edgefix_privcritic_pretanh`) move the paddle slower than the original hist4 policies in sim, and slower still on the real robot, where they juggle poorly. The hist4 policies perform about the same in sim and real.
  - What causes the ceiling failure?
  - What is the sim → real slowdown factor?
  - Can it be modelled in the sim and trained on?
- **Data**:
  - real eval logs `data/ceiling_real_data/{strike_shaped,strike_term03,edgefix_privcritic_pretanh}/seed0/`
  - `data/ceiling_scaled_x2.8_y2.35/edgefix_privcritic_pretanh/seed0/data_20261030-160001`. The second run there, `data_20261030-162243`, has no episodes.
  - hist4 real logs `data/sysid_v2_real_workspace_original_hist4_ckpt1975000_newest_frame_maxr12_no_estop_count/hist4_seed{0..3}/`
- **Code** (all new, the simulator `airhockey/` is not modified):
  - `scripts/analysis/cmd_vs_realized.py`: commanded vs realized paddle motion, real logs and sim rollouts.
  - `scripts/analysis/replay_real_actions.py`: open-loop replay of logged real actions in the sim, paddle only. `--sim-override`, `--action-shift`.
  - `scripts/analysis/proxy_sim_eval.py`: policy eval on a modified sim. `--sim-override`, `--action-latency`, `--action-scale`.
  - `scripts/rma/action_latency.py` + `action_latency_steps` in `scripts/rma/train_base_policy.py` (default 0).
  - `configs/new_juggle/sysid_v2_hist4_realplant.yaml`: new file; `sysid_v2_hist4.yaml` is unchanged.
  - `target_config` key in `scripts/ceiling/sweep.py` specs.
  - `domain_random` added to the training-only keys of `scripts/ceiling/eval_target.py`.
- **Outputs**:
  - `runs/analysis/replay/`
  - `runs/analysis/cmd_vs_realized/sim_target.json`
  - `runs/analysis/proxy_sim_eval/<label>/results.json`

## Real-robot results (10 kept episodes each)

| policy | return | length | juggle success |
|---|---:|---:|---:|
| hist4 seed 0 / 1 / 2 / 3 | 135 / 73 / 98 / 82 | 223 / 134 / 185 / 135 | 50 / 40 / 40 / 30 % |
| edgefix_privcritic_pretanh seed 0 | 41 | 101 | 20 % |
| strike_shaped seed 0 | 73 | 163 | 40 % |
| strike_term03 seed 0 | 66 | 139 | 50 % |
| edgefix, actions × (2.8, 2.35) | 67 | 129 | 30 % |

- Every episode ended with `puck_hit_bottom`, except two `puck_low_motion_window_20`.
- The real loop runs at 49.1 ms per step (p95 52.7 ms), so step timing is not the problem.

## 1. Commanded vs realized, same policies in sim and real

Columns:
- "gain" is the steady-state y displacement per metre of commanded y step, from an FIR fit d_t = Σ h_k c_{t−k}. The x fit is unreliable because the 38 cm x workspace clips most x commands.
- "lag" is the step at which half of the response has arrived.

| policy | where | mean \|a\| x / y | saturated x | paddle speed mean / p90 | y gain | lag |
|---|---|---|---:|---|---:|---:|
| hist4 (4 seeds) | target sim | 0.80–0.84 / 0.65–0.71 | 54–65 % | 0.91–1.14 / 1.76–1.95 m/s | 0.39–0.45 | 1 |
| hist4 (4 seeds) | real | 0.75–0.82 / 0.63–0.73 | 51–66 % | 0.34–0.43 / 0.70–0.83 m/s | 0.11–0.15 | 4 |
| ceiling (3) | target sim | 0.27–0.36 / 0.21–0.29 | 0 % | 0.42–0.64 / 0.85–1.27 m/s | 0.53–0.56 | 1–2 |
| ceiling (3) | real | 0.34–0.41 / 0.35–0.38 | 1 % | 0.24–0.26 / 0.50–0.55 m/s | 0.16–0.17 | 4 |

Observations:
- **Action distributions are unchanged sim → real** for both families (hist4 barely moves; the ceiling policies push slightly harder in real), but the realized paddle speed drops 2.4–2.6× for hist4 and 1.7–2.6× for the ceiling policies.
- **Every policy sees a ~3× lower gain and ~3 steps (150 ms) more lag on the real robot.** The plant gap is the same for both families; what differs is how the policies use the actions.
  - hist4 policies are bang-bang: more than half their x actions saturate at ±1.
  - Ceiling policies use small proportional actions (|a| ≈ 0.3) and almost never saturate.
- **The user's ~2.8× / 2.35× ratio is a policy difference, not the plant factor.** hist4 mean |a| ÷ ceiling mean |a| in sim is 2.2–3.1 (x) and 2.3–3.4 (y). It happens to be close to the plant factor.
  - Ceiling policies trained with `actor_pre_tanh_penalty: 0.001`, which penalises saturating actions. hist4 trained without it.

## 2. Open-loop replay: the plant mismatch, independent of the policy

Setup:
- Each real episode's logged `policy_action` sequence is fed to the sim paddle, starting at the real start pose. The puck is made a sensor so it cannot push the paddle.
- The table compares the sim paddle trajectory with the logged real one, over all 8 real runs (102 episodes).

| sim plant | real ÷ sim paddle speed | traj. error x / y | best real ≈ g · sim(t − lag) |
|---|---:|---|---|
| target config (`enable_observation_delay: true`, 2 × 25 ms Box2D steps per control step) | 0.46–0.52 | 10–14 / 26–34 cm | lag 3–4, g 0.3–0.4, r² ≈ 0.6 |
| obs delay off (one 50 ms Box2D step) | 1.13–1.23 | 3–4 / 6–12 cm | lag 2, g 1.0–1.3, r² ≈ 0.9 |
| obs delay off + 2-step action latency | 1.19–1.23 | 2.6–2.9 / 9–10 cm | lag 0, g 1.0–1.3, r² 0.84–0.94 |
| **obs delay off + 2-step latency + `pid_kp` × 1.2 (9038)** | **1.00–1.08** | **2.0–2.6 / 7–12 cm** | **lag 0, g 0.9–1.1, r² 0.85–0.96** |
| obs delay off + 2-step latency + `paddle_damping` 14 / 12 | 0.62–0.68 / 0.48–0.55 | 5–9 / 20–30 cm | worse |

**The slowdown factor:** on the same commands, the real paddle moves at **0.49 ± 0.01 × the training sim's paddle speed** (range 0.46–0.52 over 8 runs), i.e. the sim is about **2×** too fast. On top of that, the real response arrives **3–4 control steps (150–200 ms) later**.

**The cause is in the sim, not the robot.**
- With `enable_observation_delay: true` + `delay_seconds: 0.025` (every hist4 and ceiling config), each 50 ms control step is advanced as two 25 ms Box2D steps.
- pybox2d damps velocity by `v *= 1 − h · paddle_damping` per Box2D step: ×0.15 once at h = 50 ms, but ×0.575² = ×0.33 at h = 25 ms. The same PID force then gives about 2× (up to 3×) the steady-state speed.
- The PID sysid (`sysid/paddle_pid`) fit kp / ki / kd with the delay off (one 50 ms step). That fit, run the way it was fit, matches the real paddle to about 20 %; the obs-delay setting broke it. See also the memory note on this from 2026-10-29.
- The remaining ~2 steps of lag on the robot are real: servoL lookahead 0.2 s, and the command referenced to the robot's TCP target pose.
- One caveat: the logged real pose is read at the start of the step, before that step's command, so one of the two steps is the observation timing the policy experiences, not robot dynamics.
- The y residual (7–12 cm drift over an episode) is not explained; x, the striking axis, matches to 2–3 cm.

## 3. Policies on proxy sims: which part of the gap breaks which policy

Setup: 20 episodes, 1000-step cap, return mean ± std err ranges over policies. Ceiling = the 3 real-tested policies + the dribble recipe `final_privcritic_pretanh` seed 0.

| sim | hist4 seeds 0–3 | ceiling ×4 |
|---|---:|---:|
| target (fast paddle, no extra lag) | 123–175 | 632–751 (95–100 % reach 1000) |
| fast paddle + 2-step latency | **29–39** | 49–147 |
| real-matched paddle, no latency | 105–234 | **34–40** |
| obs delay off + 2-step latency (kp unchanged) | 71–120 | 29–36 |
| **real proxy: real-matched paddle + 2-step latency** (`sysid_v2_hist4_realplant.yaml` + `--action-latency 2`) | **105–160** (mean 128) | **33–52** |
| real proxy + actions × (2.8, 2.35) | — | 54–86 |
| real proxy, no latency, actions × (2.8, 2.35) | — | 62–85 |
| **real robot** | **73–135** (mean 97) | **41–73**; scaled edgefix 67 |

- **The proxy reproduces what the robot showed.**
  - hist4 keeps its performance (proxy 128 vs real 97).
  - Every ceiling policy collapses from ~700 to 33–52 (real 41–73).
  - Action scaling recovers part of it, in the proxy (edgefix 37 → 86) and on the robot (41 → 67).
  - The proxy is somewhat optimistic for hist4 and pessimistic for the ceiling policies.
- **The slow paddle alone breaks the ceiling policies** (34–40 with no latency). hist4 does not mind it (105–234).
- **Latency alone breaks hist4 on the fast paddle** (29–39): a bang-bang policy on a 2×-too-fast paddle overshoots when the command arrives late. On the slow paddle, hist4 tolerates the same latency.
- **Why hist4 survives the slowdown.**
  - It saturates more than half its x actions, so what it asks for is "as fast as possible toward the puck". That intent survives a 2× slower actuator.
  - Its sim performance was modest to begin with (~140 return, ~230 steps), so there was little precision to lose.
- **Why the ceiling policies do not.**
  - Their 10,000-step juggling is a precise, small-amplitude closed-loop behaviour (|a| ≈ 0.3, never saturated), calibrated to a paddle that moves ~2× farther and reacts 3 steps sooner than the real one.
  - On the real paddle every correction falls short and late, and the 50-step history policy sees action–response pairs it never saw in training.
  - Scaling the actions fixes part of the gain error but not the lag or the speed ceiling. The scaled actions saturate (33–37 % of steps on the robot), and the real paddle cannot reach the sim's strike speeds (real max ~1.1 m/s vs sim p90 1.2–1.9 m/s).
  - The anti-saturation `actor_pre_tanh_penalty` in the ceiling recipe probably pushes them toward this gain-sensitive style. Not tested.

## Side finding: the ceiling runs never had physics DR

Every `runs/ceiling/*` training log prints `[rma] WARNING: sim config has domain_random=false`.
- The configs list `random_variables` / `random_variable_ranges` (±25 % paddle_density, puck_damping, gravity), but `domain_random` was never set, so all ceiling policies, including `final_privcritic_pretanh`, `strike_*` and `edgefix_*`, trained on one fixed plant.
- The hist4 baselines (`juggle_sysid_v2_hist4`) are also sysid-only.
- It does not explain the hist4-vs-ceiling difference, but it removes the robustness the recipe was assumed to have.

## Retraining on the real-plant proxy

`configs/ceiling/realplant_screen1.yaml` → `runs/ceiling/realplant_screen1/`. 3M steps, 3 seeds each, GPUs 0–1, started 18:23 CDT.

- **Base:** `sysid_v2_hist4_realplant.yaml` + `action_latency_steps: 2`.
- `dribble`: the `final_privcritic_pretanh` recipe, unchanged.
- `term03`: the `strike_term03` shaping.
- `dribble_dr`: dribble with `domain_random: true` over paddle_density, puck_damping, gravity and pid_kp ±25 %.

Results: [`2026-10-31_02-00_new-sim-retraining-and-finetuning.md`](2026-10-31_02-00_new-sim-retraining-and-finetuning.md). From scratch only one `term03` seed learned; fine-tuning the old policies worked on every seed.

Note: the hist4 "proxy 128" in section 3 is at a 1000-step cap with fresh envs per episode. With the 2026-10-18 protocol (cap 10k in the new sim) hist4 gives 94.0 ± 11.8 vs 97.2 ± 13.7 real; see the follow-up note.

## Next

- Real-robot test of the best proxy-trained policy. That is the only check that the proxy is right for juggling, as opposed to paddle replay plus 4 + 4 policies.
- If it transfers, widen the DR to latency 1–3 steps and kp ±25 %, to cover the proxy's remaining error (y drift, hist4 optimism).
- Move the action latency into the sim config (an opt-in key) once the user agrees to a simulator change. Until then it lives in the trainer and the eval script.
