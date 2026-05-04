# Outline

Working outline. One claim per line; each claim points at the evidence. Update as the story firms up.

Format:
- `Claim text` — `evidence: <path>` — `status: [draft|verified|TODO]`

## Title / framing
- **Working title:** *Sim-to-Real with RL Fine-Tuning in Dynamic Air Hockey Tasks* (set in `main.tex`).
- **Target venue:** CoRL 2026 (anonymous initial submission).
- **Framing (per current abstract):** sim-to-real first to get a passable policy, then online RL fine-tuning to push past human-level juggling. Compares against (a) demonstration methods (BC, offline RL) and (b) pure online RL from scratch.

## 1. Introduction
- Air hockey as a fast, contact-rich, sim2real benchmark. — evidence: `../CLAUDE.md`, `notes/docs/repo/project-goal-and-safety.md` — status: TODO
- TD3 with dual-head critics + transformed Bellman targets is the active algorithm. — evidence: `notes/docs/training/td3-algorithm.md` — status: TODO
- Contributions (placeholder): (a) residual recipe for sim2sim/sim2real, (b) ablations on depth/updates/exploration, (c) real-robot deployment with sysid'd sim. — evidence: across multiple docs — status: TODO

## 2. Related work
- _TBD_ — needs lit search.

## 3. Setup
### 3.1 Environment
- Box2D simulator with sysid'd parameters. — evidence: `notes/docs/environments/box2d/simulator-essentials.md` — status: TODO
- Real-world stack: UR5 + paddle, safety constraints. — evidence: `notes/docs/environments/real-world/overview.md`, `notes/docs/repo/project-goal-and-safety.md` — status: TODO
- Observation: 30-dim history (5 frames × {paddle, puck} × {x, y, valid}); actor sees +last action = 32-dim. — evidence: `notes/docs/environments/observation-action-spaces.md` — status: TODO
- Action: 2-dim normalized displacement → PID. — evidence: same — status: TODO

### 3.2 System identification
- Real→sim parameter fit (gravity, puck damping, paddle density, PID gains). — evidence: `notes/docs/environments/real-world/puck-system-id.md`, `…/teleop-system-id.md`, `configs/.../sysid_best_params.yaml` — status: TODO

## 4. Method  (now in `main.tex` §Method, draft skeleton from user 2026-04-30)

### 4.1 Loose system identification (expanded 2026-04-30)
- Puck dynamics fit: damped-kinematic model `dv/dt = -γv - g`, closed-form solution, 2D grid over `(gx, γ)` with linear LSQ inner fit per segment, 10 teleop puck segments. Best `gx=-0.661, γ=0.178`, ~2.86 cm mean error. — evidence: `notes/docs/environments/real-world/puck-system-id.md`, `sysid/puck_grid_search.py` — status: draft
- Paddle controller fit: PID-on-position-error producing force, coarse-to-fine grid over `(Kp, Kd, paddle_density)` plus separate Ki sweep, 8 teleop segments. Final operating point `Kp=9000, Kd=50, Ki=0, density=3000`. — evidence: `notes/docs/environments/real-world/teleop-system-id.md`, `sysid/teleop/system_id3/` — status: draft
- What we don't fit: wall/contact restitution eyeballed, friction/spin not modeled. — evidence: same docs — status: draft

### 4.2 Sim training with domain randomization
- Starting distribution: 15% near paddle spawn, 85% randomized near top of table. — evidence: sim config — status: draft
- Observation noise on puck position + occlusions (heavier when puck near paddle). — evidence: sim config / env code — status: draft
- Observation delay with per-episode randomized magnitude. — evidence: sim config / env code — status: draft
- Action randomization: 30% per-step probability, rescale to 25–75% of magnitude. — evidence: sim config / env code — status: draft
- Collision randomization: wall bounces ±10° cone; paddle-puck ±10° cone + random strength. — evidence: sim config / env code — status: draft

### 4.3 Online fine-tuning with residual RL
- Frozen base policy + learned residual head; primary axes of variation are residual action scale and replay-buffer data balance. — evidence: `notes/docs/training/residual-rl-recipe.md` — status: draft
- Small-gap vs big-gap recipes diverge: small uses `success_top_fraction 0.5`; big uses v25 (sf=0.15 + priority_age_decay=1e-4 + q_updates=1). — evidence: same + `notes/scratch/residual_rl_paddle50_log.md` §8.13 — status: TODO

### 4.4 Algorithm details (NOT YET in main.tex — decide whether to surface)
- TD3 with dual-head critics + transformed Bellman targets. — evidence: `notes/docs/training/td3-algorithm.md` — status: TODO
- Network architecture (history encoder, dual heads). — evidence: `notes/docs/training/network-architecture.md`, `…/architecture.md` — status: TODO
- Reward shaping. — evidence: `notes/docs/training/reward-shaping.md` — status: TODO

## 5. Experiments

### 5.0 Evaluation protocol  (now in `main.tex` §Experimental Results, from user 2026-04-30)
- Task success = juggle ≥3 times in an episode; reported as sliding-window success rate. — evidence: user spec — status: draft
- Task reward summarized via mean/std/min/max over sliding window. — evidence: user spec — status: draft
- Sliding window sizes: 5, 10, 25, 50. — evidence: user spec — status: draft

### 5.1 Sim2sim transfer (used as cheap proxy during method development)
- Protocol + 400k extension results. — evidence: `notes/docs/training/sim2sim.md` — status: TODO
- Big-gap (paddle -50%) study; drift root cause = critic Q-overestimation. — evidence: `notes/scratch/residual_rl_paddle50_log.md` — status: TODO

### 5.2 Sim2real (headline result)
- _TBD_ — what real-robot numbers do we have? Need to ask the user.

## 6. Ablations  (now in `main.tex` §Ablations, from user 2026-04-30)

### 6.1 Zero-shot sim-to-real (training-pipeline) ablations
- System identification (fitted vs Box2D defaults). — evidence: `sysid_best_params.yaml`, `notes/docs/environments/real-world/puck-system-id.md` — status: draft
- Collision randomization (wall ±10° cone; paddle-puck ±10° cone + strength U[0.5, 1.0]). — evidence: env code (`enable_paddle_puck_strength_randomization` etc. in `airhockey/sims/airhockey_box2d.py`) — status: draft (see §verify-randomization-values TODO)
- Action randomization (30% prob, force scaled U[0.25, 0.75]). — evidence: `enable_action_force_attenuation` in sysid_best_params.yaml — status: draft
- Starting data distribution (15% near paddle / 85% near top). — evidence: `puck_spawn_near_paddle_prob` in sysid_best_params.yaml — status: draft
- Real-world starting states (init paddle/puck from empirical real-rollout distribution vs synthetic mixture). — evidence: aspirational; no config implementing this exists yet (2026-04-30) — status: draft
- Observation randomization (noise std 0.01m; occlusion 2.5% base, 3x near-paddle; delay 25ms ±25%). — evidence: sysid_best_params.yaml — status: draft

### 6.2 Online fine-tuning ablations (evaluated sim2sim w/ 35% smaller puck + sim2real)
- Exploration (Gaussian noise std; primitive toggles). — evidence: `td3-exploration-ablations.md`, residual configs — status: draft
- Residual scale (sweep {0.05, 0.15, 0.30}). — evidence: `residual-rl-recipe.md` + paddle50 log — status: draft
- Replay-buffer staleness (`priority_age_decay`, `success_top_fraction`). — evidence: paddle50 log §8.13 (v25) — status: draft
- Q overestimation (min-of-N for N∈{2,3,5}; REDQ at N∈{5,10}). — evidence: `td3_residual_v26-v29` configs (configs added 2026-04-30, results pending) — status: draft

## 7. Discussion (was §6)
- Why residual works where full fine-tune drifts (Q1 grows 2.6-4×, residual head norm 5-10× without the fix). — evidence: `notes/scratch/residual_rl_paddle50_log.md` — status: TODO
- Limitations / safety story. — evidence: `notes/docs/repo/project-goal-and-safety.md` — status: TODO

## 8. Conclusion (was §7)
- _TBD_

## Open questions for the user (also tracked in notes.md)
- ~~Target venue / page limit / format~~ — **resolved**: CoRL 2026, LaTeX with `corl_2026.sty`. Page limit TBD (CoRL has historically been 8 pages + refs/appendix).
- Which sim2real results count as "in" the paper vs follow-up?
- Author list, acknowledgements (anonymous for initial submission anyway).
- Is the residual recipe the headline contribution, or are the algorithmic choices (dual-head critics + transformed Bellman) co-equal? Current abstract leans heavily on the sim-to-real → fine-tune story; algorithmic choices aren't yet pitched as contributions.
- What baselines are actually run vs claimed? Abstract names BC, offline RL, and pure online RL — need to confirm which we have numbers for.
