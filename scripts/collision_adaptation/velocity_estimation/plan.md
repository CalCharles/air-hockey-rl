# Air hockey — implementation plan (scratch)

Condensed from notebook notes: sim fidelity, rewards, real-robot constraints, and iteration order.

## 1. Simulation fidelity (Box2D)

**Near term**

- Align **paddle boundaries** with real table / UR5 workspace.
- **Paddle–puck contact:** real contact is low-friction and overly sensitive in sim; first try **lower paddle density** before deep friction/restitution work.
- Keep **termination / episode logic** explicit (new termination conditions) so training and eval stay aligned.

**Later**

- **Delay and actuation/observation noise** in sim.
- **Domain randomization:** inconsistent delay, dynamics perturbations, mild boundary variation, paddle/puck drift, wall bounce / damping variation, optional observation bias by location.

## 2. Reward design and credit assignment

- Compare **single combined reward** vs **additive separate heads** (motion / smoothness vs task); focus on **weighting**, not only discount.
- **Discount γ** shapes credit assignment; **e-stop already terminates** the episode, so safety and task interact through termination.
- Open question: **decouple e-stop from “task failure” for learning** (e.g. bootstrapping / use of truncated data) while keeping e-stop a **hard** environment stop—validate in your RL stack.

## 3. Real robot: safety, cost, exploration

- **Tension:** discourage e-stops vs complete the task; rollouts are **expensive**; **reset policy** needs improvement.
- **Hardware:** clamping and setup change dynamics; hope **lateral randomization** is enough; **damping** is especially important.
- **Learning:** **targeted exploration** so useful rewards and constraint structure appear often enough (low data, partial observability).
- **E-stop → termination** is a strong signal; pair with exploration so the policy does not linger in bad regions.
- **Sim proxy:** perturb low-level control (e.g. PID noise) to iterate quickly before hardware.

## 4. Instrumentation

- Optional **real-world metrics** (e.g. positional logging)—only if needed.
- Prefer a **messy sim** for fast validation of reward / termination / exploration ideas before polishing realism.

## 5. Sequencing

| Phase | Focus |
|-------|--------|
| **A** | Paddle bounds + contact (density first) + clear termination semantics. |
| **B** | Reward ablations: combined vs separate heads; sweep motion vs task weights (fix γ unless studying horizon). |
| **C** | Sim delay / noise / drift / randomization after A/B are stable. |
| **D** | Hardware: resets, damping/clamp discipline, exploration, metrics; optional e-stop + bootstrapping study. |
| **E** | Defer **robosuite** (etc.) until decoupled reward/safety story is validated in the current pipeline.

## 6. Summary

Align sim geometry and contact with minimal physics changes, then structured reward and weight sweeps, then sim messiness for robustness; on hardware, reduce iteration cost (resets, exploration) and treat e-stops as hard stops while deciding how terminations feed the learner.
