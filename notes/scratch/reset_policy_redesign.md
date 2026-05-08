# Reset policy redesign — design doc

**Audience:** future implementer (likely a new contributor) tasked with replacing or substantially upgrading the between-episode puck reset policy on the real UR5 air-hockey rig. This doc is a *forward-looking* design brief — for a factual reference of what's deployed today, see [`notes/docs/environments/real-world/reset-fsm.md`](../docs/environments/real-world/reset-fsm.md) and the source files cited below.

---

## 1. How the current reset policy works

The reset policy runs between policy episodes on the real robot to bring the puck from "stuck somewhere at the bottom of the table" back into a state where the juggling policy can take over (puck airborne in the upper half, off the side walls). It is invoked by the async real-world collector at episode boundaries.

It is implemented as a **hand-coded, closed-loop, five-phase finite state machine** — `ResetPolicyFSM`. Phase transitions are driven by paddle position and the recent puck-history window read from the env state. There is no learned component.

**Phases (high level):**

1. **`goto_start`** — paddle navigates to a randomized bottom corner (left or right) of an edge-following path.
2. **`edge_loop`** — paddle sweeps along the bottom boundary along ~44 pre-computed waypoints offset from the wall by `off_wall_abs_y_m` (default 0.35 m), the intent being to "scoop" the puck off the bottom wall.
3. **`upward_burst`** — 5 quick flicks in the −x direction (toward the upper half) at fixed magnitude `burst_action_m = 0.2`.
4. **`wait_for_puck`** — hold position and watch puck history; succeed if the puck enters a proximity window with downward (falling) velocity.
5. **`strike`** — ramped strike `[-0.3, -0.6, -0.8, -1.0, -1.0, …]` to send the puck decisively into the upper half.

**Success detection:** after each upward motion (burst or strike), a 20-step post-window checks that the puck crossed the success line (default: 50% of table height from the bottom) AND was off the side walls (≥ `off_wall_abs_y_m` from center y) for at least 5 consecutive steps. On failure, the FSM cycles back into stage 2 (wait → strike) up to `max_stage2_cycles = 5` retries before giving up and triggering a hard reset (operator pause).

**Where to read further:**

- Reference doc with the parameter table and integration overview: [`notes/docs/environments/real-world/reset-fsm.md`](../docs/environments/real-world/reset-fsm.md).
- FSM implementation: `scripts/real/rollout_reset_policy_real.py` — `ResetPolicyFSM` class (lines 57–156 init, `step()` at 729–815, edge-loop path builder `_build_edge_loop_path` at 239–266, post-upward success check `_step_post_upward_window` at 554–622).
- Collector integration / hard-vs-soft routing: `scripts/smooth_policy/amp_history/amp_training/td3/helper/real_reset_runner.py` — `run_reset_fsm` (lines 173–254), `ResetRunner.run` (lines 381–515).
- Post-FSM soft re-init / paddle-history priming: `scripts/smooth_policy/amp_history/amp_training/td3/helper/real_collector_reset.py`.
- Where reset sits in the episode loop: [`notes/docs/environments/real-world/episode-lifecycle.md`](../docs/environments/real-world/episode-lifecycle.md).

---

## 2. Qualitative issues with the current policy

### 2a. Not robust to the puck being off the wall; success rate is mediocre

The phase 1–2 design assumes the puck starts pinned against (or very close to) the bottom wall, so the paddle's bottom-edge sweep can scoop it. In practice the puck frequently ends up a few cm off the wall — e.g., after a failed strike, after a soft collision off a side wall, or after a previous reset that "almost worked". When that happens:

- The edge-loop sweep path runs *under* the puck and never makes useful contact.
- The burst phase fires regardless of whether the puck is in front of the paddle, often producing an empty flick.
- The retry cycle re-runs the same scripted motion against the same puck configuration, so retries don't meaningfully improve the per-attempt success probability — they mostly help only when the puck happens to drift back toward the wall on its own.

The net effect is that the success rate is noticeably lower than it should be, and failures bias toward "puck just sat off-wall the whole time" rather than near-misses we could tune our way out of.

### 2b. Reset trajectories are static / hard-coded, with little variety

The edge-loop waypoints, burst magnitude/count, and strike ramp are fixed scalars. Run-to-run randomization is limited to which corner the FSM starts in. As a consequence:

- The puck gets launched into the upper half along a narrow distribution of trajectories (similar speeds, similar entry angles).
- The policy that takes over after reset sees a less diverse initial state distribution than the simulator provides during training, which probably hurts robustness on the real rig.
- We have no knob to deliberately vary "how the reset feels" — e.g., to set up a slower or faster handoff for a given experiment.

This isn't a bug — the policy works when it works — but it leaves performance on the table both for reset success and for downstream training data quality.

---

## 3. Goals for the redesign

### 3a. Primary: higher success rate, robust to off-wall pucks

The redesigned reset policy should succeed on a substantially larger fraction of attempts than the current FSM, and in particular should handle pucks that start off the bottom wall. Concretely, a rewrite should:

- **Sense before acting.** Use the puck history (already available in env state — see paddle/puck slices in `CLAUDE.md`) to localize the puck before committing to a sweep direction or strike, rather than running the same scripted path regardless of where the puck actually is.
- **Plan to make contact.** The motion before the strike should be a path that *brings the paddle to the puck*, wherever it is in the lower region of the table, not a fixed wall-hugging sweep. The corner-randomized edge loop should be replaced or generalized.
- **Recover meaningfully on retry.** When an attempt fails, the next attempt should reflect what happened — e.g., re-localize the puck and re-plan — rather than re-running the same motion.
- **Be measurable.** Define a reset-success metric (and probably a per-phase failure-mode breakdown) up front so the new policy can be compared apples-to-apples against the FSM. The `episode_summaries.jsonl` plumbing already exists; piggyback on it.

A learned reset policy is worth considering, but not required — a sensor-driven scripted policy that actually plans to where the puck is would already address the main failure mode. If we go learned, the small action space and short horizon make it tractable, but data collection on the real rig is the bottleneck and should be planned carefully.

### 3b. Stretch: qualitative control over how the reset hands off

Beyond just succeeding, it would be valuable to have *parameterized* control over the reset's outcome — e.g., asking for the puck to be handed off:

- **At a target speed** (slow / fast), so we can stress-test the policy on harder initial conditions or set up easier ones for early training.
- **At a target position / region** of the upper half, so we can sweep the initial state distribution deliberately.
- **At a target angle / trajectory shape**, so the post-reset state distribution looks more like sim's juggle-init distribution.

This is harder than 3a — it implies either a richer scripted strike (with a controller mapping `(target speed, target position) → strike action sequence`) or a goal-conditioned learned policy. We don't need to commit to an approach now; flag it as a follow-up once 3a lands and we have a working success metric to optimize against.

---

## 4. Suggested next steps for the implementer

1. **Prototype in simulation first.** The Box2D env supports the same observation/action interface as the real rig, so a candidate reset policy can be developed and iterated on entirely in sim before any real-robot time. Set up a sim scenario where the puck starts in adversarial bottom-of-table configurations (pinned to wall, a few cm off the wall, drifting, etc.), drop in the new reset policy, and measure success rate against the same metric you'll use on the real rig. To simulate the *handoff* — i.e., what happens after reset succeeds — you can hand control over to any working juggling policy: the checkpoints under `latest_model/` (the `hist2_*` variants are the canonical hist2 policies) are fine for this, or train your own, or use anything else that runs in this env. The juggling policy doesn't need to be SOTA; it just needs to produce a realistic post-reset episode so you can confirm the reset is actually leaving the puck in a state the downstream policy can pick up. Only after the sim version clearly beats the current FSM should you spend real-rig time on it.

2. Once the sim prototype looks good, run the current reset on the real rig for ~50 attempts and log per-attempt outcomes + the puck's position at FSM entry. This pins down the actual failure-mode breakdown (off-wall vs. on-wall, stage-2 retries used, etc.) so you can confirm the sim-prototyped redesign actually targets the real failure modes — and not ones that only show up in sim.
3. Decide scripted-vs-learned for goal 3a based on (1)+(2). Either way, the new entrypoint should be a drop-in replacement for `ResetPolicyFSM` so the collector integration in `real_reset_runner.py` doesn't need to change.
4. Defer goal 3b until 3a's success metric is in place and the new policy beats the FSM.
