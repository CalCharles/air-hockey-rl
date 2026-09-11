# TD3 + hindsight experience replay on the sparse puck-goal position / velocity tasks

- **Date**: 2026-09-10 06:30 UTC start (rounds 1–6 launched 20:20–23:10 UTC)
- **Status**: done — position and speed tasks learnt; velocity-vector task unsolved (all 19 runs finished)
- **Run dirs**: `runs/her/round{1..6}_20260910/` (one sub-dir per run, `summary.md` per round)
- **Configs**: `configs/td3/her/*.yaml`; sim configs `configs/new_juggle/tasks/sim_sysid_puck_goal*.yaml`
- **Code**: `scripts/td3/td3_training_her.py`, `scripts/td3/helper/td3_her.py`, `scripts/td3/helper/her_eval.py`, `airhockey/airhockey_tasks/puck_goal_sparse.py`, `airhockey/airhockey_rewards/goal_task_rewards/puck_goal_sparse_reward.py`; doc `notes/docs/training/her.md`; tests `scripts/td3/tests/test_her_relabel.py`
- **Deployable policies**: `latest_models/her/{puck_goal_position_hist2,puck_goal_position_hist4,puck_goal_speed_hist2}/`
- **Cleanup (2026-09-11)**: the velocity-vector task, its `box` / `shot` / `intercept_shot` velocity samplers and the per-round variant configs named below were removed from the tree; commit 467da75 has the code and configs exactly as these runs used them (each run dir also keeps its `config.yaml` / `args.yaml`). Kept: `puck_goal_position_sparse` and `puck_goal_position_speed_sparse`, now suite tasks `puck_goal` / `puck_goal_vel` (`configs/td3/tasks/{puck_goal,puck_goal_vel}_{sysid,dr}.yaml`; hist2 variants and the k0 ablation under `configs/td3/her/`).

## Question
Can the canonical TD3 recipe learn the goal-conditioned puck tasks (send the
puck to a goal position in the upper half; the same with a goal velocity)
from the sparse +10 goal reward when every episode is hindsight-relabelled,
and what does the task definition have to look like for that to work?

## Setup
Recipe = `configs/td3/tasks/*_sysid.yaml` (q=25 / actor=6 updates per
episode, batch 512, PER, single flat 1M buffer, q_weight_decay 0, primitive
exploration, γ 0.975), networks 128 (actor) / 256 (critic) × 2 residual
blocks, 1M env steps (2M where noted), sysid physics, goal appended to the
30-dim history obs. HER `future` strategy, k = 4 (8 where noted),
done-on-success. Position radius 0.10 m; velocity radius 0.5 m/s. Eval: 20
fresh episodes per 25k-step checkpoint, 100 episodes at the end; success =
the task's sparse goal test (return = 10 × success). Single seed per cell.

Random-action baselines: position 4 % (round-1 definition) / 0 % (contact
required); velocity 0 %.

Task definitions changed between rounds (all variants remain selectable by
config):

| Round | Position success | Velocity goal sampling | Hindsight goal filter |
|---|---|---|---|
| 1 | puck within radius **and moving up** (repo's old puck-goal convention) | `box` (independent uniform velocity) | none (classic HER) |
| 2–3 | puck within radius **after a paddle contact** in the episode | `box` | `goal_in_distribution` (upper half, velocity set) |
| 4 | same | `shot` (state of a simulated shot from a random workspace point) | same |
| 5 | same | `shot`; also the **speed** task (scalar speed) | same |
| 6 | same | `intercept_shot` (shot launched from this episode's puck intercept point) | same |

## Results

| run | task | hist | vel goals | k | filter | r_pos | r_vel | steps | best ckpt (20 ep) | last-4 ckpts | train succ (last 100k) | final (100 ep) |
|---|---|---|---|---|---|---|---|---:|---|---|---|---|
| round1/puck_goal_sysid | position (moving-up rule) | 4 | - | 4 | no | 0.1 | - | 1M | 0.75@850k | 0.55 | 0.30 | **0.44** |
| round1/puck_goal_sysid_k0 | position (moving-up rule) | 4 | - | 0 | no | 0.1 | - | 1M | 0.05@25k | 0.00 | 0.02 | 0.03 |
| round1/puck_goal_vel_sysid | velocity | 4 | box | 4 | no | 0.1 | 0.5 | 1M | 0.00 | 0.00 | 0.01 | 0.00 |
| round1/puck_goal_vel_sysid_k0 | velocity | 4 | box | 0 | no | 0.1 | 0.5 | 1M | 0.00 | 0.00 | 0.00 | 0.00 |
| round2/puck_goal_sysid_filter | position (contact) | 4 | - | 4 | yes | 0.1 | - | 1M | 0.90@925k | 0.71 | 0.54 | **0.64** |
| round2/puck_goal_sysid_filter_hist2 | position (contact) | 2 | - | 4 | yes | 0.1 | - | 1M | 0.85@375k | 0.71 | 0.62 | **0.79** |
| round2/puck_goal_vel_sysid_filter | velocity | 4 | box | 4 | yes | 0.1 | 0.5 | 1M | 0.10@250k | 0.03 | 0.02 | 0.03 |
| round2/puck_goal_vel_sysid_filter_hist2 | velocity | 2 | box | 4 | yes | 0.1 | 0.5 | 1M | 0.10@100k | 0.04 | 0.02 | 0.01 |
| round3/puck_goal_vel_hist2_2M | velocity | 2 | box | 4 | yes | 0.1 | 0.5 | 1.02M (stopped) | 0.10@200k | 0.03 | 0.02 | - |
| round3/puck_goal_vel_hist2_k8 | velocity | 2 | box | 8 | yes | 0.1 | 0.5 | 1M | 0.10@250k | 0.04 | 0.02 | 0.02 |
| round3/puck_goal_vel_hist2_nofilter | velocity | 2 | box | 4 | no | 0.1 | 0.5 | 1M | 0.10@400k | 0.00 | 0.01 | 0.02 |
| round3/puck_goal_vel_hist2_tol075 | velocity | 2 | box | 4 | yes | 0.1 | 0.75 | 1M | 0.20@625k | 0.03 | 0.04 | 0.04 |
| round4/puck_goal_vel_shot | velocity | 4 | shot | 4 | yes | 0.1 | 0.5 | 1M | 0.15@550k | 0.05 | 0.02 | 0.00 |
| round4/puck_goal_vel_shot_hist2 | velocity | 2 | shot | 4 | yes | 0.1 | 0.5 | 1M | 0.10@550k | 0.04 | 0.05 | 0.03 |
| round4/puck_goal_vel_shot_hist2_2M | velocity | 2 | shot | 4 | yes | 0.1 | 0.5 | 2M | 0.20@775k | 0.04 | 0.02 | 0.01 |
| round5/puck_goal_speed_shot_hist2 | **speed** | 2 | shot | 4 | yes | 0.1 | 0.5 | 1M | 0.60@775k | 0.45 | 0.30 | **0.40** |
| round5/puck_goal_vel_shot_hist2_r015 | velocity | 2 | shot | 4 | yes | 0.15 | 0.5 | 1M | 0.10@175k | 0.05 | 0.03 | 0.04 |
| round6/puck_goal_vel_ishot_hist2 | velocity | 2 | intercept_shot | 4 | yes | 0.1 | 0.5 | 1M | 0.20@800k | 0.06 | 0.05 | 0.06 |
| round6/puck_goal_vel_ishot_hist2_2M | velocity | 2 | intercept_shot | 4 | yes | 0.1 | 0.5 | 2M | 0.10@325k | 0.05 | 0.04 | 0.06 |

Trajectory shapes (20-episode checkpoint evals, so ±0.1 noise):

- **Position, contact rule (round 2)**: hist4 0.40 @100k → 0.55 @200k → 0.70 @300k → 0.85 @700k → 0.55 @900k, final 0.64; hist2 0.35 @100k → 0.75 @200k → 0.60–0.75 through 700k → 0.45 @600k, final **0.79**. Round 1's moving-up rule without the filter learnt the same thing ~5× later (0 until 400k, 0.35 @600k, final 0.44). k = 0 never leaves the random-action floor: the sparse reward alone gives nothing to learn from.
- **Speed (round 5)**: 0.20 @200k → 0.35 @300k → 0.50 @500k → 0.30–0.40 afterwards, final 0.40, with the policy drifting into juggling (contacts per episode 3 → 7, episode length 85 → 165, 8/20 episodes hitting the 250-step limit at 700k).
- **Velocity vector**: every variant peaks at 0.10–0.20 on a single 20-episode checkpoint and sits at 0.02–0.05 on the 100-episode final. Contacts per episode climb to 3–6 (juggling) without the velocity being controlled.

Diagnostics that drove the iterations (scratch scripts in the session scratchpad; the numbers are reproducible from the checkpoints):

1. **Free successes** (round 1 → 2): with the "moving up" rule, 16 % of random-action episodes scored before the fix to `vx < 0` and the puck's own spawn drift (random heading, ≤ 0.5 m/s) still produced "achieved" goals the policy did nothing for; the rolling training success stayed ≤ 0.05 for 400k steps. Requiring a paddle contact in the achieved goal (and filtering hindsight goals to the goal region) moved the first 0.40 eval from 600k to 100k steps.
2. **Box velocity goals are infeasible** (round 2/3 → 4): the round-2 velocity policies reached the goal *position* in 40 % of episodes but with a 2.0 m/s median velocity error there and corr(goal v, achieved v) ≈ 0. Only 6 % of states at reached position goals had a velocity inside the goal box at all. The direction of the puck at the goal is dictated by where it was hit from, so an independent velocity cannot be requested.
3. **The critic does see the goal velocity** (round 2 hist2 final replay buffer, 4 000 stored successful transitions): Q = 10.0 on the stored goal, 2.7 with the goal position perturbed by 0.3 m, 3.3 with the goal velocity perturbed by 1 m/s, 4.1 with the velocity sign-flipped. So the success test is learnt; what is not learnt is the credit assignment from the hit (paddle velocity over the 2–3 steps before contact) to the velocity 10–30 steps later.
4. **Shot / intercept-shot goals** made every sampled goal physically consistent but did not change the picture within 1M steps (peaks 0.10–0.20, finals ≤ 0.03); the speed-only goal, which removes the direction constraint, is the variant that learns.

Throughput: 200–290 training-phase SPS per run with 8–12 runs sharing 4 GPUs / 64 cores (≈ 1–1.5 h per 1M steps).

## Conclusion
- **HER is necessary and sufficient for the position task.** With the contact-based success rule and the goal-region filter the canonical recipe reaches 64 % (hist4) / **79 % (hist2)** final success from a purely sparse reward; without relabelling (k = 0) it stays at 3 %. Best single checkpoints are 0.85–0.90 on 20 episodes; the late-training dip (85 → 55 % on hist4) is the juggling attractor, so a best-checkpoint pick is worth ~+10 points.
- **The velocity-vector task as posed is not learnt by this recipe in 1–2M steps** (≤ 5 % final under every variant: goal filter, k = 8, tolerance 0.75, position radius 0.15, feasible `shot` / `intercept_shot` goals, hist2/hist4, 2M steps). The **speed** variant (goal = position + |v|) reaches 40 % final / 50–60 % at its best checkpoint, so the recipe can shape *how hard* it hits but not the full outgoing direction+speed within the tolerance.
- The task definitions matter more than the RL knobs: contact-based success (vs. "moving up") and goal-region filtering changed learning speed ~5×; k, tolerance, radius and physics variants changed nothing on the velocity task.
- hist2 vs hist4 physics: hist2 was better on the position task (79 vs 64 % final), consistent with the 2026-09-04 five-task finding; no difference on velocity.
- Single seed per cell; the 20-episode checkpoint evals have ±0.1 noise and the 100-episode finals ±0.05.

## Next
- Velocity-vector task: (a) the 2M `shot` / `intercept_shot` runs finished at 1 % and 6 % final, so more steps alone do not help; (b) try a shaped curriculum on the tolerance (start at 1.0 m/s, anneal to 0.5) — a task-side knob, not a recipe change; (c) a per-step time cost would break the juggling attractor but changes the reward scheme, so it needs a decision; (d) hitting-specific exploration (vary paddle stroke speed at contact).
- Multi-seed confirmation of the position result (3 seeds hist2) before it goes in the paper.
- Best-checkpoint selection for the deployable policies (the final checkpoint is below the best one on every run).
