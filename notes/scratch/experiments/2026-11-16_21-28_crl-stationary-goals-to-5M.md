# Training the stationary-goal CRL runs on to 5M steps does not help: stop-on-target stays at 2–4 %, far below the 4-D-goal policies

- **Date**: 2026-11-16 21:28 UTC start (15:28 CST); runs done 2026-11-17 08:52 UTC; eval 2026-11-17
- **Status**: done
- **Follows**: [`2026-11-15_23-47_crl-stationary-goals.md`](2026-11-15_23-47_crl-stationary-goals.md) (same runs, which had
  stopped on the convergence rule at 2.0–3.5M)
- **Run dirs**: `runs/crl/crl_{paper,c0}_still_s{0..4}` (`final_converged.pt` = the earlier stop, `final_5M.pt` = this one,
  `train_resume.log`); evals `runs/crl/still5M_eval_{prev,apex}/summary.json`
- **Code**: `train_crl.py --resume <ckpt> --no-convergence --final-name`; launcher
  `scripts/goal_conditioned_rl/launch_crl_still_resume5M.sh`

## Question
The stationary-goal runs stopped early because their losses and episode length had plateaued. Does training them on
to 5M steps make them better at stopping the puck on target?

## Setup
- Resumed each run from its converged checkpoint (weights, optimizer state, step count) to 5M env steps with the
  convergence stop off; everything else unchanged. The replay buffer had not been saved, so it was refilled by the
  loaded policy (10k steps, no random-action phase) and rebuilt to 1M over the first ~1M steps.
- 10 jobs on GPUs 0 and 1 (5 per GPU, c0 split 2/3). Throughput check on GPUs 2/3 first: one GPU does ~160 c0
  steps/s in total whether 1, 2, 3 or 5 c0 jobs share it (the GPU is saturated), and paper 613 / 677 / 725 / 766
  steps/s total with 1 / 2 / 3 / 5 jobs, so running fewer jobs at once would not have finished sooner.
- Wall clock: paper 5.8–7.6 h, c0 9.6–11.4 h.
- Eval identical to the previous note: 100 goals per set with velocity zeroed, paired control goal, horizon 7000,
  `final_5M.pt`.

## Results
Mean ± std err over 5 seeds. "conv" = the earlier converged checkpoint, "4-D" = the 4-D-goal 2M policies commanded the same
stop goals (both from the previous note).

**Set B (previous positions, v = 0)**

| policy | stopped on target (%) | stop lift (pp) | within 10 cm (%) | paired gap (cm) | episode steps |
|---|---|---|---|---|---|
| paper, stationary, conv (2.0–2.5M) | 5.4 ± 0.9 | +5.0 ± 1.2 | 33.6 ± 2.5 | 4.0 ± 0.5 | 153 ± 4 |
| paper, stationary, 5M | 4.0 ± 0.9 | +3.4 ± 1.2 | 34.6 ± 3.7 | 5.1 ± 1.2 | 151 ± 22 |
| paper, 4-D goals | 9.4 ± 1.0 | +6.4 ± 1.2 | 47.0 ± 3.1 | 6.1 ± 0.6 | 398 ± 27 |
| c0, stationary, conv (2.5–3.5M) | 2.2 ± 0.9 | +0.8 ± 1.0 | 30.8 ± 1.1 | 0.9 ± 0.7 | 179 ± 12 |
| c0, stationary, 5M | 2.2 ± 0.6 | +1.4 ± 0.7 | 28.2 ± 1.8 | 2.3 ± 0.9 | 172 ± 15 |
| c0, 4-D goals | **12.4 ± 1.1** | **+7.6 ± 1.1** | 56.6 ± 2.1 | 6.6 ± 1.2 | 554 ± 48 |

**Set A (juggler apex positions, v = 0)**

| policy | stopped on target (%) | stop lift (pp) | within 10 cm (%) | paired gap (cm) | episode steps |
|---|---|---|---|---|---|
| paper, stationary, conv | 2.4 ± 0.5 | +0.2 ± 0.6 | 24.6 ± 2.6 | 1.2 ± 1.3 | 142 ± 4 |
| paper, stationary, 5M | 1.6 ± 0.2 | +0.0 ± 0.3 | 20.0 ± 2.9 | 1.3 ± 0.6 | 125 ± 15 |
| c0, stationary, conv | 1.8 ± 0.4 | +0.4 ± 0.5 | 15.6 ± 0.9 | 0.5 ± 0.3 | 169 ± 16 |
| c0, stationary, 5M | 2.2 ± 0.4 | +0.8 ± 0.7 | 14.8 ± 0.9 | −0.7 ± 1.5 | 160 ± 18 |
| c0, 4-D goals | **16.0 ± 1.2** | +5.4 ± 2.7 | 49.2 ± 4.3 | 2.6 ± 1.5 | 587 ± 73 |

Training-time rollout episode length (last 200k steps): paper 124 → 140 (all 5 seeds up), c0 119 → 106 (3 of 5 down).

## Conclusion
- 2–3M more steps changed nothing measurable: every eval metric of both stationary-goal variants is within about one
  std err of its converged value. The convergence rule had stopped them on a real plateau.
- The stationary-goal policies remain far behind the 4-D-goal policies on their own task (stop on target 2–4 % vs
  9–16 %) and survive ~150 vs ~400–590 steps. The extra training did not close any of that gap.
- paper's longer training-time episodes (stochastic actor) did not carry over to the deterministic eval (151 steps).

## Next
- Commanding (x, y, 0, 0) during training is the problem, not the training length: try oversampling slow future states
  as critic goals, or keep 4-D training and command stop goals only at eval (the best policy so far).
