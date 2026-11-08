# New sim, trained from scratch: cap 7000 + SWA (5 seeds)

Best recipe (`term03_nopretanh`) trained from scratch on the new sim: `configs/new_juggle/sysid_v2_hist4_realplant.yaml`
plus 2-step action latency. No warm start, training episode cap 7000, trained until convergence.
- Spec: `configs/ceiling/scratch_newsim.yaml`, variant `cap7000`.
- Write-up: `notes/scratch/experiments/2026-11-03_08-19_new-sim-from-scratch-convergence-cap7000.md`.

Each `seedN/` is a full checkpoint. `model.pth` is the SWA actor: an average of the actor checkpoints over the last 40 %
of training. The averaged checkpoints are listed in `sources.txt`, and the original SWA dir is in `swa_source_dir.txt`.

The critics were fitted afterwards (2026-11-03) for this exact actor, in `runs/ceiling/swa_critic_fit/`:
- **Actor frozen.** Both `model.pth` and `actor_target.pth` are the SWA weights; after the run both are verified
  byte-identical to the SWA actor.
- **Critics.** `qf1/qf2` and their targets were warm-started from the seed's last raw checkpoint, then trained 2M steps
  on SWA-actor data only. Same new-sim, term03, cap-7000 settings as the actor's training. Primitive chance was fixed
  at 0.05 and there were no intermediate checkpoints.
- **Convergence.** All five stopped at 2M: `q_loss` changed 1.2 / 2.6 / 3.8 / 4.7 / 7.9 % between the halves of the
  last 1M steps (s2 / s0 / s3 / s4 / s1).
- **Files.** `training_state.pth`, `args.yaml` and `config.yaml` are from that critic-fit run.
  `args_actor_training.yaml` holds the SWA actor's original training args.
- **Value accuracy.** min(Q1, Q2) is about 50 % below the actual discounted shaped return on SWA-actor rollouts
  (fitted −47 to −56 %, warm-start critics −46 to −55 %). A matched raw checkpoint with its own actor is −47 %.
  This underestimation is a property of the recipe's critics, not of the pairing. Details are in
  `runs/ceiling/swa_critic_fit/cap7000_sN.post.json`.

Final test: `scripts.analysis.new_sim_eval`, 100 episodes, 10k cap, seeds 1000000+. ± is std err over the 100 episodes.

| Seed | Stopped | SWA window | Reach 5k | Reach 10k |
|---|---|---|---|---|
| 0 | 5M | 3.0–5.0M | 96 ± 2 % | 96 ± 2 % |
| 1 | 7M | 4.2–7.0M | 92 ± 3 % | 88 ± 3 % |
| 2 | 13M | 7.8–13.0M | 95 ± 2 % | 94 ± 2 % |
| 3 | 9M | 5.4–9.0M | 96 ± 2 % | 96 ± 2 % |
| 4 | 7M | 4.2–7.0M | 96 ± 2 % | 96 ± 2 % |
| Mean (± std err over seeds) | | | 95.0 ± 0.8 % | 94.0 ± 1.5 % |

All five policies have 0 % edge hits.
