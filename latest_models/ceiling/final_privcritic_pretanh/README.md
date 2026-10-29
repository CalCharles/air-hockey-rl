# Juggling ceiling — final recipe (5 seeds)

Best juggling policies on the target sim `configs/new_juggle/sysid_v2_hist4.yaml`: 50-step memory actor
(`rma_mode: history_privileged`, deployable as a plain `HistoryActor` on the standard observations),
true-state critic during training, `actor_pre_tanh_penalty: 0.001`, 3M steps.
Recipe: `configs/ceiling/final_privcritic_pretanh.yaml` (expanded: `configs/ceiling/generated/final_privcritic_pretanh/`).
Write-up: `notes/scratch/experiments/2026-10-29_07-49_juggle-ceiling-final-recipe.md`.

Each `seed<N>/` is the checkpoint chosen on the selection_long protocol (32 episodes, 10,000-step cap),
then tested on 100 fresh episodes at a 10,000-step cap:

| Seed | Checkpoint | Survived 10,000 steps | Mean steps |
|---|---|---:|---:|
| 0 | 2.8M | 98 % | 9846.6 |
| 1 | 2.9M | 100 % | 10000.0 |
| 2 | 2.8M | 100 % | 10000.0 |
| 3 | 2.8M | 98 % | 9902.9 |
| 4 | 2.5M | 99 % | 9961.4 |
| **5 seeds** | | **99.0 ± 0.4 %** (std err) | 9942.2 ± 29.8 |

Files: `model.pth` (actor), `rma_meta.json` (architecture), `args.yaml`, `config.yaml` (sim config).
Critics and training state stay in `runs/ceiling/final_privcritic_pretanh/` (not tracked).
Re-evaluate: `python -m scripts.ceiling.eval_target --protocol final --label <name> <seed dir>`.
