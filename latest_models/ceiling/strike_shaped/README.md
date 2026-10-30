# Strike-shaped sustained juggling (5 seeds)

Final ceiling recipe + training-only motion shaping `edgehit3_weak3_pout2` (`configs/ceiling/shape_screen3.yaml`;
seed 0 from `shape_screen2.yaml`): strikes the puck ~14 cm inside the workspace at ~0.7 m/s instead of dribbling at
the top edge. Write-up: `notes/scratch/experiments/2026-10-30_04-22_natural-strike-shaping.md`.
Each `seed<N>/` is the checkpoint chosen on selection_long, then tested on 100 fresh episodes at a 10,000-step cap
on the unmodified target sim:

| Seed | Checkpoint | Survived 10,000 steps | Mean steps |
|---|---|---:|---:|
| 0 | 2.6M | 61 % | 7809.6 |
| 1 | 2.8M | 78 % | 8837.3 |
| 2 | 2.4M | 74 % | 8685.6 |
| 3 | 1.6M | 64 % | 7951.3 |
| 4 | 2.9M | 88 % | 9287.1 |
| **5 seeds** | | **73.0 ± 4.9 %** (std err) | 8514.2 ± 277.9 |
