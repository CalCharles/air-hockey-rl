# Final ceiling recipe retrained with the top-edge velocity fix (5 seeds, Task 2)

`configs/ceiling/task2_edgefix.yaml`: `final_privcritic_pretanh` + `edge_velocity_fix: true`
(`scripts/rma/edge_velocity_fix.py`, training-side instance patch; `airhockey/` unchanged). Still dribbles near the
top edge. Write-up: `notes/scratch/experiments/2026-10-30_12-54_edge-velocity-fix-retrain.md`.
Final test (100 episodes, 10,000-step cap), reached the cap, fixed sim / unmodified sim:
seed0 2.1M 96/99 %, seed1 2.6M 100/100 %, seed2 2.7M 98/99 %, seed3 2.7M 95/89 %, seed4 2.4M 84/88 %;
5 seeds 94.6 ± 2.8 % / 95.0 ± 2.7 % (std err).
