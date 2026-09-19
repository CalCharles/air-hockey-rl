# sysid/puck_dynamics — puck free flight → `gravity`, `puck_damping`

Fits the damped free-flight model `a = g − γ v` of the Box2D puck to clean free-flight clips
harvested from real recordings, with the recordings split into train / validation and the grid
search validated on the held-out recordings.

```
code/     puck_dynamics_fit.py   grid search (coarse + fine), metrics, plots, key figures
          fit_puck.py            entrypoint: split by recording → grid on train, scored on val → validation → summary
          render_overlays.py     real-vs-Box2D overlays of representative validation windows (GIF + PNG + mosaic + table)
data/     <name>/free_fall/*.hdf5, manifest.{csv,json}, summary.md   (stage 1 of sysid/common/extract_sysid_sections.py)
          <name>/recordings → the raw recordings the clips came from
results/  <name>/summary.md, results.json, split.json, sim_config_fitted.yaml, fit_grid.png,
          validation_percentiles.png, puck_final_displacement_*.{png,pdf}
          <name>/overlays/  gifs/, png/, mosaic.png, overlay_summary.{md,csv}   (render_overlays.py)
```

```bash
# everything from raw recordings (also runs the wall fit): sysid/common/run_puck_wall_sysid.py --input-dir … --name <name>
python sysid/puck_dynamics/code/fit_puck.py --sections-dir sysid/puck_dynamics/data/<name> --out sysid/puck_dynamics/results/<name>
# variants: --window-frames 30, --seed 1, --objective fit_rel, --g-range -0.9 -0.5 --gamma-range 0 0.3, --figures-dir paper/figures/sysid
python sysid/puck_dynamics/code/render_overlays.py --results-dir sysid/puck_dynamics/results/<name>   # 5 val windows, real vs Box2D, → results/<name>/overlays/
```

**Latest result** (`results/mouse_dataset/`, 100 mouse-teleop recordings, 542 clips, 80 / 20 recordings,
20-sample windows): **g = −0.73 m/s², γ = 0.11 1/s** vs canonical −0.661 / 0.178 — held-out
prediction error 1.82 → 1.51 cm over 0.5 s; the selection is at the validation oracle, the canonical
values are 8–17 % worse. Not promoted into `configs/`. Details: [`notes/docs/environments/real-world/sysid/puck-free-flight.md`](../../notes/docs/environments/real-world/sysid/puck-free-flight.md);
notes [`2026-09-10_01-30`](../../notes/scratch/experiments/2026-09-10_01-30_sysid-train-val-pipeline.md),
[`2026-09-10_03-20`](../../notes/scratch/experiments/2026-09-10_03-20_sysid-normalised-metrics-percentile-validation.md).

The wall fit (`sysid/wall_collision/`) replays its bounces with this fit's g / γ (`--puck-results results/<name>/results.json`).
