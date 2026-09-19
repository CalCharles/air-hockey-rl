# sysid/wall_collision — puck–wall bounces → `side_wall_restitution`, `end_wall_restitution`

Replays real wall bounces in Box2D from the fitted pre-impact state and sweeps the restitution of
the side (y±) and end (x±) walls, with the recordings split into train / validation and each sweep
validated on the held-out recordings. Needs a puck free-flight model (g, γ) for the replays — by
default the one fitted in `sysid/puck_dynamics/`.

```
code/     wall_restitution_fit.py   bounce states, Box2D replay, restitution sweep, metrics, plots, key figures
          fit_walls.py              entrypoint: split by recording → sweep per wall kind → validation → apparent wall lines → summary
          render_overlays.py        real-vs-Box2D overlays of representative validation bounces (GIF + PNG + mosaic + table)
data/     <name>/wall/*.hdf5, manifest.{csv,json}, summary.md   (stage 1 of sysid/common/extract_sysid_sections.py --wall-out)
          <name>/recordings → the raw recordings the bounces came from
results/  <name>/summary.md, results.json, split.json, sim_config_fitted.yaml, fit_sweeps.png,
          validation_percentiles.png, wall_side_exit_*.{png,pdf}
          <name>/overlays/  gifs/, png/, mosaic.png, overlay_summary.{md,csv}   (render_overlays.py)
```

```bash
# everything from raw recordings: sysid/common/run_puck_wall_sysid.py --input-dir … --name <name>
python sysid/wall_collision/code/fit_walls.py --sections-dir sysid/wall_collision/data/<name> \
    --puck-results sysid/puck_dynamics/results/<name>/results.json --out sysid/wall_collision/results/<name>
# or with an explicit puck model: --puck-params -0.73 0.11
# variants: --objective speed_err, --restitution-range 0.5 1.0 0.01, --max-side-frames 6, --figures-dir paper/figures/sysid
python sysid/wall_collision/code/render_overlays.py --results-dir sysid/wall_collision/results/<name>   # 5 val side-wall bounces, real vs Box2D, → results/<name>/overlays/
```

The train / val recordings are identical to the puck fit's for the same `--seed` / `--val-fraction`
(both fits draw the split from the manifest's list of all input recordings).

**Latest result** (`results/mouse_dataset/`, 133 bounces, 80 / 20 recordings, puck model g = −0.73, γ = 0.11):
**`side_wall_restitution` = 0.90** vs canonical 0.99 — val exit-speed error 20 % → 16 % of the real
exit speed; the data only pin it to 0.85–0.95. `end_wall_restitution` = 0.55 vs 0.70 is **not
identified** (17 / 6 bounces, monotone curve) — keep 0.70. The sim has no tangential loss at the walls,
so the exit angle prefers a different restitution than the exit speed. Not promoted into `configs/`.
Details: [`notes/docs/environments/real-world/sysid/puck-wall-collision.md`](../../notes/docs/environments/real-world/sysid/puck-wall-collision.md);
notes [`2026-09-10_01-30`](../../notes/scratch/experiments/2026-09-10_01-30_sysid-train-val-pipeline.md),
[`2026-09-10_03-20`](../../notes/scratch/experiments/2026-09-10_03-20_sysid-normalised-metrics-percentile-validation.md).
