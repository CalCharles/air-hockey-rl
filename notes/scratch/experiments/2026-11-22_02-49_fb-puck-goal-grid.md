# Frozen FB policies do not yet show reliable puck-goal aiming

- **Date**: 2026-11-22 02:49 UTC
- **Status**: done
- **Run dirs**: `runs/fb/puck_grid_true_state_20261122`
- **Checkpoints**: full-history final 1M; full-history 50k; current-only backward 50k
- **Protocol**: [FB puck-grid evaluation](../../docs/training/fb-puck-grid.md)

## Question

Can reward-inferred goal latents cause the FB actor to hit the puck into a
requested upper-half region, beyond incidental visits from juggling?

## Setup

5×5 goals, 5 cm radius, 20 paired seeds per goal (500 goal-conditioned episodes
per checkpoint), plus 20 juggling-control episodes scored against every goal.
Same training simulator/latency and episode endings, frozen deterministic policies.
Goal inference uses Gaussian weights on visible saved puck observations. Success
requires paddle contact first and is scored against the raw Box2D puck center at
control-step boundaries. Secondary 10 cm tolerance uses identical trajectories.

Public simulator `current_state` already includes observation corruption; an
initial partial evaluation using it was stopped and superseded by the raw-physics
run. Only `puck_grid_true_state_20261122` is the reported result. The smoke and
partial `puck_grid_20261122` outputs are not ground-truth evaluations.

## Results

Rates average uniformly over 25 goals. One training seed per model; 20 shared
episode seeds. These are not multi-training-seed estimates.

| Model | Training steps | 5 cm goal success | 10 cm goal success | Juggling-control 5 cm visits | Contact rate under goal latents |
|---|---:|---:|---:|---:|---:|
| Full-history B | 1,000,000 | 13.4% | 23.6% | 22.8% | 81.2% |
| Full-history B | 50,000 | 0.2% | 0.6% | 1.8% | 22.2% |
| Current-only B | 50,000 | 0.0% | 0.2% | 0.0% | 20.0% |

For full-history 1M, goal conditioning minus juggling-control visits is -9.4
percentage points. A paired bootstrap over episode seeds, retaining all goals
within each seed (10,000 resamples), gives an approximate 95% interval of
[-17.4, -2.0] percentage points. This is uncertainty over these episode starts,
not over trained models. Individual goal rates have only 20 trials each.

Best full-history 1M cell: 40% at table (x=-0.6830, y=0), versus 10% under the
juggling control. This isolated cell does not offset the worse grid-wide result.

Four evaluator tests passed: contact gating/latching, exclusion of occlusion
placeholders during latent inference, grid bounds, and raw-body position scoring.
The complete evaluation ran on CPU. Heatmaps and a representative GIF frame
were visually inspected. Training and Slurm jobs were not stopped or modified.

## Artifacts

- `success_heatmaps.png`: requested-goal rates and juggling control, all models.
- `index.html`: heatmaps and 15 corner/center example GIFs (first paired seed,
  no selection for success), with green target and blue true-puck dot.
- `summary.json`: goal rates, contact rates and inference support.
- `comparison.json`: aggregate paired bootstrap comparison.
- Per-model `episodes.jsonl`: all paired episode distances for further analysis.

## Conclusion

This zero-shot goal-inference method does not yield reliable aiming yet. The
1M full-history model frequently contacts the puck but visits requested goals
less often overall than its juggling latent. Both architectures are too early
at 50k to draw a useful architecture comparison. This does not establish that FB
cannot learn aiming: it evaluates the existing reward-free checkpoints and this
specific inference method, without goal-conditioned fine-tuning.

## Next

Repeat at matched later checkpoints, especially current-only 1M. Investigate
inference support and spatial discrimination in B before changing training.
A tighter physical-hit test could record substep trajectories; endpoint checks
can miss fast crossings, even at 10 cm tolerance.
