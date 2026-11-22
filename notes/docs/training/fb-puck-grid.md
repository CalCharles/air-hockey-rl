# Zero-shot FB puck-goal grid evaluation

`scripts/fb/evaluate_puck_grid.py` evaluates frozen FB policies without training
or changing the running jobs. It supports full-history and current-only backward
models through the shared checkpoint loader. Evaluation runs on CPU with one
PyTorch thread.

```bash
.venv/bin/python -m scripts.fb.evaluate_puck_grid \
  --checkpoints runs/fb/realplant_seed0_20261121/final \
    runs/fb/realplant_seed0_20261121/step_000050000 \
    runs/fb/realplant_current_backward_seed0_20261122/step_000050000 \
  --labels full_1M full_50k current_50k \
  --output runs/fb/puck_grid_evaluation \
  --grid-size 5 --radius 0.05 --episodes 20
```

## Protocol

- Goals form a 5×5 grid in the upper half of the table. Centers are inset by
  puck radius + goal radius. Coordinates are table/base-frame metres.
- Infer a separate latent for each goal from the checkpoint's saved inference
  sample, using `exp(-0.5 * (distance / radius)^2)` on the newest observed puck
  position. Exclude occluded observations. This marginalizes over the other
  backward-model inputs rather than constructing an arbitrary full target history.
  No goal-conditioned policy fine-tuning is performed.
- The saved samples lack raw physics puck positions or an episode-level contact
  flag. Therefore inference uses observed position only, without a contact gate.
  Report sample counts within the radius and effective weighted sample counts.
- Score success only after an actual paddle–puck collision in the episode, when
  the **raw physics puck center** is within 5 cm of the requested goal at a
  control-step boundary. Also report 10 cm tolerance on the same trajectories.
  This is a point check, not continuous trajectory intersection; fast crossings
  between steps can be missed.
- Raw physics is read from `env.simulator.pucks` and converted with
  `_box2d_to_base_coords`. The public `current_state` is **not ground truth**:
  this simulator has already applied delay/noise/occlusion to it.
- Reuse the original juggling spawn distribution, real-plant physics, two-step
  action latency, observation pipeline, and episode endings. Goal position is
  conveyed through z, not appended to observations. Keep rolling after a hit
  until the ordinary episode end to measure visits to other goals as well.
- Use 20 paired seeds (default 300000–300019), deterministic actions, and the
  same seeds for every goal and model. A juggling-latent control uses those same
  seeds and is scored against all goals, exposing incidental goal visits.
- Evaluate full and current-only at the same training step for an architecture
  comparison. A 1M-step full model versus a 50k current model is an unequal-budget
  deployment comparison, not evidence of architecture superiority.

## Artifacts

`success_heatmaps.png` compares goal conditioning against the juggling control.
`summary.json` and per-model summaries contain per-goal rates, contact rates,
closest distances after contact, inference support, and a cross-goal visitation
matrix. `episodes.jsonl` retains paired episode-level distances for analysis.
`index.html` displays the heatmaps and representative clips: corners and center,
first evaluation seed, without selecting for success. Green marks the goal;
a blue dot marks the true puck position over the observation-based renderer.

With 20 episodes per goal, rates change in 5-percentage-point increments.
Uncertainty and the control comparison matter more than small cell differences.
