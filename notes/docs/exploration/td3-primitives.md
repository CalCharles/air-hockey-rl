# TD3 Exploration Primitives (High Level)

This note summarizes the exploration primitives used by TD3 in
`scripts/td3/helper/exploration_selector.py`.

As of 2026-05-20 the primitive set was reduced to **two** primitives
(stand_still + same_direction). The earlier `y_aligned` and
`target_position_directional` primitives were removed after an ablation
showed the 2-primitive subset matches the full 4-primitive set with no
regression — see
[`notes/scratch/experiments/2026-05-20_07-39_simplified-exploration-ablation.md`](../../scratch/experiments/2026-05-20_07-39_simplified-exploration-ablation.md).

## How primitive takeover works

- A primitive is sampled with probability `exploration_primitive_chance`.
- The chosen primitive controls actions for `exploration_primitive_steps`.
- Primitive selection is a weighted mixture from `exploration_primitive_weight_*`.
- `exploration_primitive_chance_start -> exploration_primitive_chance` controls annealing.

## Primitive types

### 0) Stand still

- Purpose: briefly stabilize and reduce aggressive motion.
- Behavior: outputs near-zero action.
- Main knob: `exploration_primitive_weight_stand_still`.

### 1) Same direction

- Purpose: maintain coherent movement direction for a few steps.
- Behavior: samples one direction + magnitude, then repeats that directional move.
- Main knobs:
  - weight: `exploration_primitive_weight_same_direction`
  - legacy sampling skew: `exploration_direction_y_component_weight`
  - optional simulator-space range: `exploration_same_direction_*` (angle/magnitude bounds)

## same_direction sampling modes

- **Legacy (range fields unset / `None`)**: uniform unit direction (with a
  y-component-weight bias from `exploration_direction_y_component_weight`) ×
  uniform magnitude in [0.1, 1.0].
- **Simulator-space range (all four range fields set)**: direction sampled
  from `[min_angle_deg, max_angle_deg]`, magnitude from
  `[min_magnitude, max_magnitude]` (simulator-space per-step displacement),
  then projected into the action box defined by `exploration_action_delta_x`
  / `exploration_action_delta_y`. Setting any one range field requires all
  four. Angles are in simulator displacement coordinates.

The real-world collector (`configs/td3_real_world/td3_online.yaml`) uses the
simulator-space range mode; the canonical sim configs use the legacy mode.
