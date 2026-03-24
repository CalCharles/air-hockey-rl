# TD3 Exploration Primitives (High Level)

This note summarizes the exploration primitives used by TD3 in
`scripts/smooth_policy/amp_history/amp_training/td3/helper/exploration_selector.py`.

## How primitive takeover works

- A primitive is sampled with probability `exploration_primitive_chance`.
- The chosen primitive controls actions for `exploration_primitive_steps` (or target-specific steps).
- Primitive selection is a weighted mixture from `exploration_primitive_weight_*`.
- `exploration_primitive_chance_start -> exploration_primitive_chance` controls annealing.

## Primitive types

### 1) Stand still

- Purpose: briefly stabilize and reduce aggressive motion.
- Behavior: outputs near-zero action.
- Main knob: `exploration_primitive_weight_stand_still`.

### 2) Same direction

- Purpose: maintain coherent movement direction for a few steps.
- Behavior: samples one direction + magnitude, then repeats that directional move.
- Main knobs:
  - weight: `exploration_primitive_weight_same_direction`
  - optional range: `exploration_same_direction_*` (angle/magnitude bounds)

### 3) Y-aligned direction

- Purpose: bias exploration along table y direction (often useful for interception).
- Behavior: samples y-oriented motion; sign can align with puck-relative y context.
- Main knobs:
  - weight: `exploration_primitive_weight_y_aligned`
  - optional range: `exploration_y_aligned_*`

### 4) Policy takeover (warm policy)

- Purpose: inject known-good behavior from a warm-start policy.
- Behavior: uses a loaded policy action instead of random primitive action.
- Main knobs:
  - enable: `exploration_policy_takeover_enabled`
  - weight: `exploration_primitive_weight_policy_takeover`
  - model/config paths: `exploration_policy_takeover_*`

### 5) Target-position directional

- Purpose: move toward a sampled local target position.
- Behavior: samples a displacement target, projects to feasible action, then repeats.
- Main knobs:
  - weight: `exploration_primitive_weight_target_position_directional`
  - steps: `exploration_target_position_steps`
  - workspace bounds: `exploration_target_position_delta_x`, `exploration_target_position_delta_y`
  - optional range: `exploration_target_position_directional_*`

### 6) Pre-contact hit variant

- Purpose: trigger an upward-contact style variant near puck contact conditions.
- Behavior: activates only under trigger conditions (distance, upward move, puck motion).
- Main knobs:
  - chance: `exploration_pre_contact_hit_variant_chance`
  - steps: `exploration_pre_contact_hit_variant_steps`
  - trigger thresholds/scales: `exploration_pre_contact_hit_variant_*`

## Notes on angle/magnitude range parameters

- For each primitive family (`same_direction`, `y_aligned`, `target_position_directional`):
  - if range fields are unset (`None`), legacy sampling is used;
  - if set, all four fields are required: min/max angle and min/max magnitude.
- Angles are in simulator displacement coordinates.
- Magnitudes represent simulator-space step displacement before action-space projection.
