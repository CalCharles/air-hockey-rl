# Project goal and safety



## Project goal

Build a puck-juggle policy (referencing LinearTop) in Box2D that transfers to a real UR5 air-hockey setup on a tilted table.

## Training strategy

- Use demonstration-constrained discriminator training so policy motion stays behaviorally similar to expert trajectories.
- Use RMA for robustness to dynamics variation and improved sim2real transfer.
- Prioritize stable, smooth juggling behavior over aggressive short-term corrections.

## Real-robot constraints

- The UR5 platform has e-stops; avoid jerky, high-acceleration, fast-adjusting actions.
- Prefer control and reward choices that reduce motion spikes while maintaining successful juggling.
- When uncertain, favor safer/smoother behavior that preserves hardware safety.
