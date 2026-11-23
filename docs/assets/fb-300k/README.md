# FB performance with 300k reward-inference samples

Recorded 2026-11-23. Full checkpoint: 29,621,664 steps; current-only: 29,966,976.
Each model uses its saved 300k-sample reward-inferred latent for the task and deterministic mean actions.
Episode seeds 100000 and 100001 are shared across models. These are fixed episodes, not selected best performances.

| Task | Full episode 0 | Full episode 1 | Current episode 0 | Current episode 1 |
|---|---|---|---|---|
| juggle | [full 0](full_juggle_ep00.gif) | [full 1](full_juggle_ep01.gif) | [current 0](current_juggle_ep00.gif) | [current 1](current_juggle_ep01.gif) |
| touch | [full 0](full_touch_ep00.gif) | [full 1](full_touch_ep01.gif) | [current 0](current_touch_ep00.gif) | [current 1](current_touch_ep01.gif) |
| reach | [full 0](full_reach_ep00.gif) | [full 1](full_reach_ep01.gif) | [current 0](current_reach_ep00.gif) | [current 1](current_reach_ep01.gif) |

[Controlled 10k vs 300k reward comparison](../../../notes/scratch/experiments/2026-11-23_16-28_fb-10k-vs-300k-rewards.md).
