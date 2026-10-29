# Repository agent instructions

Read [CLAUDE.md](CLAUDE.md) for repository context and existing conventions.

## Juggling simulator contract

Stay on the current branch for this work. Preserve the dynamics and observations of Carl/Rohan's merged simulator at `94acd81932c59f3cd3b827d6f39a0283747a1beb` and the canonical `configs/new_juggle/sysid_v2_hist2.yaml` sensing/control recipe. This includes noise, occlusions, delay/interpolation, observation history sampling, PID, action smoothing, geometry, contacts and workspace limits. Do not simplify sensing or dynamics to improve juggling. Algorithm, policy and reward changes are authorized; clean-sim trials are historical diagnostics, not the deployment objective. Existing optional simulator modifications are not authorization to use them. Compare effective configs and simulator code before future training. Preserve the established randomization distribution unless the user requests otherwise.

New juggling policies must train from scratch: no pretrained actor/critic, checkpoint warm start or residual base policy.

## Logbook maintenance

- Interpret “update log books” and spelling variants as a request to reconcile both `docs/experiment_logbook.md` and `docs/agent_logbook.md` against work since their last entries. Inspect conversation, changes, commits and artifacts; add only substantive new work.
- Record substantive agent tasks in the agent logbook and experiments run or analyzed in the experiment logbook. Documentation-only work needs no experiment. Maintenance does not authorize additional experiments.
- Maintain reverse chronological order (newest first) in entries and TOCs and permanent independent sequential IDs (`E001`, `A001`), explicit stable anchors, and TOCs covering every entry.
- Start each entry with a one-sentence finding/task summary. Headings include date, time, numeric UTC offset, ID and title. Use America/Chicago and distinguish actual timestamps from historical author-commit recording milestones; never invent run times.
- Distinguish completed, running, failed, interrupted and planned work. Record evidence, configurations, commands, limitations and outcomes truthfully. Label reconstructed agent tasks and unknown identities explicitly.
- Preserve history; add dated corrections/status updates instead of silently rewriting findings. Avoid duplicates and cross-link related records.
- Embed existing plots/figures with relative image links and explain them. Link available raw data, configs, checkpoints and reproduction commands. State when figures are unavailable. Label newly generated historical figures with source and generation date.
- Put extensive supporting material under `docs/` and link it. Preserve existing dated experiment notes as sources, following CLAUDE.md for new detailed notes.
- Validate relative links, image existence, unique anchors and TOC coverage. Run implementation tests when code changes warrant them; do not rerun expensive training for documentation or present historical checks as fresh checks.
