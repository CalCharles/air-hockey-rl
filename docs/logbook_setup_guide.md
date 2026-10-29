# Instructions for setting up research and agent logbooks

Give this entire file to an agent working in another repository. It is standalone;
no access to the source repository is required.

## Requested outcome

Create `docs/experiment_logbook.md` and `docs/agent_logbook.md`, each with a table
of contents and dated, timed entries. Create persistent repository instructions
so that future requests such as “update log books” update both documents using
actual work and evidence. Add discoverable links from the repository README.
Proceed with the documentation work without launching new experiments.

## Setup procedure

1. Read applicable repository instructions and inspect README, Git history,
   experiment scripts, saved results, figures, and existing documentation.
2. Preserve existing documentation. Create the two logbooks under `docs/` and
   add or merge the maintenance rules below into the root `AGENTS.md`; do not
   overwrite unrelated instructions.
3. Backfill meaningful experiments and agent tasks supported by evidence. Group
   related experimental runs into studies when useful, listing individual runs
   in a table or auxiliary document. Do not equate every commit with an experiment.
   If original prompts or agent identities are missing, label task entries as
   reconstructions from commits rather than claiming knowledge of agent activity.
4. Label backfilled timestamps explicitly: author commit timestamps are recording
   milestones, not experiment start/end times. State the backfill date and sources.
   Use actual timestamps for new work and explicit numeric UTC offsets. Use the
   user's timezone when known, otherwise UTC. Never fabricate exact run times.
5. Embed saved experiment plots and figures with relative Markdown image links,
   meaningful alt text, and an explanation of what they show. Include failures
   and negative results. If no figures exist, say so; do not fabricate evidence.
   Label any new figure generated from historical data with its source and date.
6. Cross-link related experiment and task entries. Put extensive analyses,
   derivations, and large result tables in auxiliary files under `docs/`, retaining
   a concise finding and a link in the logbook.
7. Validate relative links, image existence, unique IDs/anchors, and TOC coverage.
   Do not rerun expensive training or claim old checks as fresh checks. Report
   files created, the historical scope covered, and validation actually performed.

## Persistent maintenance rules

Place these rules in the repository's existing agent-instruction convention:

- Interpret “update log books” and spelling variants as a request to reconcile
  both logbooks against work since the last recorded entries. Inspect recent
  conversation, changes, commits, and artifacts; append only new substantive work.
- Record substantive agent tasks in the agent logbook and experiments run or
  analyzed in the experiment logbook. Documentation-only work needs no experiment.
  Logbook maintenance does not authorize additional experiments.
- Maintain chronological order and permanent sequential IDs, independently for
  experiments (`E001`) and tasks (`A001`). Include a stable explicit anchor for
  every entry and update the TOC when adding entries.
- Start every entry with a single sentence summarizing its task or finding, then
  provide details. Headings contain date, time, numeric UTC offset, ID, and title.
- Distinguish completed, running, failed, interrupted, and planned work. Report
  evidence, configurations, commands, limitations, and outcomes truthfully.
- Preserve history. Add dated corrections or status updates instead of silently
  rewriting prior findings. Avoid duplicate entries; link related records.
- Keep plots visible inside experiment Markdown, with links to raw data,
  configuration, checkpoints, and reproduction commands when available.
- Store long supporting material in `docs/` and link it from the relevant entry.
- Check links and TOCs before finishing. Run implementation tests when warranted
  by code changes, not merely for prose edits.

## Experiment logbook template

Replace placeholders with evidence-backed values. Duplicate the entry structure
for each study, assigning the next ID and adding its TOC link.

```markdown
# Experiment logbook

Scope, timezone convention, historical backfill provenance, and links to related logs.

## Table of contents

- [E001 — Short title](#e001)

<a id="e001"></a>

## E001 — YYYY-MM-DD HH:MM:SS ±HH:MM — Short title

One sentence stating the experiment and its main finding or current status.

**Question:** What hypothesis or comparison motivated this experiment?

**Timing and status:** Actual start/end if known, or explicitly labeled recording
 time; completed/running/failed/interrupted/planned.

**Setup:** Environment, dataset/coverage, model, objective, seeds, hyperparameters,
 budgets, baseline, and evaluation protocol.

**Results:** Quantitative metrics with denominators, horizons, and seed variability;
 distinguish saved historical results from fresh measurements.

![Description of the actual figure](visualizations/study/figure.png)

Explain the figure and link raw data, including failures. If no figure exists,
 replace the image with an explicit statement that none is available.

**Interpretation and limits:** What the result supports and what remains unresolved.

**Artifacts and reproduction:** Relative links and the actual invocation or script.

**Related task:** [A001](agent_logbook.md#a001).
```

## Agent logbook template

```markdown
# Agent logbook

Scope, timezone convention, and historical reconstruction provenance.

## Table of contents

- [A001 — Short title](#a001)

<a id="a001"></a>

## A001 — YYYY-MM-DD HH:MM:SS ±HH:MM — Short title

One sentence describing the task and outcome or current status.

**Scope:** Requested work and relevant constraints.

**Work and decisions:** Changes, investigation, and rationale.

**Deliverables:** Relative file links, commits, or other reviewable artifacts.

**Validation:** Commands/checks actually performed, results, and verification limits.

**Unresolved work:** Remaining issues or next steps when applicable.

**Related experiment:** [E001](experiment_logbook.md#e001), if applicable.
```
