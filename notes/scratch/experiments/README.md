# Experiment notes — additive-file convention

**Read this before writing any experiment notes or follow-ups.**

We've had repeated git merge conflicts because multiple agents (and the
human operator) all tried to append to the same long-lived files like
`residual_rl_paddle50_log.md`. Going forward, **every experiment writeup
goes in its own new file in this directory**, dated. Existing long-lived
files are frozen — read them for history, don't append.

## Naming

```
notes/scratch/experiments/YYYY-MM-DD_HH-MM_<topic-slug>.md
```

- `YYYY-MM-DD_HH-MM` — UTC start time of the experiment / writeup. Time
  granularity prevents collisions if two agents write at the same date.
- `<topic-slug>` — kebab-case, max ~40 chars. Examples:
  `v30-explore-5seed-validation`, `from-scratch-5M-saturation`,
  `hist2_motion0-v2-retrain`.

Examples:
- `2026-05-04_02-40_v30-explore-lite-5seed.md`
- `2026-05-04_04-21_from-scratch-5M-saturation.md`
- `2026-05-05_02-55_hist2-motion0-v2-retrain.md`

## File contents

Lead with a frontmatter-style header:

```markdown
# <topic, full sentence>

- **Date**: YYYY-MM-DD HH:MM UTC start
- **Status**: in-flight | done | superseded-by `<file>` | abandoned
- **Supersedes**: `<file>` (if applicable)
- **Run dirs**: `runs/td3/...` paths
- **Configs**: `scripts/.../*.yaml` paths

## Question
1-3 sentences on what we wanted to learn.

## Setup
Recipe / config / budget. Link to canonical recipe doc, don't restate.

## Results
Tables, numbers, trajectories. n-seed values must include std.

## Conclusion
What we now believe. Be honest about confidence and what's untested.

## Next
Open follow-ups. Each will get its own file in this directory.
```

## Rules

1. **Never edit a prior experiment file.** If results need correction,
   write a new file with `Supersedes:` pointing at the old one. The old
   one stays frozen as the historical record.
2. **Don't bulk-append to the long-form logs.** Files like
   `residual_rl_paddle50_log.md` are now read-only history. Future
   experiments go here.
3. **Cross-link, don't merge.** If your experiment relates to another,
   link to it. Don't combine writeups.
4. **One file per experiment.** Even tiny ones. If you re-run a recipe
   with one knob changed, that's a new file.
5. **Update `INDEX.md` when you finish.** One line, additive only.
6. **Reflect headline findings in canonical docs.** The recipe doc
   (`notes/docs/training/residual-rl-recipe.md`) and `CLAUDE.md` are
   still maintained — but only update them with **stable conclusions**,
   not ongoing experiment chatter. Reference the experiment file as the
   source of truth for the data.

## Index

Maintain `notes/scratch/experiments/INDEX.md` — one line per file,
chronological. Format: `- YYYY-MM-DD HH:MM | <topic> | <one-sentence outcome> | [file](path)`.

The index is additive: only insert at the bottom of the list. Past
entries are immutable.
