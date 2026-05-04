# Paper — Air Hockey RL

Working folder for the paper. Keep it simple.

**Target venue:** CoRL 2026 — initial submission is anonymous (`\usepackage{corl_2026}` without options). Switch to `[final]` for camera-ready or `[preprint]` for arxiv. Provisional title: *Sim-to-Real with RL Fine-Tuning in Dynamic Air Hockey Tasks*.

## Layout

```
paper/
├── README.md       ← you are here. Read first.
├── main.tex        ← THE PAPER. LaTeX, CoRL 2026 style. Source of truth for prose.
├── outline.md      ← evolving outline + claim list. Edit freely.
├── notes.md        ← scratch — open questions, TODOs, things to ask the user.
├── figures/        ← plots, GIFs, diagrams. Source scripts live next to the figure.
└── data/           ← raw numbers backing each claim (CSVs, eval logs, tables).
```

**Build:** from `paper/`, run `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`. Output is `main.pdf`.

`corl_2026.sty` and `corlabbrvnat.bst` are the **official CoRL 2026 files** (provided by user 2026-04-30). The submission build (no package option) auto-applies double-blind anonymization (`Anonymous Author(s)` overrides `\author{}`), adds line numbers, and prints the "Submitted to the 10th Conference on Robot Learning (CoRL 2026). Do not distribute." footer on the first page. Switch to `[final]` for camera-ready or `[preprint]` for arxiv.

Promote sections out of `main.tex` into a `sections/` directory and `\input{}` them only once `main.tex` is too big to navigate (>~600 lines). Don't pre-split.

## What this paper is about

Reinforcement learning for a physical air-hockey robot (UR5 + paddle), with **TD3 + dual-head critics + transformed Bellman targets**, including a residual-RL recipe that survives sim2sim and sim2real distribution gaps. See `../CLAUDE.md` for the project-level summary.

## How an agent should work in this folder

1. **Before writing anything**, read `outline.md` to see the current claim structure, then `notes.md` for open questions. Don't redo work already logged there.
2. **Source of truth for claims** is the docs under `../notes/docs/` and the experiment logs under `../notes/scratch/`. Always cite the doc you pulled a number from in `data/` (filename + section).
3. **Numbers go in `data/`, prose goes in `main.tex`.** When you state a result in the draft, the supporting numbers (with seed counts, SE, checkpoint info) must exist in `data/` under a matching filename. If a claim has no backing file, mark it `\todo{CITE}` (or a plain `% TODO:` comment) in the .tex and add a TODO to `notes.md`.
4. **Figures**: drop the image in `figures/` and the script that produced it next to the image (e.g. `figures/residual_drift.pdf` + `figures/residual_drift.py`). Prefer PDF/PNG over GIF for the paper itself; for Box2D GIFs in supplementary material follow the project convention (BGR→RGB, width 160, fps 20 — see `../.cursor/rules/box2d-environment.mdc`).
5. **Don't fabricate or estimate numbers.** If a result isn't in `notes/docs/` or `notes/scratch/`, flag it in `notes.md` and ask before filling it in.
6. **Anonymity for initial submission.** CoRL 2026 initial submission is double-blind. Don't add author names, affiliations, or self-citations in a non-anonymous form. The `\author{}` placeholder is only rendered with `[final]` or `[preprint]` options.
7. **Don't fix typos or rewrite the user's existing prose** unless explicitly asked. Log issues to `notes.md` instead.

## Key source material (in `../notes/docs/`)

Headline results likely to anchor the paper:

- `training/residual-rl-recipe.md` — winning recipe for sim2sim/sim2real residual fine-tuning. Headline numbers and the small-gap vs big-gap distinction live here.
- `training/sim2sim.md` — sim2sim transfer protocol + 400k extension results.
- `training/td3-algorithm.md` — algorithm description (dual-head critics, transformed Bellman).
- `training/td3-ablations-updates-and-depth.md` — depth / update-ratio ablations.
- `training/td3-exploration-ablations.md` — exploration ablations.
- `training/architecture.md` · `training/network-architecture.md` — model architecture.
- `training/reward-shaping.md` — reward formulation.
- `environments/real-world/puck-system-id.md` · `environments/real-world/teleop-system-id.md` — sysid (justifies the sim parameters).
- `environments/real-world/overview.md` — real-robot stack (UR5, safety, latencies).
- `environments/box2d/simulator-essentials.md` — sim description.
- `repo/project-goal-and-safety.md` — safety story for the real-robot section.

Raw experiment logs (use these for per-seed numbers and trajectory shape, not just summary tables):

- `../notes/scratch/residual_rl_drift_fix_log.md` — small-gap residual recipe campaign.
- `../notes/scratch/residual_rl_paddle50_log.md` — big-gap (paddle -50%) campaign; v25 is the current winner.
- `../notes/scratch/exploration_optimization_plan.md` — exploration sweep status.

## Conventions for this folder

- One claim per line in `outline.md`, each with a pointer to where the evidence lives (doc path or scratch log).
- Don't reduce a noisy multi-seed trajectory to a single peak number — show shape (or min/max post-peak) and always note SE for single-seed claims. (See user feedback memory `feedback_no_clean_summary_overclaim`.)
- Convert relative dates to absolute (`2026-04-30`, not "yesterday") when logging in `notes.md`.
- Prefer editing `main.tex` and `outline.md` over creating new top-level files.
- Use `\citep{}` / `\citet{}` (CoRL natbib style); add bib entries to `example.bib` (the file `main.tex` references).
