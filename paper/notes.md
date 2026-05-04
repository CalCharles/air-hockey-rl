# Notes — open questions, TODOs, scratch

Use absolute dates (`2026-04-30`), not "yesterday". Don't delete answered questions; mark them resolved with the answer.

## Open questions for the user

- [x] **Target venue / format.** ~~Resolved 2026-04-30:~~ CoRL 2026, LaTeX with `corl_2026` style. Initial submission is anonymous. Page limit: TBD (confirm against the official CoRL 2026 CFP).
- [ ] **Headline framing.** Current `main.tex` abstract pitches the story as sim-to-real → online RL fine-tune beats demo-based and pure-online baselines. Memory-side framing emphasized residual RL as the main recipe. Are these the same story (residual RL = the "fine-tune" mechanism) or two different angles to reconcile?
- [ ] **Sim2real scope.** Which real-robot results are in the paper vs follow-up work? Do we have evaluation numbers from the real UR5 yet, or only sim2sim as a proxy? Abstract claims "exceeding even the capabilities of a human player" — need the underlying number.
- [ ] **Baselines.** Abstract names BC, offline RL, and pure online RL. Confirm which baselines have actual run numbers vs are aspirational.
- [ ] **Authors / acks.** Anonymous for initial submission, but need the full list locked in for `[final]`/`[preprint]` switch.

## TODOs

### Build / infra
- [x] ~~Replace `corl_2026.sty` with the official one~~ — done 2026-04-30. Official `corl_2026.sty` and `corlabbrvnat.bst` are now in `paper/`. Stub removed.
- [x] ~~Create `example.bib`~~ — done 2026-04-30 with placeholder Gauss1857 / Lagrange1788 entries to satisfy the template's `\citep`/`\citet` examples. Replace with real references as the bib grows. May want to rename to `references.bib` later.
- [ ] Decide whether `main.tex` should be renamed (e.g. `paper.tex` / `airhockey_corl2026.tex`). Default of `main.tex` is fine.
- [ ] Add `.gitignore` entries for LaTeX build artifacts (`*.aux`, `*.log`, `*.bbl`, `*.blg`, `*.out`, `*.synctex.gz`, `*.fdb_latexmk`, `*.fls`, `main.pdf` if you want to keep it out of the repo).
- Build command (from `paper/`): `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`. Current state: 6 pages, clean build.
- Note: `main.tex` preamble had `\usepackage{amsmath}` added (in addition to `booktabs`) because the official `corl_2026.sty` doesn't load `amsmath`, but the ablation table uses `\text{ms}`.

### Content (don't act on these without user confirmation — see open questions)
- [ ] Pull headline numbers from `notes/docs/training/residual-rl-recipe.md` into `data/residual-rl-headline.md` with provenance.
- [ ] Pull sim2sim 400k numbers from `notes/docs/training/sim2sim.md` into `data/sim2sim.md`.
- [ ] Decide section split for ablations (depth+updates vs exploration — one section or two?).
- [ ] Lit search for related work.

### Method-section follow-ups (added 2026-04-30 with user's method dump)
- [x] **Verify the method-section randomization values against the actual sim config.** ~~Resolved 2026-04-30:~~ randomization knobs (start dist, obs noise, occlusion, delay, action attenuation) match `sysid_best_params.yaml`. Collision randomization (10° cones + strength `U[0.5, 1.0]`) is currently `false` in the sysid_best_params config, but **user confirmed (2026-04-30) the runs reported in the paper will have it enabled** — assume it's part of the standard pipeline and don't caveat in the prose.
- [ ] **Decide whether to surface TD3-specific algorithm details** (dual-head critics, transformed Bellman) in the Method section. User's method dump didn't mention them — they may be intentionally omitted, or just not in scope yet. Ask before adding.
- [ ] Pin down what "loose" means quantitatively for the sysid section — which parameters did we tune and which did we leave alone?
- [ ] Evaluation protocol mentions sliding windows of 5/10/25/50. Confirm which window size is the headline metric vs supplementary.

### Ablation-section follow-ups (added 2026-04-30 with user's ablation dump)
- [x] **Confirm "online fine-tuning" sim2sim setting.** ~~Resolved 2026-04-30:~~ user clarified the spec is **35% smaller paddle**, not puck (was a typo). `main.tex` updated. Note: the existing paddle50 ensemble experiments (v26-v29) use a 50% smaller paddle, not 35% — the paper's setting is a new variant.
- [ ] **Online finetuning standard values quoted in the table** are pulled from the v25/v26 residual configs (residual_scale=0.15, sf=0.15, age_decay=1e-4, exploration_noise=0.05, num_critics=2). Confirm these are the "standard" baseline the user wants reported — these are big-gap numbers; small-gap recipe uses sf=0.5 and different settings.
- [ ] **Ensemble ablation results not in yet** — v26 (N=3), v27 (N=5), v28 (REDQ-5), v29 (REDQ-10) configs added but results pending as of 2026-04-30.
- [ ] System-identification ablation (off-the-shelf Box2D defaults vs sysid'd) — confirm we actually have run this, vs it being an aspirational ablation.
- [ ] **Real-world starting-states ablation (added 2026-04-30)** — described in §Ablations as initializing the sim's paddle/puck from the empirical distribution of real-world rollouts. No config or env code implements this yet. Need to: (1) decide the source distribution (teleop traces? prior real TD3 rollouts? both?); (2) add a sim option to sample the starting state from this distribution; (3) run the ablation against the synthetic mixture baseline.
- [ ] The exploration ablation in `main.tex` mentions "structured exploration primitives (directional pushes, target-position drives)" — these are off in residual configs but on in from-scratch training. Confirm whether the residual exploration ablation should toggle these, or only sweep noise std.
- [ ] **New paddle-35%-smaller sim2sim runs needed.** Per user 2026-04-30, the paper's online-finetuning sim2sim ablations should use a 35%-smaller paddle. No config exists for this yet (current paddle50 = 50% smaller). Will need a new sim2sim config + a fresh round of v25/v26-style runs against it.

### Things to fix later in the abstract / current draft (DO NOT silently fix — user said "don't add anything yet")
- Typo: `meningful` → `meaningful` (abstract).
- Typo: `acheives` → `achieves` (abstract).
- Grammar: `substantially improvement performance` → `substantially improved performance` (abstract).
- The `\author{}` block has a missing comma between `University of California Berkeley` and `United States`.
- The placeholder `\citep{Gauss1857}` and `\citet{Lagrange1788}` examples in §Citations will need to be removed before submission.
- §Experimental Results and §Conclusion are still lorem ipsum — replace wholesale, don't edit.

## Scratch

- _Empty for now. Use this for working thoughts that don't fit elsewhere._
