# post_polyak_fix — coarse rerun of v25–v30 paddle50 residual configs

These configs re-test claims that were originally measured against frozen
target networks (the Polyak gate bug fixed in `td3_training.py` on 2026-05-06).

**Design doc / question / results table**:
[`notes/scratch/experiments/2026-05-06_18-29_post-polyak-fix-rerun.md`](../../../../../../../../notes/scratch/experiments/2026-05-06_18-29_post-polyak-fix-rerun.md).

**Launcher** (from repo root):
```bash
bash scripts/smooth_policy/run_post_polyak_fix.sh <gpu_id>   # 0..3
```

5 configs, 1 seed each at 300k steps. Each row varies a single knob from
`fix_v27_baseline` so each delta isolates one previously-frozen-target finding.
