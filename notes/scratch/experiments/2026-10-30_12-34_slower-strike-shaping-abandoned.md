# Slower strikes for more sustainable juggling (abandoned)

- **Date**: 2026-10-30 12:34 CDT start, abandoned 2026-10-30 ~16:30 CDT
- **Status**: abandoned by the user after real-robot tests ("I don't think slowing down is a good direction"). No 5-seed confirmation, no saved policies.
- **Follows**: [`2026-10-30_12-47_strike-shaping-edge-termination-5seeds.md`](2026-10-30_12-47_strike-shaping-edge-termination-5seeds.md) (`strike_term03`, 79.6 ± 4.8 % reach 10k, puck out 4.15 m/s)
- **Run dirs**: `runs/ceiling/slow_screen1/` (finished), `runs/ceiling/slow_screen2/` (stopped at ~1.9M of 3M steps, not evaluated)
  - evals: `runs/ceiling/evals/selection_long/sellong__slow_screen1/`
  - hit metrics: `runs/ceiling/hit_motion/{slow_refs,slow_screen1_top,slow_screen1_mid}/`
- **Configs**: `configs/ceiling/slow_screen1.yaml`, `configs/ceiling/slow_screen2.yaml`
- **Code**: `fast` term in `scripts/rma/motion_shaping.py` (`shape_fast_coef/min/ref`, default off): on contact steps, −fast_coef · clip((|v_puck| − fast_min) / fast_ref, 0, 1)

## Question

The strike recipes send the puck out at ~4.2 m/s. Does a lower puck-exit speed, with strikes kept, make juggling more sustainable (goal ~99 % reach 10k)?

## Setup

`term03_weak3_pout2` base (edge time penalty 0.5, edge-contact termination at 3 cm, weak-contact penalty 3 below 0.3 m/s), with the puck-exit bonus capped lower (`shape_pout_ref` 2.5 / 3.0) and/or the new penalty on exits above 3.0 / 3.5 m/s. 10 variants, seed 0, 3M steps.

- Selection: `selection_long`, 32 episodes, 10k cap, checkpoints 1.5M–2.9M plus the final model.
- Hit metrics: `hit_motion`, 10 episodes, 2k cap, on the best checkpoints.

## Results (single seed, single checkpoint each)

| group | puck exit (median) | paddle speed into puck | reached 10k (32 eps) |
|---|---:|---:|---:|
| `strike_term03` seed 0 (reference) | 3.96 m/s | 0.62 m/s | 80 % (final test) |
| little or no exit penalty (p30, p30_f35c1, p30_f35_v05) | 3.9–4.4 | 0.5–0.8 | 62–81 % |
| middle band (p25_f35, p30_f35c3r1, edgehit3_v05_p30_f35) | 3.0–3.5 | 0.3–0.4 | 84–94 % |
| exit penalty above 3.0 (nopout_f30, p25_f30*) | 1.8–2.7 | 0.03–0.3 | 97–100 % |

- Every checkpoint had 0 % contacts within 2 cm of the edge, so none went back to edge dribbling.
- Across the 5 `strike_term03` seeds, the two hardest hitters (median exit about 4.5 m/s) were the two lowest at 10k (75 %, 64 %).

## Conclusion

- Lower exit speed and higher sim survival go together.
- The near-100 % variants get there by blocking (the paddle almost stops) rather than striking softer.
- The middle band (exit 3.0–3.5 m/s) reached 84–94 % on one seed, but the paddle also slowed to 0.3–0.4 m/s.
- Screen 2 was meant to keep the exit band and require a real swing (weak-contact penalty 5 below 0.4 m/s, paddle-speed bonus). It was stopped before evaluation, after the user's real-robot runs pointed away from slower play.
