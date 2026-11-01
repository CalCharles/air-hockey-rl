# Exact action replays distinguish actual PID-target changes from a displacement-change proxy

- **Date**: 2026-11-01 22:59 UTC retrospective diagnostic recording.
- **Status**: done
- **Run dirs**: original history seed 0 `final_6m_paired_20k_eval/` and penalty-0.05 history seed 0 `paired_6000000_online_20k_eval/` under `runs/td3/`.
- **Related**: [strong-penalty results](2026-11-01_17-06_smooth-history-results.md), [active weaker sweep](2026-11-01_22-48_weaker-smoothness-sweep.md).

## Measurement correction

The existing evaluator's `target_delta_rms_m` computes RMS of `(action[t] - action[t-1]) * [action_x_ratio, action_y_ratio]`. This is a scaled **displacement-request change**, not change in the absolute filtered/clipped PID target. The distinction matters because target construction also includes the current paddle position, workspace clipping and the established action-history filter. Historical raw values remain intact; interpret earlier target-change labels as this proxy, not a direct measurement of the PID target.

The standalone [replay diagnostic](../../../scripts/analysis/juggle_control_replay.py) replays saved actions and reads cached `last_target_position` plus physical body velocities. It does not call sensor-sampling `get_current_state`, change simulation/controller code, or load/train a policy. It verifies every pre/post observation, reward, terminal/truncation flag, true paddle position and true puck position against saved arrays. All **32 baseline episodes / 7,076 replayed steps** and **32 strong-penalty episodes / 3,035 replayed steps** matched exactly. Replay was limited to the first 250 steps of each episode or its actual end, whichever came first. This is additional diagnostic simulation of existing action sequences, not a new policy-quality test.

## Results

Both policies are final 6M online actors from seed 0, evaluated on the same 32 reused starts 700000–700031. The reported RMS values average per-episode RMS values, not pooled timestep RMS. Both direct target differences and the same-prefix displacement proxy below exclude the undefined reset-target/first-action transition.

| Metric over replay prefixes | Original baseline | Penalty 0.05 |
|---|---:|---:|
| Mean actual PID-target change RMS (m) | 0.054785 | 0.014940 |
| Mean scaled displacement-change RMS (m), same edges | 0.076177 | 0.009675 |
| Median episode p95 physical speed (m/s), cached velocity | 1.8254 | 0.9430 |
| Median episode p95 acceleration (m/s²), velocity differences | 19.3528 | 5.6634 |
| Median episode p95 jerk (m/s³), second velocity differences | 510.0387 | 87.5536 |

The strong penalty reduced sampled motion in these observed prefixes but still lost juggling performance. The matched 20k-protocol means remain **77.53 vs 0.84 physical juggles**, **5,447 vs 94.84 episode steps** for baseline vs strong penalty. These diagnostic changes do not make the strong-penalty policy useful.

The cached target is the final PID update's target within a control step; it is not a complete substep target trace. Physical derivatives use cached body velocity at 20 Hz, differing from the position-finite-difference diagnostic introduced for the sweep. Neither measures UR5 joint loads, controller tracking error or protective-stop risk. Different episode lengths and visited states remain confounders: limiting each episode to a maximum prefix does not equalize exposure when episodes end sooner. No binary acceptance gate is used.

## Existing GIF now available in tracked docs

![Strong-penalty final online episode](../../../docs/assets/juggling/penalty-0p05-6m-online-episode700000.gif)

This is the existing evaluation GIF, copied byte-for-byte on 2026-11-01, not a new rendered trajectory. It shows all 106 steps (5.3 s at 20 fps) of seed 700000, with **one physical juggle**. It illustrates limited movement followed by puck loss, not the 32-episode average. [GIF metadata/source/hash](../../../docs/assets/juggling/control-replay-gif-provenance.json). For a separate historical high-performance clip, see the [3.6M target segment](../../../docs/assets/juggling/noisy-history-target-3m6-end.gif); that is a different checkpoint and selected episode, not this audit's baseline 6M online actor. The original baseline matched 6M evaluation did not save a GIF, so no equivalent clip is claimed here.

## Sources and reproduction

- [Baseline exact-replay rows and trajectory hashes](../../../docs/assets/juggling/baseline-pid-target-replay.json).
- [Strong-penalty exact-replay rows and trajectory hashes](../../../docs/assets/juggling/strong-smooth-pid-target-replay.json).
- [Replay source](../../../scripts/analysis/juggle_control_replay.py).
- Original ignored result JSON/NPZ sources are named and hashed in the exported records; source paths remain unchanged.

From the repository root, run `.venv/bin/python -m scripts.analysis.juggle_control_replay --results RESULTS_JSON --out OUTPUT_JSON --prefix-steps 250`. It requires the saved original trajectory files and canonical simulator. No actor checkpoint is needed. The same-prefix scaled proxy was calculated from the verified saved action arrays; the script reproduces that calculation. Direct physical telemetry is kept outside policy inputs and all running trainer/evaluator sources remain unchanged.

Checks actually performed: 64 episodes with exact replay-array comparisons, source compilation, output/source hashes, reviewed first/last GIF frames, GIF frame count/duration, relative links/images, ordered logbook TOCs/unique anchors, and whitespace. No new training for this audit. All six learners and CPU-only long reference job remained live at inspection; weaker-trial first 100k results are preliminary and do not justify early stopping.
