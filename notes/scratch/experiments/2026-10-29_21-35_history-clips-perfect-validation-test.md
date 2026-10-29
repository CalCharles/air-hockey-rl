# Verified 3M rollout clips and the first perfect validation milestone

Recorded 2026-10-29 16:35:02 -0500. Two rendered 3M episodes reproduce the original independent-test trajectories exactly. Separately, the first 8/8 capped validation checkpoint at3.6M is frozen for a new independent test, currently running.

## Completed renderer checks: 3M checkpoint

All eight saved arrays match over the full episode for both replays. [Array digests and provenance](../../../docs/juggling-history-3m-gif-provenance.json). The full3M independent test remains5/32caps at20k and27failures; these clips are not new independent evidence.

![Selected 3M success, final20seconds](../../../docs/assets/juggling/noisy-history-3m-cap-end.gif)

Seed600013 was selected after reaching the20k cap:310physical juggles,1000simulated seconds. This clip ends at censoring, not failure.

![First 3M independent-test seed, final20seconds](../../../docs/assets/juggling/noisy-history-3m-first-failure.gif)

Seed600000 is the first predeclared test seed: failed at8862steps,133physical juggles (443.1simulated seconds). This is a different episode. Both clips show the last20seconds at160×357,10fps,64colors, with durations preserved. Generated2026-10-29; both last decoded frames were inspected. On-screen Contacts counts contact timesteps, not distinct juggles.

```sh
# Same frozen3M model, one seed at a time, choose a new output directory.
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/train/checkpoint_3000000 --out-dir NEW_OUTPUT --scenario-set training --horizon 20000 --episodes 1 --seed-base 600013 --gif-steps 400
# Repeat with seed600000 for the failed episode.
```

Local replay roots: `selected_demo_3m_seed600013_20k/` and `first_demo_3m_seed600000_20k/` under the history seed0 run. Demo directories remain excluded from aggregate benchmark discovery.

## Running independent3.6M test

Checkpoint3.6M is the first saved history checkpoint with all8reused validation starts reaching5k and satisfying the physical repeat-juggle heuristic. All earlier saved cap fractions were checked below1.0. This is a validation milestone, not proof of reliability.

Selected that checkpoint before a new32-start20k test on seeds700000–700031. Scanned existing noisy-campaign result ranges first; none overlapped. Recorded model SHA256 `97e1184364f9f6a0c00406c8608bbc8190a82e60629f4a593eb483125ce95351` before launch and verified canonical saved environment. [Pre-test provenance](../../../docs/juggling-history-3m6-test-provenance.json). Raw output: `runs/td3/noisy_history_20261029_seed0/independent_3m6_20k_eval/`. Status **running; no result yet**. These seeds become consumed research starts after use. Do not interpret repeated validation success as independent acceptance. The fixed4M watcher remains live and unchanged.

No new training or simulator/control/DR changes. Three history learners continue. Checks: all full trajectory arrays, frozen-model hash, prior validation fractions, seed-range audit, GIF duration/decoded frames, saved-data report discovery, relative links/images, anchors/TOCs and whitespace. No implementation tests or expensive training reruns for documentation.
