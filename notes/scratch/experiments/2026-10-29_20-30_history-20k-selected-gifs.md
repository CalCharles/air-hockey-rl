# Selected 20,000-step history rollout GIFs

Recorded 2026-10-29 15:30:37 -0500. Replayed checkpoint2.1M on known successful seed500014 solely to render its clips. All eight saved arrays (observations, next observations, actions, rewards, terminal/truncation flags, true paddle/puck positions) exactly match the previous full20000-step trajectory. This verifies rendering did not change this episode's behavior. No training or simulator change.

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/train/checkpoint_2100000 --out-dir runs/td3/noisy_history_20261029_seed0/selected_2m1_cap_demo --scenario-set training --horizon 20000 --episodes 1 --seed-base 500014 --gif-steps 400
```

Local output contains raw GIFs, results and NPZ. This was a deliberately selected success, not a fresh independent test: full reused32-start20k benchmark has4caps and28failures (E020). Episode500014 has295physical-trajectory juggles and487contact timesteps; the GIF's Contacts counter counts timesteps, not distinct juggles. The cap is1000simulated seconds (16minutes40seconds), not proof of indefinite survival.

![Selected cap first10seconds](../../../docs/assets/juggling/noisy-history-20k-start.gif)

![Selected cap final20seconds](../../../docs/assets/juggling/noisy-history-20k-end.gif)

There is a time gap between these clips of the same episode. The second clip ends at the evaluation cap, not a failure. Generated2026-10-29; compacted to160×357,64colors,10fps with durations10s/20s preserved. A decoded frame was inspected. [Aggregate data](../../../docs/juggling-history-20k-comparison.csv), [earlier failure clips](../../../docs/assets/juggling/README.md).

The progress report now excludes demo directories from automatic benchmark discovery, alongside selected stretches and diagnostics; otherwise this selected1/1cap could misleadingly appear beside aggregate tests. Regeneration from saved data checks that exclusion. Documentation links, images, anchors, TOCs and whitespace checked; no expensive training launched for documentation.
