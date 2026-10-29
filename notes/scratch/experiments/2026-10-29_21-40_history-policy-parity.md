# Training/evaluation policy-memory parity

Recorded 2026-10-29 16:40:22 -0500. The training wrapper and evaluation agent produce exactly equal memory windows and actions on the tested noisy stream, including history rollover and three automatic episode resets. No mismatch was found in those adapters.

[Verification driver](../../../scripts/analysis/verify_history_policy_parity.py), [raw result](../../../docs/juggling-history-policy-parity.json). Frozen checkpoint3.6M, SHA25697e1184364f9f6a0c00406c8608bbc8190a82e60629f4a593eb483125ce95351. This is a diagnostic, not a survival benchmark or new training.

```sh
.venv/bin/python -m scripts.analysis.verify_history_policy_parity --checkpoint runs/td3/noisy_history_20261029_seed0/train/checkpoint_3600000 --out docs/juggling-history-policy-parity.json
```

The check feeds the same actual noisy observations to the existing `HistoryEnvVector`/actor entry point and `HistoryAgent`. It verifies the full50×8window before each action, compares deterministic actions exactly, checks terminal-observation windows, then checks padding after reset. Maximum action difference0.0 over256steps, with3automatic resets and3terminal-window checks. A73-step diagnostic episode budget exercises rollover beyond50frames and repeated reset handling; only this temporary evaluation budget differs from the canonical config. All sensing, physics, controller and DR remain canonical. A previously consumed research seed600013 is used, never a new acceptance seed. No diagnostic samples enter replay or training.

Scope is **CPU eager adapter parity on a shared sensor stream**. It does not establish compiled/GPU numerical equivalence, independently compare two environment RNG streams, or measure survival quality. Existing canonical-observation identity tests and the stored replay checks address separate questions; they were not rerun here. No source under `airhockey/` or training/evaluation adapters was changed.

The3.6M independent20k test remains running as of this record; no aggregate outcome is claimed. All three history learners and fixed4M watcher remain live. No new GIF for this diagnostic; E031 has the latest verified clips. Checks: actual256-step comparison, frozen hash and output review, live process checks, links/images, unique anchors/TOCs and whitespace. No expensive training rerun.
