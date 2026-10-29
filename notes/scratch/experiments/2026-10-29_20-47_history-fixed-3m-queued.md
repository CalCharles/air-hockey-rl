# Fixed history 3M benchmark queued

Recorded 2026-10-29 15:47:58 -0500. The benchmark is **waiting**, not completed. Live watcher PID1302296 waits for the existing history learner's `train/checkpoint_3000000/survival_eval/results.json`, validates the saved canonical environment (normalizing only its bookkeeping seed), then evaluates that exact frozen checkpoint on32reused starts300000–300031 at horizon10000. No new training, simulator change or checkpoint selection occurs.

Local watcher source/log/status: `runs/td3/noisy_history_20261029_seed0/milestone_3m_watch.py`, `milestone_3m_watch.log`, `milestone_3m_status.json`. Its process was confirmed live and status waiting. It refuses an existing output and records failure if training terminates before the required validation. Planned output: `runs/td3/noisy_history_20261029_seed0/milestone_3m_benchmark`.

Equivalent frozen evaluation after the checkpoint is available (choose new output):

```sh
.venv/bin/python -m scripts.td3.juggle_survival --run-dir runs/td3/noisy_history_20261029_seed0/train/checkpoint_3000000 --out-dir NEW_OUTPUT --scenario-set training --horizon 10000 --episodes 32 --seed-base 300000 --gif-steps 0
```

The run was last verified beyond2.8M; all three learner processes are live. These seeds are a reused shared research benchmark, not fresh acceptance. No numerical result or new GIF is claimed yet. [Earlier completed results](../../../docs/experiment_logbook.md#e020).
