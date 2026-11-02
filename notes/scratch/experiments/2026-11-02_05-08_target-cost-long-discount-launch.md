# Two fresh longer-discount learners test the same PID-target reward cost.

- **Date**: 2026-11-02 05:08:56 UTC documentation; actual submissions below.
- **Status**: in-flight;115928/115929 confirmed Slurm RUNNING and learning past40k steps.
- **Run dirs**: `runs/td3/noisy_history_target_cost_100_g0999_20261102_seed0`, `...seed1`.
- **Configs**: [seed0](../../../configs/td3/sustained/noisy_history_target_cost_100_g0999_seed0.yaml), [seed1](../../../configs/td3/sustained/noisy_history_target_cost_100_g0999_seed1.yaml), [hashed launch/batch records](../../../docs/juggling-target-cost-longdiscount-launch.json).

## Hypothesis / authorization / initialization

The completed6M unregularized .999 target has more endurance and lower command changes than its .9956M control (198.625 vs138.40625 physical juggles,14141.53125 vs9859.8125steps on32 reused20k starts). Its long outcomes were not collected at submission; the [completed paired result](2026-11-02_05-20_late-discount-six-million-long.md) is reconciled later in this publication. This motivates a two-seed experiment combining the same constant target-jump reward cost with longer discounting, not a claim of success from the [early500k data](2026-11-02_05-08_first-target-cost-checkpoints.md). Algorithm/reward changes and server jobs within16 GPUs are authorized by the active goal; no simulator modification or pretrained policy is used.

- CPU/GPU job **115928**, seed0, submitted2026-11-02T04:53:49.837564+00:00; `/scratch/cluster/mrudolph/documents/air-hockey-rl/runs/td3/noisy_history_target_cost_100_g0999_20261102_seed0`; user GPUs before submission9.
- CPU/GPU job **115929**, seed1, submitted2026-11-02T04:53:50.161312+00:00; `/scratch/cluster/mrudolph/documents/air-hockey-rl/runs/td3/noisy_history_target_cost_100_g0999_20261102_seed1`; user GPUs before submission10.

Before each submission, total user GPU requests—including unrelated jobs and pending allocations—were counted,9 then10; each added one, reaching11, below16. Both new processes validate scratch initialization (`model_path:null`, no history/actor/critic warm start or residual) and canonical simulator config. Effective configs and all canonical simulator/trainer/loader hashes match the existing [contract review](../../../docs/juggling-target-cost-contract-review-20261102.json). No source file used by a live learner was edited.

## Exact controlled change

Against its same-seed gamma .995 reward-cost config, each new YAML differs only in **gamma .995→.999 and run name**. Architecture50-step history/64x2, optimizer, exploration, replay capacity, cost100, actor-delta penalty0,6M budget, checkpoint cadence500k, raw8-start10000/5k evaluations and32-start700000/20k online/target1M/3M/6M tests are unchanged. The target cost stays constant; no curriculum or extra action filter. Saved replay-buffer copies remain disabled for storage; all policy/config/evaluation artifacts remain saved. All four reward-cost learners initialize randomly, separately.

Canonical noisy observations, occlusions, delay/interpolation, hist2 split sampling, PID, smoothing, geometry/contact/workspace and established DR remain intact. Discount only changes the learning objective. Coefficient400 configs remain planned and inactive. No running original trial was restarted/canceled, and no hardware was accessed.

## Pending evidence / reproduction

At the initial launch confirmation recorded in the manifest, no fixed-policy checkpoint result or performance GIF existed. Durable logs separately record original/shaped reward, cost and target-change statistics; the first fixed-policy raw juggles/steps/reward arrive at500k. The common report recognizes the new launch manifest and labels each discount/method separately, omitting missing points. Statistical claims require future data; same seeds across discount groups are comparisons, not pretrained initialization.

```sh
squeue -j 115928,115929 -o '%.18i %.9T %.10M %b %j'
# Once complete reward blocks exist, analyze this pair separately:
.venv/bin/python -m scripts.analysis.plot_juggle_target_cost_training --launch docs/juggling-target-cost-longdiscount-launch.json --out-prefix /tmp/target-cost-g999
```

Do not resubmit live jobs. Continue full budgets and compare raw endurance, sampled physical motion and actual controller-tail evidence. No infinite-juggling or hardware-stop conclusion is justified.
