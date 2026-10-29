# A fixed nominal-physics puck observer is not useful enough for policy integration

- **Date**: 2026-10-29 19:09 UTC
- **Status**: done (negative offline diagnostic); not integrated into any policy
- **Run dirs**: `runs/td3/noisy_scratch_20261028_seed0/filter_observability/`
- **Configs**: Declared fixed parameters in [raw results](../../../docs/juggling-filter-observability.json)

## Question
Could a compact deterministic state estimate avoid the high-dimensional history critic and help a future scratch actor? Test one simple nominal-physics Kalman observer before spending GPU time on a policy using it.

## Setup
Same 64 saved trajectories as E010 (old selected/final scratch actors, two actors per seed). Reused comparison split: seeds 300000–300023 in the earlier training group; 300024–300031 in the comparison group (16 episodes, eight paired environment seeds). These are an exploratory reused diagnostic benchmark, not untouched acceptance starts.

The observer consumes only latest puck xy/occlusion flag. Fixed dt .05 s, latency compensation .025 s, measurement std .01 m, process acceleration std 4 m/s², canonical mean gravity +.73 in base x and damping .11. Nominal table dimensions/radius support an approximate hard-wall reflection; exact contact-solver details and paddle impacts are not modeled. Actual episode DR parameters are not read. No parameter search, model fit, actor/critic training, environment rollout or policy checkpoint load. Current actor inputs/actions and all simulator settings are unchanged. Experimental code remains under scripts/analysis and is not connected to a learner.

As in E010, compare physical position before action t (saved post-action t−1) and preceding-step average velocity from 50 ms body-position differences. The filter predicts current velocity, so the finite-step velocity target is an imperfect proxy near impacts; the position target is directly aligned. The neural regressors also see paddle observations and previous actions, while this simple filter sees only puck measurements; the comparison does not isolate architecture or prove all filters inferior.

## Results

| Method | Mean episode 2D position RMSE (cm) | Prior-step velocity proxy RMSE (m/s) |
|---|---:|---:|
| E010 current observation, width 64 | 4.4436 | 0.7694 |
| E010 current observation, width 128 | 4.2508 | 0.7438 |
| E010 12-step memory, width 64 | 4.1023 | 0.6587 |
| Fixed nominal-physics filter | **5.1748** | **1.2656** |

![Fixed-observer negative comparison](../../../docs/juggling-filter-observability.png)

Figure generated 2026-10-29 from [new raw results](../../../docs/juggling-filter-observability.json) and [prior E010 metrics](../../../docs/juggling-observability.json). Dots are eight comparison seeds, each averaging two frozen-actor episodes, with paired lines. These are state-estimation errors, not policy returns or survival measurements. One fixed filter setting and one prior regressor training seed do not exhaust the method family.

## Validation and reproduce

```sh
.venv/bin/python -u -m scripts.analysis.juggle_filter_observability --roots runs/td3/noisy_scratch_20261028_seed0/selected_paired_eval runs/td3/noisy_scratch_20261028_seed0/final_paired_eval --out <new-results.json>
.venv/bin/python -m unittest scripts.td3.tests.test_juggle_continuing scripts.td3.tests.test_puck_state_filter -q
```

All 26 tests passed. Three new observer tests cover delayed constant-velocity prediction through occlusions, no mutation of input/RNG, and finite positive-semidefinite wall prediction during occlusion. Those three were rechecked after moving the experimental class from an uncommitted helper path to scripts/analysis. No simulator implementation changed. Only a metric figure is available; no policy GIF because no filter policy was trained.

## Conclusion
Do not integrate this fixed observer into an active actor. Smooth nominal dynamics plus approximate walls and no paddle-impact/input model do not give competitive estimates on these saved trajectories. This negative result does not rule out adaptive, contact-aware or learned observers. The existing baseline, learned-history and survival-focused RL trials continue; do not initialize any actor from diagnostic models.
