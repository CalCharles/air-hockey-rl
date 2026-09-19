## A. Phase 1 — privileged RMA policy vs plain-DR TD3 (fixed eval envs)

| Run | 250k | 500k | 750k | 1000k | 1250k | 1500k | 1750k | 2000k | back-half mean | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rma_priv:juggle_dr_phase1_seed0 | 52.0 | 57.2 | 91.1 | 94.5 | 138.2 | 133.1 | 159.8 | 156.3 | 133.6 | 177.4 |
| rma_priv:juggle_dr_phase1_seed1 | 37.8 | 65.1 | 70.2 | 79.8 | 93.6 | 122.6 | 104.4 | 158.9 | 121.9 | 158.9 |
| history:history_juggle_dr_seed0 | 38.8 | 61.4 | 70.3 | 71.2 | 145.8 | 161.4 | 144.8 | 195.2 | 159.6 | 199.1 |
| history:history_juggle_dr_seed1 | 56.6 | 89.9 | 89.5 | 121.5 | 157.8 | 179.9 | 190.9 | 194.6 | 178.1 | 200.2 |
| td3_dr:juggle_dr | 53.8 | 83.8 | 110.2 | 97.0 | 118.5 | 100.3 | 119.9 | 116.0 | 117.3 | 137.2 |
| td3_dr:td3_dr_juggle_seed1 | 53.8 | 66.8 | 85.1 | 129.5 | 120.4 | 141.8 | 128.6 | 132.2 | 130.2 | 158.9 |
| td3_dr:td3_dr_juggle_seed2 | 54.8 | 77.5 | 90.0 | 122.5 | 124.8 | 118.8 | 138.0 | 139.8 | 125.4 | 162.7 |

Mean return of the privileged RMA policy (rma_priv) / the long-history control (history) / the plain-DR actor (td3_dr) on the fixed 5-env eval set (eval_param_seed 12345, 4 episodes per env per checkpoint; the 2000k column is the final policy); back-half = mean over all 25k-checkpoints in (1000k, 2000k].

