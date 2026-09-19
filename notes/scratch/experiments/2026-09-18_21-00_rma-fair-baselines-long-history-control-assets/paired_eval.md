Paired eval: 5 ID envs + 5 OOD envs, 20 episodes per env, eval_param_seed 12345, call_index 1000 (same episode seeds for every agent); ± = SEM over episodes.

| agent | ID return ± SEM | ID success | ID ep len | OOD return ± SEM | OOD success | OOD ep len |
|---|---:|---:|---:|---:|---:|---:|
| adapted:juggle_dr_phase1_seed0 | 149.0 ± 4.1 | 1.00 | 212 | 120.0 ± 5.6 | 0.91 | 172 |
| privileged:juggle_dr_phase1_seed0 | 148.6 ± 4.5 | 0.99 | 212 | 121.9 ± 6.2 | 0.89 | 175 |
| nominal:juggle_dr_phase1_seed0 | 156.6 ± 4.4 | 0.98 | 220 | 94.1 ± 5.0 | 0.84 | 144 |
| adapted:juggle_dr_phase1_seed1 | 150.0 ± 4.7 | 0.97 | 213 | 120.3 ± 5.5 | 0.95 | 178 |
| privileged:juggle_dr_phase1_seed1 | 138.5 ± 4.7 | 0.98 | 198 | 49.2 ± 3.0 | 0.66 | 88 |
| nominal:juggle_dr_phase1_seed1 | 143.9 ± 4.4 | 0.98 | 208 | 112.7 ± 4.5 | 0.97 | 169 |
| history:history_juggle_dr_seed0 | 192.5 ± 1.7 | 1.00 | 250 | 141.3 ± 5.8 | 0.93 | 199 |
| history:history_juggle_dr_seed1 | 192.5 ± 2.1 | 1.00 | 249 | 119.9 ± 5.8 | 0.92 | 171 |
| td3_dr:juggle_dr | 125.1 ± 5.0 | 0.97 | 181 | 87.6 ± 4.8 | 0.88 | 140 |
| td3_dr:td3_dr_juggle_seed1 | 148.7 ± 4.0 | 1.00 | 213 | 88.3 ± 4.8 | 0.88 | 140 |
| td3_dr:td3_dr_juggle_seed2 | 125.9 ± 4.7 | 0.99 | 185 | 94.6 ± 5.0 | 0.89 | 145 |

Mean over seeds per agent kind (± = std across seeds):

| kind | n seeds | ID return | OOD return |
|---|---:|---:|---:|
| adapted | 2 | 149.5 ± 0.5 | 120.1 ± 0.2 |
| history | 2 | 192.5 ± 0.0 | 130.6 ± 10.7 |
| nominal | 2 | 150.2 ± 6.3 | 103.4 ± 9.3 |
| privileged | 2 | 143.6 ± 5.0 | 85.6 ± 36.3 |
| td3_dr | 3 | 133.2 ± 11.0 | 90.2 ± 3.1 |
