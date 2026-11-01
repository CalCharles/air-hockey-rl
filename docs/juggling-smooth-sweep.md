# Smoothness sweep progress

Generated 2026-11-01T23:11:16.026076+00:00 from saved artifacts; no new rollouts.

Original sweep: each coefficient (0.0005, 0.002, 0.008) has two fresh training seeds at gamma 0.995. An additional pair uses coefficient 0.0005 and gamma 0.999. Original checkpoints are every 100k steps; the additional pair saves every 500k for storage. All budgets are 6M. Simulator, sensing, control and DR remain unchanged. GPU ceiling: 16 total user allocations, including other jobs. Training seeds remain separate curves; starts within one run are not training-seed replications.

| Run | Slurm job | Saved runner status |
|---|---|---|
| noisy_history_smooth_0005_20261101_seed0 | 115858 | training |
| noisy_history_smooth_0005_20261101_seed1 | 115859 | training |
| noisy_history_smooth_002_20261101_seed0 | 115860 | training |
| noisy_history_smooth_002_20261101_seed1 | 115861 | training |
| noisy_history_smooth_008_20261101_seed0 | 115862 | training |
| noisy_history_smooth_008_20261101_seed1 | 115863 | training |
| noisy_history_smooth_0005_g0999_20261101_seed0 | 115866 | training |
| noisy_history_smooth_0005_g0999_20261101_seed1 | 115867 | training |

![Checkpoint metrics](juggling-smooth-sweep.png)

[Raw summary CSV](juggling-smooth-sweep.csv); [original launch provenance](juggling-smooth-sweep-launch.json); [longer-discount pair](juggling-smooth-longdiscount-launch.json).

Plots use saved 8-start 5k online-actor validation only. The CSV separately labels 32-start 20k online/target tests; do not mix these protocols. Physical juggles use true simulated trajectories; reward is the environment episode sum. Recording limits truncate counts and durations. Action-change RMS is a command diagnostic, not measured energy or UR5 safety. Physical speed/acceleration/jerk distributions are stored in each matched test’s local `motion_metrics.json`, with first-250-step and whole-episode estimates. Short episodes still have different exposure and visited states.

Curves are descriptive repeated-validation results. Independent longer tests and qualitative review follow candidate comparison; no binary acceptance gate is used.

Measurement clarification: [actual PID-target replay audit](../notes/scratch/experiments/2026-11-01_22-59_pid-target-replay-audit.md). Legacy `target_delta_rms_m` is a scaled displacement-change proxy, not direct PID-target telemetry.
