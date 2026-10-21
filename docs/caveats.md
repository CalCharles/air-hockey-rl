# Caveats and research context

Recorded 2026-10-21. Resolve disagreements using the selected configuration and executing code; old documentation often describes a valid historical experiment rather than the current default.

## Documentation drift verified during review

- `scripts/td3/README.md` says “dual-head critics”; `TD3QNetwork` actually returns one scalar per critic. The trainer defaults to two critics and supports a larger ensemble.
- The external-trainer guide calls v1 `sysid_best_params_hist2.yaml` canonical and discusses task/motion heads. `CLAUDE.md`, the sysid index, and the v2 config designate `sysid_v2_hist2.yaml` as the newer identified base. Follow source/config for critic behavior.
- `hist_len` is controller smoothing, not observation-history length. Standard `history` observations always take five samples per object in `airhockey/utils.py`.
- `CLAUDE.md` discusses both historical canonical source checkpoints and the need to retrain after environment simplification. Its task inventory also predates some currently registered scoring/triangle tasks. Inspect the factory and actual YAMLs before assuming a task is unconfigured or a checkpoint is deployment-compatible.
- The older `configs/td3/tasks/` family uses v1/hist4 recipes; `tasks_v2/` contains the later generated identified-physics campaign. Neither directory name alone identifies the right checkpoint for a deployment.
- The root README mentions `paper/`, but it is ignored and was absent from this checkout. Historical experiment notes also reference generated configs and runs that may not be present.
- The project-goal safety page contains an older discriminator-training strategy. The active source and root guidance use TD3; consult that page for robot constraints, not current algorithm selection.

## Physics and timing

The inspected v2 hist2 base sets gravity −0.73, puck damping 0.11, wall restitution 0.90/0.55, puck restitution 1.2626, paddle restitution 0, and PID gains about 7531.5/1928.7/0. It also includes measured table offsets, hist2 control smoothing, puck noise, random occlusions, and a shared 25 ms observation delay. Check the YAML for full workspace geometry rather than copying numeric limits from prose.

System-ID limitations are documented in [`sysid/README.md`](../sysid/README.md): end-wall restitution is weakly identified; head-on paddle collisions identify a gain rather than restitution and mass independently; offset collisions reveal an angular-response mismatch. Do not treat the fitted simulator as exact hardware ground truth.

Historical hist4 smoothing comparisons were worse than hist2 in simulation, with controller gains changed at the same time. Later experiments nevertheless investigate hist4 policies for real transfer. Use the matching training, controller, and rollout configuration; do not silently interchange them.

Recent work centers on camera timing and the simulation-to-real gap. Start with these existing reports:

- [Real camera frame staleness](../notes/scratch/experiments/2026-10-18_07-51_real-camera-frame-staleness.md).
- [Puck camera lag in sim2sim](../notes/scratch/experiments/2026-10-18_08-03_puck-camera-lag-sim2sim.md).
- [Newest-frame camera experiment](../notes/scratch/experiments/2026-10-18_09-29_newest-frame-camera-sim2sim.md).

The newest-frame report found strong sensitivity to puck-only lag across five policies and 40 paired episodes each. It distinguishes shared observation delay, camera frame period, camera latency, and queued-frame age. It also explicitly leaves the sim/robot history-spacing difference unmodeled. These are experimental findings, not evidence that the current hardware camera latency has been resolved. Inspect [`latest_frame_capture.py`](../airhockey/sims/real/latest_frame_capture.py) alongside simulator lag controls for timing changes.

## Working conventions

The user explicitly requested these reference files under `docs/`; existing repository guidance normally places formal documentation in `notes/docs/`. Keep these notes compact and cross-link detailed sources instead of duplicating research logs.

New experiment writeups follow [`notes/scratch/experiments/README.md`](../notes/scratch/experiments/README.md): new dated files and additive index updates. Older long logs are historical references. This onboarding task did not run an experiment.

Real hardware operation is a separate task. Before touching robot control, read the [hardware overview](../notes/docs/environments/real-world/overview.md), [protective-stop history](../notes/docs/environments/real-world/protective-stops.md), and [known issues](../notes/docs/known-issues.md). The documented clamping coverage gap concerns force-mode timeout during reset/idle periods; inspect current control code before assuming a historical issue is fixed. Existing e-stops, protective stops, and smooth-motion constraints matter when changing the control path.

The working tree already contained an untracked `configs/new_juggle/sysid_best_params_hist2_puck_score.yaml` at review start. It was left untouched; it is not part of these documentation changes.
