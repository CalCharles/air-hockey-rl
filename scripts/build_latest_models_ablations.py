#!/usr/bin/env python3
"""
Populate `latest_models/ablations/` with deployment-ready training_state.pth
files + standardized READMEs for each ablation we've trained.

For each ablation:
  1. Copy the chosen training_state.pth from runs/td3/.../checkpoint_<N>/ into
     latest_models/ablations/<name>/training_state.pth
  2. Write latest_models/ablations/<name>/README.md with:
     - 1-2 sentence high-level description
     - Standardized detailed config block (always-on defaults, knobs ON,
       knobs OFF, DR settings if applicable, source paths, training-time
       metric at the picked ckpt)

Also writes a top-level latest_models/ablations/README.md index.

Source ckpt selection:
  - Single-knob + isolation 700k runs -> checkpoint_675000 (highest
    available; trainer doesn't write checkpoint_700000 due to off-by-one)
  - paramrand_pm25 (2M run)            -> checkpoint_1000000 (peak per
    rolling-5 analysis: rolling-mean = 132.7, single-ckpt = 145.5,
    per-env spread tightest at 6.5 across 5 dynamics envs)

Run from repo root:
    .venv/bin/python scripts/build_latest_models_ablations.py
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "latest_models" / "ablations"
RUNS_700K = REPO / "runs/td3/zeroshot_ablations_700k"
RUNS_PARAMRAND = REPO / "runs/td3/zeroshot_paramrand"
SIM_DIR = REPO / "scripts/smooth_policy/amp_history/configs/new_juggle/zeroshot_ablations"
TD3_DIR_700K = REPO / "scripts/smooth_policy/amp_history/configs/td3/zeroshot_ablations_700k"
TD3_DIR_PARAMRAND = REPO / "scripts/smooth_policy/amp_history/configs/td3/zeroshot_paramrand"
CANONICAL_SIM = REPO / "scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist2.yaml"


@dataclass
class Ablation:
    name: str            # destination folder name
    kind: str            # "baseline" | "single-knob" | "isolation" | "paramrand"
    high_level: str      # 1-2 sentence summary
    knobs_on: list[str]  # human-readable list of knobs ON in this ablation
    knobs_off: list[str] # human-readable list of knobs OFF in this ablation
    sim_yaml: Path       # path to sim YAML
    td3_args_yaml: Path  # path to TD3 args YAML
    run_dir: Path        # path to the original run dir
    ckpt_step: int       # picked checkpoint step
    extras: dict         # any extra info (DR ranges, eval settings, etc.)


# Always-on defaults shared across all 16 ablations (the project standard).
ALWAYS_ON_DEFAULTS = """\
- 85/15 starting distribution (`puck_spawn_near_paddle_prob: 0.15`) — wide data distribution
- Sysid params (gravity=-0.661, puck_damping=0.178, paddle_density=3000, pid_kp=9000, pid_kd=50)
- `enable_observation_delay: true` (project default — see `feedback_obs_delay_default_on` memory; do NOT flip)
- TD3 recipe (`td3_hist2_motion0_v2.yaml`): 2-layer 64-wide actor + Q, q_updates=25, actor_updates_per_iteration=6, hist_len=2
- Per-checkpoint single-env eval (4 episodes) — except `paramrand_pm25` which uses 5×4 multi-env eval
"""

CANONICAL_KNOBS_ON = [
    "`puck_noise: true` (additive Gaussian puck-position noise, std=0.01)",
    "`enable_random_occlusions: true`",
    "`randomize_delay: true` (±25% per-step jitter on the 25 ms observation delay)",
    "`enable_action_force_attenuation: true` (30% chance to attenuate the commanded force by 25-75%)",
    "`enable_paddle_puck_strength_randomization: true` (paddle-puck collision impulse magnitude × U[0.5, 1.0])",
    "`enable_paddle_puck_direction_randomization: true` (paddle-puck impulse direction ± 10° cone)",
    "`enable_wall_direction_randomization: true` (wall-collision direction ± 10° cone)",
]


def _ablations() -> list[Ablation]:
    """Hand-curated list of 16 ablations with high-level descriptions and
    knobs-on/off lists derived from each ablation's sim YAML."""
    return [
        # ----- 12 single-knob 700k continuations (sweep 1) -----
        Ablation(
            name="baseline",
            kind="baseline",
            high_level=(
                "Canonical TD3 recipe with every randomization knob at its default sysid+DR "
                "setting. Reference baseline for the +200k continuation sweep — every other "
                "single-knob ablation should be compared against this one."
            ),
            knobs_on=CANONICAL_KNOBS_ON.copy(),
            knobs_off=[],
            sim_yaml=CANONICAL_SIM,
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_baseline_extend.yaml",
            run_dir=RUNS_700K / "baseline" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="sysid_off",
            kind="single-knob",
            high_level=(
                "Reverts the 5 sysid-tuned physics parameters back to legacy off-the-shelf "
                "values to test how much real-world system identification matters for transfer."
            ),
            knobs_on=CANONICAL_KNOBS_ON.copy(),
            knobs_off=[
                "Sysid REPLACED with legacy values: paddle_density 3000→1000, "
                "gravity -0.661→-0.65, puck_damping 0.178→0.25, pid_kp 9000→5000, pid_kd 50→200",
            ],
            sim_yaml=SIM_DIR / "sim_sysid_off.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_sysid_off_extend.yaml",
            run_dir=RUNS_700K / "sysid_off" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="no_paddle_puck_strength",
            kind="single-knob",
            high_level=(
                "Disables paddle-puck collision STRENGTH randomization (impulse magnitude × U[0.5, 1.0]). "
                "Direction-cone randomization stays on. Tests whether stochastic restitution is "
                "necessary for transfer."
            ),
            knobs_on=[k for k in CANONICAL_KNOBS_ON if "strength" not in k],
            knobs_off=["`enable_paddle_puck_strength_randomization: false`"],
            sim_yaml=SIM_DIR / "sim_no_paddle_puck_strength.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_no_paddle_puck_strength_extend.yaml",
            run_dir=RUNS_700K / "no_paddle_puck_strength" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="no_paddle_puck_direction",
            kind="single-knob",
            high_level=(
                "Disables paddle-puck collision DIRECTION randomization (±10° impulse-direction cone). "
                "Strength randomization stays on. Tests whether stochastic collision angles are "
                "necessary for transfer."
            ),
            knobs_on=[k for k in CANONICAL_KNOBS_ON if "paddle-puck impulse direction" not in k],
            knobs_off=["`enable_paddle_puck_direction_randomization: false`"],
            sim_yaml=SIM_DIR / "sim_no_paddle_puck_direction.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_no_paddle_puck_direction_extend.yaml",
            run_dir=RUNS_700K / "no_paddle_puck_direction" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="no_wall_direction",
            kind="single-knob",
            high_level=(
                "Disables WALL collision direction randomization (±10° puck-wall bounce-angle cone). "
                "Tests whether wall-bounce angle stochasticity matters for transfer."
            ),
            knobs_on=[k for k in CANONICAL_KNOBS_ON if "wall-collision" not in k],
            knobs_off=["`enable_wall_direction_randomization: false`"],
            sim_yaml=SIM_DIR / "sim_no_wall_direction.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_no_wall_direction_extend.yaml",
            run_dir=RUNS_700K / "no_wall_direction" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="no_action_attenuation",
            kind="single-knob",
            high_level=(
                "Disables stochastic action-force attenuation (the canonical setup randomly drops "
                "commanded paddle force to 25-75% with 30% probability per step). Tests whether "
                "actuator-noise simulation is necessary for transfer."
            ),
            knobs_on=[k for k in CANONICAL_KNOBS_ON if "action_force_attenuation" not in k],
            knobs_off=["`enable_action_force_attenuation: false`"],
            sim_yaml=SIM_DIR / "sim_no_action_attenuation.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_no_action_attenuation_extend.yaml",
            run_dir=RUNS_700K / "no_action_attenuation" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="start_100_near_top",
            kind="single-knob",
            high_level=(
                "Starts every episode with the puck near the top of the table (puck_spawn_near_paddle_prob=0). "
                "Removes the 15% near-paddle starts from the canonical 85/15 mixture. Tests whether "
                "the warm-start curriculum matters."
            ),
            knobs_on=CANONICAL_KNOBS_ON.copy(),
            knobs_off=["`puck_spawn_near_paddle_prob: 0.0` (was 0.15)"],
            sim_yaml=SIM_DIR / "sim_start_100_near_top.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_start_100_near_top_extend.yaml",
            run_dir=RUNS_700K / "start_100_near_top" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="start_100_near_paddle",
            kind="single-knob",
            high_level=(
                "Starts every episode with the puck near the paddle (puck_spawn_near_paddle_prob=1.0). "
                "Inverts the 85/15 default — now no near-top starts. Tests the opposite extreme of "
                "the start-state curriculum."
            ),
            knobs_on=CANONICAL_KNOBS_ON.copy(),
            knobs_off=["`puck_spawn_near_paddle_prob: 1.0` (was 0.15)"],
            sim_yaml=SIM_DIR / "sim_start_100_near_paddle.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_start_100_near_paddle_extend.yaml",
            run_dir=RUNS_700K / "start_100_near_paddle" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="no_puck_noise",
            kind="single-knob",
            high_level=(
                "Disables additive Gaussian puck-position observation noise (std=0.01 m). Tests "
                "whether observation-noise simulation is necessary for transfer."
            ),
            knobs_on=[k for k in CANONICAL_KNOBS_ON if "puck_noise" not in k],
            knobs_off=["`puck_noise: false`"],
            sim_yaml=SIM_DIR / "sim_no_puck_noise.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_no_puck_noise_extend.yaml",
            run_dir=RUNS_700K / "no_puck_noise" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="no_occlusions",
            kind="single-knob",
            high_level=(
                "Disables random puck-observation occlusions (the canonical setup drops puck "
                "observations at 2.5%/step rate, with 3× boost near the paddle). Tests whether "
                "vision-dropout simulation is necessary for transfer."
            ),
            knobs_on=[k for k in CANONICAL_KNOBS_ON if "occlusions" not in k],
            knobs_off=["`enable_random_occlusions: false`"],
            sim_yaml=SIM_DIR / "sim_no_occlusions.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_no_occlusions_extend.yaml",
            run_dir=RUNS_700K / "no_occlusions" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="no_obs_delay_randomization",
            kind="single-knob",
            high_level=(
                "Disables per-step JITTER on the 25 ms observation delay (delay mechanism stays on, "
                "fixed at 25 ms). Replaces the broken `no_obs_delay` ablation from the 500k sweep "
                "which collapsed puck_history density due to an env-side coupling."
            ),
            knobs_on=[k for k in CANONICAL_KNOBS_ON if "randomize_delay" not in k],
            knobs_off=["`randomize_delay: false` (delay mechanism stays on; only ±25% per-step jitter removed)"],
            sim_yaml=SIM_DIR / "sim_no_obs_delay_randomization.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_no_obs_delay_randomization.yaml",
            run_dir=RUNS_700K / "no_obs_delay_randomization" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        # NOTE: the original `all_sysid_no_rand` (700k continuation from a 500k
        # base) is intentionally OMITTED here — it inherited the legacy
        # `enable_observation_delay: false` semantics from the 500k base, which
        # collapses `puck_history` temporal density (env-coupling bug, see
        # `feedback_obs_delay_default_on` memory). The replacement is
        # `all_sysid_no_rand_v2` below — fresh 700k run with the obs-delay
        # convention right (`enable_observation_delay: true` kept on,
        # `randomize_delay: false` for the actual jitter ablation).

        Ablation(
            name="all_sysid_no_rand_v2",
            kind="isolation",
            high_level=(
                "Sysid params kept at best-fit values, but all engineered randomization OFF "
                "(paddle-puck strength + direction, wall direction, action attenuation, puck "
                "noise, occlusions, delay-jitter). Replaces the legacy `all_sysid_no_rand` run, "
                "which had `enable_observation_delay: false` and inherited the puck_history-"
                "density coupling bug. This v2 keeps the obs-delay mechanism on (the project "
                "default per `feedback_obs_delay_default_on` memory) and only flips "
                "`randomize_delay: false`."
            ),
            knobs_on=["(only the always-on defaults — see project standards)"],
            knobs_off=[
                "`puck_noise: false`",
                "`enable_random_occlusions: false`",
                "`enable_action_force_attenuation: false`",
                "`enable_paddle_puck_strength_randomization: false`",
                "`enable_paddle_puck_direction_randomization: false`",
                "`enable_wall_direction_randomization: false`",
                "`randomize_delay: false` (delay mechanism stays on per project default; only ±25% per-step jitter removed)",
            ],
            sim_yaml=SIM_DIR / "sim_all_sysid_no_rand_v2.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_all_sysid_no_rand_v2.yaml",
            run_dir=RUNS_700K / "all_sysid_no_rand_v2" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        # ----- 3 isolation 700k runs (sweep 2) -----
        Ablation(
            name="only_obs_noise_occlusion",
            kind="isolation",
            high_level=(
                "Isolation study: only `puck_noise` + `enable_random_occlusions` ON; all collision "
                "randomization (×3), action attenuation, and delay-jitter OFF. Tests whether "
                "perception-side randomization alone suffices for transfer."
            ),
            knobs_on=[
                "`puck_noise: true`",
                "`enable_random_occlusions: true`",
            ],
            knobs_off=[
                "`enable_paddle_puck_strength_randomization: false`",
                "`enable_paddle_puck_direction_randomization: false`",
                "`enable_wall_direction_randomization: false`",
                "`enable_action_force_attenuation: false`",
                "`randomize_delay: false`",
            ],
            sim_yaml=SIM_DIR / "sim_only_obs_noise_occlusion.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_only_obs_noise_occlusion.yaml",
            run_dir=RUNS_700K / "only_obs_noise_occlusion" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="only_action_attenuation",
            kind="isolation",
            high_level=(
                "Isolation study: only `enable_action_force_attenuation` ON; all collision "
                "randomization (×3), puck noise, occlusions, and delay-jitter OFF. Tests whether "
                "actuator-side randomization alone suffices. (At 700k, this single-knob isolation "
                "achieved the highest training-end mean of all 15 ablations — 134 vs 88-122 for the "
                "single-knob removals.)"
            ),
            knobs_on=[
                "`enable_action_force_attenuation: true`",
            ],
            knobs_off=[
                "`puck_noise: false`",
                "`enable_random_occlusions: false`",
                "`enable_paddle_puck_strength_randomization: false`",
                "`enable_paddle_puck_direction_randomization: false`",
                "`enable_wall_direction_randomization: false`",
                "`randomize_delay: false`",
            ],
            sim_yaml=SIM_DIR / "sim_only_action_attenuation.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_only_action_attenuation.yaml",
            run_dir=RUNS_700K / "only_action_attenuation" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        Ablation(
            name="only_action_attenuation_obs_noise_occlusion",
            kind="isolation",
            high_level=(
                "Isolation study: action attenuation + puck noise + occlusions ON; collision "
                "randomization (×3) and delay-jitter OFF. Tests the union of perception + "
                "actuator randomization without any collision-physics noise."
            ),
            knobs_on=[
                "`enable_action_force_attenuation: true`",
                "`puck_noise: true`",
                "`enable_random_occlusions: true`",
            ],
            knobs_off=[
                "`enable_paddle_puck_strength_randomization: false`",
                "`enable_paddle_puck_direction_randomization: false`",
                "`enable_wall_direction_randomization: false`",
                "`randomize_delay: false`",
            ],
            sim_yaml=SIM_DIR / "sim_only_action_attenuation_obs_noise_occlusion.yaml",
            td3_args_yaml=TD3_DIR_700K / "td3_zeroshot_only_action_attenuation_obs_noise_occlusion.yaml",
            run_dir=RUNS_700K / "only_action_attenuation_obs_noise_occlusion" / "seed0",
            ckpt_step=675_000,
            extras={},
        ),
        # ----- Physics-parameter domain randomization 2M run (paramrand) -----
        Ablation(
            name="paramrand_pm25",
            kind="paramrand",
            high_level=(
                "Physics-parameter domain randomization: paddle_density / puck_damping / gravity "
                "are drawn uniform within ±25% of their sysid values per episode reset. Engineered "
                "randomization (collision×3, action attenuation, delay-jitter) is OFF — paramrand "
                "is meant as an ALTERNATIVE. Picked at 1M steps (rolling-5 mean = 132.7, single-ckpt "
                "= 145.5, per-env spread compressed to 6.5 across 5 dynamics envs) — the paramrand "
                "trajectory plateaus around 1M and does not meaningfully improve through 2M."
            ),
            knobs_on=[
                "`puck_noise: true`",
                "`enable_random_occlusions: true`",
            ],
            knobs_off=[
                "`enable_paddle_puck_strength_randomization: false`",
                "`enable_paddle_puck_direction_randomization: false`",
                "`enable_wall_direction_randomization: false`",
                "`enable_action_force_attenuation: false`",
                "`randomize_delay: false`",
            ],
            sim_yaml=SIM_DIR / "sim_paramrand_pm25.yaml",
            td3_args_yaml=TD3_DIR_PARAMRAND / "td3_paramrand_pm25.yaml",
            run_dir=RUNS_PARAMRAND / "paramrand_pm25" / "seed0",
            ckpt_step=1_000_000,
            extras={
                "domain_randomization": (
                    "Per `env.reset()`, draws each variable uniform in `[low, high]` from "
                    "`random_variable_ranges`, reassigns to `simulator_params`, then rebuilds the "
                    "Box2D simulator. The agent has consistent dynamics within an episode but they "
                    "shift between episodes — implicit meta-learning over the 5-step paddle/puck "
                    "history."
                ),
                "ranges": {
                    "paddle_density": ([2250.0, 3750.0], "sysid 3000, ±25%"),
                    "puck_damping":   ([0.1335, 0.2225], "sysid 0.178, ±25%"),
                    "gravity":        ([-0.826, -0.496], "sysid -0.661, ±25%"),
                },
                "eval_settings": (
                    "5 fixed param-dicts seed-sampled from the same ±25% ranges using "
                    "`np.random.RandomState(eval_param_seed=12345)` at training start; held "
                    "constant for all evaluations. Per checkpoint: 4 episodes × 5 envs = 20 "
                    "episodes, aggregated to a single mean. Eval-env starts vary per ckpt via a "
                    "per-call seed shift (otherwise the deterministic env replays identical "
                    "trajectories)."
                ),
                "trainer": "Custom entrypoint `td3_training_dr.py` (wraps `td3_training.py` via monkey-patch on `evaluate_agent`).",
            },
        ),
    ]


def _read_training_metric_at_ckpt(run_dir: Path, ckpt_step: int) -> str:
    """Best-effort: parse the trainer log for the `Step <ckpt>: Rolling(2k) Avg Return` line
    closest to the picked checkpoint, return as a one-line summary. Skipped if no log."""
    candidates = list(run_dir.parent.parent.parent.glob("**/*.log"))
    # Try notes/scratch logs by run-dir-tail name.
    name = run_dir.parent.name  # e.g. "baseline" or "paramrand_pm25"
    log_dirs = [
        REPO / "notes/scratch/zeroshot_ablation_700k_logs",
        REPO / "notes/scratch/zeroshot_paramrand_logs",
        REPO / "notes/scratch/zeroshot_ablation_logs",
    ]
    for ld in log_dirs:
        for candidate_name in (
            f"{name}.log",
            f"{name}_extend.log",
            "paramrand_pm25.log",
        ):
            p = ld / candidate_name
            if not p.exists():
                continue
            # Find lines around ckpt_step (within 2k of it)
            with open(p) as f:
                last_match = None
                for line in f:
                    m = re.match(
                        r"^Step (\d+): Rolling\(2k\) Avg Return: ([\d.\-]+).*Success Rate: ([\d.]+).*Avg Episode Length: ([\d.]+)",
                        line,
                    )
                    if m:
                        step = int(m.group(1))
                        if abs(step - ckpt_step) <= 1000:
                            last_match = (step, m.group(2), m.group(3), m.group(4))
                if last_match:
                    s, r, sr, el = last_match
                    return f"Step {s}: Rolling(2k) Avg Return = {r}, Success Rate = {sr}, Avg Episode Length = {el} (single-env training-time metric)"
    return "(training metric not found in logs)"


def _multi_env_eval_at_ckpt(run_dir: Path, ckpt_step: int) -> str:
    """For paramrand_pm25 only: pull the multi_env_eval.json from the picked ckpt."""
    p = run_dir / f"checkpoint_{ckpt_step}" / "multi_env_eval.json"
    if not p.exists():
        return ""
    d = json.load(open(p))
    a = d["aggregate"]
    per_env = a["per_env_mean_return"]
    spread = max(per_env) - min(per_env)
    return (
        f"5-env eval at this ckpt: mean_return = {a['mean_return_across_envs']:.2f}, "
        f"mean_success = {a['mean_success_across_envs']:.3f}, "
        f"per_env_returns = [{', '.join(f'{x:.1f}' for x in per_env)}] (spread = {spread:.1f})"
    )


def write_readme(a: Ablation) -> str:
    rel_sim = a.sim_yaml.relative_to(REPO)
    rel_args = a.td3_args_yaml.relative_to(REPO)
    rel_run = a.run_dir.relative_to(REPO)
    rel_ckpt = (a.run_dir / f"checkpoint_{a.ckpt_step}" / "training_state.pth").relative_to(REPO)

    knobs_on_md = "\n".join(f"- {k}" for k in a.knobs_on) if a.knobs_on else "- (none beyond always-on defaults)"
    knobs_off_md = "\n".join(f"- {k}" for k in a.knobs_off) if a.knobs_off else "- (none — canonical setup)"

    extras_md = ""
    if a.extras:
        if "domain_randomization" in a.extras:
            extras_md += "\n### Domain randomization (per-reset)\n\n"
            extras_md += a.extras["domain_randomization"] + "\n\n"
            extras_md += "| variable | range | reference |\n|---|---|---|\n"
            for var, (rng, ref) in a.extras["ranges"].items():
                extras_md += f"| `{var}` | [{rng[0]}, {rng[1]}] | {ref} |\n"
        if "eval_settings" in a.extras:
            extras_md += "\n### Eval-time multi-env settings\n\n"
            extras_md += a.extras["eval_settings"] + "\n"
        if "trainer" in a.extras:
            extras_md += f"\n### Trainer\n\n{a.extras['trainer']}\n"

    metric_line = _read_training_metric_at_ckpt(a.run_dir, a.ckpt_step)
    multi_env_line = _multi_env_eval_at_ckpt(a.run_dir, a.ckpt_step)
    metrics_md = f"- {metric_line}\n"
    if multi_env_line:
        metrics_md += f"- {multi_env_line}\n"

    return f"""# {a.name}

## High-level description

{a.high_level}

---

## Standardized configuration

| Field | Value |
|---|---|
| Ablation type | {a.kind} |
| Source checkpoint | `{rel_ckpt}` |
| Training step at this ckpt | {a.ckpt_step:,} |
| Source run dir | `{rel_run}` |
| Sim config (Box2D env) | `{rel_sim}` |
| TD3 args (recipe) | `{rel_args}` |
| Deployment file (here) | `training_state.pth` |

### Always-on defaults (project standard, all 16 ablations)

{ALWAYS_ON_DEFAULTS}
### Knobs ON in this ablation (beyond defaults)

{knobs_on_md}

### Knobs OFF in this ablation (vs canonical)

{knobs_off_md}
{extras_md}
### Training metric at the picked checkpoint

{metrics_md}
---

## Deployment

Real-world rollout via `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_eval.py`
(or `async_td3_real_modular.py` for further fine-tuning). The `training_state.pth`
contains actor + Q networks + replay buffers + optimizer state + RNG + the saved
`args` dict — everything needed to load or resume the policy. The original sim
config and TD3 args YAMLs are referenced above for full reproducibility.
"""


def write_index(ablations: list[Ablation]) -> str:
    rows = []
    for a in ablations:
        kind_label = {
            "baseline": "baseline",
            "single-knob": "single-knob",
            "isolation": "isolation",
            "paramrand": "paramrand",
        }[a.kind]
        rows.append(
            f"| `{a.name}` | {kind_label} | {a.ckpt_step:,} | {a.high_level.split('.')[0]}. |"
        )
    rows_md = "\n".join(rows)
    return f"""# Ablation deployment models

Sixteen TD3 policies trained for the CoRL-2026 zero-shot sim2real ablation
study (`paper/main.tex` §Ablations:zeroshot). Each subdirectory contains:

- **`training_state.pth`** — full TD3 training-state checkpoint (actor + targets +
  Q networks + optimizer + replay + RNG + saved `args`). Suitable for real-world
  rollout via `scripts/smooth_policy/amp_history/amp_training/td3/extras/async_td3_real_eval.py`
  or further fine-tuning via `async_td3_real_modular.py`.
- **`README.md`** — 1-2 sentence high-level description + standardized detailed
  config block (always-on defaults, knobs ON, knobs OFF, source paths, training
  metric at the picked checkpoint).

Source-checkpoint convention:
- 15 ablations trained to a 700k budget — picked **`checkpoint_675000`** (highest
  available; the trainer's off-by-one means there's no `checkpoint_700000`).
- `paramrand_pm25` trained to 2M — picked **`checkpoint_1000000`** (rolling-5 mean
  peaked there at 132.7 with a single-ckpt high of 145.5; the trajectory plateaus
  around 1M and does not improve meaningfully through 2M).

| Folder | Type | Source step | Summary |
|---|---|---:|---|
{rows_md}

---

## Reproduction

Each model's `README.md` lists the exact `Sim config` and `TD3 args` paths used
to train it. To reproduce a run:

```bash
.venv/bin/python -m scripts.smooth_policy.amp_history.amp_training.td3.td3_training \\
  --args-file <td3 args yaml from the README>
```

(For `paramrand_pm25`, the entrypoint is `td3_training_dr` instead of `td3_training`.)

The original 700k continuation runs each `full_resume`d from a 500k baseline run
in `runs/td3/zeroshot_ablations/...` — see the corresponding `model_path:` field
in their TD3 args YAML.

## Source experiments (background context)

- `notes/scratch/experiments/2026-05-05_17-38_zero-shot-sim2real-ablations.md` —
  the 500k base sweep (12 single-knob ablations; `no_obs_delay` failed to train
  due to env-coupling bug, was replaced in the 700k extension by
  `no_obs_delay_randomization`).
- `notes/scratch/experiments/2026-05-09_18-50_zeroshot-ablations-700k.md` — +200k
  continuation sweep producing the 12 `*_extend` runs (means 88-122 at 675k).
- `notes/scratch/experiments/2026-05-10_*_isolation_*.md` — 3 fresh isolation runs
  at 700k (only_obs_noise_occlusion 94, only_action_attenuation 134,
  only_action_attenuation_obs_noise_occlusion 88).
- `notes/scratch/experiments/2026-05-10_*_paramrand_2M.md` — physics-parameter DR
  2M run (peak 145.5 at 1M, back-half plateau ~118).

(Some experiment writeups may not yet exist if this folder was built before the
matching writeup landed.)
"""


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    ablations = _ablations()

    for a in ablations:
        src_ckpt = a.run_dir / f"checkpoint_{a.ckpt_step}" / "training_state.pth"
        if not src_ckpt.exists():
            print(f"[SKIP] {a.name}: source missing {src_ckpt}")
            continue

        out_dir = DEST / a.name
        out_dir.mkdir(parents=True, exist_ok=True)

        dst_ckpt = out_dir / "training_state.pth"
        if dst_ckpt.exists() and dst_ckpt.stat().st_size == src_ckpt.stat().st_size:
            print(f"[skip-copy] {a.name}: training_state.pth already present (same size)")
        else:
            shutil.copyfile(src_ckpt, dst_ckpt)
            print(f"[copy] {a.name}: {src_ckpt.relative_to(REPO)} -> {dst_ckpt.relative_to(REPO)}")

        readme = out_dir / "README.md"
        readme.write_text(write_readme(a))
        print(f"[readme] {a.name}: wrote {readme.relative_to(REPO)}")

    index = DEST / "README.md"
    index.write_text(write_index(ablations))
    print(f"[index] wrote {index.relative_to(REPO)}")


if __name__ == "__main__":
    main()
