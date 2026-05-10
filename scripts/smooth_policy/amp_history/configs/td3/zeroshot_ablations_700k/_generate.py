#!/usr/bin/env python3
"""
Generate TD3 args YAMLs for the +200k extension of the zero-shot sim2real
ablation sweep (500k -> 700k).

11 of the 12 original ablations are CONTINUATIONS: they full_resume from
the 500k `training_state.pth` saved at the end of the original run, then
train another 200k env steps to global_step=700000. The remaining slot
replaces the broken `no_obs_delay` ablation with `no_obs_delay_randomization`
(see sim_no_obs_delay_randomization.yaml — flips `randomize_delay: true ->
false` while leaving `enable_observation_delay: true` so the puck_history
temporal density is unchanged). That ablation is a FRESH 700k run with no
warmstart.

Output dirs go to runs/td3/zeroshot_ablations_700k/<name>/seed0/ so the
500k final-state files (model.pth, training_state.pth, ...) at the
original path stay intact.

GPU balance (per-run wall clock extrapolated from the 500k base = 1h45m):
  - 200k continuation ~ 42m each
  - fresh 700k        ~ 2h27m

  cuda:0 (~5h15m): no_obs_delay_randomization (fresh) + 4 continuations
  cuda:1 (~4h54m): 7 continuations

Run from repo root:
    .venv/bin/python scripts/smooth_policy/amp_history/configs/td3/zeroshot_ablations_700k/_generate.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
BASE = REPO_ROOT / "scripts/smooth_policy/amp_history/configs/td3/td3_hist2_motion0_v2.yaml"
OUT_DIR = Path(__file__).resolve().parent
SIM_DIR = REPO_ROOT / "scripts/smooth_policy/amp_history/configs/new_juggle/zeroshot_ablations"
CANONICAL_SIM = REPO_ROOT / "scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist2.yaml"
SOURCE_500K_DIR = "runs/td3/zeroshot_ablations"  # where the 500k training_state.pth files live


@dataclass
class Run:
    name: str  # filename suffix and run_name suffix
    sim_path: Path  # which sim YAML to point `config:` at
    description: str
    device: str  # "cuda:0" or "cuda:1"
    is_continuation: bool  # True => warmstart from 500k training_state.pth; False => fresh
    in_queue: bool = True  # False => YAML still regenerates, but the run isn't added to a queue file


# Original 12 ablations launched 2026-05-09 13:36 UTC, completed 18:49 UTC,
# all exit 0. Kept here for reproducibility (re-running this generator
# regenerates their YAMLs) but `in_queue=False` so they don't get re-queued.
ORIGINAL_ABLATIONS = [
    # (name, sim_path, description, device, is_continuation)
    ("no_obs_delay_randomization", SIM_DIR / "sim_no_obs_delay_randomization.yaml",
     "FRESH 700k — replaces broken `no_obs_delay`. Flips randomize_delay only; delay stays on so puck_history density is unchanged.",
     "cuda:0", False),
    ("baseline", CANONICAL_SIM,
     "Continuation 500k->700k — canonical hist2 sysid sim, no knobs flipped.",
     "cuda:0", True),
    ("no_paddle_puck_strength", SIM_DIR / "sim_no_paddle_puck_strength.yaml",
     "Continuation 500k->700k — paddle-puck strength randomization OFF.",
     "cuda:0", True),
    ("no_wall_direction", SIM_DIR / "sim_no_wall_direction.yaml",
     "Continuation 500k->700k — wall-direction randomization OFF.",
     "cuda:0", True),
    ("start_100_near_top", SIM_DIR / "sim_start_100_near_top.yaml",
     "Continuation 500k->700k — 100% near-top starts.",
     "cuda:0", True),
    ("sysid_off", SIM_DIR / "sim_sysid_off.yaml",
     "Continuation 500k->700k — sysid params reverted to legacy off-the-shelf values.",
     "cuda:1", True),
    ("no_paddle_puck_direction", SIM_DIR / "sim_no_paddle_puck_direction.yaml",
     "Continuation 500k->700k — paddle-puck direction randomization OFF.",
     "cuda:1", True),
    ("no_action_attenuation", SIM_DIR / "sim_no_action_attenuation.yaml",
     "Continuation 500k->700k — action force-attenuation OFF.",
     "cuda:1", True),
    ("start_100_near_paddle", SIM_DIR / "sim_start_100_near_paddle.yaml",
     "Continuation 500k->700k — 100% near-paddle starts.",
     "cuda:1", True),
    ("no_puck_noise", SIM_DIR / "sim_no_puck_noise.yaml",
     "Continuation 500k->700k — additive Gaussian puck noise OFF.",
     "cuda:1", True),
    ("no_occlusions", SIM_DIR / "sim_no_occlusions.yaml",
     "Continuation 500k->700k — random puck occlusions OFF.",
     "cuda:1", True),
    ("all_sysid_no_rand", SIM_DIR / "sim_all_sysid_no_rand.yaml",
     "Continuation 500k->700k — sysid kept; ALL collision/action/observation randomization OFF (kept original `enable_observation_delay: false` semantics for back-compat with the 500k checkpoint).",
     "cuda:1", True),
]

# 2026-05-09 isolation studies — 3 fresh 700k runs ("only X is on, everything
# else off"). All keep `enable_observation_delay: true` (project default per
# the obs-delay-default-on memory) and `randomize_delay: false`.
# GPU split: 2 on cuda:0 (~5h), 1 on cuda:1 (~2h27m).
ISOLATION_ABLATIONS = [
    ("only_obs_noise_occlusion", SIM_DIR / "sim_only_obs_noise_occlusion.yaml",
     "FRESH 700k — isolation: only puck_noise + enable_random_occlusions ON. Everything else (collision-3, action attenuation, delay-jitter) OFF.",
     "cuda:0", False),
    ("only_action_attenuation_obs_noise_occlusion",
     SIM_DIR / "sim_only_action_attenuation_obs_noise_occlusion.yaml",
     "FRESH 700k — isolation: only puck_noise + enable_random_occlusions + enable_action_force_attenuation ON. Collision-3 + delay-jitter OFF.",
     "cuda:0", False),
    ("only_action_attenuation", SIM_DIR / "sim_only_action_attenuation.yaml",
     "FRESH 700k — isolation: only enable_action_force_attenuation ON. Everything else OFF.",
     "cuda:1", False),
]

ABLATIONS = (
    [(*entry, False) for entry in ORIGINAL_ABLATIONS]      # in_queue=False (historical)
    + [(*entry, True) for entry in ISOLATION_ABLATIONS]    # in_queue=True (current)
)

RUNS: list[Run] = [
    Run(name=name, sim_path=sim_path, description=description, device=device,
        is_continuation=is_continuation, in_queue=in_queue)
    for (name, sim_path, description, device, is_continuation, in_queue) in ABLATIONS
]


def render(base_text: str, run: Run) -> str:
    rel_sim = run.sim_path.relative_to(REPO_ROOT)
    rel_log_parent = f"runs/td3/zeroshot_ablations_700k/{run.name}/seed0"
    run_name = (
        f"td3_zeroshot_{run.name}_extend" if run.is_continuation else f"td3_zeroshot_{run.name}"
    )

    text = base_text

    patches = [
        ("total_timesteps: 1000000\n", "total_timesteps: 700000\n"),
        (
            'config: "scripts/smooth_policy/amp_history/configs/new_juggle/sysid_best_params_hist2.yaml"\n',
            f'config: "{rel_sim}"\n',
        ),
        (
            'log_parent_dir: "runs/td3/hist_motion_collision/hist2_motion0_v2/seed0"\n',
            f'log_parent_dir: "{rel_log_parent}"\n',
        ),
        (
            'run_name: "td3_hist2_motion0_v2"\n',
            f'run_name: "{run_name}"\n',
        ),
        ('device: "cuda:0"\n', f'device: "{run.device}"\n'),
    ]

    if run.is_continuation:
        # full_resume from the 500k training_state.pth; full_checkpoint_load defaults to "full_resume".
        resume_path = f"{SOURCE_500K_DIR}/{run.name}/seed0/training_state.pth"
        patches.append(
            ("model_path: null\n", f'model_path: "{resume_path}"\n'),
        )

    for old, new in patches:
        if text.count(old) != 1:
            raise AssertionError(
                f"[{run.name}] expected exactly 1 match for {old.strip()!r} in base td3 args"
            )
        text = text.replace(old, new, 1)

    header_lines = [
        f"# Zero-shot sim2real ablation (700k extension): {run.description}",
        f"# Auto-generated by _generate.py from td3_hist2_motion0_v2.yaml.",
        f"# Diff vs base recipe: 700k steps (was 1M); config -> {rel_sim};",
        f"# log/run dir -> {rel_log_parent}; device {run.device}.",
    ]
    if run.is_continuation:
        header_lines.append(
            f"# CONTINUATION: full_resume from {SOURCE_500K_DIR}/{run.name}/seed0/training_state.pth"
        )
        header_lines.append(
            f"# (global_step restored to 500000; loop runs to 700000 -> +200000 extra steps)."
        )
    else:
        header_lines.append("# FRESH 700k run (no warmstart).")
    header = "\n".join(header_lines) + "\n#\n"

    body_lines = text.splitlines(keepends=True)
    body_start = 0
    for i, line in enumerate(body_lines):
        if line.startswith("#") or line.strip() == "":
            continue
        body_start = i
        break
    return header + "\n" + "".join(body_lines[body_start:])


def main() -> None:
    base_text = BASE.read_text()
    for run in RUNS:
        out_path = OUT_DIR / f"td3_zeroshot_{run.name}{'_extend' if run.is_continuation else ''}.yaml"
        out_path.write_text(render(base_text, run))
        kind = "continue" if run.is_continuation else "fresh   "
        print(f"wrote {out_path.relative_to(REPO_ROOT)}  [{run.device}, {kind}]")

    for gpu in (0, 1):
        names_in_order = [
            (r.name + ("_extend" if r.is_continuation else "")) for r in RUNS
            if r.device == f"cuda:{gpu}" and r.in_queue
        ]
        queue_path = OUT_DIR / f"_queue_gpu{gpu}.txt"
        queue_path.write_text("\n".join(names_in_order) + "\n")
        print(f"wrote {queue_path.relative_to(REPO_ROOT)}  ({len(names_in_order)} runs)")


if __name__ == "__main__":
    main()
