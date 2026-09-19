#!/usr/bin/env python3
"""Generate the sysid-v2 policy-campaign configs (2026-09-19) from `configs/new_juggle/sysid_v2_hist2.yaml`.

One cell per (task, method):

  task    : juggle, puck_vel, puck_goal, puck_goal_vel   (2026-09-19: touch / reach / reach_vel dropped from the campaign)
  method  : sysid      — v2 parameters, no randomization, 1M steps            (scripts/td3/td3_training.py, HER for the goal tasks)
            low05 / low10 / low25 — every identified parameter 5 / 10 / 25 % lower than v2, 1M
                                                                              (same trainer; the "wrong sysid" baselines)
            dr5_3p     — ±25 % per-reset DR on the 3 canonical parameters (paddle_density / puck_damping / gravity), 2M,
                         5-frame history obs                                   (scripts/td3/td3_training_dr.py, HER for the goal tasks)
            dr5_full   — ±25 % DR on the full identified set (+ side / end wall restitution, paddle–puck restitution,
                         pid_kp, pid_ki; masses excluded), 2M                  (same)
            drlong_3p / drlong_full — same DR, long-history TD3 (deployable RMA architecture, 50-step window,
                         trained end-to-end without privileged input), 2M     (scripts/rma/train_base_policy.py, rma_mode: history)
            rma_3p / rma_full — RMA: privileged phase 1 (2M) + adaptation module phase 2 (400k)
                                                                              (scripts/rma/train_base_policy.py + train_adaptation_module.py)
  The goal tasks run under every method: the plain trainers use td3_training_her, the long-history / RMA
  trainer has its own HER path (scripts/rma/train_base_policy.py, active when the sim config sets return_goal_obs).

Outputs (all regenerated, idempotent):
  configs/new_juggle/tasks_v2/sim_{sysid,low05,low10,low25,dr3,drfull}_<task>.yaml     sim configs
  configs/td3/tasks_v2/<task>_<method>.yaml                                trainer args (config: -> the sim config above)
  configs/rma/tasks_v2/<task>_<rma method>_phase2.yaml                     RMA phase-2 args
  configs/td3/tasks_v2/manifest.json                                       the cell list scripts/td3/extras/run_sysid_v2_campaign.py runs

    .venv/bin/python scripts/td3/extras/make_sysid_v2_configs.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List

import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
V2_BASE = "configs/new_juggle/sysid_v2_hist2.yaml"
SIM_OUT = "configs/new_juggle/tasks_v2"
TD3_OUT = "configs/td3/tasks_v2"
RMA_OUT = "configs/rma/tasks_v2"

TASKS: List[str] = ["juggle", "puck_vel", "puck_goal", "puck_goal_vel"]
GOAL_TASKS = {"puck_goal", "puck_goal_vel"}
# task -> the existing sim config whose task-specific keys (task, max_timesteps, num_pucks, terminations, goal keys) are kept
TASK_TEMPLATE = {
    "juggle": "configs/new_juggle/tasks/sim_sysid_juggle.yaml",
    "touch": "configs/new_juggle/tasks/sim_sysid_touch.yaml",
    "reach": "configs/new_juggle/tasks/sim_sysid_reach.yaml",
    "reach_vel": "configs/new_juggle/tasks/sim_sysid_reach_vel.yaml",
    "puck_vel": "configs/new_juggle/tasks/sim_sysid_puck_vel.yaml",
    "puck_goal": "configs/new_juggle/tasks/sim_sysid_puck_goal_hist2.yaml",
    "puck_goal_vel": "configs/new_juggle/tasks/sim_sysid_puck_goal_vel_hist2.yaml",
}
TASK_DESC = {
    "juggle": "puck_juggle_upper_half_reward (canonical juggle)",
    "touch": "puck_touch (+1 on contact)",
    "reach": "paddle_reach_position (+10 at the goal)",
    "reach_vel": "paddle_reach_position_velocity (+10 at goal position and velocity)",
    "puck_vel": "puck_velocity (10 x upward puck displacement per step)",
    "puck_goal": "puck_goal_position_sparse (HER)",
    "puck_goal_vel": "puck_goal_position_speed_sparse (HER)",
}

# DR sets. ±25 % around the v2 value; the side-wall upper bound is capped at 1.0 (0.9 × 1.25 = 1.125 would add
# energy at every side-wall bounce). pid_kd is 0 in v2 (a ±25 % range would be degenerate) and the masses
# (paddle_density is kept in the canonical 3-parameter set only because that set predates v2; puck_density) are
# not identified, so the "full" set adds the restitutions and the two non-zero PID gains.
DR_3P = ["paddle_density", "puck_damping", "gravity"]
DR_FULL = DR_3P + ["side_wall_restitution", "end_wall_restitution", "puck_restitution", "pid_kp", "pid_ki"]
DR_FRACTION = 0.25
RESTITUTION_CAP = {"side_wall_restitution": 1.0, "end_wall_restitution": 1.0}
# The lowXX baselines scale exactly the full DR set by 1 − XX/100 (gravity: XX % weaker pull).
LOW25_PARAMS = DR_FULL
LOW_FRACTIONS = {"low05": 0.05, "low10": 0.10, "low25": 0.25}

METHODS = ["sysid", "low05", "low10", "low25", "dr5_3p", "dr5_full", "drlong_3p", "drlong_full", "rma_3p", "rma_full"]
METHOD_SIM = {"sysid": "sysid", "low05": "low05", "low10": "low10", "low25": "low25", "dr5_3p": "dr3", "dr5_full": "drfull",
              "drlong_3p": "dr3", "drlong_full": "drfull", "rma_3p": "dr3", "rma_full": "drfull"}
PLAIN_METHODS = ("sysid", "low05", "low10", "low25")     # no randomization: the 1M-step plain recipe
METHOD_DESC = {
    "sysid": "sysid v2 physics, no randomization, 1M steps",
    "low05": "sysid v2 physics with every identified parameter 5 % lower (wrong-sysid baseline), 1M steps",
    "low10": "sysid v2 physics with every identified parameter 10 % lower (wrong-sysid baseline), 1M steps",
    "low25": "sysid v2 physics with every identified parameter 25 % lower (wrong-sysid baseline), 1M steps",
    "dr5_3p": "+-25 % per-reset DR on paddle_density / puck_damping / gravity, 5-frame history TD3, 2M steps",
    "dr5_full": "+-25 % per-reset DR on the full identified set (3 canonical + wall / paddle-puck restitution + pid_kp / pid_ki), 5-frame history TD3, 2M steps",
    "drlong_3p": "+-25 % DR on the 3 canonical parameters, long-history TD3 (50-step window, no privileged input), 2M steps",
    "drlong_full": "+-25 % DR on the full identified set, long-history TD3 (50-step window, no privileged input), 2M steps",
    "rma_3p": "+-25 % DR on the 3 canonical parameters, RMA phase 1 (privileged encoder) 2M + phase 2 adaptation module 400k",
    "rma_full": "+-25 % DR on the full identified set, RMA phase 1 (privileged encoder) 2M + phase 2 adaptation module 400k",
}
HISTORY_KEYS = {"rma_mode": "history", "rma_latent_dim": 8, "history_len": 50, "step_features": "latest_frame",
                "history_embed_dim": 32, "history_conv_channels": 32}
RMA_KEYS = {"rma_mode": "privileged", "rma_latent_dim": 8, "rma_encoder_hidden": [256, 128]}


def _load(path: str) -> Dict[str, Any]:
    with open(os.path.join(REPO, path)) as f:
        return yaml.safe_load(f)


def _dump(path: str, header: str, data: Dict[str, Any]) -> None:
    full = os.path.join(REPO, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        for line in header.strip("\n").split("\n"):
            f.write(f"# {line}\n" if line else "#\n")
        yaml.safe_dump(data, f, sort_keys=False)


def dr_ranges(sp: Dict[str, Any], names: List[str]) -> Dict[str, List[float]]:
    out = {}
    for n in names:
        v = float(sp[n])
        lo, hi = sorted([v * (1 - DR_FRACTION), v * (1 + DR_FRACTION)])
        if n in RESTITUTION_CAP:
            hi = min(hi, RESTITUTION_CAP[n])
        out[n] = [round(lo, 6), round(hi, 6)]
    return out


def build_sim_configs(v2: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Returns {task: {variant: path}}."""
    v2_sp = v2["air_hockey"]["simulator_params"]
    paths: Dict[str, Dict[str, str]] = {}
    for task in TASKS:
        tmpl = _load(TASK_TEMPLATE[task])["air_hockey"]
        base = copy.deepcopy(tmpl)
        for k in ("domain_random", "random_variables", "random_variable_ranges", "random_variable_ranges_OOD"):
            base.pop(k, None)
        base["simulator_params"].update(copy.deepcopy(v2_sp))
        paths[task] = {}
        variants = {
            "sysid": ("sysid v2 physics, no randomization", None, {}),
            **{name: (f"sysid v2 physics with {', '.join(LOW25_PARAMS)} scaled by {1 - frac:g} (wrong-sysid baseline)", None,
                      {n: round(float(v2_sp[n]) * (1 - frac), 6) for n in LOW25_PARAMS}) for name, frac in LOW_FRACTIONS.items()},
            "dr3": ("+-25 % per-reset DR on the 3 canonical parameters around sysid v2", DR_3P, {}),
            "drfull": ("+-25 % per-reset DR on the full identified set around sysid v2 (side-wall upper bound capped at 1.0)", DR_FULL, {}),
        }
        for variant, (desc, dr_names, overrides) in variants.items():
            cfg = copy.deepcopy(base)
            cfg["simulator_params"].update(overrides)
            if dr_names:
                cfg["domain_random"] = True
                cfg["random_variables"] = list(dr_names)
                cfg["random_variable_ranges"] = dr_ranges(v2_sp, dr_names)
            header = (f"Task '{task}' ({TASK_DESC[task]}), {desc}.\n"
                      f"Generated by scripts/td3/extras/make_sysid_v2_configs.py from {V2_BASE} (physics) and {TASK_TEMPLATE[task]} (task keys).\n"
                      f"hist_len 2. Do not edit by hand — edit the generator or the v2 base and regenerate.")
            path = f"{SIM_OUT}/sim_{variant}_{task}.yaml"
            _dump(path, header, {"air_hockey": cfg})
            paths[task][variant] = path
    return paths


def build_trainer_configs(sim_paths: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    phase2_tmpl = _load("configs/rma/rma_juggle_dr_phase2.yaml")
    for task in TASKS:
        for method in METHODS:
            recipe = _load(f"configs/td3/tasks/{task}_{'sysid' if method in PLAIN_METHODS else 'dr'}.yaml")
            args = copy.deepcopy(recipe)
            args["config"] = sim_paths[task][METHOD_SIM[method]]
            args["run_name"] = f"{task}_{method}"
            args["log_parent_dir"] = None
            args["model_path"] = None
            if method.startswith("drlong"):
                args.update(HISTORY_KEYS)
            if method.startswith("rma"):
                args.update(RMA_KEYS)
            if method.startswith("drlong") or method.startswith("rma"):
                trainer = "scripts.rma.train_base_policy"
            elif task in GOAL_TASKS:
                trainer = "scripts.td3.td3_training_her"
            elif method in PLAIN_METHODS:
                trainer = "scripts.td3.td3_training"
            else:
                trainer = "scripts.td3.td3_training_dr"
            header = (f"Task '{task}' ({TASK_DESC[task]}) — method '{method}': {METHOD_DESC[method]}.\n"
                      f"Recipe = configs/td3/tasks/{task}_{'sysid' if method in PLAIN_METHODS else 'dr'}.yaml; sim config {args['config']}.\n"
                      f"Trainer: {trainer}. Generated by scripts/td3/extras/make_sysid_v2_configs.py; campaign runner\n"
                      f"scripts/td3/extras/run_sysid_v2_campaign.py. Do not edit by hand.")
            path = f"{TD3_OUT}/{task}_{method}.yaml"
            _dump(path, header, args)
            cell = {"task": task, "method": method, "name": f"{task}_{method}", "args_file": path, "sim_config": args["config"],
                    "trainer": trainer, "total_timesteps": int(args["total_timesteps"]), "goal_task": task in GOAL_TASKS}
            if method.startswith("rma"):
                p2 = copy.deepcopy(phase2_tmpl)
                p2.update({"phase1_dir": "", "config": None, "log_parent_dir": None, "run_name": f"{task}_{method}_phase2",
                           "eval_ood": False, "baseline_model_path": None, "gifs": True})
                p2_path = f"{RMA_OUT}/{task}_{method}_phase2.yaml"
                _dump(p2_path, f"RMA phase 2 (adaptation module) for cell {task}_{method}; phase-1 dir is passed on the CLI by the campaign runner.\n"
                               f"Template configs/rma/rma_juggle_dr_phase2.yaml (20 x 20k on-policy steps). Generated by make_sysid_v2_configs.py.", p2)
                cell["phase2_args_file"] = p2_path
            cells.append(cell)
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    v2 = _load(V2_BASE)
    sim_paths = build_sim_configs(v2)
    cells = build_trainer_configs(sim_paths)
    v2_sp = v2["air_hockey"]["simulator_params"]
    manifest = {
        "v2_base": V2_BASE,
        "v2_params": {k: v2_sp[k] for k in DR_FULL + ["pid_kd", "paddle_restitution", "puck_density", "hist_len"]},
        "dr_fraction": DR_FRACTION, "dr_3p": DR_3P, "dr_full": DR_FULL,
        "dr_ranges_3p": dr_ranges(v2_sp, DR_3P), "dr_ranges_full": dr_ranges(v2_sp, DR_FULL),
        "low25_params": LOW25_PARAMS, "low_fractions": LOW_FRACTIONS, "tasks": TASKS, "methods": METHODS, "method_desc": METHOD_DESC,
        "sim_configs": sim_paths, "cells": cells,
    }
    with open(os.path.join(REPO, TD3_OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    readme = [f"# configs/td3/tasks_v2 — sysid v2 policy campaign (generated, do not edit)", "",
              f"Generated by `scripts/td3/extras/make_sysid_v2_configs.py` from `{V2_BASE}`; run with",
              "`scripts/td3/extras/run_sysid_v2_campaign.py`. Sim configs in `configs/new_juggle/tasks_v2/`, RMA phase-2 args in `configs/rma/tasks_v2/`.", "",
              "| method | what | trainer | steps |", "|---|---|---|---|"]
    for m in METHODS:
        c = next(x for x in cells if x["method"] == m)
        readme.append(f"| `{m}` | {METHOD_DESC[m]} | `{c['trainer']}` | {c['total_timesteps'] // 1000}k |")
    readme += ["", f"{len(cells)} cells. The two goal tasks use HER under every method (td3_training_her for the plain "
                   "trainers; the long-history / RMA trainer relabels itself when the sim config sets return_goal_obs).", "",
               "DR ranges (±25 % around v2, side-wall upper bound capped at 1.0):", "", "| parameter | v2 | 3-parameter set | full set |", "|---|---|---|---|"]
    full = dr_ranges(v2_sp, DR_FULL)
    for n in DR_FULL:
        readme.append(f"| `{n}` | {v2_sp[n]} | {'[%g, %g]' % tuple(full[n]) if n in DR_3P else '–'} | [{full[n][0]:g}, {full[n][1]:g}] |")
    with open(os.path.join(REPO, TD3_OUT, "README.md"), "w") as f:
        f.write("\n".join(readme) + "\n")
    print(f"{len(cells)} cells -> {TD3_OUT}/manifest.json; sim configs in {SIM_OUT}/, phase-2 args in {RMA_OUT}/")


if __name__ == "__main__":
    main()
