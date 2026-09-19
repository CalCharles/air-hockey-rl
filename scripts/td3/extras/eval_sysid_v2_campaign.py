#!/usr/bin/env python3
"""Paired final evaluation of the sysid-v2 campaign: every final policy of every method on the SAME envs and episode seeds.

    .venv/bin/python -m scripts.td3.extras.eval_sysid_v2_campaign --out-root runs/td3/sysid_v2_20260919 [--tasks ...] [--methods ...]

Per task, every finished cell's final policy (plain TD3 actor for sysid / low25 / dr5_*; `HistoryActor` for drlong_*;
RMA adapted policy phi + pi for rma_* — its phase-2 output) is rolled on three eval sets. Goal-conditioned tasks run
through `FlatGoalEnv` (flat [observation, goal] obs, the layout every trainer used; success = goal reached).

  nominal   the task's sysid-v2 sim (no randomization), `--eps-nominal` episodes
  id_3p     the 5 fixed DR envs of the 3-parameter set  (eval_param_seed 12345 -> the same 5 physics overrides the dr5_3p /
            drlong_3p / rma_3p runs were evaluated on during training), `--eps-per-env` episodes each
  id_full   the 5 fixed DR envs of the full set (same seed; the envs the *_full runs were evaluated on)

Episode seeds are identical for every policy (seed = eval_param_seed * 1e5 + env_idx * 1000 + call_index, as in
td3_training_dr), so differences between methods are paired. Writes `<out-root>/final_eval/final_eval.json` and
`final_eval/summary.md` (one table per task: rows = methods, columns = the three sets; return ± SEM over episodes,
success rate, episode length).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scripts.rma.bundle import load_adaptation_module, load_phase1  # noqa: E402
from scripts.rma.env_wrapper import EnvParamNormalizer  # noqa: E402
from scripts.rma.evaluate import AdaptedRMAAgent, HistoryAgent, PlainTD3Agent, evaluate_multi_env, make_eval_env  # noqa: E402

DEFAULT_MANIFEST = "configs/td3/tasks_v2/manifest.json"
SETS = ("nominal", "id_3p", "id_full")


def _yaml(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def eval_configs(manifest: Dict, task: str) -> Dict[str, Dict[str, Any]]:
    """The air_hockey config of each eval set. `nominal` = the sysid sim with zero-width DR ranges on the 3
    canonical parameters, so the same override / seed machinery yields exactly the nominal physics."""
    sims = manifest["sim_configs"][task]
    sysid = _yaml(os.path.join(REPO, sims["sysid"]))["air_hockey"]
    nominal = copy.deepcopy(sysid)
    nominal["random_variables"] = list(manifest["dr_3p"])
    nominal["random_variable_ranges"] = {k: [float(sysid["simulator_params"][k])] * 2 for k in manifest["dr_3p"]}
    return {"nominal": nominal,
            "id_3p": _yaml(os.path.join(REPO, sims["dr3"]))["air_hockey"],
            "id_full": _yaml(os.path.join(REPO, sims["drfull"]))["air_hockey"]}


def _cell_run_dir(out_root: str, cell: Dict) -> Optional[str]:
    d = os.path.join(out_root, cell["name"])
    if cell["method"].startswith("rma"):
        return d if os.path.isfile(os.path.join(d, "phase2", "adaptation_module.pth")) else None
    return d if os.path.isfile(os.path.join(d, "model.pth")) else None


def build_agent_factory(cell: Dict, run_dir: str, normalizer_full: EnvParamNormalizer, obs_dim: int):
    """Returns (factory, normalizer) for a cell. ``obs_dim`` = the flat task observation (30, plus the goal for
    goal-conditioned tasks — every trainer sees that same layout)."""
    m = cell["method"]
    if m.startswith("rma"):
        bundle = load_phase1(run_dir)
        phi, phi_meta = load_adaptation_module(os.path.join(run_dir, "phase2"))
        actor = bundle["actor"]
        return (lambda a=actor, p=phi, sf=phi_meta["step_features"]: AdaptedRMAAgent(a, p, sf)), bundle["normalizer"]
    if m.startswith("drlong"):
        bundle = load_phase1(run_dir)
        if bundle["mode"] != "history":
            raise ValueError(f"{run_dir} is not a history-mode run")
        actor = bundle["actor"]
        return (lambda a=actor: HistoryAgent(a)), normalizer_full
    args = _yaml(os.path.join(run_dir, "args.yaml"))
    model_path = os.path.join(run_dir, "model.pth")
    return (lambda mp=model_path, ul=bool(args.get("use_last_action_in_policy_state", True)), h=int(args["agent_hidden_layer_size"]),
            n=int(args["agent_num_hidden_layers"]), od=int(obs_dim): PlainTD3Agent(mp, od, 2, ul, h, n)), normalizer_full


def summary_markdown(results: Dict[str, Dict[str, Dict[str, Any]]], manifest: Dict, cli) -> str:
    L = ["# sysid v2 campaign — paired final evaluation", "",
         f"Every finished policy on the same envs and episode seeds (eval_param_seed {cli.eval_param_seed}, call_index {cli.call_index}). "
         f"`nominal` = the task's sysid-v2 sim, {cli.eps_nominal} episodes; `id_3p` / `id_full` = the 5 fixed DR envs of the 3-parameter / "
         f"full set, {cli.eps_per_env} episodes each. Cells: return ± SEM over episodes · success rate · mean episode length. "
         "The RMA rows are the adapted policy (phase 2); drlong = long-history TD3.", ""]
    for task, per_method in results.items():
        L += [f"## {task}", "", "| method | " + " | ".join(f"{s} return | {s} success | {s} len" for s in SETS) + " |", "|---|" + "---:|---:|---:|" * len(SETS)]
        for method in manifest["methods"]:
            if method not in per_method:
                continue
            cells = []
            for s in SETS:
                r = per_method[method].get(s)
                if r is None or "error" in r:
                    cells.append("– | – | –")
                    continue
                a = r["aggregate"]
                cells.append(f"{a['mean_return_across_envs']:.1f} ± {a['return_sem_across_episodes']:.1f} | {a['mean_success_across_envs']:.2f} | {a['mean_ep_length_across_envs']:.0f}")
            L.append(f"| {method} | " + " | ".join(cells) + " |")
        L.append("")
    missing = [f"{t}:{m}" for t, pm in results.items() for m, r in pm.items() if any("error" in (r.get(s) or {}) for s in SETS)]
    if missing:
        L += ["Cells with an evaluation error: " + ", ".join(missing), ""]
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--eps-nominal", type=int, default=50)
    ap.add_argument("--eps-per-env", type=int, default=10)
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--eval-param-seed", type=int, default=12345)
    ap.add_argument("--call-index", type=int, default=1000)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    cli = ap.parse_args(argv)
    import torch

    torch.set_num_threads(1)      # rollout-bound; the default thread pool would grab every core
    manifest = json.load(open(os.path.join(REPO, cli.manifest)))
    tasks = cli.tasks or manifest["tasks"]
    methods = cli.methods or manifest["methods"]
    out_root = os.path.abspath(cli.out_root)
    out_dir = os.path.join(out_root, "final_eval")
    os.makedirs(out_dir, exist_ok=True)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    t0 = time.time()
    for task in tasks:
        cfgs = eval_configs(manifest, task)
        normalizer_full = EnvParamNormalizer.from_air_hockey_config(cfgs["id_full"])
        probe = make_eval_env(dict(cfgs["nominal"], domain_random=False))
        obs_dim = int(np.prod(probe.observation_space.shape))     # 30, or 30 + goal_dim (FlatGoalEnv) for the goal tasks
        probe.close()
        results[task] = {}
        for cell in manifest["cells"]:
            if cell["task"] != task or cell["method"] not in methods:
                continue
            run_dir = _cell_run_dir(out_root, cell)
            if run_dir is None:
                print(f"[final eval] {cell['name']}: no finished policy, skipped", flush=True)
                continue
            per_set: Dict[str, Any] = {}
            for s in cli.sets:
                n_envs, eps = (1, cli.eps_nominal) if s == "nominal" else (cli.n_envs, cli.eps_per_env)
                try:
                    factory, normalizer = build_agent_factory(cell, run_dir, normalizer_full, obs_dim)
                    p = evaluate_multi_env(cfgs[s], {cell["method"]: factory}, normalizer, eval_param_seed=cli.eval_param_seed, n_envs=n_envs,
                                           eps_per_env=eps, call_index=cli.call_index, save_dir=None, log_prefix=f"[final eval {task}/{s}]")
                    r = {"aggregate": p["aggregate"], "per_env": p["per_env"], "overrides": p["overrides"]}
                    a = r["aggregate"]
                    print(f"[final eval] {cell['name']} {s}: return {a['mean_return_across_envs']:.1f} ± {a['return_sem_across_episodes']:.1f} "
                          f"success {a['mean_success_across_envs']:.2f} len {a['mean_ep_length_across_envs']:.0f}", flush=True)
                except Exception as exc:  # keep going, report in the table
                    import traceback
                    traceback.print_exc()
                    r = {"error": f"{type(exc).__name__}: {exc}"}
                per_set[s] = r
            results[task][cell["method"]] = per_set
            with open(os.path.join(out_dir, "final_eval.json"), "w") as f:
                json.dump({"settings": vars(cli), "results": results}, f, indent=1)
        with open(os.path.join(out_dir, "summary.md"), "w") as f:
            f.write(summary_markdown(results, manifest, cli) + "\n")
    print(summary_markdown(results, manifest, cli))
    print(f"[final eval] wrote {out_dir} ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
