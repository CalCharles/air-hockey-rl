"""Paired evaluation of any set of RMA / baseline policies on the fixed DR eval envs.

Every agent runs on the same physics overrides **and the same episode
seeds**, in distribution (``random_variable_ranges``) and, if the sim config
has ``random_variable_ranges_OOD``, out of distribution.

    .venv/bin/python -m scripts.rma.eval_paired \
        --rma-phase2 runs/rma/juggle_dr_phase1_seed0/phase2 runs/rma/juggle_dr_phase1_seed1/phase2 \
        --history-runs runs/rma/history_juggle_dr_seed0 runs/rma/history_juggle_dr_seed1 \
        --td3-runs runs/td3/tasks_20260904/dr/juggle_dr runs/td3/dr_juggle_seed1 \
        --eps-per-env 20 --out-dir runs/rma/paired_eval

Agents added per flag (label = basename of the run dir, or of its parent for
``.../phase2``):
  --rma-phase2 DIR ...   adapted:<label> (phi + pi), privileged:<label>, nominal:<label>
  --rma-phase1 DIR ...   privileged:<label>, nominal:<label>   (phase-1 only)
  --history-runs DIR ... history:<label>   (long-history TD3 control)
  --td3-runs DIR|model.pth ...   td3_dr:<label> (canonical DeterministicAgent)

Writes ``<out-dir>/paired_eval.json`` (+ ``_ood``) and ``paired_eval.md``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List

import numpy as np
import yaml

from scripts.rma.bundle import load_adaptation_module, load_phase1
from scripts.rma.env_wrapper import EnvParamNormalizer
from scripts.rma.evaluate import (
    AdaptedRMAAgent,
    EvalAgent,
    HistoryAgent,
    NominalLatentAgent,
    PlainTD3Agent,
    PrivilegedRMAAgent,
    evaluate_multi_env,
)


def _label(path: str) -> str:
    path = path.rstrip("/")
    base = os.path.basename(path)
    if base.startswith("phase2") or base == "model.pth":
        return os.path.basename(os.path.dirname(path))
    return base


def _yaml(path: str):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rma-phase2", nargs="*", default=[])
    ap.add_argument("--rma-phase1", nargs="*", default=[])
    ap.add_argument("--history-runs", nargs="*", default=[])
    ap.add_argument("--td3-runs", nargs="*", default=[])
    ap.add_argument("--config", default=None, help="sim config YAML (default: config.yaml of the first RMA / history run)")
    ap.add_argument("--eval-param-seed", type=int, default=12345)
    ap.add_argument("--n-envs", type=int, default=5)
    ap.add_argument("--eps-per-env", type=int, default=20)
    ap.add_argument("--call-index", type=int, default=1000)
    ap.add_argument("--no-ood", action="store_true")
    ap.add_argument("--ood-n-envs", type=int, default=5)
    ap.add_argument("--td3-hidden", type=int, default=64)
    ap.add_argument("--td3-layers", type=int, default=2)
    ap.add_argument("--gifs", action="store_true")
    ap.add_argument("--out-dir", required=True)
    cli = ap.parse_args()

    agents: Dict[str, Callable[[], EvalAgent]] = {}
    config = _yaml(cli.config) if cli.config else None
    normalizer = None

    for p2 in cli.rma_phase2:
        p2_args = _yaml(os.path.join(p2, "args.yaml"))
        bundle = load_phase1(p2_args["phase1_dir"])
        phi, phi_meta = load_adaptation_module(p2)
        label = _label(p2)
        actor = bundle["actor"]
        agents[f"adapted:{label}"] = (lambda a=actor, m=phi, sf=phi_meta["step_features"]: AdaptedRMAAgent(a, m, sf))
        agents[f"privileged:{label}"] = (lambda a=actor: PrivilegedRMAAgent(a))
        agents[f"nominal:{label}"] = (lambda a=actor: NominalLatentAgent(a))
        config = config or bundle["config"]
        normalizer = normalizer or bundle["normalizer"]
    for p1 in cli.rma_phase1:
        bundle = load_phase1(p1)
        label = _label(p1)
        actor = bundle["actor"]
        agents[f"privileged:{label}"] = (lambda a=actor: PrivilegedRMAAgent(a))
        agents[f"nominal:{label}"] = (lambda a=actor: NominalLatentAgent(a))
        config = config or bundle["config"]
        normalizer = normalizer or bundle["normalizer"]
    for h in cli.history_runs:
        bundle = load_phase1(h)
        if bundle["mode"] != "history":
            raise ValueError(f"{h} is not a history-mode run")
        actor = bundle["actor"]
        agents[f"history:{_label(h)}"] = (lambda a=actor: HistoryAgent(a))
        config = config or bundle["config"]
        normalizer = normalizer or bundle["normalizer"]
    if config is None:
        raise ValueError("pass --config or at least one RMA / history run")
    air_hockey = config["air_hockey"]
    normalizer = normalizer or EnvParamNormalizer.from_air_hockey_config(air_hockey)
    obs_dim = 30
    for t in cli.td3_runs:
        model_path = t if t.endswith(".pth") else os.path.join(t, "model.pth")
        run_dir = os.path.dirname(model_path)
        use_last = True
        try:
            use_last = bool(_yaml(os.path.join(run_dir, "args.yaml")).get("use_last_action_in_policy_state", True))
        except OSError:
            pass
        agents[f"td3_dr:{_label(t)}"] = (lambda mp=model_path, ul=use_last: PlainTD3Agent(mp, obs_dim, 2, ul, cli.td3_hidden, cli.td3_layers))
    if not agents:
        raise ValueError("no agents given")

    os.makedirs(cli.out_dir, exist_ok=True)
    gif_agents = tuple(agents) if cli.gifs else ()
    results = {
        "id": evaluate_multi_env(air_hockey, agents, normalizer, eval_param_seed=cli.eval_param_seed, n_envs=cli.n_envs, eps_per_env=cli.eps_per_env,
                                 call_index=cli.call_index, save_dir=cli.out_dir, primary=next(iter(agents)), gif_agents=gif_agents, json_name="paired_eval.json")
    }
    if not cli.no_ood and air_hockey.get("random_variable_ranges_OOD"):
        results["ood"] = evaluate_multi_env(air_hockey, agents, normalizer, eval_param_seed=cli.eval_param_seed, n_envs=cli.ood_n_envs, eps_per_env=cli.eps_per_env,
                                            call_index=cli.call_index, save_dir=cli.out_dir, primary=next(iter(agents)), gif_agents=(),
                                            ranges_key="random_variable_ranges_OOD", json_name="paired_eval_ood.json")

    lines: List[str] = [f"Paired eval: {cli.n_envs} ID envs" + (f" + {cli.ood_n_envs} OOD envs" if "ood" in results else "") +
                        f", {cli.eps_per_env} episodes per env, eval_param_seed {cli.eval_param_seed}, call_index {cli.call_index} (same episode seeds for every agent); ± = SEM over episodes.", ""]
    lines.append("| agent | " + " | ".join(f"{s.upper()} return ± SEM | {s.upper()} success | {s.upper()} ep len" for s in results) + " |")
    lines.append("|---|" + "---:|---:|---:|" * len(results))
    for name in agents:
        cells = []
        for split in results:
            agg = results[split]["agents"][name]["aggregate"]
            cells.append(f"{agg['mean_return_across_envs']:.1f} ± {agg['return_sem_across_episodes']:.1f} | {agg['mean_success_across_envs']:.2f} | {agg['mean_ep_length_across_envs']:.0f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    # group means over seeds per agent kind
    kinds = sorted({n.split(":")[0] for n in agents})
    if any(sum(n.startswith(k + ":") for n in agents) > 1 for k in kinds):
        lines += ["", "Mean over seeds per agent kind (± = std across seeds):", ""]
        lines.append("| kind | n seeds | " + " | ".join(f"{s.upper()} return" for s in results) + " |")
        lines.append("|---|---:|" + "---:|" * len(results))
        for k in kinds:
            names = [n for n in agents if n.startswith(k + ":")]
            cells = []
            for split in results:
                vals = [results[split]["agents"][n]["aggregate"]["mean_return_across_envs"] for n in names]
                cells.append(f"{np.mean(vals):.1f} ± {np.std(vals):.1f}" if len(vals) > 1 else f"{vals[0]:.1f}")
            lines.append(f"| {k} | {len(names)} | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    with open(os.path.join(cli.out_dir, "paired_eval.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
