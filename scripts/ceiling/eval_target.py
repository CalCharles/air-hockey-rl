"""Fixed-protocol evaluation of juggling policies on the target sim.

Every policy in the ceiling search is scored here, on exactly
``configs/new_juggle/sysid_v2_hist4.yaml`` (only the episode cap is raised).

    selection:      32 episodes, cap 2,000 steps, seeds 500000+ (reused for every candidate)
    selection_long: 32 episodes, cap 10,000 steps, same seeds; for candidates that saturate the 2,000-step cap
    final:          100 episodes, cap 10,000 steps, seeds 900000+ (only for chosen policies)

Run with ``python -m scripts.ceiling.eval_target --protocol selection --label <name> <checkpoint_dir> ...``.
Policies are evaluated in parallel, one process each. Results land in
``runs/ceiling/evals/<protocol>/<label>/<policy>/`` (``results.json``, ``summary.md``, GIFs).
``--edge-velocity-fix`` evaluates on the target sim with the counterfactual top-edge velocity fix of
``scripts/rma/edge_velocity_fix.py`` (instance patch; use a separate label).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
TARGET_CONFIG = REPO / "configs/new_juggle/sysid_v2_hist4.yaml"
EVAL_ROOT = REPO / "runs/ceiling/evals"
PROTOCOLS = {
    "selection": {"episodes": 32, "horizon": 2000, "seed_base": 500000},
    "selection_long": {"episodes": 32, "horizon": 10000, "seed_base": 500000},
    "final": {"episodes": 100, "horizon": 10000, "seed_base": 900000},
}
# Keys that may legitimately differ between a policy's training config and the target:
# the reset seed, the training episode cap and training-only reward shaping. Evaluation
# always runs on the unmodified target config. ``random_variables`` / ``random_variable_ranges``
# are inert while ``domain_random`` is off (which is checked); the RMA/history trainer needs them.
IGNORED_KEYS = {"seed", "max_timesteps", "enable_survival_bonus", "survival_bonus_per_step", "puck_loss_penalty",
                "random_variables", "random_variable_ranges"}


def config_differences(cfg, target, prefix=""):
    """Paths where ``cfg`` differs from ``target``, other than the ignored keys."""
    out = []
    for key in sorted(set(cfg) | set(target)):
        a, b = cfg.get(key), target.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out += config_differences(a, b, f"{prefix}{key}.")
        elif a != b and key not in IGNORED_KEYS:
            out.append(f"{prefix}{key}: {a!r} (policy) vs {b!r} (target)")
    return out


def policy_name(run_dir):
    """Short unique name: the path below runs/, with separators flattened."""
    path = Path(run_dir).resolve()
    for root in (REPO / "runs", REPO):
        try:
            path = path.relative_to(root)
            break
        except ValueError:
            pass
    return "__".join(path.parts)


def _evaluate_chunk(run_dir, out_dir, protocol, episodes, seed_base, gif_steps, edge_velocity_fix=False):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if edge_velocity_fix:
        from airhockey.airhockey_base import AirHockeyBaseEnv
        from scripts.rma.edge_velocity_fix import patch_env_class

        patch_env_class(AirHockeyBaseEnv)
    from scripts.td3.juggle_survival import evaluate_checkpoint

    target = yaml.safe_load(TARGET_CONFIG.read_text())["air_hockey"]
    result = evaluate_checkpoint(
        run_dir, out_dir=out_dir, config_path=str(TARGET_CONFIG),
        scenarios={"target": copy.deepcopy(target)}, gif_steps=gif_steps,
        horizon=PROTOCOLS[protocol]["horizon"], episodes=episodes, seed_base=seed_base,
    )
    return result


def merge_chunks(out_dir, chunk_results, protocol):
    """Combine per-chunk results into one ``results.json`` / ``summary.md`` for the policy."""
    from scripts.td3.juggle_survival import summarize_episodes

    rows = []
    for i, chunk in enumerate(chunk_results):
        for row in chunk["scenarios"]["target"]["episodes"]:
            rows.append({**row, "trajectory": f"chunk{i}/{row['trajectory']}"})
    rows.sort(key=lambda row: row["seed"])
    summary = summarize_episodes(rows, PROTOCOLS[protocol]["horizon"])
    merged = {**chunk_results[0], "episodes_per_scenario": len(rows), "seed_base": PROTOCOLS[protocol]["seed_base"]}
    merged["scenarios"] = {"target": {**chunk_results[0]["scenarios"]["target"], "summary": summary, "episodes": rows}}
    Path(out_dir, "results.json").write_text(json.dumps(merged, indent=2) + "\n")
    Path(out_dir, "summary.md").write_text(
        f"# {merged['run_dir']} ({protocol})\n\n{len(rows)} episodes, cap {summary['horizon']} steps.\n\n"
        f"Mean steps {summary['restricted_mean_steps']:.1f}, median {summary['median_observed_steps']:.0f}, "
        f"reached cap {summary['horizon_survival_fraction']:.1%}, juggles {summary['mean_juggles']:.1f}, "
        f"failures {summary['failures']}, survival {summary['survival_at_steps']}.\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", help="checkpoint directories (model.pth + args.yaml)")
    parser.add_argument("--protocol", choices=sorted(PROTOCOLS), default="selection")
    parser.add_argument("--label", required=True, help="group name, e.g. baseline_hist4 or the sweep name")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--gif-steps", type=int, default=400)
    parser.add_argument("--chunks", type=int, default=1,
                        help="split each policy's episodes over this many parallel processes")
    parser.add_argument("--allow-config-mismatch", action="store_true",
                        help="evaluate policies trained on a different sim config (e.g. privileged teachers)")
    parser.add_argument("--edge-velocity-fix", action="store_true",
                        help="remove the paddle's outward velocity at the workspace clip (scripts/rma/edge_velocity_fix.py)")
    cli = parser.parse_args()

    target = yaml.safe_load(TARGET_CONFIG.read_text())["air_hockey"]
    for run_dir in cli.run_dirs:
        cfg_path = Path(run_dir) / "config.yaml"
        if cfg_path.exists():
            diffs = config_differences(yaml.safe_load(cfg_path.read_text())["air_hockey"], target)
            if diffs and not cli.allow_config_mismatch:
                raise SystemExit(f"{run_dir} was trained on a different sim:\n  " + "\n  ".join(diffs))
            if diffs:
                print(f"warning: {run_dir} trained on a different sim ({len(diffs)} keys differ)")

    root = EVAL_ROOT / cli.protocol / cli.label
    jobs = {run_dir: root / policy_name(run_dir) for run_dir in cli.run_dirs}
    protocol = PROTOCOLS[cli.protocol]
    per_chunk = -(-protocol["episodes"] // cli.chunks)
    with ProcessPoolExecutor(max_workers=min(cli.workers, len(jobs) * cli.chunks)) as pool:
        futures = {}
        for run_dir, out in jobs.items():
            if cli.chunks == 1:
                futures[run_dir] = [pool.submit(_evaluate_chunk, str(run_dir), str(out), cli.protocol,
                                                protocol["episodes"], protocol["seed_base"], cli.gif_steps,
                                                cli.edge_velocity_fix)]
                continue
            futures[run_dir] = []
            for i in range(cli.chunks):
                start = i * per_chunk
                count = min(per_chunk, protocol["episodes"] - start)
                if count > 0:
                    futures[run_dir].append(pool.submit(
                        _evaluate_chunk, str(run_dir), str(out / f"chunk{i}"), cli.protocol, count,
                        protocol["seed_base"] + start, cli.gif_steps if i == 0 else 0, cli.edge_velocity_fix))
        for run_dir, chunk_futures in futures.items():
            results = [f.result() for f in chunk_futures]
            s = (results[0]["scenarios"]["target"]["summary"] if cli.chunks == 1
                 else merge_chunks(jobs[run_dir], results, cli.protocol))
            print(f"{run_dir}: mean steps {s['restricted_mean_steps']:.1f}, median {s['median_observed_steps']:.0f}, "
                  f"reached cap {s['horizon_survival_fraction']:.1%}, juggles {s['mean_juggles']:.1f}", flush=True)
    (root / "protocol.json").write_text(json.dumps(
        {"protocol": cli.protocol, **PROTOCOLS[cli.protocol], "edge_velocity_fix": cli.edge_velocity_fix,
         "target_config": str(TARGET_CONFIG.relative_to(REPO)),
         "policies": {str(r): str(o.relative_to(REPO)) for r, o in jobs.items()}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
