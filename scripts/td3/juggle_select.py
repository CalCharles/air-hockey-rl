"""Evaluate a validation-selected scratch checkpoint, optionally after a live run.

This never trains or warm-starts a policy. Selection uses only saved validation
results; the requested evaluation seeds must be disjoint from those starts and
the run's final-actor benchmark. Output is durable and existing output is refused.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import yaml

from scripts.td3.juggle_original_sensing import assert_original_sensing
from scripts.td3.juggle_survival import evaluate_checkpoint


def select_checkpoint(root, seed_base, episodes):
    candidates, used_seeds = [], set()
    for path in (root / "train").glob("checkpoint_*/survival_eval/results.json"):
        payload = json.loads(path.read_text())
        used_seeds.update(range(payload["seed_base"], payload["seed_base"] + payload["episodes_per_scenario"]))
        summary = payload["scenarios"]["training"]["summary"]
        rank = (summary.get("true_state_sustained_juggling_fraction", summary["sustained_juggling_fraction"]),
                summary["restricted_mean_steps"])
        candidates.append((rank, path.parents[1], summary))
    final = root / "final_eval/results.json"
    if final.exists():
        payload = json.loads(final.read_text())
        used_seeds.update(range(payload["seed_base"], payload["seed_base"] + payload["episodes_per_scenario"]))
    if used_seeds.intersection(range(seed_base, seed_base + episodes)):
        raise ValueError("Selected evaluation seeds overlap validation or final-actor starts")
    if not candidates:
        raise ValueError("No completed training-distribution checkpoint validation")
    _, checkpoint, summary = max(candidates, key=lambda item: (item[0], item[1].name))
    config = yaml.safe_load((checkpoint / "config.yaml").read_text())["air_hockey"]
    # The trainer saves its current evaluation RNG seed in this field. The
    # survival evaluator replaces it per episode; no physical/sensor key varies.
    assert_original_sensing(dict(config, seed=0))
    return checkpoint, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=400000)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=10000)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "selected_eval"
    out.mkdir(parents=True, exist_ok=False)
    state = {"status": "waiting", "started": datetime.now(timezone.utc).isoformat()}

    def save(status):
        state.update(status=status, updated=datetime.now(timezone.utc).isoformat())
        temporary = out / "status.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(out / "status.json")

    try:
        save("waiting")
        if args.wait:
            while True:
                training = json.loads((root / "status.json").read_text())
                if training["status"] == "complete":
                    break
                if training["status"] == "failed":
                    raise RuntimeError("Training runner failed: " + str(training.get("error")))
                time.sleep(20)
        checkpoint, summary = select_checkpoint(root, args.seed_base, args.episodes)
        state.update(checkpoint=str(checkpoint), validation_summary=summary,
                     seed_base=args.seed_base, episodes=args.episodes, horizon=args.horizon)
        save("evaluating")
        evaluate_checkpoint(checkpoint, episodes=args.episodes, horizon=args.horizon,
                            seed_base=args.seed_base, scenario_set="training", out_dir=out)
        save("complete")
    except Exception as exc:
        state["error"] = repr(exc)
        save("failed")
        raise


if __name__ == "__main__":
    main()
