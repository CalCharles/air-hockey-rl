"""Extract 500k metrics for Phase-1 exploration ablation table."""

from __future__ import annotations

import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "E0 anchor (P1a)": "runs/td3/sysid_params/upd_sweep",
    "E2 no-warmstart": "runs/td3/sysid_params/expl_no_warmstart",
    "E5 warmstart-heavy": "runs/td3/sysid_params/expl_warmstart_heavy",
    "E4 no-bootstrap": "runs/td3/sysid_params/expl_no_bootstrap",
}

CUTOFF = 500_000


def load(path: str) -> dict:
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        out[tag] = [(e.step, e.value, e.wall_time) for e in events]
    return out


def within(events, max_step):
    return [(s, v, t) for (s, v, t) in events if s <= max_step]


def value_at_or_before(events, step):
    evs = [e for e in events if e[0] <= step]
    if not evs:
        return None
    return evs[-1][1]


def max_value(events, max_step):
    evs = within(events, max_step)
    if not evs:
        return None
    return max(v for (_, v, _) in evs)


def first_step_ge(events, threshold, max_step):
    for s, v, _ in events:
        if s > max_step:
            break
        if v >= threshold:
            return s
    return None


def wall_seconds(events, max_step):
    evs = within(events, max_step)
    if len(evs) < 2:
        return None
    return evs[-1][2] - evs[0][2]


def tail_mean(events, max_step, n):
    evs = within(events, max_step)
    if len(evs) < 1:
        return None
    tail = evs[-n:]
    return sum(v for (_, v, _) in tail) / len(tail)


def extract(path: str) -> dict:
    m = load(path)
    ep_ret = m.get("charts/episodic_return", [])
    max_ret = m.get("charts/max_episodic_return", [])
    avg_ret = m.get("charts/avg_episodic_return", [])
    rolling = m.get("charts/rolling2k_avg_episode_return", [])
    pos = m.get("rewards/sampled_task_reward_positive_fraction", [])

    return {
        "ret@250k (rolling2k)": value_at_or_before(rolling, 250_000),
        "ret@500k (rolling2k)": value_at_or_before(rolling, CUTOFF),
        "tail10@500k": tail_mean(ep_ret, CUTOFF, 10),
        "tail50@500k": tail_mean(ep_ret, CUTOFF, 50),
        "max_ret<=500k": max_value(max_ret, CUTOFF) if max_ret else max_value(ep_ret, CUTOFF),
        "pos_frac@500k": value_at_or_before(pos, CUTOFF),
        "step_ret>=50": first_step_ge(ep_ret, 50, CUTOFF),
        "step_ret>=100": first_step_ge(ep_ret, 100, CUTOFF),
        "wall_sec<=500k": wall_seconds(ep_ret, CUTOFF),
        "total_ep_ret_events": len([e for e in ep_ret if e[0] <= CUTOFF]),
    }


def fmt(v, kind="float"):
    if v is None:
        return "—"
    if kind == "float":
        return f"{v:.2f}"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "pct":
        return f"{v:.3f}"
    if kind == "hours":
        return f"{v/3600:.2f}h"
    return str(v)


def main():
    for name, path in RUNS.items():
        if not Path(path).exists():
            print(f"{name}: MISSING PATH {path}")
            continue
        r = extract(path)
        print(f"\n=== {name} ({path}) ===")
        print(f"  ret@250k (rolling2k):   {fmt(r['ret@250k (rolling2k)'])}")
        print(f"  ret@500k (rolling2k):   {fmt(r['ret@500k (rolling2k)'])}")
        print(f"  tail10 @500k:           {fmt(r['tail10@500k'])}")
        print(f"  tail50 @500k:           {fmt(r['tail50@500k'])}")
        print(f"  max_ret in [0, 500k]:   {fmt(r['max_ret<=500k'])}")
        print(f"  pos_frac @500k:         {fmt(r['pos_frac@500k'])}")
        print(f"  step ret>=50:           {fmt(r['step_ret>=50'], 'int')}")
        print(f"  step ret>=100:          {fmt(r['step_ret>=100'], 'int')}")
        print(f"  wall time to 500k:      {fmt(r['wall_sec<=500k'], 'hours')}")
        print(f"  num ep_return events:   {r['total_ep_ret_events']}")


if __name__ == "__main__":
    main()
