#!/usr/bin/env python3
"""
Standalone Box2D collision diagnostic.

Creates minimal Box2D worlds with only two dynamic circles (paddle + puck),
sweeps across density ratios and speed distributions, and compares post-collision
velocities to analytical rigid-body predictions.

No PID, no force limits, no env machinery -- pure Box2D collision resolution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import os
import numpy as np
from Box2D.b2 import world, contactListener
from Box2D import b2CircleShape, b2FixtureDef, b2Filter, b2_dynamicBody

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

PADDLE_RADIUS = 0.0508
PUCK_RADIUS = 0.03175
RESTITUTION = 1.0
FRICTION = 0.0


@dataclass
class CollisionResult:
    paddle_density: float
    puck_density: float
    mass_ratio: float
    rel_speed: float
    speed_label: str
    paddle_v_in: float
    puck_v_in: float
    paddle_v_out: float
    puck_v_out: float
    paddle_v_expected: float
    puck_v_expected: float
    puck_error_pct: float
    paddle_error_pct: float
    ke_ratio: float
    p_ratio: float
    collision_detected: bool


class CollisionDetector(contactListener):
    def __init__(self):
        contactListener.__init__(self)
        self.contact_count = 0

    def PostSolve(self, contact, impulse):
        self.contact_count += 1


def analytical_1d_collision(m1: float, m2: float, v1: float, v2: float, e: float = 1.0):
    """
    1D collision with coefficient of restitution e.
    Returns (v1_final, v2_final).
    """
    denom = m1 + m2
    v1f = ((m1 - e * m2) / denom) * v1 + ((1 + e) * m2 / denom) * v2
    v2f = ((1 + e) * m1 / denom) * v1 + ((m2 - e * m1) / denom) * v2
    return v1f, v2f


def run_single_collision(
    paddle_density: float,
    puck_density: float,
    paddle_v: float,
    puck_v: float,
    restitution: float = RESTITUTION,
    dt: float = 1.0 / 2000.0,
    vel_iters: int = 100,
    pos_iters: int = 100,
    max_steps: int = 20000,
    use_listener_fix: bool = False,
) -> dict:
    """
    Set up two circles approaching head-on along y-axis, step until collision
    resolves, return pre/post velocities.
    """
    w = world(gravity=(0, 0), doSleep=False)

    if use_listener_fix:
        from airhockey.sims.airhockey_box2d import CollisionForceListener
        listener = CollisionForceListener()
        w.contactListener = listener
        detector_contact_count = lambda: len(listener.collision_forces)
    else:
        detector = CollisionDetector()
        w.contactListener = detector
        detector_contact_count = lambda: detector.contact_count

    gap = 0.01
    separation = PADDLE_RADIUS + PUCK_RADIUS + gap

    paddle_body = w.CreateDynamicBody(
        fixtures=b2FixtureDef(
            shape=b2CircleShape(radius=PADDLE_RADIUS),
            density=paddle_density,
            restitution=restitution,
            friction=FRICTION,
            filter=b2Filter(maskBits=1, categoryBits=1),
        ),
        bullet=True,
        position=(0, -separation / 2),
        linearVelocity=(0, paddle_v),
        linearDamping=0.0,
        userData="paddle_ego",
    )

    puck_body = w.CreateDynamicBody(
        fixtures=b2FixtureDef(
            shape=b2CircleShape(radius=PUCK_RADIUS),
            density=puck_density,
            restitution=restitution,
            friction=FRICTION,
            filter=b2Filter(maskBits=1, categoryBits=1),
        ),
        bullet=True,
        position=(0, separation / 2),
        linearVelocity=(0, puck_v),
        linearDamping=0.0,
        userData="puck_0",
    )

    pre_paddle_v = paddle_v
    pre_puck_v = puck_v

    collision_happened = False
    post_paddle_v = None
    post_puck_v = None

    settle_steps_after_collision = 0
    settle_target = 50

    for step_i in range(max_steps):
        w.Step(dt, vel_iters, pos_iters)
        w.ClearForces()

        if not collision_happened and detector_contact_count() > 0:
            collision_happened = True
            settle_steps_after_collision = 0

        if collision_happened:
            settle_steps_after_collision += 1
            if settle_steps_after_collision >= settle_target:
                post_paddle_v = float(paddle_body.linearVelocity.y)
                post_puck_v = float(puck_body.linearVelocity.y)
                break

    if post_paddle_v is None:
        post_paddle_v = float(paddle_body.linearVelocity.y)
        post_puck_v = float(puck_body.linearVelocity.y)

    w.contactListener = None

    return {
        "pre_paddle_v": pre_paddle_v,
        "pre_puck_v": pre_puck_v,
        "post_paddle_v": post_paddle_v,
        "post_puck_v": post_puck_v,
        "collision_detected": collision_happened,
        "paddle_mass": paddle_body.mass,
        "puck_mass": puck_body.mass,
    }


def compute_result(
    paddle_density: float,
    puck_density: float,
    rel_speed: float,
    speed_label: str,
    paddle_v: float,
    puck_v: float,
    restitution: float = RESTITUTION,
    use_listener_fix: bool = False,
) -> CollisionResult:
    raw = run_single_collision(
        paddle_density, puck_density, paddle_v, puck_v, restitution,
        use_listener_fix=use_listener_fix,
    )

    m1 = raw["paddle_mass"]
    m2 = raw["puck_mass"]
    paddle_v_exp, puck_v_exp = analytical_1d_collision(
        m1, m2, raw["pre_paddle_v"], raw["pre_puck_v"], restitution
    )

    ke_pre = 0.5 * m1 * raw["pre_paddle_v"] ** 2 + 0.5 * m2 * raw["pre_puck_v"] ** 2
    ke_post = 0.5 * m1 * raw["post_paddle_v"] ** 2 + 0.5 * m2 * raw["post_puck_v"] ** 2
    ke_ratio = ke_post / ke_pre if ke_pre > 1e-12 else float("nan")

    p_pre = m1 * raw["pre_paddle_v"] + m2 * raw["pre_puck_v"]
    p_post = m1 * raw["post_paddle_v"] + m2 * raw["post_puck_v"]
    p_ratio = p_post / p_pre if abs(p_pre) > 1e-12 else float("nan")

    def pct_err(actual, expected):
        if abs(expected) < 1e-8:
            return abs(actual - expected) * 100.0
        return (actual - expected) / abs(expected) * 100.0

    return CollisionResult(
        paddle_density=paddle_density,
        puck_density=puck_density,
        mass_ratio=m1 / m2 if m2 > 0 else float("inf"),
        rel_speed=rel_speed,
        speed_label=speed_label,
        paddle_v_in=raw["pre_paddle_v"],
        puck_v_in=raw["pre_puck_v"],
        paddle_v_out=raw["post_paddle_v"],
        puck_v_out=raw["post_puck_v"],
        paddle_v_expected=paddle_v_exp,
        puck_v_expected=puck_v_exp,
        puck_error_pct=pct_err(raw["post_puck_v"], puck_v_exp),
        paddle_error_pct=pct_err(raw["post_paddle_v"], paddle_v_exp),
        ke_ratio=ke_ratio,
        p_ratio=p_ratio,
        collision_detected=raw["collision_detected"],
    )


def build_speed_cases(rel_speed: float) -> list[tuple[str, float, float]]:
    """
    For a given relative speed, return (label, paddle_vy, puck_vy) tuples.
    Paddle moves in +y, puck in -y (head-on).
    """
    return [
        ("paddle_only", rel_speed, 0.0),
        ("mostly_paddle", 0.75 * rel_speed, -0.25 * rel_speed),
        ("equal_split", 0.5 * rel_speed, -0.5 * rel_speed),
        ("mostly_puck", 0.25 * rel_speed, -0.75 * rel_speed),
        ("puck_only", 0.0, -rel_speed),
    ]


DENSITY_RATIOS = [
    ("1:1", 250.0, 250.0),
    ("5:1", 1250.0, 250.0),
    ("25:1", 6250.0, 250.0),
]

REL_SPEEDS = [0.25, 0.5, 0.75, 1.0, 1.05, 1.1, 1.5, 2.0, 4.0]


def run_sweep(use_listener_fix: bool = False) -> list[CollisionResult]:
    results = []
    for ratio_label, pd, pkd in DENSITY_RATIOS:
        for rs in REL_SPEEDS:
            for label, pv, pkv in build_speed_cases(rs):
                r = compute_result(pd, pkd, rs, label, pv, pkv,
                                   use_listener_fix=use_listener_fix)
                results.append(r)
    return results


def find_velocity_threshold() -> float:
    """Binary-search for the relative speed at which Box2D switches from inelastic to elastic."""
    lo, hi = 0.1, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        raw = run_single_collision(250.0, 250.0, mid, 0.0)
        stuck = abs(raw["post_paddle_v"] - raw["post_puck_v"]) < 1e-4
        if stuck:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fmt(val: float, width: int = 8, prec: int = 4) -> str:
    s = f"{val:>{width}.{prec}f}"
    return s


def print_tables(results: list[CollisionResult]) -> None:
    header = (
        f"{'rel_spd':>8} {'distribution':>14} "
        f"{'pdl_v_in':>9} {'pck_v_in':>9} "
        f"{'pdl_v_out':>10} {'pck_v_out':>10} "
        f"{'pck_v_exp':>10} {'pck_err%':>9} "
        f"{'pdl_v_exp':>10} {'pdl_err%':>9} "
        f"{'KE_ratio':>9} {'p_ratio':>8}"
    )
    sep = "-" * len(header)

    grouped: dict[str, list[CollisionResult]] = {}
    for r in results:
        key = f"{r.paddle_density:.0f}:{r.puck_density:.0f}"
        grouped.setdefault(key, []).append(r)

    for density_key, group in grouped.items():
        ratio = group[0].mass_ratio
        print(f"\n{'='*len(header)}")
        print(f"  Density paddle:puck = {density_key}   |   Mass ratio = {ratio:.2f}:1")
        print(f"{'='*len(header)}")
        print(header)
        print(sep)

        prev_rs = None
        for r in group:
            if prev_rs is not None and r.rel_speed != prev_rs:
                print(sep)
            prev_rs = r.rel_speed

            collision_marker = "" if r.collision_detected else " [NO HIT]"
            print(
                f"{r.rel_speed:>8.2f} {r.speed_label:>14} "
                f"{r.paddle_v_in:>9.4f} {r.puck_v_in:>9.4f} "
                f"{r.paddle_v_out:>10.4f} {r.puck_v_out:>10.4f} "
                f"{r.puck_v_expected:>10.4f} {r.puck_error_pct:>8.2f}% "
                f"{r.paddle_v_expected:>10.4f} {r.paddle_error_pct:>8.2f}% "
                f"{r.ke_ratio:>9.4f} {r.p_ratio:>8.4f}"
                f"{collision_marker}"
            )


def print_summary(results: list[CollisionResult]) -> None:
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)

    grouped: dict[float, list[CollisionResult]] = {}
    for r in results:
        grouped.setdefault(r.mass_ratio, []).append(r)

    print(f"\n{'mass_ratio':>11} {'mean|pck_err%|':>15} {'max|pck_err%|':>14} "
          f"{'mean|pdl_err%|':>15} {'max|pdl_err%|':>14} "
          f"{'mean_KE':>8} {'mean_p':>8}")
    print("-" * 90)
    for mr in sorted(grouped.keys()):
        grp = grouped[mr]
        pck_errs = [abs(r.puck_error_pct) for r in grp if r.collision_detected]
        pdl_errs = [abs(r.paddle_error_pct) for r in grp if r.collision_detected]
        ke_vals = [r.ke_ratio for r in grp if r.collision_detected and not math.isnan(r.ke_ratio)]
        p_vals = [r.p_ratio for r in grp if r.collision_detected and not math.isnan(r.p_ratio)]
        print(
            f"{mr:>10.1f}:1 "
            f"{np.mean(pck_errs):>14.4f}% {np.max(pck_errs):>13.4f}% "
            f"{np.mean(pdl_errs):>14.4f}% {np.max(pdl_errs):>13.4f}% "
            f"{np.mean(ke_vals):>8.4f} {np.mean(p_vals):>8.4f}"
        )


def print_analysis(results: list[CollisionResult]) -> None:
    print("\n" + "=" * 90)
    print("  ANALYSIS")
    print("=" * 90)

    valid = [r for r in results if r.collision_detected]
    if not valid:
        print("  No valid collisions detected.")
        return

    puck_errors = [r.puck_error_pct for r in valid]
    positive = sum(1 for e in puck_errors if e > 0.01)
    negative = sum(1 for e in puck_errors if e < -0.01)
    near_zero = len(puck_errors) - positive - negative
    print(f"\n  1. Systematic bias in puck post-collision speed:")
    print(f"     Overshooting (actual > expected): {positive}/{len(puck_errors)}")
    print(f"     Undershooting (actual < expected): {negative}/{len(puck_errors)}")
    print(f"     Near-zero error (<0.01%):          {near_zero}/{len(puck_errors)}")

    print(f"\n  2. Error vs relative speed (the critical relationship):")
    by_speed: dict[float, list[float]] = {}
    by_speed_ke: dict[float, list[float]] = {}
    for r in valid:
        by_speed.setdefault(r.rel_speed, []).append(abs(r.puck_error_pct))
        if not math.isnan(r.ke_ratio):
            by_speed_ke.setdefault(r.rel_speed, []).append(r.ke_ratio)
    for rs in sorted(by_speed.keys()):
        errs = by_speed[rs]
        ke = by_speed_ke.get(rs, [])
        ke_str = f"mean_KE={np.mean(ke):.4f}" if ke else "no KE data"
        print(f"     rel_speed={rs:>5.2f}: mean|err|={np.mean(errs):>8.4f}%, "
              f"max={np.max(errs):>8.4f}%,  {ke_str}")

    # Identify the threshold
    sorted_speeds = sorted(by_speed.keys())
    below_threshold = [rs for rs in sorted_speeds if np.mean(by_speed[rs]) > 1.0]
    above_threshold = [rs for rs in sorted_speeds if np.mean(by_speed[rs]) < 0.01]
    if below_threshold and above_threshold:
        transition_lo = max(below_threshold)
        transition_hi = min(above_threshold)
        print(f"\n     >>> VELOCITY THRESHOLD DETECTED <<<")
        print(f"     All relative speeds <= {transition_lo:.2f} m/s: INELASTIC (e=0, bodies stick)")
        print(f"     All relative speeds >= {transition_hi:.2f} m/s: ELASTIC (e=1, correct physics)")
        print(f"     This is Box2D's b2_velocityThreshold (compiled C++ constant = 1.0 m/s)")
        print(f"     Setting Box2D.b2_velocityThreshold in Python does NOT change the C++ value.")

    print(f"\n  3. Error vs speed distribution (who is moving):")
    by_label: dict[str, list[float]] = {}
    for r in valid:
        by_label.setdefault(r.speed_label, []).append(abs(r.puck_error_pct))
    for lbl in ["paddle_only", "mostly_paddle", "equal_split", "mostly_puck", "puck_only"]:
        if lbl in by_label:
            errs = by_label[lbl]
            print(f"     {lbl:>14}: mean|err|={np.mean(errs):.4f}%, max={np.max(errs):.4f}%")

    print(f"\n  4. Error vs density ratio:")
    by_ratio: dict[float, list[float]] = {}
    for r in valid:
        by_ratio.setdefault(r.mass_ratio, []).append(abs(r.puck_error_pct))
    for mr in sorted(by_ratio.keys()):
        errs = by_ratio[mr]
        print(f"     ratio={mr:.1f}:1: mean|err|={np.mean(errs):.4f}%, max={np.max(errs):.4f}%")

    ke_vals = [r.ke_ratio for r in valid if not math.isnan(r.ke_ratio)]
    p_vals = [r.p_ratio for r in valid if not math.isnan(r.p_ratio)]
    print(f"\n  5. Conservation laws:")
    print(f"     Kinetic energy ratio (should be 1.0 for elastic):")
    print(f"       mean={np.mean(ke_vals):.6f}, std={np.std(ke_vals):.6f}, "
          f"min={np.min(ke_vals):.6f}, max={np.max(ke_vals):.6f}")
    print(f"     Momentum ratio (should be 1.0 always):")
    print(f"       mean={np.mean(p_vals):.6f}, std={np.std(p_vals):.6f}, "
          f"min={np.min(p_vals):.6f}, max={np.max(p_vals):.6f}")

    print(f"\n  6. Effective coefficient of restitution (from actual speeds):")
    for mr in sorted(by_ratio.keys()):
        cors_below = []
        cors_above = []
        for r in valid:
            if abs(r.mass_ratio - mr) < 0.01:
                v_rel_in = r.paddle_v_in - r.puck_v_in
                v_rel_out = r.puck_v_out - r.paddle_v_out
                if abs(v_rel_in) > 1e-8:
                    e_eff = v_rel_out / v_rel_in
                    if r.rel_speed <= 1.0:
                        cors_below.append(e_eff)
                    else:
                        cors_above.append(e_eff)
        if cors_below:
            print(f"     ratio={mr:.1f}:1 (rel_speed<=1.0): mean_e={np.mean(cors_below):.6f}, "
                  f"range=[{np.min(cors_below):.4f}, {np.max(cors_below):.4f}]")
        if cors_above:
            print(f"     ratio={mr:.1f}:1 (rel_speed> 1.0): mean_e={np.mean(cors_above):.6f}, "
                  f"range=[{np.min(cors_above):.4f}, {np.max(cors_above):.4f}]")

    print(f"\n  7. Impact on air hockey gameplay:")
    print(f"     Default max_paddle_vel = 2.0 m/s. Typical relative approach speeds")
    print(f"     at contact are often 0.3-1.5 m/s, meaning many paddle-puck collisions")
    print(f"     fall below the 1.0 m/s threshold and become perfectly inelastic.")
    print(f"     This causes the puck to 'stick' to the paddle instead of bouncing off,")
    print(f"     producing unrealistic low-energy post-contact puck speeds.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Box2D collision physics diagnostic")
    parser.add_argument("--json-out", type=str, default="",
                        help="Save all results as JSON to this path.")
    parser.add_argument("--skip-threshold-search", action="store_true",
                        help="Skip the binary-search for the velocity threshold.")
    parser.add_argument("--use-listener-fix", action="store_true",
                        help="Use CollisionForceListener with paddle-puck restitution fix.")
    args = parser.parse_args()

    print("Running Box2D collision diagnostic sweep...")
    print(f"  Paddle radius: {PADDLE_RADIUS}, Puck radius: {PUCK_RADIUS}")
    print(f"  Restitution: {RESTITUTION}, Friction: {FRICTION}")
    print(f"  Density ratios: {[d[0] for d in DENSITY_RATIOS]}")
    print(f"  Relative speeds: {REL_SPEEDS}")
    print(f"  Speed distributions: 5 per relative speed")
    print(f"  Total cases: {len(DENSITY_RATIOS) * len(REL_SPEEDS) * 5}")

    if not args.skip_threshold_search:
        print("\nSearching for Box2D velocity threshold (binary search)...")
        threshold = find_velocity_threshold()
        print(f"  Detected threshold: {threshold:.6f} m/s")
        print(f"  (Box2D ignores restitution when relative approach speed <= this value)")
        import Box2D
        print(f"  Box2D.b2_velocityThreshold Python attribute: {Box2D.b2_velocityThreshold}")
        print(f"  NOTE: setting this attribute in Python does NOT change the C++ constant.")

    if args.use_listener_fix:
        print("\n  >>> Using CollisionForceListener with paddle-puck restitution fix <<<")

    results = run_sweep(use_listener_fix=args.use_listener_fix)

    print_tables(results)
    print_summary(results)
    print_analysis(results)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nJSON results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
