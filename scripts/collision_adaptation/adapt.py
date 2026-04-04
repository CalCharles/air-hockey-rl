"""
Scale update rule for collision adaptation.

Given oracle and learner per-tier paddle collision stats, computes new paddle
restitution scales so the learner's outgoing puck speed per tier converges
toward the oracle's.

Update rule (per tier t):
    ratio_t  = oracle_mean_out_t / learner_mean_out_t
    scale_t' = scale_t * (1 + lr * (ratio_t - 1))   # multiplicative
    scale_t' = clamp(scale_t', min_scale, max_scale)

Tiers with fewer than min_count collisions in either sim are skipped.
"""

from __future__ import annotations

_TIERS = ("low", "mid", "high")


def compute_scale_updates(
    oracle_stats: dict,
    learner_stats: dict,
    current_scales: list[float],
    lr: float = 0.2,
    min_count: int = 3,
    min_scale: float = 0.3,
    max_scale: float = 3.0,
) -> tuple[list[float], dict]:
    """
    Compute updated paddle restitution scales.

    Parameters
    ----------
    oracle_stats    : {"paddle": {"low": {"count", "mean_speed_in", "mean_speed_out"}, ...}}
    learner_stats   : same structure
    current_scales  : [low_scale, mid_scale, high_scale] (current learner paddle scales)
    lr              : learning rate for multiplicative update
    min_count       : minimum collision count in BOTH sims to update a tier
    min_scale       : lower clamp for any scale
    max_scale       : upper clamp for any scale

    Returns
    -------
    new_scales : list[float], length 3
    update_info : dict with per-tier debug info
    """
    new_scales = list(current_scales)
    update_info: dict = {}

    for i, tier in enumerate(_TIERS):
        o_bucket = oracle_stats["paddle"][tier]
        l_bucket = learner_stats["paddle"][tier]
        o_count = int(o_bucket.get("count", 0))
        l_count = int(l_bucket.get("count", 0))

        if o_count < min_count or l_count < min_count:
            update_info[tier] = {
                "skipped": True,
                "reason": f"oracle_count={o_count} learner_count={l_count} < min_count={min_count}",
                "scale_before": current_scales[i],
                "scale_after": current_scales[i],
            }
            continue

        o_out = float(o_bucket.get("mean_speed_out", 0.0))
        l_out = float(l_bucket.get("mean_speed_out", 0.0))

        if l_out < 1e-8:
            update_info[tier] = {
                "skipped": True,
                "reason": f"learner mean_speed_out ≈ 0 (l_out={l_out:.6f})",
                "scale_before": current_scales[i],
                "scale_after": current_scales[i],
            }
            continue

        ratio = o_out / l_out
        raw_new = current_scales[i] * (1.0 + lr * (ratio - 1.0))
        clamped_new = float(max(min_scale, min(max_scale, raw_new)))
        new_scales[i] = clamped_new

        update_info[tier] = {
            "skipped": False,
            "oracle_count": o_count,
            "learner_count": l_count,
            "oracle_mean_out": o_out,
            "learner_mean_out": l_out,
            "ratio": ratio,
            "scale_before": current_scales[i],
            "scale_after": clamped_new,
            "clamped": clamped_new != raw_new,
        }

    return new_scales, update_info


def max_abs_ratio_minus_one(update_info: dict) -> float:
    """
    Convergence metric: max(|ratio_t - 1|) across non-skipped tiers.
    Returns 0.0 if all tiers were skipped.
    """
    values = []
    for info in update_info.values():
        if not info.get("skipped", True):
            values.append(abs(info["ratio"] - 1.0))
    return max(values) if values else 0.0
