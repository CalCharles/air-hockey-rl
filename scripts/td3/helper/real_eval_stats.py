"""Distribution stats + serialization for fixed-policy real-world evaluations.

Used by ``extras/async_td3_real_eval.py`` to summarize a batch of N
*kept* (validator-passed) policy episodes collected against a frozen
checkpoint. Kept deliberately env-agnostic: the eval entrypoint hands
this module a list of per-episode dicts and gets back a JSON-friendly
aggregate + a summary writer. Future in-training-loop evaluation can
reuse the same helpers without depending on the entrypoint.

Numeric series tracked over the eval batch:

* ``episode_return``       — total return.
* ``episode_reward``       — sum of per-step rewards (== episode_return for
                              single-objective TD3).
* ``episode_length``       — policy steps in the kept episode.
* ``episode_juggles``      — paddle-puck juggles (see helper/juggle_counter.py).
* ``episode_contacts``     — paddle-puck contacts.

Each series gets ``count / mean / std / min / max / median / p25 / p75``.
Boolean / event-count fields (``had_protective_stop``,
``had_controller_disconnect``, ``readiness_fail_estop``,
``episode_juggle_success``, ``episode_success``) are summarized as
``count / rate`` so e-stop / juggle-success frequency is directly
readable. ``estop_total`` collapses the three e-stop event flags into
one count so the answer to "how many e-stops?" is a single number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------------------------
# Schema — one place to list which fields get summarized as what.
# ---------------------------------------------------------------------------


# Numeric per-episode fields → summarized with full distribution stats.
# Order chosen so the printed/JSON summary reads "what the user asked for"
# first (return, juggles, contacts), then secondary metrics.
NUMERIC_SERIES_FIELDS: tuple[str, ...] = (
    "episode_return",
    "episode_juggles",
    "episode_contacts",
    "episode_reward",
    "episode_length",
)

# Boolean / 0-or-1 per-episode fields → summarized as count + rate.
RATE_FIELDS: tuple[str, ...] = (
    "episode_juggle_success",
    "episode_success",
    "had_protective_stop",
    "had_controller_disconnect",
    "readiness_fail_estop",
)


# ---------------------------------------------------------------------------
# Stats primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesSummary:
    """Distribution summary for one numeric series over the eval batch."""

    count: int
    mean: float
    std: float
    min: float
    max: float
    median: float
    p25: float
    p75: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min),
            "max": float(self.max),
            "median": float(self.median),
            "p25": float(self.p25),
            "p75": float(self.p75),
        }

    @classmethod
    def empty(cls) -> "SeriesSummary":
        return cls(count=0, mean=0.0, std=0.0, min=0.0, max=0.0, median=0.0, p25=0.0, p75=0.0)


def summarize_series(values: Sequence[float]) -> SeriesSummary:
    """Eight-number summary. Empty input → all-zero ``SeriesSummary``."""
    if len(values) == 0:
        return SeriesSummary.empty()
    # Local import so the helper has no module-level numpy dependency for
    # callers that only need the dataclass shape.
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return SeriesSummary(
        count=int(arr.shape[0]),
        mean=float(np.mean(arr)),
        std=float(np.std(arr)),
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        median=float(np.median(arr)),
        p25=float(np.percentile(arr, 25.0)),
        p75=float(np.percentile(arr, 75.0)),
    )


# ---------------------------------------------------------------------------
# Aggregate over a batch of per-episode records
# ---------------------------------------------------------------------------


def _extract_floats(records: Sequence[Mapping[str, Any]], field: str) -> list[float]:
    out: list[float] = []
    for rec in records:
        if field in rec and rec[field] is not None:
            out.append(float(rec[field]))
    return out


def _extract_bools(records: Sequence[Mapping[str, Any]], field: str) -> list[int]:
    out: list[int] = []
    for rec in records:
        if field in rec and rec[field] is not None:
            out.append(1 if bool(rec[field]) else 0)
    return out


def compute_eval_aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the aggregate stats dict for a batch of *kept* eval episodes.

    Each record is the per-episode dict produced by the eval entrypoint
    (one per kept episode). Returns a JSON-friendly dict with:

      ``n_episodes``           — kept count actually evaluated.
      ``series.<field>``       — full ``SeriesSummary`` for every entry of
                                 ``NUMERIC_SERIES_FIELDS`` (count / mean /
                                 std / min / max / median / p25 / p75).
      ``rates.<field>``        — ``{count, total, rate}`` for every entry
                                 of ``RATE_FIELDS``.
      ``estop_total``          — number of episodes flagged with *any*
                                 e-stop class (protective / controller-
                                 disconnect / readiness-fail). Collapses
                                 the three rate fields so the common
                                 question "how many e-stops?" has one
                                 answer.
    """
    n = len(records)
    series: dict[str, dict[str, float | int]] = {}
    for field in NUMERIC_SERIES_FIELDS:
        series[field] = summarize_series(_extract_floats(records, field)).to_dict()

    rates: dict[str, dict[str, float | int]] = {}
    for field in RATE_FIELDS:
        flags = _extract_bools(records, field)
        total = int(sum(flags))
        rates[field] = {
            "count": int(len(flags)),
            "total": total,
            "rate": float(total) / float(len(flags)) if flags else 0.0,
        }

    # Collapsed e-stop counter — one row per episode is "estop_total"-flagged
    # iff any of the three classes triggered. We OR the bools per episode so
    # an episode that hit two classes is still counted once.
    estop_total = 0
    for rec in records:
        if (
            bool(rec.get("had_protective_stop", False))
            or bool(rec.get("had_controller_disconnect", False))
            or bool(rec.get("readiness_fail_estop", False))
        ):
            estop_total += 1

    return {
        "n_episodes": int(n),
        "series": series,
        "rates": rates,
        "estop_total": int(estop_total),
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def write_eval_summary_json(
    path: Path,
    *,
    run_meta: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    per_episode: Sequence[Mapping[str, Any]],
) -> None:
    """Write a single self-contained ``eval_summary.json``.

    Layout::

        {
          "run_meta":  {model_path, run_data_dir, n_target, n_attempts,
                        n_discarded, started_iso, finished_iso, ...},
          "aggregate": {n_episodes, series, rates, estop_total},
          "per_episode": [ {episode_id, episode_return, ...}, ... ]
        }

    A consumer that wants raw individual statistics for further analysis
    only needs ``per_episode``; ``aggregate`` is the
    print-ready summary; ``run_meta`` lets a reader trace which
    checkpoint / config / run produced these numbers.
    """
    payload = {
        "run_meta": dict(run_meta),
        "aggregate": dict(aggregate),
        "per_episode": [dict(r) for r in per_episode],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Console formatter
# ---------------------------------------------------------------------------


def format_eval_summary_console(
    aggregate: Mapping[str, Any],
    *,
    n_target: int,
    n_attempts: int,
    n_discarded: int,
) -> str:
    """Multi-line human-readable digest for the end-of-run print.

    Format mirrors the rolling-window console line so eyes already trained
    on training output find what they expect (avg / std / [min..max] /
    median).
    """
    lines: list[str] = []
    n_kept = int(aggregate.get("n_episodes", 0))
    estop_total = int(aggregate.get("estop_total", 0))
    lines.append(
        "[eval_summary] "
        f"kept={n_kept}/{n_target} attempts={n_attempts} discarded={n_discarded} "
        f"estop_total={estop_total}"
    )

    series: Mapping[str, Mapping[str, float]] = aggregate.get("series", {})
    for field in NUMERIC_SERIES_FIELDS:
        s = series.get(field)
        if not s:
            continue
        # Choose precision per field: integer-y counts get .2f, returns / rewards
        # / lengths get .3f. Falls back to .3f if unknown.
        if field in ("episode_juggles", "episode_contacts"):
            avg_p, lim_p, med_p, std_p = ".2f", ".0f", ".1f", ".2f"
        elif field == "episode_length":
            avg_p, lim_p, med_p, std_p = ".1f", ".0f", ".1f", ".1f"
        else:
            avg_p, lim_p, med_p, std_p = ".3f", ".3f", ".3f", ".3f"
        lines.append(
            f"[eval_summary] {field:>22s}: "
            f"mean={s['mean']:{avg_p}} std={s['std']:{std_p}} "
            f"[{s['min']:{lim_p}}..{s['max']:{lim_p}}] "
            f"median={s['median']:{med_p}} "
            f"p25={s['p25']:{med_p}} p75={s['p75']:{med_p}}"
        )

    rates: Mapping[str, Mapping[str, float]] = aggregate.get("rates", {})
    for field in RATE_FIELDS:
        r = rates.get(field)
        if not r:
            continue
        lines.append(
            f"[eval_summary] {field:>22s}: "
            f"total={int(r['total'])}/{int(r['count'])} rate={float(r['rate']):.3f}"
        )
    return "\n".join(lines)
