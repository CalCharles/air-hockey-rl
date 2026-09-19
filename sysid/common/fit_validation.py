"""Did the parameter search find anything meaningful?

Given the validation error of *every* candidate the search evaluated (the (g, γ)
grid for the puck, the restitution sweep for a wall), the value the search
selected on the training data is compared with the distribution of validation
errors over the whole candidate set:

* ``selected_beats`` — fraction of candidates whose validation error is worse
  than the selected one (1.0 = the selected point is the validation oracle);
* ``oracle`` — the best validation error any candidate reaches (what the
  search could have got with the answer key);
* the candidates at the 50th / 75th / 90th percentile of the ranking (a
  candidate better than 50 / 75 / 90 % of the others) and how much of the
  oracle each reaches, ``fraction_of_best = oracle_err / err``.

If a candidate that merely beats 90 % of the grid already reaches ~99 % of the
oracle, the landscape is flat and the search is not resolving the parameter;
if the selected point beats > 95 % of the grid while the p90 point sits well
below it, the search did something the data can validate. Lower error is
better throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class PercentileReport:
    name: str                       # e.g. "puck fit rms (cm)" or "side walls speed rel err"
    unit: str
    n_candidates: int
    selected_err: float
    selected_beats: float           # fraction of candidates with worse val error
    selected_fraction_of_best: float
    oracle_err: float
    oracle_params: dict
    worst_err: float
    median_err: float
    percentiles: dict = field(default_factory=dict)   # q -> {"err", "fraction_of_best", "gain_vs_median", "params"}
    canonical_err: Optional[float] = None
    canonical_beats: Optional[float] = None
    canonical_fraction_of_best: Optional[float] = None
    selected_gain_vs_median: Optional[float] = None
    canonical_gain_vs_median: Optional[float] = None
    val_errors: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)     # every candidate, finite only
    train_errors: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)   # same candidates, train error (may be empty)
    selected_train_err: float = float("nan")
    canonical_train_err: float = float("nan")

    def to_json(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k not in ("val_errors", "train_errors")}


def _gain_vs_median(err: float, median: float, oracle: float) -> float:
    """Fraction of the median → oracle improvement realised by ``err``
    (1 = oracle, 0 = median, negative = worse than the median candidate)."""
    span = median - oracle
    return float((median - err) / span) if span > 1e-12 else float("nan")


def percentile_report(name: str, unit: str, val_errors: np.ndarray, selected_err: float,
                      params_of: Callable[[int], dict], canonical_err: Optional[float] = None,
                      percentiles=(50, 75, 90), train_errors: Optional[np.ndarray] = None,
                      selected_train_err: float = float("nan"), canonical_train_err: float = float("nan")) -> PercentileReport:
    """``val_errors``: flat validation error of every candidate (NaN allowed,
    ignored). ``selected_err`` / ``canonical_err``: validation error of the
    selected / canonical parameters evaluated exactly (they need not lie on
    the candidate set). ``params_of(flat_index)`` names a candidate.
    ``train_errors`` (same shape as ``val_errors``) is kept for the plots only."""
    errs = np.asarray(val_errors, dtype=float).ravel()
    finite = np.flatnonzero(np.isfinite(errs))
    e = errs[finite]
    n = len(e)
    tr = np.asarray(train_errors, dtype=float).ravel()[finite] if train_errors is not None else np.zeros(0)
    if n == 0:
        nan = float("nan")
        return PercentileReport(name, unit, 0, selected_err, nan, nan, nan, {}, nan, nan)
    oracle_i = int(finite[np.argmin(e)])
    oracle = float(e.min())
    worst = float(e.max())
    median = float(np.median(e))
    beats = lambda x: float(np.mean(e > x))
    frac = lambda x: float(oracle / x) if x > 0 else float("nan")
    pcts = {}
    order = np.argsort(e)                      # best first
    for q in percentiles:
        # candidate better than q % of the candidates: rank (1-q) from the best
        k = int(min(n - 1, max(0, round((1 - q / 100) * (n - 1)))))
        idx = int(finite[order[k]])
        err_q = float(e[order[k]])
        pcts[int(q)] = {"err": err_q, "fraction_of_best": frac(err_q),
                        "gain_vs_median": _gain_vs_median(err_q, median, oracle), "params": params_of(idx)}
    rep = PercentileReport(name, unit, n, float(selected_err), beats(selected_err), frac(selected_err),
                           oracle, params_of(oracle_i), worst, median, pcts,
                           selected_gain_vs_median=_gain_vs_median(selected_err, median, oracle), val_errors=e,
                           train_errors=tr, selected_train_err=float(selected_train_err), canonical_train_err=float(canonical_train_err))
    if canonical_err is not None and np.isfinite(canonical_err):
        rep.canonical_err = float(canonical_err)
        rep.canonical_beats = beats(canonical_err)
        rep.canonical_fraction_of_best = frac(canonical_err)
        rep.canonical_gain_vs_median = _gain_vs_median(canonical_err, median, oracle)
    return rep


def summarize_reports(reports: list[PercentileReport]) -> list[str]:
    """Markdown table rows: one line per report."""
    lines = ["| validation metric | selected | beats grid | of oracle | canonical | beats grid | of oracle | oracle | p50 (of oracle) | p75 (of oracle) | p90 (of oracle) | worst | n |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        f = (lambda x: f"{x:.3f}") if r.unit != "cm" else (lambda x: f"{x:.2f}")
        pc = lambda q: f"{f(r.percentiles[q]['err'])} ({100 * r.percentiles[q]['fraction_of_best']:.0f} %)" if q in r.percentiles else "–"
        can = (f"{f(r.canonical_err)} | {100 * r.canonical_beats:.0f} % | {100 * r.canonical_fraction_of_best:.0f} %"
               if r.canonical_err is not None else "– | – | –")
        lines.append(f"| {r.name} [{r.unit}] | **{f(r.selected_err)}** | {100 * r.selected_beats:.0f} % | {100 * r.selected_fraction_of_best:.0f} % | {can} | "
                     f"{f(r.oracle_err)} ({_fmt_params(r.oracle_params)}) | {pc(50)} | {pc(75)} | {pc(90)} | {f(r.worst_err)} | {r.n_candidates} |")
    return lines


def _fmt_params(p: dict) -> str:
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())


def plot_percentiles(reports: list[PercentileReport], out_path, ncols: int = 4):
    """One panel per report: every candidate's error **relative to the best
    candidate of that split** (``err / min err``, 1 = best), sorted, as a
    function of the candidate's percentile rank (100 = best candidate), for
    validation and — when available — training separately, each ranked and
    normalised on its own split. The 50th / 75th / 90th percentiles are
    marked, together with the selected and canonical parameters placed at
    their validation rank."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    reports = [r for r in reports if r.n_candidates > 0]
    if not reports:
        return
    ncols = min(ncols, len(reports))
    nrows = int(np.ceil(len(reports) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows) if len(reports) > 1 else (6.0, 4.4), squeeze=False)
    for ax in axes.ravel()[len(reports):]:
        ax.axis("off")
    for ax, r in zip(axes.ravel(), reports):
        for errs, color, label in ((r.val_errors, "tab:red", "validation"), (r.train_errors, "tab:blue", "training")):
            if len(errs) == 0 or not np.isfinite(errs).any():
                continue
            e = np.sort(errs[np.isfinite(errs)])[::-1]            # worst … best
            best = e[-1]
            pct = 100 * np.arange(len(e)) / max(1, len(e) - 1)      # 0 = worst, 100 = best
            ax.plot(pct, e / best, "-", color=color, lw=1.4, label=f"{label} ({len(e)} candidates, best {best:.3g} {r.unit})")
            for q, mk in zip(sorted(r.percentiles), ("^", "s", "D")):
                k = int(round(q / 100 * (len(e) - 1)))
                ax.plot(pct[k], e[k] / best, mk, color=color, ms=6, mfc="white", mew=1.3,
                        label=f"{label} p{q}: {100 * best / e[k]:.0f} % of best" if label == "validation" else None)
        vb = r.oracle_err
        ax.plot(100 * r.selected_beats, r.selected_err / vb, "*", color="tab:red", ms=13, mec="k",
                label=f"selected @ val rank {100 * r.selected_beats:.0f} % ({100 * r.selected_fraction_of_best:.0f} % of best)")
        if r.canonical_err is not None:
            ax.plot(100 * r.canonical_beats, r.canonical_err / vb, "o", color="gray", ms=8, mec="k",
                    label=f"canonical @ val rank {100 * r.canonical_beats:.0f} % ({100 * r.canonical_fraction_of_best:.0f} % of best)")
        for q in sorted(r.percentiles):
            ax.axvline(q, color="k", lw=0.6, ls=":", alpha=0.6)
        ax.axhline(1.0, color="k", lw=0.6, ls=":", alpha=0.6)
        ax.set_xlabel("candidate percentile (100 = best of the split)"); ax.set_ylabel("error / best error of the split")
        ax.set_title(r.name, fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=6.2, loc="upper right")
        ax.set_xlim(0, 100)
        tops = [np.nanmax(r.val_errors / vb), r.selected_err / vb, (r.canonical_err / vb) if r.canonical_err else 1.0]
        if len(r.train_errors) and np.isfinite(r.train_errors).any():
            tops.append(np.nanmax(r.train_errors) / np.nanmin(r.train_errors))
        ax.set_ylim(0.95, min(np.nanmax(tops) * 1.05, 4.0))
    fig.tight_layout(); fig.savefig(out_path, dpi=150 if len(reports) == 1 else 110); plt.close(fig)


PERCENTILE_LEGEND = ("`beats grid` = share of searched candidates with a worse validation error than that parameter set; "
                     "`of oracle` = best validation error over the candidates / this error; p50 / p75 / p90 = the candidate better than "
                     "50 / 75 / 90 % of the candidates and the share of the oracle it reaches. If p90 already reaches ≈ 100 % of the oracle the "
                     "validation data cannot tell the candidates apart; an oracle at the range edge means a monotone curve (parameter not identified). "
                     "`validation_percentiles.png`: every candidate's error relative to the best of its split vs percentile rank, train and val separately.")
