"""Puck free-flight sysid: grid search over (gravity_x, damping) on
fixed-length free-fall windows.

Same procedure as ``sysid/puck_grid_search.py``: for each candidate (g, γ)
the damped model is linear in (p0, v0), so every window is fitted by a small
least-squares solve on all its samples (the *full fit*). Two scores, each the
mean over windows:

* ``fit_rms_cm`` — rms position residual of the in-window fit (the original
  grid-search criterion and the selection objective);
* ``fit_rel``    — **final displacement error**: distance between the model
  and the measured puck at the last sample of the window, divided by the
  distance the puck actually travelled over the window (path length of the
  measured trajectory). End-of-horizon drift as a fraction of the motion, so
  fast and slow windows weigh the same.

The whole coarse grid is scored on **both** train and val windows with both
metrics (``PuckFitResult.grids``); the parameter is selected on the train
``objective`` (fine grid refinement around the coarse optimum), and the
validation landscape is then used by ``fit_validation.percentile_report`` to
check that the selection is better than a random good grid point.

Physics sign convention: the model is ``a = g − γ v`` in the sim frame
(``gravity_x`` < 0). ``sysid/puck_grid_search.py`` uses the opposite sign in the
raw base frame; the two agree on |g|.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sysid.common.fit_validation import PercentileReport, percentile_report, plot_percentiles
from sysid.common.sysid_dataset import FreeFallWindow
from sysid.common.trajectory_segmentation import SegmentationConfig, fit_damped, model_state

METRICS = ("fit_rms_cm", "fit_rel")
METRIC_LABEL = {"fit_rms_cm": "fit rms", "fit_rel": "final displacement err / distance"}
METRIC_UNIT = {"fit_rms_cm": "cm", "fit_rel": "fraction"}


def _path_length(xy: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def score_windows(windows: list[FreeFallWindow], g: float, gamma: float) -> dict:
    """Mean scores over windows for one (g, γ): ``{"fit_rms_cm", "fit_rel"}``."""
    cfg = SegmentationConfig(gravity_x=g, damping=gamma)
    fit_r, fit_rel = [], []
    for w in windows:
        f = fit_damped(w.t, w.xy, cfg)
        fit_r.append(f["rms"])
        final_err = float(np.linalg.norm(model_state(f, w.t[-1], cfg)[0] - w.xy[-1]))
        fit_rel.append(final_err / max(1e-6, _path_length(w.xy)))
    nan = float("nan")
    return {"fit_rms_cm": 100 * float(np.mean(fit_r)) if fit_r else nan,
            "fit_rel": float(np.mean(fit_rel)) if fit_rel else nan}


@dataclass
class PuckFitResult:
    gravity_x: float
    damping: float
    objective: str                      # metric minimised on train
    grid_g: np.ndarray
    grid_gamma: np.ndarray
    grids: dict                         # metric -> {"train": (Ng, Nγ) array, "val": ...}
    train: dict                         # metric -> score at the fitted params
    val: dict
    n_train: int
    n_val: int
    canonical: dict = field(default_factory=dict)      # {"gravity", "damping", "train": {...}, "val": {...}}
    validation: dict = field(default_factory=dict)     # metric -> PercentileReport (val landscape)

    def to_json(self) -> dict:
        return {"gravity_x": self.gravity_x, "damping": self.damping, "objective": self.objective,
                "grid_g": self.grid_g.tolist(), "grid_gamma": self.grid_gamma.tolist(),
                "grids": {m: {s: a.tolist() for s, a in d.items()} for m, d in self.grids.items()},
                "train": self.train, "val": self.val, "n_train": self.n_train, "n_val": self.n_val,
                "canonical": self.canonical, "validation": {m: r.to_json() for m, r in self.validation.items()}}


def grid_search(train: list[FreeFallWindow], val: list[FreeFallWindow],
                g_range=(-1.0, -0.4), gamma_range=(0.0, 0.4), coarse_step=(0.02, 0.02),
                fine_step=(0.005, 0.005), fine_halfwidth=(0.06, 0.06), objective: str = "fit_rms_cm",
                canonical: tuple[float, float] | None = None, verbose: bool = True) -> PuckFitResult:
    """Score the coarse grid on train and val (all metrics), pick the train
    minimum of ``objective``, refine it on a fine grid, and build the
    validation percentile reports against the coarse-grid landscape."""
    assert objective in METRICS, objective
    gs = np.arange(g_range[0], g_range[1] + 1e-9, coarse_step[0])
    gams = np.arange(gamma_range[0], gamma_range[1] + 1e-9, coarse_step[1])
    grids = {m: {"train": np.full((len(gs), len(gams)), np.nan), "val": np.full((len(gs), len(gams)), np.nan)} for m in METRICS}
    for i, g in enumerate(gs):
        for j, gm in enumerate(gams):
            s = score_windows(train, float(g), float(gm))
            for m in METRICS:
                grids[m]["train"][i, j] = s[m]
            if val:
                s = score_windows(val, float(g), float(gm))
                for m in METRICS:
                    grids[m]["val"][i, j] = s[m]
        if verbose and (i % 5 == 0 or i == len(gs) - 1):
            print(f"  grid row {i + 1}/{len(gs)} (g={g:+.3f})", flush=True)
    R = grids[objective]["train"]
    i, j = np.unravel_index(int(np.nanargmin(R)), R.shape)
    best_g, best_gm, best_r = float(gs[i]), float(gams[j]), float(R[i, j])
    for g in np.arange(best_g - fine_halfwidth[0], best_g + fine_halfwidth[0] + 1e-9, fine_step[0]):
        for gm in np.arange(max(0.0, best_gm - fine_halfwidth[1]), best_gm + fine_halfwidth[1] + 1e-9, fine_step[1]):
            r = score_windows(train, float(g), float(gm))[objective]
            if r < best_r:
                best_g, best_gm, best_r = float(g), float(gm), r
    tr = score_windows(train, best_g, best_gm)
    va = score_windows(val, best_g, best_gm) if val else {m: float("nan") for m in METRICS}
    res = PuckFitResult(best_g, best_gm, objective, gs, gams, grids, tr, va, len(train), len(val))
    if canonical is not None:
        cg, cgm = canonical
        res.canonical = {"gravity": float(cg), "damping": float(cgm), "train": score_windows(train, cg, cgm),
                         "val": score_windows(val, cg, cgm) if val else {m: float("nan") for m in METRICS}}
    if val:
        params_of = lambda k: {"g": float(gs[k // len(gams)]), "gamma": float(gams[k % len(gams)])}
        for m in METRICS:
            res.validation[m] = percentile_report(f"puck {METRIC_LABEL[m]}", METRIC_UNIT[m], grids[m]["val"], va[m], params_of,
                                                  canonical_err=res.canonical["val"][m] if res.canonical else None,
                                                  train_errors=grids[m]["train"], selected_train_err=tr[m],
                                                  canonical_train_err=res.canonical["train"][m] if res.canonical else float("nan"))
    return res


def _draw_grid_panel(ax, res: PuckFitResult, m: str, split: str, fontsize: float = 6.5):
    import matplotlib.pyplot as plt
    ext = [res.grid_g[0], res.grid_g[-1], res.grid_gamma[0], res.grid_gamma[-1]]
    Z = res.grids[m][split]
    n = res.n_train if split == "train" else res.n_val
    if not np.isfinite(Z).any():
        ax.set_title(f"{split} {METRIC_LABEL[m]} — no windows"); ax.axis("off"); return None
    im = ax.imshow(Z.T, origin="lower", aspect="auto", extent=ext, cmap="viridis")
    ax.plot(res.gravity_x, res.damping, "r*", ms=12, label=f"selected g={res.gravity_x:.3f}, γ={res.damping:.3f}")
    if res.canonical:
        ax.plot(res.canonical["gravity"], res.canonical["damping"], "wo", mfc="none", ms=8, mew=1.5,
                label=f"canonical g={res.canonical['gravity']:.3f}, γ={res.canonical['damping']:.3f}")
    oi, oj = np.unravel_index(int(np.nanargmin(Z)), Z.shape)
    ax.plot(res.grid_g[oi], res.grid_gamma[oj], "c^", ms=8, mfc="none", mew=1.5,
            label=f"{split} min {Z[oi, oj]:.3g} @ g={res.grid_g[oi]:.2f}, γ={res.grid_gamma[oj]:.2f}")
    ax.set_xlabel("gravity_x [m/s²]"); ax.set_ylabel("damping γ [1/s]")
    ax.set_title(f"{split} {METRIC_LABEL[m]} [{METRIC_UNIT[m]}], {n} windows", fontsize=10)
    ax.legend(fontsize=fontsize, loc="upper right"); plt.colorbar(im, ax=ax)
    return im


def plot_grid(res: PuckFitResult, out_path):
    """2 × 2 panels: rows = rms (cm) / final displacement error over distance,
    columns = train / val. Marks: selected (red star), canonical (white
    circle), that panel's own minimum (cyan triangle)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for r_i, m in enumerate(METRICS):
        for c_i, split in enumerate(("train", "val")):
            _draw_grid_panel(axes[r_i, c_i], res, m, split)
    fig.suptitle(f"Puck (g, γ) grid — rms error (top) and final displacement error / distance travelled (bottom); selected on train {res.objective}", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_paper_figures(res: PuckFitResult, out_dir, stem: str = "puck_final_displacement") -> list:
    """The individual figures kept for the paper (PNG + PDF each):
    ``<stem>_vs_percentile``  train + val final-displacement error relative to
    the best candidate of its split vs candidate percentile;
    ``<stem>_grid_train`` / ``<stem>_grid_val``  the (g, γ) grids of the
    final displacement error / distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if "fit_rel" in res.validation:
        for ext in ("png", "pdf"):
            plot_percentiles([res.validation["fit_rel"]], out_dir / f"{stem}_vs_percentile.{ext}", ncols=1)
        written.append(out_dir / f"{stem}_vs_percentile.png")
    for split in ("train", "val"):
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        _draw_grid_panel(ax, res, "fit_rel", split, fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}_grid_{split}.{ext}", dpi=150)
        plt.close(fig)
        written.append(out_dir / f"{stem}_grid_{split}.png")
    return written
