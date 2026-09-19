"""CMA-ES over (paddle–puck restitution e, mass ratio r = m_paddle / m_puck).

Search space: the unit square ``z ∈ [0, 1]²`` mapped to

    e = e_min + z0 · (e_max − e_min)                 (linear)
    r = r_min · (r_max / r_min)^z1                   (log-uniform)

Objective = RMS over the training collisions of ``speed_out_sim − speed_out_real`` (m/s), with
the sim collision orchestrated by ``sim_collision.HeadOnCollider`` at every trial's measured
paddle speed and incoming puck speed. The validation error of every candidate is recorded so
the search can be validated afterwards (percentile report), and ``grid_scan`` evaluates the
same objective on a regular (e, r) grid to expose the ``(1 + e) · r / (r + 1) = const`` ridge
along which head-on data cannot separate the two parameters.

Candidates are evaluated in parallel with a ``fork`` process pool; every worker owns one
``HeadOnCollider``.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np

from .sim_collision import CollisionParams, HeadOnCollider, evaluate_measurements


@dataclass
class ParamBounds:
    restitution: tuple[float, float] = (0.0, 1.5)
    mass_ratio: tuple[float, float] = (0.25, 1000.0)

    def to_params(self, z) -> CollisionParams:
        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
        e = self.restitution[0] + z[0] * (self.restitution[1] - self.restitution[0])
        r = self.mass_ratio[0] * (self.mass_ratio[1] / self.mass_ratio[0]) ** z[1]
        return CollisionParams(restitution=float(e), mass_ratio=float(r))

    def to_unit(self, p: CollisionParams) -> np.ndarray:
        e = np.clip(p.restitution, *self.restitution)
        r = np.clip(p.mass_ratio, *self.mass_ratio)
        z = np.array([(e - self.restitution[0]) / (self.restitution[1] - self.restitution[0]),
                      np.log(r / self.mass_ratio[0]) / np.log(self.mass_ratio[1] / self.mass_ratio[0])])
        return np.clip(z, 0.0, 1.0)


# ---- parallel evaluation ------------------------------------------------------------------
_W: dict = {}


def _init_worker(sim_cfg, train, val, collider_kwargs):
    _W["col"] = HeadOnCollider(sim_cfg, **(collider_kwargs or {}))
    _W["train"], _W["val"] = train, val


def _eval_candidate(params: CollisionParams) -> tuple[float, float]:
    tr = evaluate_measurements(_W["col"], _W["train"], params)["rms_err"]
    va = evaluate_measurements(_W["col"], _W["val"], params)["rms_err"] if _W["val"] else float("nan")
    return float(tr), float(va)


class CandidateEvaluator:
    """Evaluates lists of ``CollisionParams`` → (train_rms, val_rms); ``workers <= 1`` runs inline."""

    def __init__(self, sim_cfg: dict, train: list, val: list, workers: int = 1, collider_kwargs: Optional[dict] = None):
        self.workers = max(1, int(workers))
        self._pool = None
        if self.workers > 1:
            ctx = mp.get_context("fork")
            self._pool = ctx.Pool(self.workers, initializer=_init_worker, initargs=(sim_cfg, train, val, collider_kwargs))
        else:
            _init_worker(sim_cfg, train, val, collider_kwargs)

    def __call__(self, candidates: list[CollisionParams]) -> list[tuple[float, float]]:
        if self._pool is None:
            return [_eval_candidate(c) for c in candidates]
        return self._pool.map(_eval_candidate, candidates)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


# ---- the search ---------------------------------------------------------------------------
@dataclass
class CandidateRecord:
    restart: int
    generation: int
    restitution: float
    mass_ratio: float
    gain: float
    train_err: float
    val_err: float


@dataclass
class CMAESResult:
    best: CollisionParams                   # lowest training error over every candidate
    best_train_err: float
    best_val_err: float
    x0: CollisionParams
    bounds: ParamBounds
    sigma0: float
    popsize: int
    restarts: int
    n_evaluations: int
    wall_seconds: float
    stop_reasons: list = field(default_factory=list)
    candidates: list[CandidateRecord] = field(default_factory=list)
    generation_best: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"best": self.best.as_dict(), "best_gain": self.best.gain(), "best_train_err": self.best_train_err,
                "best_val_err": self.best_val_err, "x0": self.x0.as_dict(), "bounds": asdict(self.bounds),
                "sigma0": self.sigma0, "popsize": self.popsize, "restarts": self.restarts,
                "n_evaluations": self.n_evaluations, "wall_seconds": self.wall_seconds,
                "stop_reasons": self.stop_reasons, "generation_best": self.generation_best}


def run_cmaes(evaluate: Callable[[list[CollisionParams]], list[tuple[float, float]]], x0: CollisionParams,
              bounds: ParamBounds = ParamBounds(), sigma0: float = 0.3, popsize: int = 12, max_iter: int = 40,
              seed: int = 0, restarts: int = 1, tolfun: float = 1e-3,
              log: Optional[Callable[[str], None]] = print) -> CMAESResult:
    """Ask/tell CMA-ES in the unit square. Restarts (IPOP-style, population doubled each time)
    start from the best point so far with a fresh random seed."""
    import cma

    t0 = time.time()
    records: list[CandidateRecord] = []
    gen_log: list[dict] = []
    stop_reasons = []
    best_p, best_tr, best_va = x0, float("inf"), float("nan")
    z_start = bounds.to_unit(x0)
    pop = int(popsize)
    n_eval = 0
    for r in range(max(1, int(restarts))):
        es = cma.CMAEvolutionStrategy(z_start.tolist(), sigma0, {
            "bounds": [0.0, 1.0], "popsize": pop, "seed": int(seed) + 1000 * r + 1, "maxiter": int(max_iter),
            "tolfun": float(tolfun), "tolx": 1e-4, "verbose": -9, "verb_log": 0})
        gen = 0
        while not es.stop():
            Z = es.ask()
            cands = [bounds.to_params(z) for z in Z]
            scores = evaluate(cands)
            n_eval += len(cands)
            f = [s[0] if np.isfinite(s[0]) else 1e9 for s in scores]
            es.tell(Z, f)
            for p, (tr, va) in zip(cands, scores):
                records.append(CandidateRecord(r, gen, p.restitution, p.mass_ratio, p.gain(), tr, va))
                if tr < best_tr:
                    best_p, best_tr, best_va = p, tr, va
            i_gen = int(np.argmin(f))
            gen_log.append({"restart": r, "generation": gen, "n_evaluations": n_eval,
                            "gen_best_train_err": float(f[i_gen]), "gen_best_val_err": float(scores[i_gen][1]),
                            "gen_median_train_err": float(np.median(f)),
                            "best_so_far_train_err": best_tr, "best_so_far_val_err": best_va,
                            "sigma": float(es.sigma), "best_so_far": best_p.as_dict(), "best_so_far_gain": best_p.gain()})
            if log:
                log(f"[restart {r} gen {gen:3d} | {n_eval:5d} evals | {time.time() - t0:6.1f}s] "
                    f"gen best train {f[i_gen]:.4f} m/s (val {scores[i_gen][1]:.4f}) | best so far "
                    f"e={best_p.restitution:.4f} r={best_p.mass_ratio:8.2f} gain={best_p.gain():.4f} "
                    f"train {best_tr:.4f} val {best_va:.4f} | sigma {es.sigma:.3f}")
            gen += 1
        stop_reasons.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                             for k, v in es.stop().items()})
        z_start = bounds.to_unit(best_p)
        pop *= 2
    return CMAESResult(best=best_p, best_train_err=best_tr, best_val_err=best_va, x0=x0, bounds=bounds,
                       sigma0=sigma0, popsize=popsize, restarts=restarts, n_evaluations=n_eval,
                       wall_seconds=time.time() - t0, stop_reasons=stop_reasons, candidates=records,
                       generation_best=gen_log)


def grid_scan(evaluate: Callable[[list[CollisionParams]], list[tuple[float, float]]], bounds: ParamBounds,
              n_restitution: int = 31, n_mass_ratio: int = 31) -> dict:
    """Train / val RMS on a regular grid (restitution linear, mass ratio log) — the landscape plot."""
    e_grid = np.linspace(bounds.restitution[0], bounds.restitution[1], int(n_restitution))
    r_grid = np.geomspace(bounds.mass_ratio[0], bounds.mass_ratio[1], int(n_mass_ratio))
    cands = [CollisionParams(float(e), float(r)) for r in r_grid for e in e_grid]
    scores = evaluate(cands)
    train = np.array([s[0] for s in scores]).reshape(len(r_grid), len(e_grid))
    val = np.array([s[1] for s in scores]).reshape(len(r_grid), len(e_grid))
    i = np.unravel_index(int(np.nanargmin(train)), train.shape)
    return {"restitution": e_grid.tolist(), "mass_ratio": r_grid.tolist(), "train_err": train.tolist(),
            "val_err": val.tolist(), "best": CollisionParams(float(e_grid[i[1]]), float(r_grid[i[0]])).as_dict(),
            "best_train_err": float(train[i]), "best_val_err": float(val[i])}
