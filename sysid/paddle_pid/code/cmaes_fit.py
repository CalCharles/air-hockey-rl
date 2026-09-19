"""CMA-ES over the paddle PID gains (kp, ki, kd) with the paddle mass fixed.

Search space: the unit cube ``z ∈ [0, 1]^3`` (``cma`` handles the box bounds) mapped to
gains with

    kp = kp_min · (kp_max / kp_min)^z0            (log-uniform; kp > 0 always)
    ki = expm1(z1 · log1p(ki_max))                (0 at z1 = 0, log-like above)
    kd = expm1(z2 · log1p(kd_max))                (0 at z2 = 0, log-like above)

so the three gains, which span orders of magnitude, get a comparable step size and ``ki = 0``
(the canonical value) and ``kd = 0`` (the body already carries ``paddle_damping``) are exactly
representable. Objective = mean per-step position error (mm)
over the training trials (``replay.evaluate_trials``); the validation error of every candidate
is recorded alongside so the search can be validated afterwards (percentile report).

Candidates are evaluated in parallel with a ``fork`` process pool; every worker owns one
``PaddleReplayer``.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import numpy as np

from .dataset import PaddleTrial
from .replay import PaddleReplayer, PlantParams, evaluate_trials


@dataclass
class GainBounds:
    kp: tuple[float, float] = (500.0, 1.0e5)
    ki_max: float = 1.0e5
    kd_max: float = 5.0e3

    def to_gains(self, z) -> PlantParams:
        z = np.clip(np.asarray(z, dtype=float), 0.0, 1.0)
        kp = self.kp[0] * (self.kp[1] / self.kp[0]) ** z[0]
        ki = float(np.expm1(z[1] * np.log1p(self.ki_max)))
        kd = float(np.expm1(z[2] * np.log1p(self.kd_max)))
        return PlantParams(kp=float(kp), ki=max(ki, 0.0), kd=max(kd, 0.0))

    def to_unit(self, params: PlantParams) -> np.ndarray:
        kp = np.clip(params.kp, *self.kp)
        kd = np.clip(params.kd, 0.0, self.kd_max)
        ki = np.clip(params.ki, 0.0, self.ki_max)
        z = np.array([np.log(kp / self.kp[0]) / np.log(self.kp[1] / self.kp[0]),
                      np.log1p(ki) / np.log1p(self.ki_max),
                      np.log1p(kd) / np.log1p(self.kd_max)])
        return np.clip(z, 0.0, 1.0)


# ---- parallel evaluation ------------------------------------------------------------------
_W: dict = {}


def _init_worker(sim_cfg, train, val, density, action_delay):
    _W["rep"] = PaddleReplayer(sim_cfg)
    _W["train"], _W["val"] = train, val
    _W["density"], _W["delay"] = density, action_delay


def _eval_candidate(params: PlantParams) -> tuple[float, float]:
    p = PlantParams(params.kp, params.ki, params.kd, _W["density"])
    tr = evaluate_trials(_W["rep"], _W["train"], p, _W["delay"])["mean_pos_err_mm"]
    va = evaluate_trials(_W["rep"], _W["val"], p, _W["delay"])["mean_pos_err_mm"] if _W["val"] else float("nan")
    return float(tr), float(va)


class CandidateEvaluator:
    """Evaluates lists of ``PlantParams`` → (train_err, val_err); ``workers <= 1`` runs inline."""

    def __init__(self, sim_cfg: dict, train: list[PaddleTrial], val: list[PaddleTrial],
                 paddle_density: Optional[float], action_delay_steps: int = 0, workers: int = 1):
        self.workers = max(1, int(workers))
        self._pool = None
        if self.workers > 1:
            ctx = mp.get_context("fork")
            self._pool = ctx.Pool(self.workers, initializer=_init_worker,
                                  initargs=(sim_cfg, train, val, paddle_density, action_delay_steps))
        else:
            _init_worker(sim_cfg, train, val, paddle_density, action_delay_steps)

    def __call__(self, candidates: list[PlantParams]) -> list[tuple[float, float]]:
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
    kp: float
    ki: float
    kd: float
    train_err_mm: float
    val_err_mm: float


@dataclass
class CMAESResult:
    best: PlantParams                       # lowest training error over every candidate
    best_train_err_mm: float
    best_val_err_mm: float
    x0: PlantParams
    bounds: GainBounds
    sigma0: float
    popsize: int
    restarts: int
    n_evaluations: int
    wall_seconds: float
    stop_reasons: list = field(default_factory=list)
    candidates: list[CandidateRecord] = field(default_factory=list)
    generation_best: list[dict] = field(default_factory=list)     # per generation: best-so-far etc.

    def to_json(self) -> dict:
        d = {"best": self.best.as_dict(), "best_train_err_mm": self.best_train_err_mm,
             "best_val_err_mm": self.best_val_err_mm, "x0": self.x0.as_dict(),
             "bounds": asdict(self.bounds), "sigma0": self.sigma0, "popsize": self.popsize,
             "restarts": self.restarts, "n_evaluations": self.n_evaluations,
             "wall_seconds": self.wall_seconds, "stop_reasons": self.stop_reasons,
             "generation_best": self.generation_best}
        return d


def run_cmaes(evaluate: Callable[[list[PlantParams]], list[tuple[float, float]]], x0: PlantParams,
              bounds: GainBounds = GainBounds(), sigma0: float = 0.3, popsize: int = 16,
              max_iter: int = 60, seed: int = 0, restarts: int = 1, tolfun_mm: float = 0.02,
              log: Optional[Callable[[str], None]] = print) -> CMAESResult:
    """Ask/tell CMA-ES in the unit cube. Restarts (IPOP-style, population doubled each time)
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
            "tolfun": float(tolfun_mm), "tolx": 1e-4, "verbose": -9, "verb_log": 0})
        gen = 0
        while not es.stop():
            Z = es.ask()
            cands = [bounds.to_gains(z) for z in Z]
            scores = evaluate(cands)
            n_eval += len(cands)
            f = [s[0] if np.isfinite(s[0]) else 1e9 for s in scores]
            es.tell(Z, f)
            for p, (tr, va) in zip(cands, scores):
                records.append(CandidateRecord(r, gen, p.kp, p.ki, p.kd, tr, va))
                if tr < best_tr:
                    best_p, best_tr, best_va = p, tr, va
            i_gen = int(np.argmin(f))
            gen_log.append({"restart": r, "generation": gen, "n_evaluations": n_eval,
                            "gen_best_train_err_mm": float(f[i_gen]), "gen_best_val_err_mm": float(scores[i_gen][1]),
                            "gen_median_train_err_mm": float(np.median(f)),
                            "best_so_far_train_err_mm": best_tr, "best_so_far_val_err_mm": best_va,
                            "sigma": float(es.sigma), "best_so_far": best_p.as_dict()})
            if log:
                log(f"[restart {r} gen {gen:3d} | {n_eval:5d} evals | {time.time() - t0:6.1f}s] "
                    f"gen best train {f[i_gen]:7.2f} mm (val {scores[i_gen][1]:7.2f}) | best so far "
                    f"kp={best_p.kp:8.1f} ki={best_p.ki:9.1f} kd={best_p.kd:7.2f} train {best_tr:7.2f} val {best_va:7.2f} "
                    f"| sigma {es.sigma:.3f}")
            gen += 1
        stop_reasons.append({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                             for k, v in es.stop().items()})
        z_start = bounds.to_unit(best_p)
        pop *= 2
    return CMAESResult(best=best_p, best_train_err_mm=best_tr, best_val_err_mm=best_va, x0=x0, bounds=bounds,
                       sigma0=sigma0, popsize=popsize, restarts=restarts, n_evaluations=n_eval,
                       wall_seconds=time.time() - t0, stop_reasons=stop_reasons, candidates=records,
                       generation_best=gen_log)
