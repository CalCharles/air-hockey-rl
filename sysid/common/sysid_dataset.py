"""Dataset layer for the sysid pipeline.

Reads the sections written by ``sysid/common/extract_sysid_sections.py``
(``manifest.json`` + ``free_fall/*.hdf5`` under ``sysid/puck_dynamics/data/<name>/``,
``manifest.json`` + ``wall/*.hdf5`` under ``sysid/wall_collision/data/<name>/``), splits them into
train / validation **by source recording** (so no trajectory leaks across the
split), and turns them into the two fixed-shape inputs the fits consume:

* ``FreeFallWindow`` — exactly ``window_frames`` usable puck samples cut from a
  clean free-fall clip (long clips are chopped into consecutive windows, the
  remainder is dropped) — every datapoint has the same length.
* ``WallBounce`` — one wall impact with its clean pre / post windows.

Everything is in the sim frame used by ``trajectory_segmentation.py``
(x = long axis, robot at x < 0, gravity towards x < 0). Conversions to the
Box2D "base" frame live in ``wall_restitution_fit.py``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

from .trajectory_segmentation import SegmentationConfig, _mark_stale


@dataclass
class FreeFallWindow:
    source: str
    clip: str
    start_in_clip: int
    t: np.ndarray        # (L,) seconds from first sample
    xy: np.ndarray       # (L, 2) sim frame


@dataclass
class WallBounce:
    source: str
    clip: str
    wall: str            # x+ / x- / y+ / y-
    t: np.ndarray        # (N,) seconds from first slice sample
    xy: np.ndarray       # (N, 2) sim frame, usable samples only
    usable_idx: np.ndarray  # (N,) original slice indices of the usable samples
    a: int               # slice index of the last clean pre-impact frame
    b: int               # slice index of the first clean post-impact frame
    paddle_xy_a: np.ndarray  # calibrated paddle position at frame a (sim frame)
    meta: dict = field(default_factory=dict)

    @property
    def pre_idx(self) -> np.ndarray:
        return np.flatnonzero(self.usable_idx <= self.a)

    @property
    def post_idx(self) -> np.ndarray:
        return np.flatnonzero(self.usable_idx >= self.b)


def load_manifest(sections_dir: str | Path) -> dict:
    with open(Path(sections_dir) / "manifest.json") as f:
        return json.load(f)


def split_by_source(rows: list[dict], val_fraction: float = 0.2, seed: int = 0, sources: list[str] | None = None) -> dict:
    """Group sections by their source recording and hold out a fraction of
    the recordings. Returns {"train": [...], "val": [...], "sources": {...}}.

    ``sources`` (default: the recordings present in ``rows``) is the list the
    split is drawn from; pass the manifest's ``sources`` (every input recording)
    so the puck and wall fits hold out the same recordings for a given seed."""
    sources = sorted(set(sources) if sources else {r["source"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(sources)
    n_val = max(1, int(round(val_fraction * len(sources))))
    val_src = set(sources[:n_val])
    return {
        "train": [r for r in rows if r["source"] not in val_src],
        "val": [r for r in rows if r["source"] in val_src],
        "sources": {"train": sorted(set(sources) - val_src), "val": sorted(val_src)},
        "val_fraction": val_fraction, "seed": seed,
    }


def _load_slice(path: Path, cfg: SegmentationConfig, puck_x_sign: int):
    with h5py.File(path, "r") as h:
        p = np.asarray(h["puck"][:], dtype=np.float64)
        t = np.asarray(h["cur_time"][:], dtype=np.float64).ravel()
        pose = np.asarray(h["pose"][:, :2], dtype=np.float64)
        attrs = {k: (json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v) for k, v in h.attrs.items()}
    xy = p[:, :2].copy()
    xy[:, 0] *= puck_x_sign
    valid = p[:, 2] == 0
    fresh = _mark_stale(xy, valid, cfg)
    return t - t[0], xy, valid & fresh, pose, attrs


def make_free_fall_windows(rows: list[dict], sections_dir: str | Path, cfg: SegmentationConfig,
                           window_frames: int = 10, max_span_frames: Optional[int] = None) -> list[FreeFallWindow]:
    """Chop every free-fall clip into consecutive windows of exactly
    ``window_frames`` usable samples. A window whose samples span more than
    ``max_span_frames`` raw frames (default window_frames + 2, i.e. it hides an
    occlusion) is skipped."""
    sections_dir = Path(sections_dir)
    max_span = max_span_frames or window_frames + 2
    out = []
    for r in rows:
        if r["kind"] != "free_fall":
            continue
        t, xy, usable, _, _ = _load_slice(sections_dir / r["file"], cfg, r["puck_x_sign"])
        idx = np.flatnonzero(usable)
        for k in range(0, len(idx) - window_frames + 1, window_frames):
            w = idx[k:k + window_frames]
            if w[-1] - w[0] + 1 > max_span:
                continue
            out.append(FreeFallWindow(source=r["source"], clip=r["file"], start_in_clip=int(w[0]),
                                      t=t[w] - t[w[0]], xy=xy[w]))
    return out


def make_wall_bounces(rows: list[dict], sections_dir: str | Path, cfg: SegmentationConfig) -> list[WallBounce]:
    sections_dir = Path(sections_dir)
    out = []
    for r in rows:
        if r["kind"] != "wall":
            continue
        t, xy, usable, pose, attrs = _load_slice(sections_dir / r["file"], cfg, r["puck_x_sign"])
        idx = np.flatnonzero(usable)
        a, b = int(r["impact_pre_frame"]), int(r["impact_post_frame"])
        pad_a = np.array([r["paddle_x_sign"] * pose[a, 0] + r["paddle_x_offset"], pose[a, 1]])
        out.append(WallBounce(source=r["source"], clip=r["file"], wall=r["wall"], t=t[idx], xy=xy[idx],
                              usable_idx=idx, a=a, b=b, paddle_xy_a=pad_a, meta=r))
    return out
