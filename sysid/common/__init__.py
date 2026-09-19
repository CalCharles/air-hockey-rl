"""Shared sysid library: trajectory segmentation + rendering, the sections dataset layer,
percentile validation of a search, and the recording-level tools (stage 0 / 1 of the
puck + wall pipeline). The fits themselves live in ``sysid/<fit>/code/``."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SYSID_ROOT = REPO_ROOT / "sysid"


def link_data(fit_name: str, source: Path, out: Path) -> Path | None:
    """Record where a fit's raw data came from: ``sysid/<fit_name>/data/<source.name>`` → ``source``
    (absolute symlink). Only done when ``out`` is under that fit's ``results/`` (i.e. a real run,
    not a test in a temporary directory). Returns the link path or None."""
    source = Path(source).resolve()
    results_root = (SYSID_ROOT / fit_name / "results").resolve()
    try:
        Path(out).resolve().relative_to(results_root)
    except ValueError:
        return None
    data_dir = SYSID_ROOT / fit_name / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    link = data_dir / source.name
    if link.is_symlink() or link.exists():
        return link
    os.symlink(source, link)
    return link
