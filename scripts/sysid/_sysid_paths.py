"""Shared path resolution for paddle sysid grid searches.

All paddle sysid scripts under ``scripts/sysid/`` import:

  - ``SYSID_DIR``       : where the named-category segment HDF5s live (input).
                          Override with ``AIRHOCKEY_SYSID_DATA_DIR``.
                          Default: ``<repo>/sysid/teleop/system_id3``.

  - ``DEFAULT_CONFIG``  : the sim YAML the grid sweep starts from (PID/density
                          values get overridden per combo). Override with
                          ``AIRHOCKEY_SYSID_CONFIG``.
                          Default: ``<repo>/configs/new_juggle/sysid_best_params_hist2.yaml``.

  - ``load_subset(default)`` : returns the per-script 8-segment subset list.
                          If ``AIRHOCKEY_SYSID_SUBSET`` is set, loads that JSON
                          (list of paths relative to ``SYSID_DIR``); otherwise
                          returns ``default`` unchanged.

This indirection lets a single set of grid-search scripts cover the canonical
hist2 sysid AND any new variant (e.g. ``hist_len: 4``) without forking — point
the env vars at the new data directory + config + subset file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

SYSID_DIR = Path(
    os.environ.get(
        "AIRHOCKEY_SYSID_DATA_DIR",
        str(_REPO_ROOT / "sysid/teleop/system_id3"),
    )
)

DEFAULT_CONFIG = Path(
    os.environ.get(
        "AIRHOCKEY_SYSID_CONFIG",
        str(_REPO_ROOT / "configs/new_juggle/sysid_best_params_hist2.yaml"),
    )
)


def load_subset(default: list[str]) -> list[str]:
    path = os.environ.get("AIRHOCKEY_SYSID_SUBSET")
    if not path:
        return list(default)
    with open(path) as f:
        loaded = json.load(f)
    if not isinstance(loaded, list) or not loaded:
        raise ValueError(
            f"AIRHOCKEY_SYSID_SUBSET={path} must contain a non-empty JSON list"
        )
    return [str(p) for p in loaded]
