"""Smoke test: the whole puck + wall sysid pipeline on a handful of recordings,
written into a temporary sysid tree (``--root``), so nothing under ``sysid/`` is touched.

Needs a recordings directory; set ``AIRHOCKEY_SYSID_SMOKE_DIR`` to override the
default (the shared mouse dataset). Skips when no recordings are available.
Runs in ~1-2 minutes (no GIFs, coarse restitution grid).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_DEFAULT = _REPO / "shared/mouse_state_data_all_new_len_gt130_take100_trim30/trimmed_hdf5"


@pytest.fixture(scope="module")
def recordings(tmp_path_factory):
    src = Path(os.environ.get("AIRHOCKEY_SYSID_SMOKE_DIR", _DEFAULT))
    files = sorted(src.glob("*.hdf5"))[:8] if src.exists() else []
    if not files:
        pytest.skip(f"no recordings under {src}")
    d = tmp_path_factory.mktemp("recordings")
    for f in files:
        (d / f.name).symlink_to(f)
    return d


def test_full_pipeline(recordings, tmp_path):
    root = tmp_path / "sysid"
    cmd = [sys.executable, str(_REPO / "sysid/common/run_puck_wall_sysid.py"), "--input-dir", str(recordings),
           "--name", "smoke", "--root", str(root), "--eval-sample", "1", "--window-frames", "10",
           "--segment-args", "--no-gif --no-hdf5",
           "--extract-args", "--min-free-frames 8",
           "--puck-args", "--g-range -0.9 -0.5 --gamma-range 0.0 0.3",
           "--wall-args", "--restitution-range 0.6 1.0 0.1"]
    proc = subprocess.run(cmd, cwd=_REPO, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    run, pd, wd = root / "common/runs/smoke", root / "puck_dynamics/data/smoke", root / "wall_collision/data/smoke"
    pr, wr = root / "puck_dynamics/results/smoke", root / "wall_collision/results/smoke"
    for path in (run / "README.md", run / "pipeline_run.json", run / "segmentation_eval/summary.md",
                 pd / "manifest.json", pd / "summary.md", pd / "recordings", wd / "manifest.json", wd / "summary.md", wd / "recordings",
                 pr / "results.json", pr / "sim_config_fitted.yaml", pr / "summary.md", pr / "split.json", pr / "fit_grid.png",
                 pr / "validation_percentiles.png", pr / "puck_final_displacement_vs_percentile.png", pr / "puck_final_displacement_grid_val.pdf",
                 wr / "results.json", wr / "sim_config_fitted.yaml", wr / "summary.md", wr / "split.json", wr / "fit_sweeps.png",
                 wr / "wall_side_exit_speed_sweep.png", wr / "wall_side_exit_angle_scatter.pdf"):
        assert path.exists(), path
    # the two data folders hold one section kind each and both list every recording
    pm, wm = json.load(open(pd / "manifest.json")), json.load(open(wd / "manifest.json"))
    assert pm["calibration"]["puck_x_sign"] in (1, -1)
    assert pm["sections"] and all(r["kind"] == "free_fall" for r in pm["sections"])
    assert all(r["kind"] == "wall" for r in wm["sections"])
    assert pm["sources"] == wm["sources"] and len(pm["sources"]) == 8
    # identical train / val recordings in both fits
    ps, ws = json.load(open(pr / "split.json")), json.load(open(wr / "split.json"))
    assert ps["sources"] == ws["sources"]
    puck = json.load(open(pr / "results.json"))
    assert -1.0 <= puck["gravity_x"] <= -0.4 and 0.0 <= puck["damping"] <= 0.4
    assert puck["n_train"] > 0 and puck["n_val"] > 0
    for m in ("fit_rms_cm", "fit_rel"):
        assert puck["train"][m] >= 0 and puck["val"][m] >= 0
        assert m in puck["grids"] and m in puck["validation"]
        v = puck["validation"][m]
        assert 0.0 <= v["selected_beats"] <= 1.0 and set(v["percentiles"]) == {"50", "75", "90"}
    assert set(puck["fitted_params"]) == {"gravity", "puck_damping"}
    wall = json.load(open(wr / "results.json"))
    assert wall["puck_params_used"] == {"gravity": puck["gravity_x"], "puck_damping": puck["damping"]}
    for w in wall["walls"]:
        for m in ("speed_err", "speed_rel_err", "normal_err", "normal_rel_err", "angle_err"):
            assert len(w["train"][m]) == len(w["values"]) and len(w["val"][m]) == len(w["values"])
        assert w["objective"] == "speed_rel_err"
        assert w["param"] in wall["fitted_params"]
