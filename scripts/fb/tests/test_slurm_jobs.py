"""Preparation-only tests: never submit jobs or start training."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
from scripts.fb.slurm_jobs import prepare, run_manifest


def bundle(tmp_path, **overrides):
    cfg = tmp_path / "config with spaces.yaml"
    cfg.write_text("seed: 0\nwandb: true\n")
    options = dict(folder=tmp_path / "job bundle", configs=[cfg], seeds=[0, 1, 2, 3],
                   parallel=4, account="ASC26008", partition="gh", walltime="24:00:00",
                   cpus=72, python=sys.executable)
    options.update(overrides)
    return prepare(**options)


def test_preparation_and_shell_syntax(tmp_path):
    path = bundle(tmp_path)
    manifest = json.loads(path.read_text())
    assert manifest["parallel"] == 4
    assert len(manifest["runs"]) == 4
    assert len({run["output"] for run in manifest["runs"]}) == 4
    for i, run in enumerate(manifest["runs"]):
        assert run["command"][run["command"].index("--seed") + 1] == str(i)
        assert not Path(run["output"]).exists()
    for filename in ("job.sbatch", "submit.sh"):
        subprocess.run(["bash", "-n", str(path.parent / filename)], check=True)
    assert "SLURM_JOB_ID" in (path.parent / "submit.sh").read_text()
    assert "login[0-9]*" in (path.parent / "submit.sh").read_text()
    assert not (path.parent / "results.jsonl").exists()


def test_reject_duplicate_seeds(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        bundle(tmp_path, seeds=[0, 0])


def test_reject_directive_injection(tmp_path):
    with pytest.raises(ValueError, match="account"):
        bundle(tmp_path, account="ASC26008\n#SBATCH --nodes=99")


def test_manifest_requires_allocation(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="allocated"):
        run_manifest(tmp_path / "not-read.json")
