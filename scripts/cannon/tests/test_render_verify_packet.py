"""No-SSH contract tests for the canonical ``verify.sh`` packet renderer."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RENDERER_PATH = REPO_ROOT / "scripts" / "cannon" / "render_verify_packet.py"
MODULE_SPEC = importlib.util.spec_from_file_location("render_verify_packet", RENDERER_PATH)
assert MODULE_SPEC and MODULE_SPEC.loader
renderer = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(renderer)
SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _base(tmp_path: Path) -> dict[str, str | list[str]]:
    facility = tmp_path / "facility"
    shared = tmp_path / "shared-env"
    facility.mkdir()
    shared.mkdir()
    return {
        "sha": SHA,
        "tests": ["scripts/cannon/tests/test_render_verify_packet.py"],
        "run_root": str(facility / "run"),
        "packet_dir": str(facility / "run" / "packet"),
        "facility_run_prefix": str(facility),
        "shared_env_prefix": str(shared),
        "repo_root": str(REPO_ROOT),
        "uv_bin": "/usr/bin/python3",
    }


def _spec(tmp_path: Path, *, killer: str = "test_render_verify_packet.py::test_golden_render", old: str = "old", new: str = "new") -> Path:
    path = tmp_path / "mutants.json"
    rows = [
        {"id": f"m{index}", "target": "scripts/cannon/render_verify_packet.py", "old": old, "new": new, "killer": killer}
        for index in range(3)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _render(tmp_path: Path, **overrides: object) -> subprocess.CompletedProcess[str]:
    values = _base(tmp_path)
    values.update(overrides)
    args = [sys.executable, str(RENDERER_PATH)]
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if key == "tests":
            for test_path in value:  # type: ignore[union-attr]
                args.extend([flag, str(test_path)])
        else:
            args.extend([flag, str(value)])
    return subprocess.run(args, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def test_golden_render(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    result = _render(tmp_path, mutants=spec, job_name="verify-renderer")
    assert result.returncode == 0, result.stdout + result.stderr
    packet = Path(_line(result.stdout, "READY packet="))
    body = (packet / "verify.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --output=" in body and "#SBATCH --error=" in body
    assert 'WORK_ROOT="/scratch/${USER:?USER required}/${SLURM_JOB_ID:?SLURM_JOB_ID required}"' in body
    assert 'cd "${TASK_ROOT}"' in body
    assert "--mode quick" in body and "--typecheck on" in body and "--parallel serial" in body
    assert str(packet / "stage" / "mutant-spec.json") in body
    assert (packet / "packet-manifest.sha256").is_file()
    assert "SBATCH_COMMAND=sbatch " in result.stdout
    assert 'HEADROOM_GATE=squeue -u "$USER" -h -p test | wc -l' in result.stdout


def test_missing_work_root_or_unexpanded_path_is_rejected(tmp_path: Path) -> None:
    values = _base(tmp_path)
    packet = renderer.render_packet(**values)
    body_path = packet / "verify.sbatch"
    body = body_path.read_text(encoding="utf-8").replace(
        'WORK_ROOT="/scratch/${USER:?USER required}/${SLURM_JOB_ID:?SLURM_JOB_ID required}"',
        'WORK_ROOT="/scratch/${USER:?USER required}"',
    )
    body_path.write_text(body, encoding="utf-8")
    with pytest.raises(renderer.RenderError, match="WORK_ROOT"):
        renderer._validate_body(body, body_path, packet / "stage", Path(values["run_root"]), None)


def test_single_quoted_work_root_is_rejected(tmp_path: Path) -> None:
    values = _base(tmp_path)
    packet = renderer.render_packet(**values)
    body_path = packet / "verify.sbatch"
    body = body_path.read_text(encoding="utf-8").replace(
        'WORK_ROOT="/scratch/${USER:?USER required}/${SLURM_JOB_ID:?SLURM_JOB_ID required}"',
        "WORK_ROOT='/scratch/${USER:?USER required}/${SLURM_JOB_ID:?SLURM_JOB_ID required}'",
    )
    body_path.write_text(body, encoding="utf-8")
    with pytest.raises(renderer.RenderError, match="WORK_ROOT"):
        renderer._validate_body(body, body_path, packet / "stage", Path(values["run_root"]), None)


def test_escaped_dollar_is_rejected(tmp_path: Path) -> None:
    values = _base(tmp_path)
    packet = renderer.render_packet(**values)
    body_path = packet / "verify.sbatch"
    body = body_path.read_text(encoding="utf-8").replace("HEAD_SHA=", r"HEAD_SHA=\$", 1)
    with pytest.raises(renderer.RenderError, match="backslash-dollar"):
        renderer._validate_body(body, body_path, packet / "stage", Path(values["run_root"]), None)


def test_full_path_killer_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path, killer="/absolute/test_render_verify_packet.py::test_golden_render")
    result = _render(tmp_path, mutants=spec)
    assert result.returncode != 0
    assert "killer" in result.stderr


def test_escaped_newline_mutant_is_rejected(tmp_path: Path) -> None:
    spec = _spec(tmp_path, old=r"old\ntext")
    result = _render(tmp_path, mutants=spec)
    assert result.returncode != 0
    assert "escaped newline" in result.stderr


def test_relative_spec_path_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path)
    monkeypatch.chdir(REPO_ROOT)
    result = _render(tmp_path, mutants=str(spec.relative_to(REPO_ROOT)) if spec.is_relative_to(REPO_ROOT) else "mutants.json")
    if spec.is_relative_to(REPO_ROOT):
        assert result.returncode != 0
    else:
        # A missing relative path is still fail-closed and cannot reach sbatch.
        assert result.returncode != 0


def test_existing_packet_dir_is_rejected(tmp_path: Path) -> None:
    values = _base(tmp_path)
    packet = Path(values["packet_dir"])
    packet.mkdir(parents=True)
    result = _render(tmp_path)
    assert result.returncode != 0
    assert "existing packet" in result.stderr


def test_dry_run_left_on_is_rejected(tmp_path: Path) -> None:
    values = _base(tmp_path)
    packet = renderer.render_packet(**values)
    body_path = packet / "verify.sbatch"
    body = body_path.read_text(encoding="utf-8").replace('DRY_RUN="${DRY_RUN:-0}"', "DRY_RUN=1")
    with pytest.raises(renderer.RenderError, match="DRY_RUN"):
        renderer._validate_body(body, body_path, packet / "stage", Path(values["run_root"]), None)


def test_malformed_spec_is_rejected(tmp_path: Path) -> None:
    spec = tmp_path / "malformed.json"
    spec.write_text("{not-json", encoding="utf-8")
    result = _render(tmp_path, mutants=spec)
    assert result.returncode != 0
    assert "malformed" in result.stderr


def _line(output: str, prefix: str) -> str:
    for line in output.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise AssertionError(f"missing {prefix!r} in {output!r}")
