"""No-SSH contract tests for the canonical ``verify.sh`` packet renderer."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess

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
    facility.mkdir(exist_ok=True)
    shared.mkdir(exist_ok=True)
    return {
        "sha": SHA,
        "tests": ["scripts/cannon/tests/test_render_verify_packet.py"],
        "run_root": str(facility / "run"),
        "packet_dir": str(facility / "run" / "packet"),
        "test_facility_run_prefix": str(facility),
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
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            renderer.render_packet(**values)
    except renderer.RenderError as exc:
        print(f"REJECTED {exc}", file=stderr)
        return subprocess.CompletedProcess([], 2, stdout.getvalue(), stderr.getvalue())
    return subprocess.CompletedProcess([], 0, stdout.getvalue(), stderr.getvalue())


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


def test_packet_dir_outside_run_root_is_rejected(tmp_path: Path) -> None:
    values = _base(tmp_path)
    outside = tmp_path / "outside-packet"
    result = _render(tmp_path, packet_dir=str(outside))
    assert result.returncode != 0
    assert "inside run root" in result.stderr
    assert not outside.exists()


def test_cli_rejects_run_root_outside_approved_prefix(tmp_path: Path) -> None:
    values = _base(tmp_path)
    values.pop("test_facility_run_prefix")
    with pytest.raises(renderer.RenderError, match="facility prefix"):
        renderer.render_packet(**values)


def test_relative_spec_path_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = _render(tmp_path, mutants=spec.name)
    assert result.returncode != 0
    assert "absolute" in result.stderr


def test_rendered_body_consumes_verify_interface(tmp_path: Path) -> None:
    values = _base(tmp_path)
    packet = renderer.render_packet(**values)
    task_root = packet / "stage"
    verify = task_root / "scripts" / "cannon" / "verify.sh"
    verify.write_text(
        "#!/usr/bin/env bash\n"
        "set -Eeuo pipefail\n"
        "printf 'cwd=%s\\n' \"$PWD\" > \"$STUB_LOG\"\n"
        "for name in HEAD_SHA JOB_NAME TASK_ROOT RUN_ROOT DRY_RUN PYTHON USER WORK_ROOT "
        "FACILITY_RUN_PREFIX SHARED_ENV_PREFIX UV_BIN REPO_URL; do\n"
        "  printf '%s=%s\\n' \"$name\" \"${!name:?missing $name}\" >> \"$STUB_LOG\"\n"
        "done\n"
        "for arg in \"$@\"; do printf 'arg=%s\\n' \"$arg\" >> \"$STUB_LOG\"; done\n",
        encoding="utf-8",
    )
    verify.chmod(0o750)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "mkdir").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (bin_dir / "mkdir").chmod(0o750)
    log = tmp_path / "consumer.log"
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "USER": "packet-consumer",
        "SLURM_JOB_ID": "consumer-job",
        "SLURM_JOB_PARTITION": "test",
        "DRY_RUN": "0",
        "STUB_LOG": str(log),
    }
    result = subprocess.run(
        ["/bin/bash", str(packet / "verify.sbatch")],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = log.read_text(encoding="utf-8").splitlines()
    assert observed[0] == f"cwd={task_root}"
    expected_env = {
        "HEAD_SHA": SHA,
        "JOB_NAME": "verify-" + SHA[:12],
        "TASK_ROOT": str(task_root),
        "RUN_ROOT": str(Path(values["run_root"])),
        "DRY_RUN": "0",
        "PYTHON": "/usr/bin/python3",
        "USER": "packet-consumer",
        "WORK_ROOT": "/scratch/packet-consumer/consumer-job",
        "FACILITY_RUN_PREFIX": str(Path(values["test_facility_run_prefix"])),
        "SHARED_ENV_PREFIX": str(Path(values["shared_env_prefix"])),
        "UV_BIN": "/usr/bin/python3",
        "REPO_URL": "https://github.com/YizhongHu/TPEN.git",
    }
    assert observed[1 : 1 + len(expected_env)] == [f"{key}={value}" for key, value in expected_env.items()]
    args = [line.removeprefix("arg=") for line in observed[1 + len(expected_env) :]]
    assert args == [
        "--sha",
        SHA,
        "--run-root",
        str(Path(values["run_root"])),
        "--test",
        "scripts/cannon/tests/test_render_verify_packet.py",
        "--mode",
        "quick",
        "--typecheck",
        "on",
        "--parallel",
        "serial",
    ]


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
