"""Independent P6 review probes for the executor profile capability boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.recovery import (
    CoreProfileReference,
    ProfileCheck,
    ProfileUnavailable,
    REFUSAL_CODES,
)
from tpen_executor.workers import SubprocessWorker


PROFILE = "tpen.native-recovery/v1"


def _payload(verdict: str = "supported", *, profile: str = PROFILE) -> dict[str, object]:
    """Return one canonical profile-check payload for subprocess probes."""

    return {
        "schema": "tpen.profile-check/v1",
        "profile": profile,
        "verdict": verdict,
        "state_owners": ["model", "optimizer"],
        "refusal": None if verdict == "supported" else {
            "code": "unsupported-callback",
            "reason": "review refusal",
        },
    }


def _fake_environment(tmp_path: Path, output: bytes | str, *, exit_code: int = 0) -> PreparedPythonEnvironment:
    """Build a fake core module that writes exactly the requested wire output."""

    package = tmp_path / "tpen" / "checkpoint"
    package.mkdir(parents=True)
    (tmp_path / "tpen" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "profiles.py").write_text(
        "import os, sys\n"
        "value = os.environ['PROFILE_OUTPUT']\n"
        "if os.environ.get('PROFILE_BINARY') == '1':\n"
        "    sys.stdout.buffer.write(value.encode('latin1'))\n"
        "else:\n"
        "    sys.stdout.write(value)\n"
        "sys.exit(int(os.environ.get('PROFILE_EXIT', '0')))\n",
        encoding="utf-8",
    )
    if isinstance(output, bytes):
        output_value = output.decode("latin1")
        binary = "1"
    else:
        output_value = output
        binary = "0"
    return PreparedPythonEnvironment(
        sys.executable,
        env={
            "PYTHONPATH": str(tmp_path),
            "PROFILE_OUTPUT": output_value,
            "PROFILE_BINARY": binary,
            "PROFILE_EXIT": str(exit_code),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def _read(tmp_path: Path, environment: PreparedPythonEnvironment, *, timeout: float = 10.0) -> ProfileCheck:
    """Read one subprocess-emitted profile record through the public boundary."""

    return CoreProfileReference(str(tmp_path / "config.yaml"), PROFILE).read(
        SubprocessWorker(),
        environment,
        cwd=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        timeout=timeout,
    )


def test_r1_t0_profile_response_must_match_requested_profile(tmp_path: Path) -> None:
    """RED: a valid response for another profile must not satisfy this reference."""

    with pytest.raises(ProfileUnavailable):
        _read(tmp_path, _fake_environment(tmp_path, json.dumps(_payload(profile="other/v1"))))


@pytest.mark.parametrize("verdict", ["supported", "refused"])
def test_r1_t1_valid_supported_and_refused_records_decode(tmp_path: Path, verdict: str) -> None:
    """GREEN: both typed core outcomes cross the real subprocess boundary."""

    result = _read(tmp_path, _fake_environment(tmp_path, json.dumps(_payload(verdict))))
    assert result.verdict == verdict
    assert result.profile == PROFILE


@pytest.mark.parametrize(
    "name, output, exit_code, timeout",
    [
        ("unknown-schema", json.dumps({**_payload(), "schema": "tpen.profile-check/v2"}), 0, None),
        ("unknown-refusal-code", json.dumps({**_payload("refused"), "refusal": {"code": "new", "reason": "x"}}), 0, None),
        ("malformed-types", json.dumps({**_payload(), "state_owners": {"model": True}}), 0, None),
        ("malformed-shape", json.dumps([_payload()]), 0, None),
        ("duplicate-top-level", json.dumps(_payload()).replace(
            '"schema": "tpen.profile-check/v1"',
            '"schema": "tpen.profile-check/v2", "schema": "tpen.profile-check/v1"',
            1,
        ), 0, None),
        ("duplicate-nested", json.dumps(_payload("refused")).replace('"code": "unsupported-callback"', '"code":"new","code":"unsupported-callback"'), 0, None),
        ("extra-json-noise", json.dumps(_payload()) + "\nnoise\n", 0, None),
        ("invalid-utf8", b"\xff", 0, None),
        ("empty", "", 0, None),
        ("nonzero", json.dumps(_payload()), 3, None),
        ("timed-out", "import-me-not", 0, 0.01),
    ],
)
def test_r1_t2_fail_closed_transport_and_wire_controls(
    tmp_path: Path,
    name: str,
    output: bytes | str,
    exit_code: int,
    timeout: float | None,
) -> None:
    """GREEN: malformed, noisy, undecodable, and incomplete output is unavailable."""

    if name == "timed-out":
        package = tmp_path / "tpen" / "checkpoint"
        package.mkdir(parents=True)
        (tmp_path / "tpen" / "__init__.py").write_text("", encoding="utf-8")
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "profiles.py").write_text("import time; time.sleep(2)\n", encoding="utf-8")
        environment = PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(tmp_path)})
    else:
        environment = _fake_environment(tmp_path, output, exit_code=exit_code)
    with pytest.raises(ProfileUnavailable):
        _read(tmp_path, environment, timeout=timeout)


def test_r1_t3_refusal_mapping_is_immutable_on_refused_record(tmp_path: Path) -> None:
    """GREEN: refusal immutability is exercised on an actual refused record."""

    result = _read(tmp_path, _fake_environment(tmp_path, json.dumps(_payload("refused"))))
    assert result.refusal is not None
    with pytest.raises(TypeError):
        result.refusal["code"] = "unknown"  # type: ignore[index]


@pytest.mark.parametrize("config_text, expected_verdict, expected_refusal_code", [
    ("{}\n", "supported", None),
    ("callbacks:\n  - _target_: tpen.callback.NotCertified\n", "refused", "unsupported-callback"),
    (
        "callbacks:\n"
        "  - _target_: tpen.callback.Checkpoint\n"
        "    schedule:\n"
        "      _target_: tpen.callback.cadence.StepCadenceGate\n"
        "      cadence:\n"
        "        _target_: tpen.callback.cadence.StepCadence\n"
        "        max_calls: 1\n",
        "refused",
        "unsupported-callback-cadence",
    ),
])
def test_r1_t4_real_core_profile_cli_and_refusal_codes(
    tmp_path: Path,
    config_text: str,
    expected_verdict: str,
    expected_refusal_code: str | None,
) -> None:
    """GREEN: the pinned core CLI emits the executor-consumed capability record."""

    python = os.environ["P5B_REVIEW_SCIENCE_PYTHON"]
    repo_root = str(Path(__file__).resolve().parents[2])
    config = tmp_path / "profile.yaml"
    config.write_text(config_text, encoding="utf-8")
    environment = PreparedPythonEnvironment(python, env={"PYTHONPATH": repo_root})
    result = CoreProfileReference(str(config), PROFILE).read(
        SubprocessWorker(),
        environment,
        cwd=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        timeout=10.0,
    )

    core_refusal_codes = {
        "unknown-profile",
        "unsupported-callback",
        "unsupported-callback-cadence",
        "unsupported-logger",
        "malformed-config",
    }
    assert REFUSAL_CODES == core_refusal_codes
    assert result.verdict == expected_verdict
    assert result.profile == PROFILE
    if expected_verdict == "refused":
        assert result.refusal is not None
        assert result.refusal["code"] in REFUSAL_CODES
        assert result.refusal["code"] == expected_refusal_code

