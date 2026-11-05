"""Executor tests for the core profile capability boundary."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.recovery import CoreProfileReference, ProfileCheck, ProfileUnavailable
from tpen_executor.workers import SubprocessWorker


def _fake_core(tmp_path: Path, payload: object) -> PreparedPythonEnvironment:
    package = tmp_path / "tpen" / "checkpoint"
    package.mkdir(parents=True)
    (tmp_path / "tpen" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "profiles.py").write_text(
        "import json, os, sys\n"
        "assert sys.argv[1:] == json.loads(os.environ['PROFILE_EXPECTED_ARGS'])\n"
        "print(os.environ['PROFILE_PAYLOAD'])\n",
        encoding="utf-8",
    )
    return PreparedPythonEnvironment(
        sys.executable,
        env={
            "PYTHONPATH": str(tmp_path),
            "PROFILE_PAYLOAD": json.dumps(payload),
            "PROFILE_EXPECTED_ARGS": json.dumps(
                [str(tmp_path / "config.yaml"), "--profile", "tpen.native-recovery/v1"]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def _payload(verdict: str = "supported") -> dict[str, object]:
    return {
        "schema": "tpen.profile-check/v1",
        "profile": "tpen.native-recovery/v1",
        "verdict": verdict,
        "state_owners": ["model", "optimizer"],
        "refusal": None if verdict == "supported" else {"code": "unsupported-callback", "reason": "refused"},
    }


@pytest.mark.parametrize("verdict", ["supported", "refused"])
def test_core_profile_reference_decodes_supported_and_refused(tmp_path: Path, verdict: str) -> None:
    reference = CoreProfileReference(str(tmp_path / "config.yaml"), "tpen.native-recovery/v1")
    result = reference.read(
        SubprocessWorker(),
        _fake_core(tmp_path, _payload(verdict)),
        cwd=tmp_path,
        stdout_path=tmp_path / "out",
        stderr_path=tmp_path / "err",
    )
    assert result.verdict == verdict


@pytest.mark.parametrize(
    "payload",
    [
        {**_payload(), "schema": "tpen.profile-check/v2"},
        {**_payload(), "refusal": {"code": "not-pinned", "reason": "bad"}, "verdict": "refused"},
        {**_payload(), "state_owners": "model"},
        {"not": "a profile check"},
    ],
)
def test_core_profile_reference_rejects_malformed_or_unpinned_output(tmp_path: Path, payload: object) -> None:
    reference = CoreProfileReference(str(tmp_path / "config.yaml"), "tpen.native-recovery/v1")
    with pytest.raises(ProfileUnavailable):
        reference.read(
            SubprocessWorker(),
            _fake_core(tmp_path, payload),
            cwd=tmp_path,
            stdout_path=tmp_path / "out",
            stderr_path=tmp_path / "err",
        )


def test_profile_check_hash_matches_equality_and_nested_state_is_frozen() -> None:
    first = ProfileCheck.from_mapping(_payload("refused"))
    second = ProfileCheck.from_mapping(dict(reversed(tuple(first.to_dict().items()))))
    assert first == second
    assert hash(first) == hash(second)
    assert first.refusal is not None
    with pytest.raises(TypeError):
        first.refusal["code"] = "bad"  # type: ignore[index]


@pytest.mark.parametrize("config,profile", [("relative.yaml", "tpen.native-recovery/v1"), ("/tmp/config.yaml", "")])
def test_core_profile_reference_rejects_invalid_identity(config: str, profile: str) -> None:
    with pytest.raises(ValueError):
        CoreProfileReference(config, profile)
