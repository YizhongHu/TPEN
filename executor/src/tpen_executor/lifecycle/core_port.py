"""Process boundary for core-owned admission evaluation.

The control interpreter only validates the closed request and decodes the
closed result.  All imports of :mod:`tpen` occur in the prepared science
interpreter.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..recovery import (
    CoreAdmissionUnavailable,
    MappingCoreAdmissionPort,
    MemberRecoveryInputV2,
)
from ..workers import SubprocessWorker
from ..environments import PreparedPythonEnvironment


_REQUEST_SCHEMA = "tpen-executor/core-admission-request-v1"
_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "task_id",
        "lineage_id",
        "core_run_dir",
        "science_input_digest",
        "successor_request",
        "unresolved_config",
        "profile_name",
    }
)


@dataclass(frozen=True, slots=True)
class CoreAdmissionProcessRequest:
    """Closed inputs for one science-process admission call."""

    task_id: str
    lineage_id: str
    core_run_dir: str | None
    science_input_digest: str
    successor_request: Mapping[str, Any] | None
    unresolved_config: Mapping[str, Any]
    profile_name: str
    environment: PreparedPythonEnvironment

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            raise TypeError("task_id must be a non-empty string")
        if type(self.lineage_id) is not str or not self.lineage_id:
            raise TypeError("lineage_id must be a non-empty string")
        if self.core_run_dir is not None:
            if type(self.core_run_dir) is not str or not os.path.isabs(self.core_run_dir):
                raise ValueError("core_run_dir must be absolute or None")
        if type(self.science_input_digest) is not str or len(self.science_input_digest) != 64:
            raise ValueError("science_input_digest must be a SHA-256 hex digest")
        try:
            int(self.science_input_digest, 16)
        except ValueError as exc:
            raise ValueError("science_input_digest must be a SHA-256 hex digest") from exc
        if self.successor_request is not None and not isinstance(self.successor_request, Mapping):
            raise TypeError("successor_request must be a mapping or None")
        if not isinstance(self.unresolved_config, Mapping):
            raise TypeError("unresolved_config must be a mapping")
        if type(self.profile_name) is not str or not self.profile_name:
            raise TypeError("profile_name must be a non-empty string")
        if not isinstance(self.environment, PreparedPythonEnvironment):
            raise TypeError("environment must be PreparedPythonEnvironment")

    def to_dict(self) -> dict[str, Any]:
        """Return only the context inputs; the environment is launch-local."""

        return {
            "schema": _REQUEST_SCHEMA,
            "task_id": self.task_id,
            "lineage_id": self.lineage_id,
            "core_run_dir": self.core_run_dir,
            "science_input_digest": self.science_input_digest,
            "successor_request": None if self.successor_request is None else dict(self.successor_request),
            "unresolved_config": dict(self.unresolved_config),
            "profile_name": self.profile_name,
        }


_CHILD = r'''
import json
import sys

from tpen.execution.core_admission import CoreAdmissionContextV1, evaluate_core_admission

request = json.loads(sys.argv[1])
context = CoreAdmissionContextV1(
    task_id=request["task_id"],
    lineage_id=request["lineage_id"],
    core_run_dir=request["core_run_dir"],
    science_input_digest=request["science_input_digest"],
    successor_request=request["successor_request"],
    unresolved_config=request["unresolved_config"],
    profile_name=request["profile_name"],
)
print(json.dumps(evaluate_core_admission(context), sort_keys=True, separators=(",", ":")))
'''


class ScienceProcessCoreAdmissionPort:
    """Evaluate core admission in a prepared interpreter and decode strictly."""

    def __init__(self, worker: SubprocessWorker | None = None) -> None:
        self._worker = SubprocessWorker() if worker is None else worker

    def admit(
        self,
        request: CoreAdmissionProcessRequest,
        member: MemberRecoveryInputV2,
        *,
        stdout_path: str | os.PathLike[str],
        stderr_path: str | os.PathLike[str],
        cwd: str | os.PathLike[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Return a decoded verdict, ``None`` for a missing leaf, or park error."""

        if not isinstance(request, CoreAdmissionProcessRequest):
            raise TypeError("request must be CoreAdmissionProcessRequest")
        if not isinstance(member, MemberRecoveryInputV2):
            raise TypeError("member must be MemberRecoveryInputV2")
        if request.task_id != member.task_id or request.lineage_id != member.lineage_id:
            raise CoreAdmissionUnavailable("core admission context crosses member identity")
        if request.successor_request is not None:
            predecessor = request.successor_request.get("predecessor")
            successor = request.successor_request.get("successor")
            if not isinstance(predecessor, Mapping) or not isinstance(successor, Mapping):
                raise CoreAdmissionUnavailable("core admission request identity is malformed")
            if predecessor.get("task_id") != request.task_id or successor.get("task_id") != request.task_id:
                raise CoreAdmissionUnavailable("core admission request crosses task identity")
            if request.successor_request.get("science_input_digest") != request.science_input_digest:
                raise CoreAdmissionUnavailable("core admission request crosses science digest")

        try:
            outcome = self._worker.run(
                request.environment,
                ("-c", _CHILD, json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))),
                cwd=cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout=timeout,
            )
        except Exception as exc:
            raise CoreAdmissionUnavailable("core admission process could not be launched") from exc
        if outcome.timed_out or outcome.returncode != 0:
            raise CoreAdmissionUnavailable("core admission process did not complete successfully")
        try:
            payload = Path(outcome.stdout_path).read_text(encoding="utf-8")
            decoded = json.loads(payload)
        except Exception as exc:
            raise CoreAdmissionUnavailable("core admission process output is invalid") from exc

        decoder = MappingCoreAdmissionPort(lambda _member: decoded)
        try:
            return decoder.admit(member)
        except CoreAdmissionUnavailable:
            raise
        except Exception as exc:
            raise CoreAdmissionUnavailable("core admission output unavailable") from exc


CoreAdmissionProcessPort = ScienceProcessCoreAdmissionPort

__all__ = ["CoreAdmissionProcessPort", "CoreAdmissionProcessRequest", "ScienceProcessCoreAdmissionPort"]
