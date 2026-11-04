"""Public command and evidence clients for configured TPEN child processes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, ClassVar

from .environments import PreparedPythonEnvironment
from .workers import SubprocessWorker

RUN_EVIDENCE_SCHEMA = "tpen.run-evidence/v1"
REFUSAL_CODES = frozenset(
    {
        "missing-run-dir", "missing-status", "multiple-status", "malformed-status",
        "status-io-error", "incomplete-checkpoint", "corrupt-checkpoint",
        "checkpoint-io-error", "corrupt-publication-catalog", "torn-publication-catalog",
        "missing-checkpoint", "no-train-resume-checkpoint",
    }
)


class EvidenceUnavailable(RuntimeError):
    """Raised when a child cannot provide trustworthy public evidence."""


@dataclass(frozen=True, slots=True)
class CoreRunEvidence:
    """Strict executor-owned mirror of the public core evidence record."""

    schema: str
    run_dir: str
    verdict: str
    run_status: str | None
    checkpoint: dict[str, Any] | None
    refusal: dict[str, str] | None

    SCHEMA: ClassVar[str] = RUN_EVIDENCE_SCHEMA
    VERDICTS: ClassVar[frozenset[str]] = frozenset({"completed", "incomplete", "failed", "refused"})
    REFUSAL_CODES: ClassVar[frozenset[str]] = REFUSAL_CODES

    @classmethod
    def from_mapping(cls, value: Any) -> "CoreRunEvidence":
        """Decode one strict public evidence mapping."""

        if type(value) is not dict:
            raise EvidenceUnavailable("evidence is not a JSON object")
        expected = {"schema", "run_dir", "verdict", "run_status", "checkpoint", "refusal"}
        if set(value) != expected:
            raise EvidenceUnavailable("evidence has unexpected keys")
        if value["schema"] != cls.SCHEMA:
            raise EvidenceUnavailable("unknown evidence schema")
        if type(value["run_dir"]) is not str or not value["run_dir"]:
            raise EvidenceUnavailable("run_dir must be a non-empty string")
        verdict = value["verdict"]
        if type(verdict) is not str or verdict not in cls.VERDICTS:
            raise EvidenceUnavailable("unknown evidence verdict")
        run_status = value["run_status"]
        if run_status is not None and type(run_status) is not str:
            raise EvidenceUnavailable("run_status must be a string or null")
        checkpoint = cls._checkpoint(value["checkpoint"])
        refusal = cls._refusal(value["refusal"])
        if verdict == "refused" and refusal is None:
            raise EvidenceUnavailable("refused evidence requires refusal")
        if verdict == "refused" and checkpoint is not None:
            raise EvidenceUnavailable("refused evidence cannot contain checkpoint")
        if verdict == "completed" and checkpoint is None:
            raise EvidenceUnavailable("completed evidence requires checkpoint")
        if verdict != "refused" and refusal is not None:
            raise EvidenceUnavailable("non-refused evidence cannot contain refusal")
        return cls(value["schema"], value["run_dir"], verdict, run_status, checkpoint, refusal)

    @staticmethod
    def _checkpoint(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        keys = {"checkpoint_dir", "schema_version", "next_iteration", "completed_updates"}
        if type(value) is not dict or set(value) != keys:
            raise EvidenceUnavailable("checkpoint has invalid keys")
        if type(value["checkpoint_dir"]) is not str or type(value["schema_version"]) is not str:
            raise EvidenceUnavailable("checkpoint text fields have invalid types")
        if type(value["next_iteration"]) is not int or type(value["completed_updates"]) is not int:
            raise EvidenceUnavailable("checkpoint counters have invalid types")
        if value["next_iteration"] < 0 or value["completed_updates"] < 0:
            raise EvidenceUnavailable("checkpoint counters must be nonnegative")
        return dict(value)

    @classmethod
    def _refusal(cls, value: Any) -> dict[str, str] | None:
        if value is None:
            return None
        if type(value) is not dict or set(value) != {"code", "reason"}:
            raise EvidenceUnavailable("refusal has invalid keys")
        if type(value["code"]) is not str or value["code"] not in cls.REFUSAL_CODES:
            raise EvidenceUnavailable("unknown refusal code")
        if type(value["reason"]) is not str:
            raise EvidenceUnavailable("refusal reason must be a string")
        return {"code": value["code"], "reason": value["reason"]}


@dataclass(frozen=True, slots=True)
class TPENCommand:
    """Command specification for the public configured-run entrypoint."""

    config: str
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.config) is not str or not os.path.isabs(self.config):
            raise ValueError("config must be an absolute path")
        if any(type(item) is not str for item in self.overrides):
            raise TypeError("overrides must be strings")
        object.__setattr__(self, "overrides", tuple(self.overrides))

    @property
    def args(self) -> tuple[str, ...]:
        """Return arguments following the selected Python interpreter."""

        return ("-m", "tpen.run", "--config", self.config, *self.overrides)


@dataclass(frozen=True, slots=True)
class TPENEvidence:
    """Read public run evidence through a child inspection process."""

    def read(
        self,
        worker: SubprocessWorker,
        environment: PreparedPythonEnvironment,
        run_dir: str | os.PathLike[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        stdout_path: str | os.PathLike[str],
        stderr_path: str | os.PathLike[str],
        timeout: float | None = None,
    ) -> CoreRunEvidence:
        """Launch ``tpen.inspection`` and decode exactly one JSON object."""

        outcome = worker.run(
            environment,
            ("-m", "tpen.inspection", os.fspath(run_dir)),
            cwd=cwd,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=timeout,
        )
        if outcome.timed_out or outcome.returncode != 0:
            raise EvidenceUnavailable("inspection process did not complete successfully")
        try:
            payload = Path(outcome.stdout_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise EvidenceUnavailable("inspection stdout could not be read") from exc
        if not payload.strip():
            raise EvidenceUnavailable("inspection stdout is empty")
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise EvidenceUnavailable("inspection stdout is not one JSON object") from exc
        return CoreRunEvidence.from_mapping(decoded)


__all__ = ["CoreRunEvidence", "EvidenceUnavailable", "REFUSAL_CODES", "RUN_EVIDENCE_SCHEMA", "TPENCommand", "TPENEvidence"]
