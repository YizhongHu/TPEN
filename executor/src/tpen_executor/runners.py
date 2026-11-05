"""Public command and evidence clients for configured TPEN child processes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from ._codec import canonical_bytes, decode_json
from .environments import PreparedPythonEnvironment
from .workers import SubprocessWorker

RUN_EVIDENCE_SCHEMA = "tpen.run-evidence/v1"
RUN_SHIM = "import sys; from tpen.run import main; raise SystemExit(main(sys.argv[1:]))"
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
    """Strict, immutable executor-owned mirror of core run evidence.

    Parameters
    ----------
    schema, run_dir, verdict : str
        Validated public evidence fields.
    run_status : str or None
        Core-reported run status.
    checkpoint, refusal : mapping or None
        Validated nested evidence, copied into read-only mappings.
    """

    schema: str
    run_dir: str
    verdict: str
    run_status: str | None
    checkpoint: Mapping[str, Any] | None
    refusal: Mapping[str, str] | None

    SCHEMA: ClassVar[str] = RUN_EVIDENCE_SCHEMA
    VERDICTS: ClassVar[frozenset[str]] = frozenset({"completed", "incomplete", "failed", "refused"})
    REFUSAL_CODES: ClassVar[frozenset[str]] = REFUSAL_CODES

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", self._checkpoint(self.checkpoint))
        object.__setattr__(self, "refusal", self._refusal(self.refusal))

    @classmethod
    def from_mapping(cls, value: Any) -> "CoreRunEvidence":
        """Decode one strict public evidence mapping.

        Parameters
        ----------
        value : mapping
            JSON-shaped public evidence.

        Returns
        -------
        CoreRunEvidence
            A validated record with frozen nested mappings.

        Raises
        ------
        EvidenceUnavailable
            If the schema, keys, types, or verdict invariants are invalid.
        """

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
        checkpoint = value["checkpoint"]
        refusal = value["refusal"]
        if verdict == "refused" and refusal is None:
            raise EvidenceUnavailable("refused evidence requires refusal")
        if verdict == "refused" and checkpoint is not None:
            raise EvidenceUnavailable("refused evidence cannot contain checkpoint")
        if verdict == "completed" and checkpoint is None:
            raise EvidenceUnavailable("completed evidence requires checkpoint")
        if verdict != "refused" and refusal is not None:
            raise EvidenceUnavailable("non-refused evidence cannot contain refusal")
        return cls(value["schema"], value["run_dir"], verdict, run_status, checkpoint, refusal)

    def __hash__(self) -> int:
        """Hash the canonical JSON representation used by equality fields."""

        return hash(canonical_bytes({
            "schema": self.schema,
            "run_dir": self.run_dir,
            "verdict": self.verdict,
            "run_status": self.run_status,
            "checkpoint": self.checkpoint,
            "refusal": self.refusal,
        }))

    @staticmethod
    def _checkpoint(value: Any) -> MappingProxyType | None:
        if value is None:
            return None
        keys = {"checkpoint_dir", "schema_version", "next_iteration", "completed_updates"}
        if not isinstance(value, Mapping) or set(value) != keys:
            raise EvidenceUnavailable("checkpoint has invalid keys")
        if type(value["checkpoint_dir"]) is not str:
            raise EvidenceUnavailable("checkpoint text fields have invalid types")
        if type(value["schema_version"]) is not int:
            raise EvidenceUnavailable("checkpoint schema_version must be an integer")
        if type(value["next_iteration"]) is not int or type(value["completed_updates"]) is not int:
            raise EvidenceUnavailable("checkpoint counters have invalid types")
        if value["next_iteration"] < 0 or value["completed_updates"] < 0:
            raise EvidenceUnavailable("checkpoint counters must be nonnegative")
        return MappingProxyType(dict(value))

    @classmethod
    def _refusal(cls, value: Any) -> MappingProxyType | None:
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {"code", "reason"}:
            raise EvidenceUnavailable("refusal has invalid keys")
        if type(value["code"]) is not str or value["code"] not in cls.REFUSAL_CODES:
            raise EvidenceUnavailable("unknown refusal code")
        if type(value["reason"]) is not str:
            raise EvidenceUnavailable("refusal reason must be a string")
        return MappingProxyType({"code": value["code"], "reason": value["reason"]})


@dataclass(frozen=True, slots=True)
class TPENCommand:
    """Command specification for the public configured-run entrypoint.

    Parameters
    ----------
    config : str
        Absolute path to the Hydra configuration.
    overrides : sequence of str, optional
        Positional dot-list overrides passed unchanged to core.

    Raises
    ------
    TypeError
        If overrides is a scalar string/bytes value or contains non-strings.
    ValueError
        If config is not an absolute path.
    """

    config: str
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.config) is not str or not os.path.isabs(self.config):
            raise ValueError("config must be an absolute path")
        if isinstance(self.overrides, (str, bytes, bytearray)):
            raise TypeError("overrides must be a sequence of strings")
        if any(type(item) is not str for item in self.overrides):
            raise TypeError("overrides must be strings")
        object.__setattr__(self, "overrides", tuple(self.overrides))

    @property
    def args(self) -> tuple[str, ...]:
        """Return arguments following the selected Python interpreter."""

        return ("-c", RUN_SHIM, "--config", self.config, *self.overrides)


@dataclass(frozen=True, slots=True)
class TPENEvidence:
    """Read public run evidence through a child inspection process.

    Notes
    -----
    Inspection output is obtained only from the named stdout log.  The
    decoder rejects duplicate JSON members at every depth and never opens
    ``run_dir`` directly.
    """

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
        """Launch inspection and decode exactly one JSON object.

        Parameters
        ----------
        worker : SubprocessWorker
            Worker used to launch the selected science interpreter.
        environment : PreparedPythonEnvironment
            Prepared environment consumed by the worker.
        run_dir : str or os.PathLike
            Run directory passed to public inspection; it is not read here.
        cwd : str or os.PathLike or None, optional
            Child working directory.
        stdout_path, stderr_path : str or os.PathLike
            Required named logs.
        timeout : float or None, optional
            Timeout in seconds.

        Returns
        -------
        CoreRunEvidence
            Frozen, validated public evidence.

        Raises
        ------
        EvidenceUnavailable
            If transport, process, decoding, or evidence validation fails.

        Notes
        -----
        The worker applies the owner environment, then the descriptor overlay,
        then the setup script; setup is the last writer. The timeout covers the
        worker's complete process group.
        """

        try:
            outcome = worker.run(
                environment,
                ("-m", "tpen.inspection", os.fspath(run_dir)),
                cwd=cwd,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                timeout=timeout,
            )
        except OSError as exc:
            raise EvidenceUnavailable("inspection process could not be launched") from exc
        if outcome.timed_out or outcome.returncode != 0:
            raise EvidenceUnavailable("inspection process did not complete successfully")
        try:
            payload = Path(outcome.stdout_path).read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise EvidenceUnavailable("inspection stdout could not be read") from exc
        if not payload.strip():
            raise EvidenceUnavailable("inspection stdout is empty")
        try:
            decoded = decode_json(payload)
        except (TypeError, ValueError) as exc:
            raise EvidenceUnavailable("inspection stdout is not one JSON object") from exc
        return CoreRunEvidence.from_mapping(decoded)


__all__ = ["CoreRunEvidence", "EvidenceUnavailable", "REFUSAL_CODES", "RUN_EVIDENCE_SCHEMA", "TPENCommand", "TPENEvidence"]
