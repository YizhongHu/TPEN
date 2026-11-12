"""Read-only inspection of a TPEN core run's durable evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tpen.checkpoint import (
    CheckpointRef,
    IncompletePublicationRecordError,
    PUBLICATION_CATALOG_FILENAME,
    list_complete_checkpoints,
    read_publications,
    resolve_checkpoint_dir,
)
from tpen.checkpoint.manifest import CheckpointManifest
from tpen.checkpoint.schema import validate_manifest_schema
from tpen.checkpoint.hashing import RESOLVED_CONFIG_IDENTITY_SCHEMA, resolved_config_identity
from tpen.evaluation.trajectory_records import (
    TRAJECTORY_RECORD_INSPECTION_SCHEMA,
    TamperedTrajectoryRecordError,
    TrajectoryPhysicalContext,
    TrajectoryRecordInspection,
    UnsupportedTrajectoryRecordSchemaError,
    inspect_trajectory_record,
)

RUN_EVIDENCE_SCHEMA = "tpen.run-evidence/v1"
RUN_EVIDENCE_SCHEMA_V2 = "tpen.run-evidence/v2"
_EVIDENCE_KEYS = {"schema", "run_dir", "verdict", "run_status", "checkpoint", "refusal"}
_EVIDENCE_V2_KEYS = _EVIDENCE_KEYS | {"resolved_config_identity"}
_CHECKPOINT_KEYS = {"checkpoint_dir", "schema_version", "next_iteration", "completed_updates"}
_REFUSAL_KEYS = {"code", "reason"}


@dataclass(frozen=True, slots=True)
class RunEvidence:
    """Versioned, serializable facts observed in one core run directory.

    Parameters
    ----------
    schema : str
        Evidence schema identifier.
    run_dir : str
        Run directory inspected.
    verdict : str
        One of ``completed``, ``incomplete``, ``failed``, or ``refused``.
    run_status : str or None
        The status value written by core, when a status artifact was readable.
    checkpoint : dict or None
        Selected checkpoint facts, including both independent progress counters.
    refusal : dict or None
        Stable refusal code and human-readable reason, when inspection refused.
    """

    schema: str
    run_dir: str
    verdict: str
    run_status: str | None
    checkpoint: dict[str, Any] | None
    refusal: dict[str, str] | None
    resolved_config_identity: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.schema not in {RUN_EVIDENCE_SCHEMA, RUN_EVIDENCE_SCHEMA_V2}:
            raise ValueError(f"unsupported run-evidence schema {self.schema!r}")
        if self.verdict not in {"completed", "incomplete", "failed", "refused"}:
            raise ValueError(f"unsupported run-evidence verdict {self.verdict!r}")
        if not isinstance(self.run_dir, str):
            raise TypeError("run_dir must be a string")
        if self.run_status is not None and not isinstance(self.run_status, str):
            raise TypeError("run_status must be a string or null")
        if self.verdict == "completed":
            if self.checkpoint is None or self.refusal is not None:
                raise ValueError("completed evidence requires checkpoint and no refusal")
        elif self.verdict == "refused":
            if self.refusal is None or self.checkpoint is not None:
                raise ValueError("refused evidence requires refusal and no checkpoint")
        elif self.refusal is not None:
            raise ValueError(f"{self.verdict} evidence cannot contain a refusal")
        if self.resolved_config_identity is not None:
            if set(self.resolved_config_identity) != {"schema", "sha256"}:
                raise ValueError("resolved config identity has an invalid key set")
            if self.resolved_config_identity["schema"] != RESOLVED_CONFIG_IDENTITY_SCHEMA:
                raise ValueError("resolved config identity has an unknown schema")
            digest = self.resolved_config_identity["sha256"]
            if type(digest) is not str or len(digest) != 64 or digest != digest.lower() or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("resolved config identity has an invalid sha256")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy of this evidence."""

        data = {
            "schema": self.schema,
            "run_dir": self.run_dir,
            "verdict": self.verdict,
            "run_status": self.run_status,
            "checkpoint": None if self.checkpoint is None else dict(self.checkpoint),
            "refusal": None if self.refusal is None else dict(self.refusal),
        }
        if self.schema == RUN_EVIDENCE_SCHEMA_V2:
            data["resolved_config_identity"] = (
                None if self.resolved_config_identity is None else dict(self.resolved_config_identity)
            )
        return data

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunEvidence":
        """Construct evidence while enforcing the v1 object shape."""

        if not isinstance(data, Mapping):
            raise TypeError("run evidence must be a mapping")
        schema = data.get("schema")
        expected = _EVIDENCE_KEYS if schema == RUN_EVIDENCE_SCHEMA else _EVIDENCE_V2_KEYS
        if schema not in {RUN_EVIDENCE_SCHEMA, RUN_EVIDENCE_SCHEMA_V2} or set(data) != expected:
            missing = sorted(expected - set(data))
            extra = sorted(set(data) - expected)
            raise ValueError(f"run evidence keys differ: missing={missing}, extra={extra}")
        if schema == RUN_EVIDENCE_SCHEMA_V2 and data.get("resolved_config_identity") is None:
            raise ValueError("v2 run evidence requires a non-null resolved_config_identity")
        checkpoint = data["checkpoint"]
        if checkpoint is not None:
            if not isinstance(checkpoint, Mapping) or set(checkpoint) != _CHECKPOINT_KEYS:
                raise ValueError("checkpoint evidence has an invalid key set")
            checkpoint = dict(checkpoint)
        refusal = data["refusal"]
        if refusal is not None:
            if not isinstance(refusal, Mapping) or set(refusal) != _REFUSAL_KEYS:
                raise ValueError("refusal evidence has an invalid key set")
            refusal = {str(key): str(value) for key, value in refusal.items()}
        run_status = data["run_status"]
        if run_status is not None and not isinstance(run_status, str):
            raise TypeError("run_status must be a string or null")
        identity = data.get("resolved_config_identity")
        if identity is not None and not isinstance(identity, Mapping):
            raise TypeError("resolved_config_identity must be a mapping or null")
        return cls(
            schema=data["schema"],
            run_dir=data["run_dir"],
            verdict=data["verdict"],
            run_status=run_status,
            checkpoint=checkpoint,
            refusal=refusal,
            resolved_config_identity=None if identity is None else dict(identity),
        )


class _InspectionRefusal(ValueError):
    """Internal typed refusal used to keep the public result total."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


class _TrainResumeModeGate(ValueError):
    """Internal marker for a valid artifact excluded by the train-resume gate."""


def inspect_run(path: str | Path) -> RunEvidence:
    """Inspect a core run without modifying any file.

    Parameters
    ----------
    path : str or pathlib.Path
        Run directory containing the durable status and checkpoint artifacts.

    Returns
    -------
    RunEvidence
        A completed, failed, incomplete, or refused evidence record. Expected
        artifact problems are represented as ``verdict="refused"`` rather than
        escaping as an exception. A committed train-resume checkpoint is
        reported for any status, including failed and incomplete runs. Stable
        refusal codes are ``missing-run-dir``, ``missing-status``,
        ``multiple-status``, ``status-io-error``, ``malformed-status``,
        ``missing-checkpoint``, ``incomplete-checkpoint`` (completed runs
        whose checkpoint is uncommitted), ``no-train-resume-checkpoint``
        (completed runs with only committed model-only checkpoints),
        ``checkpoint-io-error``, ``corrupt-checkpoint``,
        ``corrupt-publication-catalog``, and ``torn-publication-catalog``.
        In-flight or uncommitted checkpoint directories do not refuse
        non-completed runs; they yield a null checkpoint and an ``incomplete``
        or ``failed`` verdict.
    """

    run_dir = Path(path).resolve()
    run_status: str | None = None
    try:
        if not run_dir.is_dir():
            raise _InspectionRefusal("missing-run-dir", f"run directory not found: {run_dir}")
        status_path, status = _read_status(run_dir)
        run_dir = status_path.parent.resolve()
        identity = _read_run_identity(run_dir)
        raw_status: object = status.get("status")
        if raw_status is not None and not isinstance(raw_status, str):
            raise _InspectionRefusal("malformed-status", f"status value is not a string: {status_path}")
        run_status = raw_status
        checkpoint: dict[str, Any] | None = None
        try:
            checkpoint = _read_checkpoint(run_dir)
        except _InspectionRefusal as exc:
            if run_status != "completed" and exc.code in {
                "missing-checkpoint",
                "incomplete-checkpoint",
                "no-train-resume-checkpoint",
            }:
                checkpoint = None
            elif exc.code != "missing-checkpoint":
                raise
        if run_status == "failed":
            return RunEvidence(_evidence_schema(identity), str(run_dir), "failed", run_status, checkpoint, None, identity)
        if run_status != "completed":
            return RunEvidence(_evidence_schema(identity), str(run_dir), "incomplete", run_status, checkpoint, None, identity)
        if checkpoint is None:
            raise _InspectionRefusal("missing-checkpoint", f"no committed checkpoint under {run_dir / 'checkpoints'}")
        return RunEvidence(_evidence_schema(identity), str(run_dir), "completed", run_status, checkpoint, None, identity)
    except _InspectionRefusal as exc:
        return RunEvidence(
            _evidence_schema(None),
            str(run_dir),
            "refused",
            run_status,
            None,
            {"code": exc.code, "reason": exc.reason},
        )


def _evidence_schema(identity: dict[str, str] | None) -> str:
    """Select v2 only for runs that carry the public identity sidecar."""

    return RUN_EVIDENCE_SCHEMA_V2 if identity is not None else RUN_EVIDENCE_SCHEMA


def _read_run_identity(run_dir: Path) -> dict[str, str] | None:
    """Read and validate the identity emitted at the run-start boundary."""

    path = run_dir / "run_start.json"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeError, ValueError) as exc:
        raise _InspectionRefusal("malformed-run-identity", f"cannot read {path}: {exc}") from exc
    identity = data.get("resolved_config_identity") if isinstance(data, Mapping) else None
    if identity is None:
        return None
    try:
        candidate = RunEvidence(
            RUN_EVIDENCE_SCHEMA_V2, str(run_dir), "incomplete", None, None, None, dict(identity)
        ).resolved_config_identity
    except (TypeError, ValueError) as exc:
        raise _InspectionRefusal("malformed-run-identity", f"invalid identity in {path}: {exc}") from exc
    return candidate


def _read_status(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    status_paths = tuple(run_dir.rglob("status.json"))
    if not status_paths:
        raise _InspectionRefusal("missing-status", f"no status.json under {run_dir}")
    if len(status_paths) != 1:
        raise _InspectionRefusal(
            "multiple-status", f"expected one status.json under {run_dir}, found {len(status_paths)}"
        )
    try:
        with status_paths[0].open("r", encoding="utf-8") as handle:
            status = json.load(handle)
    except OSError as exc:
        raise _InspectionRefusal("status-io-error", f"cannot read {status_paths[0]}: {exc}") from exc
    except ValueError as exc:
        raise _InspectionRefusal("malformed-status", f"cannot read {status_paths[0]}: {exc}") from exc
    if not isinstance(status, dict):
        raise _InspectionRefusal("malformed-status", f"status artifact is not an object: {status_paths[0]}")
    return status_paths[0], status


def _read_checkpoint(run_dir: Path) -> dict[str, Any]:
    root = run_dir / "checkpoints"
    if not root.is_dir():
        raise _InspectionRefusal("missing-checkpoint", f"checkpoint root not found: {root}")

    catalog_path = root / PUBLICATION_CATALOG_FILENAME
    try:
        publications = read_publications(catalog_path) if catalog_path.is_file() else ()
    except OSError as exc:
        raise _InspectionRefusal("checkpoint-io-error", f"cannot read {catalog_path}: {exc}") from exc
    except IncompletePublicationRecordError as exc:
        raise _InspectionRefusal("torn-publication-catalog", f"cannot read {catalog_path}: {exc}") from exc
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise _InspectionRefusal("corrupt-publication-catalog", f"cannot read {catalog_path}: {exc}") from exc

    refs: list[CheckpointRef] = []
    for ref in publications:
        try:
            validated_ref = ref.validate()
            _read_train_resume_manifest(validated_ref.checkpoint_dir / "manifest.json")
            refs.append(validated_ref)
        except _TrainResumeModeGate:
            continue
        except OSError as exc:
            raise _InspectionRefusal("checkpoint-io-error", f"published checkpoint cannot be read: {exc}") from exc
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise _InspectionRefusal("corrupt-checkpoint", f"published checkpoint is invalid: {exc}") from exc

    complete_dirs = list_complete_checkpoints(root)
    for directory in complete_dirs:
        try:
            ref = CheckpointRef.from_directory(directory)
            manifest = _read_train_resume_manifest(directory / "manifest.json")
            declared_model_hash = manifest.hashes.get("model_sha256")
            if declared_model_hash is not None and declared_model_hash != ref.model_sha256:
                raise ValueError("manifest model_sha256 does not match model.pt")
            ref.validate()
            refs.append(ref)
        except _TrainResumeModeGate:
            continue
        except OSError as exc:
            raise _InspectionRefusal("checkpoint-io-error", f"checkpoint cannot be read: {exc}") from exc
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise _InspectionRefusal("corrupt-checkpoint", f"checkpoint is invalid: {exc}") from exc

    if not refs:
        if any(candidate.name.endswith(".tmp") for candidate in root.glob("step_*")):
            raise _InspectionRefusal("incomplete-checkpoint", f"no committed checkpoint under {root}")
        if any(
            candidate.is_dir() and candidate not in complete_dirs
            for candidate in root.glob("step_*")
        ):
            raise _InspectionRefusal("incomplete-checkpoint", f"no committed checkpoint under {root}")
        if complete_dirs or publications:
            raise _InspectionRefusal(
                "no-train-resume-checkpoint",
                f"committed published checkpoints under {root} do not support train resume",
            )
        try:
            resolve_checkpoint_dir(root)
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            raise _InspectionRefusal("missing-checkpoint", f"no committed checkpoint under {root}: {exc}") from exc
        raise _InspectionRefusal("missing-checkpoint", f"no committed checkpoint under {root}")

    ref = max(refs, key=lambda candidate: candidate.next_iteration)
    return {
        "checkpoint_dir": str(ref.checkpoint_dir.resolve()),
        "schema_version": ref.schema_version,
        "next_iteration": ref.next_iteration,
        "completed_updates": ref.completed_updates,
    }


def _read_train_resume_manifest(path: Path) -> CheckpointManifest:
    """Read a manifest, marking only a valid non-train-resume artifact."""

    with path.open("r", encoding="utf-8") as handle:
        manifest = CheckpointManifest.from_mapping(json.load(handle))
    validate_manifest_schema(manifest, mode="model_only", path=path)
    try:
        validate_manifest_schema(manifest, mode="train_resume", path=path)
    except ValueError as exc:
        raise _TrainResumeModeGate(str(exc)) from exc
    return manifest


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    evidence = inspect_run(args.run_dir)
    print(json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    raise SystemExit(_main())


__all__ = [
    "RUN_EVIDENCE_SCHEMA",
    "RUN_EVIDENCE_SCHEMA_V2",
    "TRAJECTORY_RECORD_INSPECTION_SCHEMA",
    "RunEvidence",
    "TamperedTrajectoryRecordError",
    "TrajectoryPhysicalContext",
    "TrajectoryRecordInspection",
    "UnsupportedTrajectoryRecordSchemaError",
    "inspect_run",
    "inspect_trajectory_record",
    "resolved_config_identity",
]
