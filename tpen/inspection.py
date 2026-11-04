"""Read-only inspection of a TPEN core run's durable evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tpen.checkpoint import (
    CheckpointRef,
    PUBLICATION_CATALOG_FILENAME,
    list_complete_checkpoints,
    read_publications,
    resolve_checkpoint_dir,
)
from tpen.checkpoint.schema import read_manifest

RUN_EVIDENCE_SCHEMA = "tpen.run-evidence/v1"
_EVIDENCE_KEYS = {"schema", "run_dir", "verdict", "run_status", "checkpoint", "refusal"}
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

    def __post_init__(self) -> None:
        if self.schema != RUN_EVIDENCE_SCHEMA:
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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe copy of this evidence."""

        return {
            "schema": self.schema,
            "run_dir": self.run_dir,
            "verdict": self.verdict,
            "run_status": self.run_status,
            "checkpoint": None if self.checkpoint is None else dict(self.checkpoint),
            "refusal": None if self.refusal is None else dict(self.refusal),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RunEvidence":
        """Construct evidence while enforcing the v1 object shape."""

        if not isinstance(data, Mapping):
            raise TypeError("run evidence must be a mapping")
        if set(data) != _EVIDENCE_KEYS:
            missing = sorted(_EVIDENCE_KEYS - set(data))
            extra = sorted(set(data) - _EVIDENCE_KEYS)
            raise ValueError(f"run evidence keys differ: missing={missing}, extra={extra}")
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
        return cls(
            schema=data["schema"],
            run_dir=data["run_dir"],
            verdict=data["verdict"],
            run_status=run_status,
            checkpoint=checkpoint,
            refusal=refusal,
        )


class _InspectionRefusal(ValueError):
    """Internal typed refusal used to keep the public result total."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


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
        reported for any status, including failed and incomplete runs; a
        corrupt checkpoint refuses inspection regardless of run status.
    """

    run_dir = Path(path).resolve()
    run_status: str | None = None
    try:
        if not run_dir.is_dir():
            raise _InspectionRefusal("missing-run-dir", f"run directory not found: {run_dir}")
        status_path, status = _read_status(run_dir)
        run_dir = status_path.parent.resolve()
        run_status = status.get("status")
        if run_status is not None and not isinstance(run_status, str):
            raise _InspectionRefusal("malformed-status", f"status value is not a string: {status_path}")
        checkpoint: dict[str, Any] | None = None
        try:
            checkpoint = _read_checkpoint(run_dir)
        except _InspectionRefusal as exc:
            if exc.code != "missing-checkpoint":
                raise
        if run_status == "failed":
            return RunEvidence(RUN_EVIDENCE_SCHEMA, str(run_dir), "failed", run_status, checkpoint, None)
        if run_status != "completed":
            return RunEvidence(RUN_EVIDENCE_SCHEMA, str(run_dir), "incomplete", run_status, checkpoint, None)
        if checkpoint is None:
            raise _InspectionRefusal("missing-checkpoint", f"no committed checkpoint under {run_dir / 'checkpoints'}")
        return RunEvidence(RUN_EVIDENCE_SCHEMA, str(run_dir), "completed", run_status, checkpoint, None)
    except _InspectionRefusal as exc:
        return RunEvidence(
            RUN_EVIDENCE_SCHEMA,
            str(run_dir),
            "refused",
            run_status,
            None,
            {"code": exc.code, "reason": exc.reason},
        )


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
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise _InspectionRefusal("corrupt-publication-catalog", f"cannot read {catalog_path}: {exc}") from exc

    refs: list[CheckpointRef] = []
    for ref in publications:
        try:
            read_manifest(ref.checkpoint_dir / "manifest.json", mode="train_resume")
            refs.append(ref.validate())
        except OSError as exc:
            raise _InspectionRefusal("checkpoint-io-error", f"published checkpoint cannot be read: {exc}") from exc
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise _InspectionRefusal("corrupt-checkpoint", f"published checkpoint is invalid: {exc}") from exc

    complete_dirs = list_complete_checkpoints(root)
    for directory in complete_dirs:
        try:
            ref = CheckpointRef.from_directory(directory)
            manifest = read_manifest(directory / "manifest.json", mode="train_resume")
            declared_model_hash = manifest.hashes.get("model_sha256")
            if declared_model_hash is not None and declared_model_hash != ref.model_sha256:
                raise ValueError("manifest model_sha256 does not match model.pt")
            ref.validate()
            refs.append(ref)
        except OSError as exc:
            raise _InspectionRefusal("checkpoint-io-error", f"checkpoint cannot be read: {exc}") from exc
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise _InspectionRefusal("corrupt-checkpoint", f"checkpoint is invalid: {exc}") from exc

    if not refs:
        if any(candidate.name.endswith(".tmp") for candidate in root.glob("step_*")):
            raise _InspectionRefusal("incomplete-checkpoint", f"no committed checkpoint under {root}")
        if any(candidate.is_dir() for candidate in root.glob("step_*")):
            raise _InspectionRefusal("incomplete-checkpoint", f"no committed checkpoint under {root}")
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


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    evidence = inspect_run(args.run_dir)
    print(json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by CLI tests
    raise SystemExit(_main())


__all__ = ["RUN_EVIDENCE_SCHEMA", "RunEvidence", "inspect_run"]
