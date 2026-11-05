"""Strict, executor-independent readers for scientific run manifests.

The toolkit deliberately decodes the published JSON instead of importing the
executor package.  The optional inspection callback is the public core
inspection seam; the default callback is loaded lazily so the toolkit remains
usable in the root experiment environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import string
from types import MappingProxyType
from typing import Any, Callable, Mapping


RUN_MANIFEST_V1 = "tpen-executor/run-manifest-v1"
RUN_MANIFEST_V2 = "tpen-executor/run-manifest-v2"
ACCEPTED_SEGMENT_SCHEMA = "tpen-executor/accepted-segment-v1"
OUTCOME_STATUSES = frozenset({"completed", "failed", "timed_out", "launch_failed"})
_HEX_DIGITS = frozenset(string.hexdigits.lower())


class ManifestRefusal(ValueError):
    """Typed refusal for an unreadable or scientifically inconsistent manifest.

    Parameters
    ----------
    code : str
        Stable machine-readable refusal category.
    reason : str
        Human-readable explanation of the rejected manifest or evidence.
    """

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason


ManifestReadError = ManifestRefusal


@dataclass(frozen=True, slots=True)
class CommittedSegment:
    """A selected segment projected from one accepted executor record.

    Parameters
    ----------
    task_id, attempt_id : str
        Logical task and executor attempt identities.
    predecessor_checkpoint : str or None
        Prior checkpoint in the committed chain, or ``None`` for a fresh start.
    checkpoint_dir : str
        Core-published checkpoint directory.
    next_iteration, completed_updates : int
        Nonnegative progress coordinates copied from core evidence.
    """

    task_id: str
    attempt_id: str
    predecessor_checkpoint: str | None
    checkpoint_dir: str
    next_iteration: int
    completed_updates: int


@dataclass(frozen=True, slots=True)
class CommittedChain:
    """The exactly-once accepted segment chain for one logical task.

    Parameters
    ----------
    task_id : str
        Logical task identity.
    segments : tuple of CommittedSegment
        Ordered, predecessor-linked accepted segments; no attempt is repeated.
    """

    task_id: str
    segments: tuple[CommittedSegment, ...]


@dataclass(frozen=True, slots=True)
class PreservedAttempt:
    """A manifest attempt retained as evidence but excluded from selection.

    Parameters
    ----------
    task_id, attempt_id, output_dir : str
        Original executor attempt identity and fenced output directory.
    outcome : str or None
        Published operational outcome status, if one exists.
    reason : str, default="not-selected"
        Why the attempt is preserved rather than counted in a committed chain.
    """

    task_id: str
    attempt_id: str
    output_dir: str
    outcome: str | None
    reason: str = "not-selected"


@dataclass(frozen=True, slots=True)
class ResultSet:
    """Selected chains plus original attempts that were not selected.

    Parameters
    ----------
    chains : tuple of CommittedChain
        Exactly-once committed coordinate chains grouped by task.
    preserved_attempts : tuple of PreservedAttempt
        Original non-selected attempts retained as evidence and never counted.
    """

    chains: tuple[CommittedChain, ...]
    preserved_attempts: tuple[PreservedAttempt, ...]

    @property
    def chains_by_task(self) -> Mapping[str, CommittedChain]:
        """Return a task-indexed view without changing the immutable result."""

        return MappingProxyType({chain.task_id: chain for chain in self.chains})

    @property
    def non_selected_attempts(self) -> tuple[PreservedAttempt, ...]:
        """Compatibility name for the preserved-evidence collection."""

        return self.preserved_attempts

    @property
    def preserved_evidence(self) -> tuple[PreservedAttempt, ...]:
        """Return original attempts retained as preserved evidence."""

        return self.preserved_attempts


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestRefusal("duplicate-json-key", f"duplicate JSON member {key!r}")
        value[key] = item
    return value


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except ManifestRefusal:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise ManifestRefusal("malformed-json", f"cannot decode manifest JSON: {error}") from error
    if not isinstance(value, dict):
        raise ManifestRefusal("manifest-not-object", "manifest JSON must be an object")
    return value


def _fields(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ManifestRefusal("malformed-record", f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ManifestRefusal(
            "malformed-record",
            f"{name} keys differ (missing={sorted(expected - actual)!r}, extra={sorted(actual - expected)!r})",
        )
    return value


def _text(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ManifestRefusal("malformed-record", f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ManifestRefusal("malformed-record", f"{name} must be a nonnegative integer")
    return value


def _sha256(value: Any, name: str) -> str:
    """Validate the lowercase SHA-256 wire representation used by executor."""

    _text(value, name)
    if len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ManifestRefusal("malformed-record", f"{name} must be a lowercase SHA-256 digest")
    if value != value.lower():
        raise ManifestRefusal("malformed-record", f"{name} must be a lowercase SHA-256 digest")
    return value


def _attempt(value: Any) -> dict[str, Any]:
    record = dict(_fields(value, {"schema", "task_id", "attempt_id", "start_receipt", "outcome_receipt", "output_dir"}, "manifest attempt"))
    if record["schema"] != "tpen-executor/manifest-attempt-v1":
        raise ManifestRefusal("unknown-schema", "manifest attempt schema is unsupported")
    for key in ("task_id", "attempt_id", "output_dir"):
        _text(record[key], f"attempt.{key}")
    start = dict(_fields(record["start_receipt"], {"schema", "task_id", "attempt_id", "allocation_id", "backend_job_id", "group_digest", "started_at_utc", "output_dir"}, "start receipt"))
    if start["schema"] != "tpen-executor/start-receipt-v1":
        raise ManifestRefusal("unknown-schema", "start receipt schema is unsupported")
    for key in ("task_id", "attempt_id", "allocation_id", "backend_job_id", "started_at_utc", "output_dir"):
        _text(start[key], f"start.{key}")
    _sha256(start["group_digest"], "start.group_digest")
    if (start["task_id"], start["attempt_id"], start["output_dir"]) != (record["task_id"], record["attempt_id"], record["output_dir"]):
        raise ManifestRefusal("attempt-receipt-mismatch", "manifest attempt disagrees with its start receipt")
    outcome = record["outcome_receipt"]
    if outcome is not None:
        outcome = dict(_fields(outcome, {"schema", "task_id", "attempt_id", "group_digest", "finished_at_utc", "status", "output_dir", "result_digest"}, "outcome receipt"))
        if outcome["schema"] != "tpen-executor/outcome-receipt-v1":
            raise ManifestRefusal("unknown-schema", "outcome receipt schema is unsupported")
        for key in ("task_id", "attempt_id", "finished_at_utc", "status", "output_dir"):
            _text(outcome[key], f"outcome.{key}")
        _sha256(outcome["group_digest"], "outcome.group_digest")
        if outcome["status"] not in OUTCOME_STATUSES:
            raise ManifestRefusal("malformed-record", "outcome.status is not a supported operational status")
        if outcome["result_digest"] is not None:
            _sha256(outcome["result_digest"], "outcome.result_digest")
        if (
            outcome["task_id"], outcome["attempt_id"], outcome["group_digest"], outcome["output_dir"]
        ) != (
            record["task_id"], record["attempt_id"], start["group_digest"], record["output_dir"]
        ):
            raise ManifestRefusal("attempt-receipt-mismatch", "manifest attempt disagrees with its outcome receipt")
    record["outcome_receipt"] = outcome
    return record


def _segment(value: Any) -> dict[str, Any]:
    record = dict(_fields(value, {"schema", "task_id", "attempt_id", "predecessor_checkpoint", "checkpoint_dir", "next_iteration", "completed_updates"}, "accepted segment"))
    if record["schema"] != ACCEPTED_SEGMENT_SCHEMA:
        raise ManifestRefusal("unknown-schema", "accepted segment schema is unsupported")
    _text(record["task_id"], "segment.task_id")
    _text(record["attempt_id"], "segment.attempt_id")
    if record["predecessor_checkpoint"] is not None:
        _text(record["predecessor_checkpoint"], "segment.predecessor_checkpoint")
    _text(record["checkpoint_dir"], "segment.checkpoint_dir")
    _nonnegative_int(record["next_iteration"], "segment.next_iteration")
    _nonnegative_int(record["completed_updates"], "segment.completed_updates")
    return record


def _manifest(path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    value = _load_json(path)
    schema = value.get("schema")
    if schema == RUN_MANIFEST_V1:
        _fields(value, {"schema", "group_id", "group_digest", "attempts"}, "run manifest")
        raw_segments: tuple[dict[str, Any], ...] = ()
    elif schema == RUN_MANIFEST_V2:
        _fields(value, {"schema", "group_id", "group_digest", "attempts", "segments"}, "run manifest")
        if isinstance(value["segments"], (str, bytes, bytearray)) or not isinstance(value["segments"], list):
            raise ManifestRefusal("malformed-record", "manifest segments must be a sequence")
        raw_segments = tuple(_segment(item) for item in value["segments"])
    else:
        raise ManifestRefusal("unknown-schema", f"unsupported run manifest schema {schema!r}")
    _text(value["group_id"], "manifest.group_id")
    _sha256(value["group_digest"], "manifest.group_digest")
    if isinstance(value["attempts"], (str, bytes, bytearray)) or not isinstance(value["attempts"], list):
        raise ManifestRefusal("malformed-record", "manifest attempts must be a sequence")
    attempts = tuple(_attempt(item) for item in value["attempts"])
    for attempt in attempts:
        if attempt["start_receipt"]["group_digest"] != value["group_digest"]:
            raise ManifestRefusal("group-digest-mismatch", "start receipt group_digest differs from manifest")
        if (
            attempt["outcome_receipt"] is not None
            and attempt["outcome_receipt"]["group_digest"] != value["group_digest"]
        ):
            raise ManifestRefusal("group-digest-mismatch", "outcome receipt group_digest differs from manifest")
    if len({item["attempt_id"] for item in attempts}) != len(attempts):
        raise ManifestRefusal("duplicate-attempt", "manifest contains duplicate attempt identities")
    return dict(value), attempts, raw_segments


def _check_segments(attempts: tuple[dict[str, Any], ...], segments: tuple[dict[str, Any], ...]) -> None:
    by_id = {attempt["attempt_id"]: attempt for attempt in attempts}
    if len({segment["attempt_id"] for segment in segments}) != len(segments):
        raise ManifestRefusal("duplicate-segment", "an attempt appears in more than one accepted segment")
    previous: dict[str, dict[str, Any]] = {}
    for segment in segments:
        attempt = by_id.get(segment["attempt_id"])
        if attempt is None:
            raise ManifestRefusal("uncommitted-segment", "accepted segment references an unknown attempt")
        if attempt["task_id"] != segment["task_id"]:
            raise ManifestRefusal("segment-task-mismatch", "accepted segment task differs from its attempt")
        outcome = attempt["outcome_receipt"]
        if outcome is None or outcome["status"] != "completed":
            raise ManifestRefusal("segment-on-failed-attempt", "accepted segment does not reference a completed attempt")
        prior = previous.get(segment["task_id"])
        if prior is None:
            if segment["predecessor_checkpoint"] is not None:
                raise ManifestRefusal("wrong-predecessor", "first accepted segment must be a fresh start")
        else:
            if segment["predecessor_checkpoint"] != prior["checkpoint_dir"]:
                raise ManifestRefusal("wrong-predecessor", "accepted segment predecessor does not match prior checkpoint")
            prior_coord = (prior["next_iteration"], prior["completed_updates"])
            coordinate = (segment["next_iteration"], segment["completed_updates"])
            if coordinate[0] < prior_coord[0] or coordinate[1] < prior_coord[1] or coordinate == prior_coord:
                raise ManifestRefusal("non-increasing-coordinate", "accepted segment progress coordinate is not strictly increasing")
        previous[segment["task_id"]] = segment


def _public_evidence(value: Any) -> tuple[str, Mapping[str, Any] | None]:
    if isinstance(value, Mapping):
        verdict = value.get("verdict")
        checkpoint = value.get("checkpoint")
    else:
        verdict = getattr(value, "verdict", None)
        checkpoint = getattr(value, "checkpoint", None)
    if type(verdict) is not str or verdict not in {"completed", "incomplete"}:
        raise ManifestRefusal("core-evidence-mismatch", "core inspection did not report completed or incomplete evidence")
    if not isinstance(checkpoint, Mapping):
        raise ManifestRefusal("core-evidence-mismatch", "core inspection did not report a checkpoint")
    expected = {"checkpoint_dir", "schema_version", "next_iteration", "completed_updates"}
    if set(checkpoint) != expected:
        raise ManifestRefusal("core-evidence-mismatch", "core checkpoint evidence has an unexpected shape")
    _text(checkpoint["checkpoint_dir"], "core.checkpoint_dir")
    if type(checkpoint["schema_version"]) is not int:
        raise ManifestRefusal("core-evidence-mismatch", "core.schema_version must be an integer")
    _nonnegative_int(checkpoint["next_iteration"], "core.next_iteration")
    _nonnegative_int(checkpoint["completed_updates"], "core.completed_updates")
    return verdict, checkpoint


def _default_inspect_run(path: str | Path) -> Any:
    """Load the public core inspector only when a caller did not inject one."""

    try:
        from tpen.inspection import inspect_run
    except ImportError as error:
        raise ManifestRefusal("core-inspector-unavailable", "public core inspection is unavailable") from error
    return inspect_run(path)


def read(
    manifest_path: str | Path,
    inspect_run: Callable[[str | Path], Any] | None = None,
) -> ResultSet:
    """Read one manifest and return its exactly-once committed result set.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Published v1 or v2 run manifest JSON.
    inspect_run : callable, optional
        Public core inspection callback.  It receives the fenced core run
        directory for each selected segment.  When omitted, the callback is
        loaded lazily from ``tpen.inspection``.

    Returns
    -------
    ResultSet
        Per-task accepted chains and non-selected original attempts.

    Raises
    ------
    ManifestRefusal
        If the manifest, chain, or public core evidence is inconsistent.
    """

    path = Path(manifest_path)
    _, attempts, segments = _manifest(path)
    _check_segments(attempts, segments)
    inspector = inspect_run or _default_inspect_run
    selected_ids = {segment["attempt_id"] for segment in segments}
    chains: list[CommittedChain] = []
    by_task: dict[str, list[CommittedSegment]] = {}
    by_attempt = {attempt["attempt_id"]: attempt for attempt in attempts}
    for segment in segments:
        attempt = by_attempt[segment["attempt_id"]]
        run_dir = Path(attempt["output_dir"]) / "core"
        try:
            evidence = inspector(run_dir)
        except ManifestRefusal:
            raise
        except Exception as error:
            raise ManifestRefusal("core-evidence-unavailable", f"core inspection failed for {run_dir}: {error}") from error
        _, checkpoint = _public_evidence(evidence)
        if dict(checkpoint) != {
            "checkpoint_dir": segment["checkpoint_dir"],
            "schema_version": checkpoint["schema_version"],
            "next_iteration": segment["next_iteration"],
            "completed_updates": segment["completed_updates"],
        }:
            raise ManifestRefusal("core-evidence-mismatch", f"core evidence disagrees with accepted segment {segment['attempt_id']!r}")
        projected = CommittedSegment(
            segment["task_id"],
            segment["attempt_id"],
            segment["predecessor_checkpoint"],
            segment["checkpoint_dir"],
            segment["next_iteration"],
            segment["completed_updates"],
        )
        by_task.setdefault(projected.task_id, []).append(projected)
    for task_id, task_segments in by_task.items():
        chains.append(CommittedChain(task_id, tuple(task_segments)))
    preserved = tuple(
        PreservedAttempt(
            attempt["task_id"],
            attempt["attempt_id"],
            attempt["output_dir"],
            None if attempt["outcome_receipt"] is None else attempt["outcome_receipt"]["status"],
            "uncommitted" if attempt["outcome_receipt"] is None else (
                "failed-or-incomplete" if attempt["outcome_receipt"]["status"] != "completed" else "not-selected"
            ),
        )
        for attempt in attempts
        if attempt["attempt_id"] not in selected_ids
    )
    return ResultSet(tuple(chains), preserved)


__all__ = [
    "ACCEPTED_SEGMENT_SCHEMA",
    "CommittedChain",
    "CommittedSegment",
    "ManifestReadError",
    "ManifestRefusal",
    "PreservedAttempt",
    "RUN_MANIFEST_V1",
    "RUN_MANIFEST_V2",
    "ResultSet",
    "read",
]
