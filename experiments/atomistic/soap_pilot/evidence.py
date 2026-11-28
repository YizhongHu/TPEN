"""Named consumer evidence for the SOAP pilot.

The producer's JSONL contract is deliberately small: one object per line with
``step``, ``namespace``, and ``metrics``.  This module owns the corresponding
consumer boundary instead of discovering files or reconstructing evidence from
other run artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple, Sequence

if TYPE_CHECKING:
    from .checks import CheckResult


class EvidenceFormatError(ValueError):
    """Raised when the declared metrics JSONL has an invalid record envelope."""


class MetricRecord(NamedTuple):
    """One persisted metric record, preserving its step and namespace."""

    step: int | None
    namespace: str
    metrics: Mapping[str, Any]


def load_metric_records(run_dir: str | Path) -> tuple[MetricRecord, ...] | None:
    """Load only ``<run_dir>/metrics.jsonl``.

    A missing declared file is represented by ``None`` so the consumer can
    classify it as ``UNMEASURED``.  Every non-empty line must be a JSON object
    with an integer-or-null ``step``, string ``namespace``, and object
    ``metrics``.  Record order and duplicate step/namespace pairs are retained.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        The named run directory.  No sibling files are discovered.

    Returns
    -------
    tuple[MetricRecord, ...] or None
        Parsed records, or ``None`` when the declared file is absent.

    Raises
    ------
    EvidenceFormatError
        If a line is not valid JSON or its record envelope is malformed.
    """

    path = Path(run_dir) / "metrics.jsonl"
    if not path.is_file():
        return None

    records: list[MetricRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise EvidenceFormatError(f"metrics.jsonl line {line_number}: blank record")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceFormatError(
                f"metrics.jsonl line {line_number}: invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise EvidenceFormatError(
                f"metrics.jsonl line {line_number}: record must be an object"
            )
        step = payload.get("step")
        if step is not None and type(step) is not int:
            raise EvidenceFormatError(
                f"metrics.jsonl line {line_number}: step must be an integer or null"
            )
        namespace = payload.get("namespace")
        if not isinstance(namespace, str):
            raise EvidenceFormatError(
                f"metrics.jsonl line {line_number}: namespace must be a string"
            )
        metrics = payload.get("metrics")
        if not isinstance(metrics, dict):
            raise EvidenceFormatError(
                f"metrics.jsonl line {line_number}: metrics must be an object"
            )
        records.append(MetricRecord(step, namespace, metrics))
    return tuple(records)


def check_metric_records(records: Sequence[MetricRecord] | None) -> "CheckResult":
    """Apply the pilot's finite-only metric check to loaded records."""

    # Import lazily so the file-format boundary remains usable without a
    # checks-module import cycle and each inner mapping reaches the checker.
    from .checks import CheckResult, Verdict, check_metric_integrity

    name = "metric-jsonl-evidence"
    if records is None:
        return CheckResult(name, Verdict.UNMEASURED, "declared metrics.jsonl evidence is absent", measured=False)
    if not records:
        return CheckResult(name, Verdict.UNMEASURED, "declared metrics.jsonl contains no records", measured=False)
    for index, record in enumerate(records):
        if not isinstance(record, MetricRecord):
            return CheckResult(name, Verdict.HARNESS_INVALID, f"metric record {index} is not a MetricRecord")
        result = check_metric_integrity(record.metrics)
        if result.verdict is not Verdict.PASS:
            return CheckResult(
                result.name,
                result.verdict,
                f"record {index} ({record.namespace!r}, step={record.step!r}): {result.detail}",
                result.measured,
                {"record_index": index, "step": record.step, "namespace": record.namespace},
            )
    return CheckResult(
        name,
        Verdict.PASS,
        f"{len(records)} JSONL metric records are finite",
        True,
        {"records": len(records), "steps": [record.step for record in records]},
    )


# Descriptive aliases keep the evidence boundary easy to find for callers.
load_metrics_jsonl = load_metric_records
check_metric_evidence = check_metric_records


__all__ = [
    "EvidenceFormatError",
    "MetricRecord",
    "check_metric_evidence",
    "check_metric_records",
    "load_metric_records",
    "load_metrics_jsonl",
]
