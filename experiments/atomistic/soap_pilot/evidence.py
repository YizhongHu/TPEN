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


def build_optimizer_group_inventory(
    rendered_config_path: str | Path,
) -> tuple[tuple[str, ...], dict[str, dict[str, int]]]:
    """Build the complete optimizer-group inventory from rendered bytes.

    The producer and this adapter deliberately share the core authority:
    ``OptimizerState`` calls ``tpen.parameter_groups.resolve_parameter_groups``
    on the live model (``tpen/callback/optimizer_state.py:92-98``).  We read
    the named, already-resolved rendered config, instantiate its model, and
    call that same resolver with its callback groups.  The resolver omits
    empty groups (``tpen/parameter_groups.py:111-118``), so this adapter puts
    those declared groups back with zero counts rather than inventing ownership
    from parameter names or a static policy.

    Parameters
    ----------
    rendered_config_path : str or pathlib.Path
        The exact rendered YAML named by the evidence packet.

    Returns
    -------
    tuple[tuple[str, ...], dict[str, dict[str, int]]]
        Declared group names and a complete ``parameters``/``elements``
        inventory, including explicitly empty groups.

    Raises
    ------
    EvidenceFormatError
        If the named file, model, or OptimizerState declaration is malformed.
    """

    path = Path(rendered_config_path)
    if not path.is_file():
        raise EvidenceFormatError(f"rendered config is absent: {path}")
    try:
        import yaml
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from tpen.parameter_groups import resolve_parameter_groups

        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping):
            raise EvidenceFormatError("rendered config must be a mapping")
        model_config = document.get("model")
        callbacks = document.get("callbacks")
        if not isinstance(model_config, Mapping) or not isinstance(callbacks, list):
            raise EvidenceFormatError("rendered config must contain model and callbacks")
        optimizer_callbacks = [
            callback
            for callback in callbacks
            if isinstance(callback, Mapping)
            and callback.get("_target_") == "tpen.callback.OptimizerState"
        ]
        if len(optimizer_callbacks) != 1:
            raise EvidenceFormatError(
                "rendered config must contain exactly one OptimizerState callback"
            )
        declared = optimizer_callbacks[0].get("groups")
        if not isinstance(declared, Mapping) or not declared:
            raise EvidenceFormatError("OptimizerState groups must be a non-empty mapping")
        if any(not isinstance(group, str) or not group for group in declared):
            raise EvidenceFormatError("OptimizerState group names must be non-empty strings")

        model = instantiate(OmegaConf.create(model_config))
        resolved = resolve_parameter_groups(model, declared)
    except EvidenceFormatError:
        raise
    except Exception as exc:
        raise EvidenceFormatError(f"could not build optimizer group inventory: {exc}") from exc

    group_names = tuple(declared)
    inventory = {
        group: {
            "parameters": len(resolved.get(group, ())),
            "elements": sum(int(parameter.numel()) for _, parameter in resolved.get(group, ())),
        }
        for group in group_names
    }
    return group_names, inventory


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
    "build_optimizer_group_inventory",
    "EvidenceFormatError",
    "MetricRecord",
    "check_metric_evidence",
    "check_metric_records",
    "load_metric_records",
    "load_metrics_jsonl",
]
