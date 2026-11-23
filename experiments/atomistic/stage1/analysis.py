"""Descriptive analysis for already-collected Stage-1 evaluations.

The collector remains the owner of admission and per-stream measurements.  This
module is a postprocessor over its JSON-safe document, which keeps analysis
usable for Hooke and roster-B collections without importing TPEN core internals.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


class AnalysisError(ValueError):
    """The collected rows or training history cannot form a defined summary."""


def _finite(value: Any, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be numeric") from exc
    if not math.isfinite(converted):
        raise AnalysisError(f"{label} must be finite")
    return converted


def _range(values: list[float]) -> dict[str, float]:
    low, high = min(values), max(values)
    return {"min": low, "max": high, "range": high - low}


def _group_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    try:
        return str(row["system_id"]), int(row["lineage"]), int(row["checkpoint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("each collection record needs system_id, lineage, and checkpoint") from exc


def _record_measurement(row: Mapping[str, Any]) -> tuple[float, float, float]:
    try:
        statistics = row["statistics"]
        reference = row["reference"]
        mean = _finite(statistics["mean"], "stream mean")
        variance = _finite(statistics["variance"], "stream variance")
        reference_energy = _finite(reference["energy_hartree"], "reference energy")
    except KeyError as exc:
        raise AnalysisError(f"collection record is missing {exc.args[0]}") from exc
    if variance < 0:
        raise AnalysisError("stream variance must be non-negative")
    return mean, variance, reference_energy


def _summarize_group(key: tuple[str, int, int], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    streams: dict[int, dict[str, Any]] = {}
    references: set[float] = set()
    for row in rows:
        try:
            stream = int(row["stream"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError("each collection record needs an integer stream") from exc
        if stream in streams:
            raise AnalysisError(f"duplicate stream {stream} in {key}")
        mean, variance, reference = _record_measurement(row)
        references.add(reference)
        streams[stream] = {"stream": stream, "mean": mean, "variance": variance,
                           "error": mean - reference, "absolute_error": abs(mean - reference)}
    if set(streams) != {0, 1, 2, 3}:
        raise AnalysisError(f"{key} requires exactly streams 0..3, got {sorted(streams)}")
    if len(references) != 1:
        raise AnalysisError(f"{key} has inconsistent reference energies")

    stream_rows = [streams[index] for index in range(4)]
    means = [row["mean"] for row in stream_rows]
    variances = [row["variance"] for row in stream_rows]
    errors = [row["error"] for row in stream_rows]
    reference = references.pop()
    mean = sum(means) / 4
    pooled_variance = sum(variance + (stream_mean - mean) ** 2
                          for variance, stream_mean in zip(variances, means)) / 4
    mean_range = _range(means)
    error_range = _range(errors)
    absolute_error_range = _range([row["absolute_error"] for row in stream_rows])
    variance_range = _range(variances)
    return {
        "system_id": key[0], "lineage": key[1], "checkpoint": key[2],
        "reference_energy_hartree": reference, "stream_count": 4,
        "streams": stream_rows,
        "mean": mean, "signed_error": mean - reference, "absolute_error": abs(mean - reference),
        "stream_mean_min": mean_range["min"], "stream_mean_max": mean_range["max"],
        "stream_mean_range": mean_range["range"],
        "stream_error_min": error_range["min"], "stream_error_max": error_range["max"],
        "stream_error_range": error_range["range"],
        "stream_absolute_error_min": absolute_error_range["min"],
        "stream_absolute_error_max": absolute_error_range["max"],
        "stream_absolute_error_range": absolute_error_range["range"],
        "stream_variance_min": variance_range["min"], "stream_variance_max": variance_range["max"],
        "stream_variance_range": variance_range["range"],
        "stream_mean_rms_spread": math.sqrt(sum((value - mean) ** 2 for value in means) / 4),
        "pooled_variance": pooled_variance,
    }


def _checkpoint_aggregate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    signed_errors = [group["signed_error"] for group in groups]
    absolute_errors = [group["absolute_error"] for group in groups]
    pooled_variances = [group["pooled_variance"] for group in groups]
    means = [group["mean"] for group in groups]
    return {
        "system_id": groups[0]["system_id"], "checkpoint": groups[0]["checkpoint"],
        "lineage_count": len(groups), "lineages": [group["lineage"] for group in groups],
        "mean": sum(means) / len(means),
        "signed_error_mean": sum(signed_errors) / len(signed_errors),
        "absolute_error_mean": sum(absolute_errors) / len(absolute_errors),
        "signed_error_min": min(signed_errors), "signed_error_max": max(signed_errors),
        "signed_error_range": max(signed_errors) - min(signed_errors),
        "absolute_error_min": min(absolute_errors), "absolute_error_max": max(absolute_errors),
        "absolute_error_range": max(absolute_errors) - min(absolute_errors),
        "pooled_variance_mean": sum(pooled_variances) / len(pooled_variances),
        "pooled_variance_min": min(pooled_variances), "pooled_variance_max": max(pooled_variances),
        "pooled_variance_range": max(pooled_variances) - min(pooled_variances),
    }


def summarize_collection(collection: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize collector records without re-reading trajectory files.

    Parameters
    ----------
    collection
        JSON-safe output from :func:`experiments.atomistic.stage1.collect.collect`.

    Returns
    -------
    dict
        ``groups`` contains one complete four-stream row per lineage/checkpoint;
        ``checkpoints`` contains ranges across lineages at each checkpoint.

    Raises
    ------
    AnalysisError
        If records are absent, incomplete, duplicated, non-finite, or disagree
        on the reference energy.
    """
    records = collection.get("records")
    if not isinstance(records, list) or not records:
        raise AnalysisError("collection records must be a non-empty list")
    grouped: dict[tuple[str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if not isinstance(record, Mapping):
            raise AnalysisError("collection records must be mappings")
        grouped[_group_key(record)].append(record)
    groups = [_summarize_group(key, grouped[key]) for key in sorted(grouped)]
    by_checkpoint: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_checkpoint[(group["system_id"], group["checkpoint"])].append(group)
    checkpoints = [_checkpoint_aggregate(by_checkpoint[key]) for key in sorted(by_checkpoint)]
    return {"schema": "stage1-analysis/v1", "collection_manifest_hash": collection.get("manifest_hash"),
            "groups": groups, "checkpoints": checkpoints}


def _training_value(row: Mapping[str, Any], name: str) -> Any:
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and name in metrics:
        return metrics[name]
    return row.get(name)


def _training_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    try:
        return str(row["system_id"]), int(row["lineage"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("training rows need system_id and lineage") from exc


def join_training_history(analysis: Mapping[str, Any],
                          training_history: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Join fresh evaluation groups with their trailing 100 training batches.

    Training JSONL records use the trainer's zero-based ``step`` and contain
    ``metrics.energy``, ``metrics.energy_variance``, and
    ``metrics.optimizer_step``.  A stored step ``j`` maps to completed step
    ``j + 1``; therefore checkpoint ``t > 0`` uses completed steps
    ``max(1, t - 99) .. t``.  Missing or non-finite windows stay unavailable.
    """
    rows_by_identity: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for raw in training_history:
        if not isinstance(raw, Mapping):
            raise AnalysisError("training rows must be mappings")
        system_id, lineage = _training_identity(raw)
        try:
            stored_step = int(raw["step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError("training rows need an integer zero-based step") from exc
        if stored_step < 0:
            raise AnalysisError("training step must be non-negative")
        mean = _finite(_training_value(raw, "energy"), "training energy")
        variance = _finite(_training_value(raw, "energy_variance"), "training energy_variance")
        if variance < 0:
            raise AnalysisError("training energy_variance must be non-negative")
        optimizer_step = _training_value(raw, "optimizer_step")
        if not isinstance(optimizer_step, bool):
            raise AnalysisError("training optimizer_step must be a boolean")
        rows_by_identity[(system_id, lineage)].append({
            "stored_step": stored_step, "completed_step": stored_step + 1,
            "energy": mean, "energy_variance": variance,
            "optimizer_step": optimizer_step, "skipped": not optimizer_step,
        })
    for rows in rows_by_identity.values():
        rows.sort(key=lambda row: row["completed_step"])

    comparisons: list[dict[str, Any]] = []
    for group in analysis.get("groups", []):
        system_id, lineage, checkpoint = group["system_id"], group["lineage"], group["checkpoint"]
        comparison: dict[str, Any] = {"system_id": system_id, "lineage": lineage,
                                     "checkpoint": checkpoint, "available": False}
        if checkpoint == 0:
            comparison["reason"] = "step zero has no trailing pre-update window"
            comparisons.append(comparison)
            continue
        start = max(1, checkpoint - 99)
        expected = set(range(start, checkpoint + 1))
        rows = rows_by_identity.get((system_id, lineage), [])
        selected = [row for row in rows if row["completed_step"] in expected]
        by_step = {row["completed_step"]: row for row in selected}
        missing = sorted(expected - set(by_step))
        duplicate = len(selected) != len(by_step)
        comparison.update({"expected_completed_steps": [start, checkpoint],
                           "missing_completed_steps": missing,
                           "duplicate_completed_steps": duplicate})
        if missing or duplicate:
            comparison["reason"] = "training window is incomplete or duplicated"
            comparisons.append(comparison)
            continue
        ordered = [by_step[step] for step in sorted(expected)]
        batch_means = [row["energy"] for row in ordered]
        batch_variances = [row["energy_variance"] for row in ordered]
        comparison.update({
            "available": True, "stored_steps": [row["stored_step"] for row in ordered],
            "completed_steps": [row["completed_step"] for row in ordered],
            "batch_means": batch_means, "batch_variances": batch_variances,
            "training_mean": sum(batch_means) / len(batch_means),
            "training_variance": sum(batch_variances) / len(batch_variances),
            "training_mean_min": min(batch_means), "training_mean_max": max(batch_means),
            "training_mean_range": max(batch_means) - min(batch_means),
            "training_variance_min": min(batch_variances), "training_variance_max": max(batch_variances),
            "training_variance_range": max(batch_variances) - min(batch_variances),
            "skipped_updates": sum(row["skipped"] for row in ordered),
            "optimizer_steps": [row["optimizer_step"] for row in ordered],
            "fresh_minus_training_mean": group["mean"] - sum(batch_means) / len(batch_means),
            "fresh_minus_training_variance": group["pooled_variance"] - sum(batch_variances) / len(batch_variances),
        })
        comparisons.append(comparison)
    return {"schema": "stage1-training-join/v1", "comparisons": comparisons}


def write_trajectory_figures(analysis: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write deterministic signed-error, absolute-error, and variance figures.

    Each lineage contributes four raw stream traces and one aggregate trace to
    every panel.  The translucent band is the corresponding raw four-stream
    min/max range.  No interpolation or epsilon is applied.
    """
    groups = list(analysis.get("groups", []))
    if not groups:
        raise AnalysisError("cannot plot an empty analysis")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise AnalysisError("matplotlib is required to render Stage-1 figures") from exc

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    for name, stream_value, aggregate_value, low, high, ylabel in (
        ("error_vs_step.png", "error", "signed_error", "stream_error_min", "stream_error_max", "signed error (Ha)"),
        ("absolute_error_vs_step.png", "absolute_error", "absolute_error", "stream_absolute_error_min", "stream_absolute_error_max", "absolute error (Ha)"),
        ("variance_vs_step.png", "variance", "pooled_variance", "stream_variance_min", "stream_variance_max", "variance (Ha^2)"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5), dpi=150)
        identities = sorted({(str(group["system_id"]), int(group["lineage"])) for group in groups})
        for system_id, lineage in identities:
            selected = sorted((group for group in groups
                               if str(group["system_id"]) == system_id and int(group["lineage"]) == lineage),
                              key=lambda group: int(group["checkpoint"]))
            steps = [group["checkpoint"] for group in selected]
            lows = [group[low] for group in selected]
            highs = [group[high] for group in selected]
            label = f"{system_id}/lineage-{lineage}"
            stream_series = [[row[stream_value] for row in group["streams"]] for group in selected]
            aggregate = [group[aggregate_value] for group in selected]
            # A single 2-D call retains four stream traces plus Ehat/pooled in
            # one lineage panel while keeping the legend label compact.
            series = [
                [stream_series[index][stream] for index in range(len(selected))]
                for stream in range(4)
            ]
            series.append(aggregate)
            lines = axis.plot(
                steps,
                list(zip(*series)),
                marker="o",
                markersize=3,
                label=f"{label} streams + aggregate",
            )
            line = lines[0]
            axis.fill_between(steps, lows, highs, color=line.get_color(), alpha=0.16)
        axis.set_xlabel("checkpoint step")
        axis.set_ylabel(ylabel)
        if ylabel.startswith("signed") and hasattr(axis, "axhline"):
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        path = output / name
        figure.savefig(path, metadata={"Software": "TPEN Stage-1 analysis"})
        plt.close(figure)
        figures[name.removesuffix(".png")] = str(path)
    return figures


__all__ = ["AnalysisError", "join_training_history", "summarize_collection",
           "write_trajectory_figures"]
