"""Strict collector for complete H-v2 draw-by-walker records.

Consumes TPEN's real per-evaluation-run artifacts directly: the trajectory
``sampled_eval_table.csv`` / ``sampled_eval_table.metadata.json`` envelope
written by ``tpen.evaluation.summaries.SampledRecordWriter`` for a complete
draw-by-walker grid (schema ``trajectory_records/v1``, see
``tpen.evaluation.trajectory_records``), the run's own ``resolved_config.yaml``
and ``metadata.json`` (written by ``tpen.callback.ResolvedConfigSnapshot`` and
``tpen.callback.Metadata``), and the restored checkpoint directory (written by
``tpen.checkpoint.save.save_checkpoint``).

Identity is reconciled at the envelope, not per row, following the he-v1
collector's precedent (resolved-config hash, csv hash, source commit): the
real per-row schema carries no per-row identity columns.

The collector is deliberately conservative: it validates every identity and
the entire grid before calculating any statistic.  It never repairs, clips,
filters, replaces, or silently accepts a final-draw-only file.

``collect_records`` additionally refuses: a row whose term energies do not
sum to its local energy or whose sign is not +-1; a model-only restore not
confirmed by the emitted ``tpen.checkpoint.events.CheckpointRestored``
occurrence (``occurrences.jsonl``); a missing or malformed per-stream sampler
draw receipt (``sampler_trajectory_diagnostics.json``, written by
``tpen.evaluation.SamplerStatsSummary``); and a whole-trajectory variance that
does not reconcile with the record manifest. ``validate_training_run``
separately refuses a dirty or incomplete training run, a checkpoint whose file
content does not match its manifest hash, and a training log that skipped an
update or fell short of the declared number of updates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.toolkit.config_identity import expected_resolved_config_identity

#: Schema and row-semantics constants mirrored from
#: ``tpen.evaluation.trajectory_records`` so this module never imports tpen
#: (see ``experiments/README.md``).
TRAJECTORY_RECORD_SCHEMA = "trajectory_records/v1"
ROW_SEMANTICS = "complete_draw_walker_grid"

#: Fully-qualified event name ``tpen.artifacts.write_occurrence_artifact``
#: writes for ``tpen.checkpoint.events.CheckpointRestored``.
CHECKPOINT_RESTORED_EVENT = "tpen.checkpoint.events.CheckpointRestored"

#: Schema mirrored from ``tpen.sampling.trajectory.SamplerTrajectoryDiagnostics``.
SAMPLER_TRAJECTORY_DIAGNOSTICS_SCHEMA = "sampler_trajectory_diagnostics/v1"

_TERM_SUM_REL_TOL = 1e-9
_TERM_SUM_ABS_TOL = 1e-9
_VARIANCE_REL_TOL = 1e-9
_VARIANCE_ABS_TOL = 1e-9


class RecordIntegrityError(ValueError):
    """Raised when a record cannot be admitted to the complete-record set."""


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RecordSpec:
    """Expected identity and shape for one complete evaluation stream."""

    source_sha: str
    checkpoint_model_sha256: str
    resolved_config_sha256: str
    walkers: int = 1024
    draws: int = 64

    @property
    def expected_rows(self) -> int:
        """Return the complete draw-by-walker row count."""

        return self.walkers * self.draws


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecordIntegrityError(message)


def validate_checkpoint_identity(checkpoint_dir: Path, spec: RecordSpec) -> None:
    """Validate the restored checkpoint's commit marker and content identity."""

    _require((checkpoint_dir / "COMPLETE").is_file(), "missing checkpoint commit marker")
    model_path = checkpoint_dir / "model.pt"
    _require(model_path.is_file(), "altered source/checkpoint/config identity")
    _require(
        file_sha256(model_path) == spec.checkpoint_model_sha256,
        "altered source/checkpoint/config identity",
    )


def validate_run_identity(
    run_metadata: dict[str, Any], resolved_config_path: Path, spec: RecordSpec
) -> None:
    """Validate the evaluation run's source commit and resolved-config identity."""

    _require(
        run_metadata.get("git_commit") == spec.source_sha,
        "altered source/checkpoint/config identity",
    )
    try:
        observed_identity = expected_resolved_config_identity_from_path(resolved_config_path)
    except (OSError, UnicodeError, yaml.YAMLError, TypeError, ValueError) as exc:
        raise RecordIntegrityError("altered source/checkpoint/config identity") from exc
    _require(observed_identity["sha256"] == spec.resolved_config_sha256, "altered source/checkpoint/config identity")


def expected_resolved_config_identity_from_path(path: Path) -> dict[str, str]:
    """Compute the public identity from one intended resolved-config YAML."""

    resolved_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return expected_resolved_config_identity(resolved_config)


def validate_restore_mode(resolved_config: dict[str, Any]) -> None:
    """Refuse a restore whose mode is absent or is not ``model_only``.

    The collector is deliberately conservative: an absent ``load.mode`` is
    refused exactly like an explicit ``train_resume`` request, rather than
    defaulting to the safe value.
    """

    mode = (resolved_config.get("load") or {}).get("mode")
    _require(mode is not None, "collector requires an explicit restore_mode")
    _require(mode == "model_only", "collector requires a model-only restore_mode")


def validate_manifest(manifest: dict[str, Any], csv_path: Path, spec: RecordSpec) -> None:
    """Validate the immutable trajectory envelope before reading rows."""

    _require(manifest.get("schema") == TRAJECTORY_RECORD_SCHEMA, "wrong record schema")
    _require(
        manifest.get("row_semantics") == ROW_SEMANTICS,
        "record is not a complete draw-by-walker grid",
    )
    _require(manifest.get("walker_count") == spec.walkers, "record shape differs from the frozen evaluation")
    _require(manifest.get("draw_count") == spec.draws, "record shape differs from the frozen evaluation")
    _require(manifest.get("row_count") == spec.expected_rows, "record manifest is not complete")
    _require(manifest.get("nonfinite_count", 1) == 0, "nonfinite record")
    _require(manifest.get("csv_sha256") == file_sha256(csv_path), "uncommitted tail")


def validate_rows(rows: list[dict[str, Any]], spec: RecordSpec) -> list[dict[str, Any]]:
    """Validate every row and return it in its original complete-grid order."""

    _require(len(rows) == spec.expected_rows, "missing draw or uncommitted tail")
    seen: set[tuple[int, int]] = set()
    ordered: list[dict[str, Any] | None] = [None] * spec.expected_rows
    for row in rows:
        try:
            draw = int(row["draw_index"])
            walker = int(row["walker_index"])
            energy = float(row["local_energy"])
            logabs = float(row["logabs"])
            coordinates = [
                float(row[f"position/electron_0/axis_{axis}"]) for axis in range(3)
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordIntegrityError("malformed complete record") from exc
        if not 0 <= draw < spec.draws or not 0 <= walker < spec.walkers:
            raise RecordIntegrityError("record index outside the complete grid")
        key = (draw, walker)
        if key in seen:
            raise RecordIntegrityError("duplicate walker record")
        seen.add(key)
        # TPEN's own ``finite`` column covers only ``local_energy``
        # (tpen/evaluation/generators/trajectory.py); the contract's
        # complete-record rule also requires finite coordinates, so they are
        # checked explicitly here.
        if (
            row.get("finite") != "True"
            or not math.isfinite(energy)
            or not math.isfinite(logabs)
            or not all(math.isfinite(coordinate) for coordinate in coordinates)
        ):
            raise RecordIntegrityError("nonfinite record")
        ordered[draw * spec.walkers + walker] = row
    if any(row is None for row in ordered):
        raise RecordIntegrityError("missing draw or uncommitted tail")
    return [row for row in ordered if row is not None]


def validate_term_sum_and_sign(rows: list[dict[str, Any]], term_names: list[str]) -> None:
    """Refuse a row whose term energies do not sum to its local energy or whose sign is not +-1."""

    _require(isinstance(term_names, list) and bool(term_names), "record is missing term names")
    for row in rows:
        try:
            energy = float(row["local_energy"])
            terms = [float(row[f"term/{name}"]) for name in term_names]
            sign = float(row["sign"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordIntegrityError("malformed term energy or sign") from exc
        _require(sign in (-1.0, 1.0), "record sign is not +-1")
        _require(
            math.isclose(energy, math.fsum(terms), rel_tol=_TERM_SUM_REL_TOL, abs_tol=_TERM_SUM_ABS_TOL),
            "local energy does not equal the sum of term energies",
        )


def validate_variance_reconciliation(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    """Refuse a record whose rows do not reconcile with the manifest's whole-trajectory variance."""

    expected = manifest.get("variance")
    _require(isinstance(expected, int | float), "record manifest is missing the trajectory variance")
    values = [float(row["local_energy"]) for row in rows]
    mean = math.fsum(values) / len(values)
    variance = math.fsum(value * value for value in values) / len(values) - mean * mean
    _require(
        math.isclose(variance, float(expected), rel_tol=_VARIANCE_REL_TOL, abs_tol=_VARIANCE_ABS_TOL),
        "whole-trajectory variance does not reconcile with the record manifest",
    )


def validate_restore_event(occurrences_path: Path, *, checkpoint_dir: Path, step: int) -> None:
    """Confirm the model-only restore from the emitted CheckpointRestored event.

    The declared ``load.mode: model_only`` config is not itself proof a restore
    happened as claimed; this binds the collector to the typed event
    ``tpen.checkpoint.events.CheckpointRestored`` actually emits to
    ``occurrences.jsonl`` (see ``tpen.artifacts.write_occurrence_artifact``).
    """

    _require(occurrences_path.is_file(), "missing checkpoint restore event log")
    reports: list[dict[str, Any]] = []
    with occurrences_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("event") == CHECKPOINT_RESTORED_EVENT:
                reports.append(record.get("fields", {}).get("report", {}))
    _require(len(reports) == 1, "evaluation must record exactly one checkpoint restore event")
    report = reports[0]
    _require(
        report.get("checkpoint_dir") == str(checkpoint_dir)
        and report.get("mode") == "model_only"
        and report.get("completed_updates") == step
        and report.get("next_iteration") == step
        and report.get("loaded_model") is True
        and not any(
            report.get(key) for key in ("loaded_optimizer", "loaded_rng", "loaded_sampler", "loaded_trainer")
        ),
        "checkpoint restore event differs from the intended fresh model-only restore",
    )


def validate_sampler_draw_receipt(sidecar_path: Path, spec: RecordSpec) -> None:
    """Validate the per-stream sampler draw receipt: draw count, burn-in, stride, acceptance."""

    _require(sidecar_path.is_file(), "missing sampler draw receipt")
    receipt = json.loads(sidecar_path.read_text(encoding="utf-8"))
    _require(
        receipt.get("schema") == SAMPLER_TRAJECTORY_DIAGNOSTICS_SCHEMA,
        "wrong sampler draw receipt schema",
    )
    retained = receipt.get("retained_draws")
    _require(
        isinstance(retained, list) and len(retained) == spec.draws,
        "sampler draw receipt draw count differs from the evaluation",
    )
    _require(receipt.get("n_walkers") == spec.walkers, "sampler draw receipt walker count differs from the evaluation")
    burn_in = receipt.get("sampler_burn_in")
    stride = receipt.get("draw_stride")
    _require(isinstance(burn_in, int) and not isinstance(burn_in, bool) and burn_in >= 0, "sampler draw receipt burn-in is invalid")
    _require(isinstance(stride, int) and not isinstance(stride, bool) and stride >= 1, "sampler draw receipt stride is invalid")
    for draw in retained:
        acceptance = draw.get("acceptance_rate")
        _require(
            isinstance(acceptance, int | float) and not isinstance(acceptance, bool) and 0.0 <= float(acceptance) <= 1.0,
            "sampler draw receipt acceptance rate is invalid",
        )


def validate_checkpoint_manifest_hashes(checkpoint_dir: Path, manifest: dict[str, Any]) -> None:
    """Validate that every checkpoint file's content hash matches its manifest."""

    files = manifest.get("files")
    hashes = manifest.get("hashes")
    _require(isinstance(files, dict) and bool(files), "checkpoint manifest has no files")
    _require(isinstance(hashes, dict), "checkpoint manifest has no hashes")
    resolved_checkpoint_dir = checkpoint_dir.resolve()
    for component, relative in files.items():
        path = checkpoint_dir / relative
        _require(
            path.resolve().is_relative_to(resolved_checkpoint_dir),
            "checkpoint component escapes its directory",
        )
        _require(path.is_file(), "missing checkpoint component file")
        expected = hashes.get(f"{component}_sha256")
        _require(
            isinstance(expected, str) and file_sha256(path) == expected,
            "checkpoint file content does not match its manifest",
        )


def validate_training_log_completeness(checkpoints_dir: Path, *, expected_updates: int) -> None:
    """Refuse a training run that skipped an update or stopped short of its contract target.

    Every checkpoint's own ``completed_updates``/``next_iteration`` counters
    must agree with its directory step: a skipped update would make
    ``completed_updates`` permanently lag ``next_iteration`` from that
    checkpoint onward, so requiring equality at every checkpoint is enough to
    rule out a skip anywhere in the run, without needing a per-update log row.
    """

    step_dirs = sorted(
        (path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.startswith("step_")),
        key=lambda path: path.name,
    )
    _require(bool(step_dirs), "training run has no checkpoints")
    reached_target = False
    for step_dir in step_dirs:
        _require((step_dir / "COMPLETE").is_file(), "incomplete checkpoint directory")
        manifest = json.loads((step_dir / "manifest.json").read_text(encoding="utf-8"))
        try:
            step = int(step_dir.name.removeprefix("step_"))
        except ValueError as exc:
            raise RecordIntegrityError("malformed checkpoint step directory name") from exc
        _require(
            manifest.get("completed_updates") == step and manifest.get("next_iteration") == step,
            "training log contains a skipped or mislabeled update",
        )
        reached_target = reached_target or step == expected_updates
    _require(reached_target, "training log does not contain the declared number of updates")


def validate_run_is_clean_and_complete(status_path: Path, metadata_path: Path) -> None:
    """Refuse a run whose source worktree was dirty or whose status is not completed."""

    status = json.loads(status_path.read_text(encoding="utf-8"))
    _require(status.get("status") == "completed", "run did not complete")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _require(metadata.get("dirty_worktree") is False, "run source worktree was dirty")


def validate_training_run(training_dir: Path, *, expected_updates: int) -> None:
    """Validate one training run's completion, cleanliness, and checkpoint integrity."""

    validate_run_is_clean_and_complete(training_dir / "status.json", training_dir / "metadata.json")
    checkpoints_dir = training_dir / "checkpoints"
    validate_training_log_completeness(checkpoints_dir, expected_updates=expected_updates)
    for step_dir in checkpoints_dir.iterdir():
        if step_dir.is_dir() and step_dir.name.startswith("step_"):
            manifest = json.loads((step_dir / "manifest.json").read_text(encoding="utf-8"))
            validate_checkpoint_manifest_hashes(step_dir, manifest)


def collect_records(
    *,
    csv_path: Path,
    metadata_path: Path,
    resolved_config_path: Path,
    run_metadata_path: Path,
    checkpoint_dir: Path,
    occurrences_path: Path,
    sampler_diagnostics_path: Path,
    checkpoint_step: int,
    spec: RecordSpec,
) -> dict[str, Any]:
    """Read and validate one complete real evaluation stream, returning a receipt."""

    validate_checkpoint_identity(checkpoint_dir, spec)
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    resolved_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8"))
    validate_run_identity(run_metadata, resolved_config_path, spec)
    validate_restore_mode(resolved_config)
    validate_restore_event(occurrences_path, checkpoint_dir=checkpoint_dir, step=checkpoint_step)
    if not metadata_path.is_file():
        raise RecordIntegrityError("uncommitted tail")
    manifest = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, csv_path, spec)
    validate_sampler_draw_receipt(sampler_diagnostics_path, spec)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    ordered = validate_rows(rows, spec)
    validate_term_sum_and_sign(ordered, manifest.get("term_names"))
    validate_variance_reconciliation(ordered, manifest)
    return {
        "schema": "tpen.h-v2.collection/v1",
        "status": "complete",
        "source_sha": spec.source_sha,
        "checkpoint_model_sha256": spec.checkpoint_model_sha256,
        "resolved_config_sha256": spec.resolved_config_sha256,
        "row_count": len(ordered),
        "csv_sha256": manifest["csv_sha256"],
        "statistic_input": "complete_draw_by_walker_grid",
    }


__all__ = [
    "CHECKPOINT_RESTORED_EVENT",
    "RecordIntegrityError",
    "RecordSpec",
    "ROW_SEMANTICS",
    "SAMPLER_TRAJECTORY_DIAGNOSTICS_SCHEMA",
    "TRAJECTORY_RECORD_SCHEMA",
    "collect_records",
    "expected_resolved_config_identity_from_path",
    "file_sha256",
    "validate_checkpoint_identity",
    "validate_checkpoint_manifest_hashes",
    "validate_manifest",
    "validate_restore_event",
    "validate_restore_mode",
    "validate_rows",
    "validate_run_identity",
    "validate_run_is_clean_and_complete",
    "validate_sampler_draw_receipt",
    "validate_term_sum_and_sign",
    "validate_training_log_completeness",
    "validate_training_run",
    "validate_variance_reconciliation",
]
