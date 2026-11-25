"""Recovery-safe reconciliation and bounded operator summaries for TPEN observability.

This module implements pure, strictly read-only reconciliation over versioned
``tpen.run-observability/v1`` bundles produced by
:class:`tpen.observability_artifacts.ObservabilityBundleWriter` and
:class:`tpen.callback.resource_samples.ResourceSamples`:

* Verifies additive recovery-safe open/finalized markers
  (``lifecycle-marker.json``) against ``manifest.json`` without mutating files
  or breaking backward compatibility with O2 bundles lacking a marker.
* Classifies each invocation bundle into one of four explicit recovery states:
  ``"clean_final"``, ``"failed"``, ``"open_interrupted"``, or
  ``"missing_final_status"``, while keeping ``lifecycle_state``,
  ``data_completeness``, and ``finalization`` distinct.
* Reconciles per-observer batch counts (``attempted``, ``written``,
  ``dropped``, ``failed``, optional ``pending``) against recovered JSONL sample
  packets, diagnosing torn tails, invalid interior rows, and count
  discrepancies separately, and reporting ``None``/``"unknown"`` rather than
  inventing ``0`` when final status or readings are missing.
* Produces bounded, JSON-safe operator summaries (:class:`BoundedOperatorSummary`)
  enforcing explicit row, quantity, observer, caveat, diagnostic, profile-row,
  and text/line limits with explicit truncation markers, full identity linkage,
  paired raw CPU core-equivalents, and strict O2/r5 metric distinctions.

This module is strictly Torch-free and ``psutil``-free, never mutates files on
disk, and imports nothing from ``experiments`` or ``tpen_executor``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from tpen.observability_artifacts import (
    LIFECYCLE_MARKER_FILENAME,
    MANIFEST_FILENAME,
    OBSERVABILITY_DATA_COMPLETENESS,
    OBSERVABILITY_LIFECYCLE_STATES,
    RUN_OBSERVABILITY_SCHEMA,
    ObservabilityArtifactError,
    ObservabilityBundleInspection,
    ObservabilityDiagnostic,
    ObservabilityInventory,
    ObservabilityLifecycleMarker,
    ObservabilityManifest,
    ObservabilityPathError,
    ObservabilityProfileStatus,
    ObservabilityProfileSummary,
    ObservabilitySample,
    ObservabilitySchemaError,
    ObservabilityStatus,
    QuantityObservationSummary,
    _bound_json_value,
    _bound_text,
    _check_no_symlinks_under,
    _loads_strict_json,
    _subject_generation_key,
    inspect_observability_bundle,
    inspect_run_observability,
)

OBSERVABILITY_RECONCILIATION_SCHEMA = "tpen.run-observability-reconciliation/v1"
OBSERVABILITY_OPERATOR_SUMMARY_SCHEMA = "tpen.run-observability-summary/v1"

OBSERVABILITY_RECONCILIATION_STATES: frozenset[str] = frozenset(
    {"clean_final", "failed", "open_interrupted", "missing_final_status"}
)
OBSERVABILITY_BATCH_STATUSES: frozenset[str] = frozenset(
    {"complete", "partial", "empty", "open_pending", "inconsistent", "unknown"}
)
OBSERVABILITY_MARKER_STATES: frozenset[str] = frozenset(
    {"open", "finalized", "unsafe", "absent", "invalid"}
)

_FAILURE_OUTCOMES: frozenset[str] = frozenset(
    {"failed", "error", "interrupted", "aborted", "killed", "crashed", "timeout"}
)
_FAILURE_WRITER_STATES: frozenset[str] = frozenset({"error", "unsafe", "failed"})
_FAILURE_HELPER_STATES: frozenset[str] = frozenset(
    {"unconfirmed_timeout", "error", "unsafe", "failed"}
)

# Canonical paired user/system CPU counter quantity names across O2 fixtures
# and O3 RUsageSource production records.
_CPU_COUNTER_PAIRS: tuple[tuple[str, str], ...] = (
    ("process.rusage.user_cpu_seconds", "process.rusage.system_cpu_seconds"),
    ("cpu_user_s", "cpu_system_s"),
)


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def _truncate_text(
    text: str,
    *,
    max_chars: int,
    reasons: list[str],
    reason_tag: str = "text_truncated",
) -> str:
    """Bound ``text`` to ``max_chars`` characters while recording truncation."""

    return _bound_text(
        text,
        max_chars=max_chars,
        reasons=reasons,
        reason_tag=reason_tag,
    )


# ---------------------------------------------------------------------------
# Public immutable reconciliation & bounded summary types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OperatorSummaryBounds:
    """Explicit row, collection, and text bounds for bounded operator summaries.

    Parameters
    ----------
    max_quantities : int, optional
        Maximum quantity summary rows included (default ``32``).
    max_observers : int, optional
        Maximum per-observer reconciliation rows included (default ``16``).
    max_diagnostics : int, optional
        Maximum diagnostic rows included (default ``32``).
    max_caveats : int, optional
        Maximum caveat strings included (default ``16``).
    max_profile_windows : int, optional
        Maximum profiling windows included (default ``8``).
    max_profile_rows : int, optional
        Maximum operator-table rows included per profile summary (default ``20``).
    max_text_chars : int, optional
        Maximum character length for bounded diagnostic/caveat/detail text
        fields (default ``256``).
    max_rendered_lines : int, optional
        Maximum lines emitted by :func:`render_operator_summary` (default ``80``).
    max_invocations : int, optional
        Maximum invocations included by :func:`reconcile_run_observability`
        (default ``16``).
    """

    max_quantities: int = 32
    max_observers: int = 16
    max_diagnostics: int = 32
    max_caveats: int = 16
    max_profile_windows: int = 8
    max_profile_rows: int = 20
    max_text_chars: int = 256
    max_rendered_lines: int = 80
    max_invocations: int = 16

    def __post_init__(self) -> None:
        _require_positive_int(self.max_quantities, "OperatorSummaryBounds.max_quantities")
        _require_positive_int(self.max_observers, "OperatorSummaryBounds.max_observers")
        _require_positive_int(self.max_diagnostics, "OperatorSummaryBounds.max_diagnostics")
        _require_positive_int(self.max_caveats, "OperatorSummaryBounds.max_caveats")
        _require_positive_int(
            self.max_profile_windows, "OperatorSummaryBounds.max_profile_windows"
        )
        _require_positive_int(
            self.max_profile_rows, "OperatorSummaryBounds.max_profile_rows"
        )
        _require_positive_int(self.max_text_chars, "OperatorSummaryBounds.max_text_chars")
        _require_positive_int(
            self.max_rendered_lines, "OperatorSummaryBounds.max_rendered_lines"
        )
        _require_positive_int(
            self.max_invocations, "OperatorSummaryBounds.max_invocations"
        )

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-safe dictionary of configured bounds."""

        return {
            "max_quantities": self.max_quantities,
            "max_observers": self.max_observers,
            "max_diagnostics": self.max_diagnostics,
            "max_caveats": self.max_caveats,
            "max_profile_windows": self.max_profile_windows,
            "max_profile_rows": self.max_profile_rows,
            "max_text_chars": self.max_text_chars,
            "max_rendered_lines": self.max_rendered_lines,
            "max_invocations": self.max_invocations,
        }


@dataclass(frozen=True)
class LifecycleMarkerVerification:
    """Verification outcome for additive ``lifecycle-marker.json`` against ``manifest.json``.

    Parameters
    ----------
    present : bool
        Whether ``lifecycle-marker.json`` existed on disk.
    state : {"open", "finalized", "unsafe", "absent", "invalid"}
        Observed marker state.
    matches_manifest : bool
        ``True`` when the marker is absent (O2 backward-compatible) or agrees
        with ``manifest.json`` lifecycle state, identities, and digest.
    manifest_sha256_verified : bool or None
        ``True`` when present and ``manifest.json`` SHA-256 + byte length match,
        ``False`` on mismatch/invalid marker, ``None`` when marker is absent.
    diagnostic_code : str or None, optional
        Machine-readable diagnostic code when verification detected an issue.
    detail : str or None, optional
        Bounded diagnostic explanation when verification detected an issue.
    """

    present: bool
    state: str
    matches_manifest: bool
    manifest_sha256_verified: bool | None
    diagnostic_code: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if type(self.present) is not bool:
            raise TypeError("LifecycleMarkerVerification.present must be bool")
        if self.state not in OBSERVABILITY_MARKER_STATES:
            raise ValueError(
                f"unsupported LifecycleMarkerVerification.state {self.state!r}"
            )
        if type(self.matches_manifest) is not bool:
            raise TypeError("LifecycleMarkerVerification.matches_manifest must be bool")
        if (
            self.manifest_sha256_verified is not None
            and type(self.manifest_sha256_verified) is not bool
        ):
            raise TypeError(
                "LifecycleMarkerVerification.manifest_sha256_verified must be bool or None"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""

        return {
            "present": self.present,
            "state": self.state,
            "matches_manifest": self.matches_manifest,
            "manifest_sha256_verified": self.manifest_sha256_verified,
            "diagnostic_code": self.diagnostic_code,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ObserverBatchReconciliation:
    """Reconciled status and packet-count accounting for one observer.

    When final status is missing, ``attempted``, ``written``, ``dropped``,
    ``failed``, ``pending``, and ``written_vs_read_delta`` remain ``None``
    (unknown) rather than inventing ``0``.
    """

    observer_id: str
    has_final_status: bool
    is_profile_only: bool
    lifecycle_state: str
    data_completeness: str
    batch_status: str
    attempted: int | None
    written: int | None
    dropped: int | None
    failed: int | None
    pending: int | None
    read_valid_packets: int
    torn_tail_count: int
    interior_error_count: int
    written_vs_read_delta: int | None
    count_mismatch: bool
    application_outcome: str
    invocation_outcome: str
    writer_state: str
    flush_confirmed: bool | None
    fsync_confirmed: bool | None
    helper_cleanup_state: str
    truncation_reasons: tuple[str, ...] = ()
    diagnostic_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.batch_status not in OBSERVABILITY_BATCH_STATUSES:
            raise ValueError(
                f"unsupported ObserverBatchReconciliation.batch_status {self.batch_status!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""

        return {
            "observer_id": self.observer_id,
            "has_final_status": self.has_final_status,
            "is_profile_only": self.is_profile_only,
            "lifecycle_state": self.lifecycle_state,
            "data_completeness": self.data_completeness,
            "batch_status": self.batch_status,
            "attempted": self.attempted,
            "written": self.written,
            "dropped": self.dropped,
            "failed": self.failed,
            "pending": self.pending,
            "read_valid_packets": self.read_valid_packets,
            "torn_tail_count": self.torn_tail_count,
            "interior_error_count": self.interior_error_count,
            "written_vs_read_delta": self.written_vs_read_delta,
            "count_mismatch": self.count_mismatch,
            "application_outcome": self.application_outcome,
            "invocation_outcome": self.invocation_outcome,
            "writer_state": self.writer_state,
            "flush_confirmed": self.flush_confirmed,
            "fsync_confirmed": self.fsync_confirmed,
            "helper_cleanup_state": self.helper_cleanup_state,
            "truncation_reasons": list(self.truncation_reasons),
            "diagnostic_codes": list(self.diagnostic_codes),
        }


@dataclass(frozen=True)
class CpuCoreEquivalentsSummary:
    """Raw paired CPU core-equivalents without guessed quota normalization.

    Computed strictly from consecutive valid sample pairs where both self user
    CPU and self system CPU readings are ``"ok"``, share clock domain and
    subject generation, have strictly sequential sequence numbers, non-negative
    deltas, and positive elapsed monotonic time.
    """

    observer_id: str
    source_id: str
    clock_domains: tuple[str, ...]
    valid_interval_count: int
    broken_interval_count: int
    user_cpu_delta_seconds: float | None
    system_cpu_delta_seconds: float | None
    total_cpu_delta_seconds: float | None
    valid_elapsed_seconds: float | None
    cpu_core_equivalents: float | None
    capacity_normalized_utilization: float | None = None
    capacity_denominator_status: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""

        return {
            "observer_id": self.observer_id,
            "source_id": self.source_id,
            "clock_domains": list(self.clock_domains),
            "valid_interval_count": self.valid_interval_count,
            "broken_interval_count": self.broken_interval_count,
            "user_cpu_delta_seconds": self.user_cpu_delta_seconds,
            "system_cpu_delta_seconds": self.system_cpu_delta_seconds,
            "total_cpu_delta_seconds": self.total_cpu_delta_seconds,
            "valid_elapsed_seconds": self.valid_elapsed_seconds,
            "cpu_core_equivalents": self.cpu_core_equivalents,
            "capacity_normalized_utilization": self.capacity_normalized_utilization,
            "capacity_denominator_status": self.capacity_denominator_status,
        }


@dataclass(frozen=True)
class ObservabilityIdentityLink:
    """Complete linked provenance identities for one invocation summary."""

    invocation_id: str
    run_id: str
    config_hash: str
    git_commit: str
    git_commit_status: str
    inventory_id: str
    descriptor_set_id: str
    observer_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    clock_domains: tuple[str, ...]
    executor_context: str | Mapping[str, Any]
    scheduler_accounting: str | Mapping[str, Any]
    billed_units: str | Mapping[str, Any]
    device_bindings: tuple[Mapping[str, Any], ...] = ()
    sharing_and_mig_caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""

        return {
            "invocation_id": self.invocation_id,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "git_commit": self.git_commit,
            "git_commit_status": self.git_commit_status,
            "inventory_id": self.inventory_id,
            "descriptor_set_id": self.descriptor_set_id,
            "observer_ids": list(self.observer_ids),
            "source_ids": list(self.source_ids),
            "clock_domains": list(self.clock_domains),
            "executor_context": (
                self.executor_context
                if isinstance(self.executor_context, str)
                else dict(self.executor_context)
            ),
            "scheduler_accounting": (
                self.scheduler_accounting
                if isinstance(self.scheduler_accounting, str)
                else dict(self.scheduler_accounting)
            ),
            "billed_units": (
                self.billed_units
                if isinstance(self.billed_units, str)
                else dict(self.billed_units)
            ),
            "device_bindings": [dict(item) for item in self.device_bindings],
            "sharing_and_mig_caveats": list(self.sharing_and_mig_caveats),
        }


@dataclass(frozen=True)
class BoundedOperatorQuantityRow:
    """Bounded operator-facing quantity row preserving every O2/r5 distinction."""

    observer_id: str
    source_id: str
    quantity_id: str
    unit: str
    statistic: str
    subject_kind: str
    window: str
    clock_domains: tuple[str, ...]
    sample_count: int
    ok_count: int
    unavailable_count: int
    error_count: int
    reason_codes: tuple[str, ...]
    caveats: tuple[str, ...]
    # Gauge fields
    sampled_min: int | float | None = None
    sampled_max: int | float | None = None
    sampled_max_is_lower_bound: bool = False
    # High-water fields
    observed_max: int | float | None = None
    # Counter fields
    valid_interval_count: int = 0
    broken_interval_count: int = 0
    valid_delta_total: float | None = None
    valid_elapsed_seconds: float | None = None
    valid_rate_per_second: float | None = None
    # Capacity fields
    latest_value: int | float | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    # Vendor-window-mean fields
    observation_min: int | float | None = None
    observation_max: int | float | None = None
    observation_mean: float | None = None
    continuous_window_integrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""

        return {
            "observer_id": self.observer_id,
            "source_id": self.source_id,
            "quantity_id": self.quantity_id,
            "unit": self.unit,
            "statistic": self.statistic,
            "subject_kind": self.subject_kind,
            "window": self.window,
            "clock_domains": list(self.clock_domains),
            "sample_count": self.sample_count,
            "ok_count": self.ok_count,
            "unavailable_count": self.unavailable_count,
            "error_count": self.error_count,
            "reason_codes": list(self.reason_codes),
            "caveats": list(self.caveats),
            "sampled_min": self.sampled_min,
            "sampled_max": self.sampled_max,
            "sampled_max_is_lower_bound": self.sampled_max_is_lower_bound,
            "observed_max": self.observed_max,
            "valid_interval_count": self.valid_interval_count,
            "broken_interval_count": self.broken_interval_count,
            "valid_delta_total": self.valid_delta_total,
            "valid_elapsed_seconds": self.valid_elapsed_seconds,
            "valid_rate_per_second": self.valid_rate_per_second,
            "latest_value": self.latest_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "observation_min": self.observation_min,
            "observation_max": self.observation_max,
            "observation_mean": self.observation_mean,
            "continuous_window_integrated": self.continuous_window_integrated,
        }


@dataclass(frozen=True)
class BoundedOperatorProfileWindow:
    """Bounded operator-facing projection of one profiling window."""

    observer_id: str
    window_id: str
    window_kind: str
    lifecycle_state: str
    data_completeness: str
    exporter: str
    export_status: str
    guarantee_class: str
    requested_activities: tuple[str, ...]
    accepted_activities: tuple[str, ...]
    total_rows: int
    rows: tuple[Mapping[str, Any], ...]
    truncated: bool
    truncation_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""

        return {
            "observer_id": self.observer_id,
            "window_id": self.window_id,
            "window_kind": self.window_kind,
            "lifecycle_state": self.lifecycle_state,
            "data_completeness": self.data_completeness,
            "exporter": self.exporter,
            "export_status": self.export_status,
            "guarantee_class": self.guarantee_class,
            "requested_activities": list(self.requested_activities),
            "accepted_activities": list(self.accepted_activities),
            "total_rows": self.total_rows,
            "rows": [dict(row) for row in self.rows],
            "truncated": self.truncated,
            "truncation_reasons": list(self.truncation_reasons),
        }


@dataclass(frozen=True)
class BoundedOperatorSummary:
    """Bounded, identity-linked operator summary over one reconciled bundle."""

    identity: ObservabilityIdentityLink
    reconciliation_state: str
    lifecycle_state: str
    data_completeness: str
    finalization: str
    assessment_reason: str | None
    lifecycle_marker: LifecycleMarkerVerification
    observers: tuple[ObserverBatchReconciliation, ...]
    total_observers: int
    cpu_core_equivalents: tuple[CpuCoreEquivalentsSummary, ...]
    quantities: tuple[BoundedOperatorQuantityRow, ...]
    total_quantities: int
    profile_windows: tuple[BoundedOperatorProfileWindow, ...]
    total_profile_windows: int
    caveats: tuple[str, ...]
    total_caveats: int
    diagnostics: tuple[ObservabilityDiagnostic, ...]
    total_diagnostics: int
    bounds: OperatorSummaryBounds
    truncated: bool
    truncation_reasons: tuple[str, ...]
    schema: str = OBSERVABILITY_OPERATOR_SUMMARY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OBSERVABILITY_OPERATOR_SUMMARY_SCHEMA:
            raise ValueError(
                f"unsupported BoundedOperatorSummary.schema {self.schema!r}"
            )
        if self.reconciliation_state not in OBSERVABILITY_RECONCILIATION_STATES:
            raise ValueError(
                f"unsupported reconciliation_state {self.reconciliation_state!r}"
            )
        if self.lifecycle_state not in OBSERVABILITY_LIFECYCLE_STATES:
            raise ValueError(
                f"unsupported lifecycle_state {self.lifecycle_state!r}"
            )
        if self.data_completeness not in OBSERVABILITY_DATA_COMPLETENESS:
            raise ValueError(
                f"unsupported data_completeness {self.data_completeness!r}"
            )
        if len(self.observers) > self.bounds.max_observers:
            raise ValueError("observers exceed bounds.max_observers")
        if len(self.quantities) > self.bounds.max_quantities:
            raise ValueError("quantities exceed bounds.max_quantities")
        if len(self.profile_windows) > self.bounds.max_profile_windows:
            raise ValueError("profile_windows exceed bounds.max_profile_windows")
        if len(self.caveats) > self.bounds.max_caveats:
            raise ValueError("caveats exceed bounds.max_caveats")
        if len(self.diagnostics) > self.bounds.max_diagnostics:
            raise ValueError("diagnostics exceed bounds.max_diagnostics")

    def quantity(
        self,
        source_id: str,
        quantity_id: str,
        *,
        observer_id: str | None = None,
    ) -> BoundedOperatorQuantityRow | None:
        """Look up one bounded quantity row by source and quantity ID."""

        for row in self.quantities:
            if row.source_id != source_id or row.quantity_id != quantity_id:
                continue
            if observer_id is not None and row.observer_id != observer_id:
                continue
            return row
        return None

    def render_text(
        self,
        *,
        max_lines: int | None = None,
        max_line_chars: int | None = None,
    ) -> str:
        """Render a bounded plain-text operator summary."""

        return render_operator_summary(
            self,
            max_lines=max_lines,
            max_line_chars=max_line_chars,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""

        return {
            "schema": self.schema,
            "identity": self.identity.to_dict(),
            "reconciliation_state": self.reconciliation_state,
            "lifecycle_state": self.lifecycle_state,
            "data_completeness": self.data_completeness,
            "finalization": self.finalization,
            "assessment_reason": self.assessment_reason,
            "lifecycle_marker": self.lifecycle_marker.to_dict(),
            "observers": [obs.to_dict() for obs in self.observers],
            "total_observers": self.total_observers,
            "cpu_core_equivalents": [item.to_dict() for item in self.cpu_core_equivalents],
            "quantities": [row.to_dict() for row in self.quantities],
            "total_quantities": self.total_quantities,
            "profile_windows": [win.to_dict() for win in self.profile_windows],
            "total_profile_windows": self.total_profile_windows,
            "caveats": list(self.caveats),
            "total_caveats": self.total_caveats,
            "diagnostics": [diag.to_dict() for diag in self.diagnostics],
            "total_diagnostics": self.total_diagnostics,
            "bounds": self.bounds.to_dict(),
            "truncated": self.truncated,
            "truncation_reasons": list(self.truncation_reasons),
        }


@dataclass(frozen=True)
class ReconciledObservabilityBundle:
    """Complete read-only reconciliation result for one invocation bundle."""

    bundle_dir: str
    invocation_id: str
    reconciliation_state: str
    lifecycle_state: str
    data_completeness: str
    finalization: str
    assessment_reason: str | None
    lifecycle_marker: LifecycleMarkerVerification
    observers: tuple[ObserverBatchReconciliation, ...]
    cpu_core_equivalents: tuple[CpuCoreEquivalentsSummary, ...]
    diagnostics: tuple[ObservabilityDiagnostic, ...]
    inspection: ObservabilityBundleInspection
    summary: BoundedOperatorSummary
    schema: str = OBSERVABILITY_RECONCILIATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != OBSERVABILITY_RECONCILIATION_SCHEMA:
            raise ValueError(
                f"unsupported ReconciledObservabilityBundle.schema {self.schema!r}"
            )
        if self.reconciliation_state not in OBSERVABILITY_RECONCILIATION_STATES:
            raise ValueError(
                f"unsupported reconciliation_state {self.reconciliation_state!r}"
            )

    def observer(self, observer_id: str) -> ObserverBatchReconciliation | None:
        """Look up one observer's batch reconciliation by ``observer_id``."""

        for item in self.observers:
            if item.observer_id == observer_id:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""

        return {
            "schema": self.schema,
            "bundle_dir": self.bundle_dir,
            "invocation_id": self.invocation_id,
            "reconciliation_state": self.reconciliation_state,
            "lifecycle_state": self.lifecycle_state,
            "data_completeness": self.data_completeness,
            "finalization": self.finalization,
            "assessment_reason": self.assessment_reason,
            "lifecycle_marker": self.lifecycle_marker.to_dict(),
            "observers": [obs.to_dict() for obs in self.observers],
            "cpu_core_equivalents": [item.to_dict() for item in self.cpu_core_equivalents],
            "diagnostics": [diag.to_dict() for diag in self.diagnostics],
            "summary": self.summary.to_dict(),
        }


@dataclass(frozen=True)
class ReconciledRunObservability:
    """Read-only reconciliation across all observability bundles in a run directory.

    Absent sidecars (legacy or uncollected runs) return ``status="absent"`` and
    ``data_completeness="unknown"`` with ``invocations=()``, never inventing
    zero utilization. Multiple invocations remain strictly isolated and are
    never merged across ``invocation_id``.
    """

    run_dir: str
    status: str
    data_completeness: str
    invocations: tuple[ReconciledObservabilityBundle, ...]
    total_invocations: int
    truncated: bool
    truncation_reasons: tuple[str, ...]
    schema: str = OBSERVABILITY_RECONCILIATION_SCHEMA

    def __post_init__(self) -> None:
        if self.status not in {"absent", "present"}:
            raise ValueError(f"unsupported ReconciledRunObservability.status {self.status!r}")
        if self.data_completeness not in OBSERVABILITY_DATA_COMPLETENESS:
            raise ValueError(
                f"unsupported ReconciledRunObservability.data_completeness {self.data_completeness!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict JSON-safe dictionary representation."""

        return {
            "schema": self.schema,
            "run_dir": self.run_dir,
            "status": self.status,
            "data_completeness": self.data_completeness,
            "invocations": [bundle.to_dict() for bundle in self.invocations],
            "total_invocations": self.total_invocations,
            "truncated": self.truncated,
            "truncation_reasons": list(self.truncation_reasons),
        }


# ---------------------------------------------------------------------------
# Internal read-only verification, batch reconciliation, and reduction helpers
# ---------------------------------------------------------------------------


def _verify_lifecycle_marker(
    bundle_dir: Path,
    manifest: ObservabilityManifest,
    diagnostics: list[ObservabilityDiagnostic],
) -> LifecycleMarkerVerification:
    """Verify optional additive ``lifecycle-marker.json`` without mutating disk."""

    marker_path = _check_no_symlinks_under(bundle_dir, [LIFECYCLE_MARKER_FILENAME])
    if not marker_path.exists():
        # O2 bundles written before O7 may omit lifecycle-marker.json; this is
        # valid and backward-compatible rather than a corruption error.
        return LifecycleMarkerVerification(
            present=False,
            state="absent",
            matches_manifest=True,
            manifest_sha256_verified=None,
            diagnostic_code=None,
            detail=None,
        )
    if not marker_path.is_file():
        raise ObservabilityPathError(
            f"{LIFECYCLE_MARKER_FILENAME} must be a regular file"
        )

    try:
        raw_text = marker_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        diag = ObservabilityDiagnostic(
            code="invalid_lifecycle_marker",
            message=f"failed to read {LIFECYCLE_MARKER_FILENAME}: {exc}",
            path=LIFECYCLE_MARKER_FILENAME,
        )
        diagnostics.append(diag)
        return LifecycleMarkerVerification(
            present=True,
            state="invalid",
            matches_manifest=False,
            manifest_sha256_verified=False,
            diagnostic_code="invalid_lifecycle_marker",
            detail=diag.message,
        )

    try:
        parsed = _loads_strict_json(raw_text, context=LIFECYCLE_MARKER_FILENAME)
    except ObservabilitySchemaError as exc:
        # Reject literal NaN/Infinity strictly; otherwise classify corrupt/torn
        # JSON marker as an invalid marker diagnostic so recovery stays readable.
        if "non-finite JSON constant" in str(exc):
            raise
        diag = ObservabilityDiagnostic(
            code="invalid_lifecycle_marker",
            message=f"corrupt JSON in {LIFECYCLE_MARKER_FILENAME}: {exc}",
            path=LIFECYCLE_MARKER_FILENAME,
        )
        diagnostics.append(diag)
        return LifecycleMarkerVerification(
            present=True,
            state="invalid",
            matches_manifest=False,
            manifest_sha256_verified=False,
            diagnostic_code="invalid_lifecycle_marker",
            detail=diag.message,
        )

    if isinstance(parsed, Mapping):
        raw_schema = parsed.get("schema")
        if isinstance(raw_schema, str) and raw_schema != RUN_OBSERVABILITY_SCHEMA:
            raise ObservabilitySchemaError(
                f"unsupported lifecycle_marker schema {raw_schema!r}; "
                f"expected {RUN_OBSERVABILITY_SCHEMA!r}"
            )

    try:
        marker = ObservabilityLifecycleMarker.from_mapping(parsed)
    except (ObservabilitySchemaError, ObservabilityPathError) as exc:
        diag = ObservabilityDiagnostic(
            code="invalid_lifecycle_marker",
            message=f"invalid {LIFECYCLE_MARKER_FILENAME}: {exc}",
            path=LIFECYCLE_MARKER_FILENAME,
        )
        diagnostics.append(diag)
        return LifecycleMarkerVerification(
            present=True,
            state="invalid",
            matches_manifest=False,
            manifest_sha256_verified=False,
            diagnostic_code="invalid_lifecycle_marker",
            detail=diag.message,
        )

    # Cross-check invocation & manifest identities.
    identity_mismatches: list[str] = []
    if marker.invocation_id != manifest.invocation_id:
        identity_mismatches.append(
            f"invocation_id {marker.invocation_id!r} != {manifest.invocation_id!r}"
        )
    if marker.run_id != manifest.run_id:
        identity_mismatches.append(f"run_id {marker.run_id!r} != {manifest.run_id!r}")
    if marker.configuration_digest != manifest.configuration_digest:
        identity_mismatches.append("configuration_digest mismatch")
    if marker.inventory_id != manifest.inventory_id:
        identity_mismatches.append(
            f"inventory_id {marker.inventory_id!r} != {manifest.inventory_id!r}"
        )
    if marker.descriptor_set_id != manifest.descriptor_set_id:
        identity_mismatches.append(
            f"descriptor_set_id {marker.descriptor_set_id!r} != {manifest.descriptor_set_id!r}"
        )
    expected_observers = tuple(
        sorted(str(obs["observer_id"]) for obs in manifest.observers)
    )
    if marker.observer_ids != expected_observers:
        identity_mismatches.append(
            f"observer_ids {marker.observer_ids!r} != {expected_observers!r}"
        )
    if marker.artifact_count != len(manifest.artifacts):
        identity_mismatches.append(
            f"artifact_count {marker.artifact_count} != {len(manifest.artifacts)}"
        )
    if identity_mismatches:
        detail = "; ".join(identity_mismatches)
        # Finalization adds the closed manifest's artifact references.  If the
        # manifest replace won but the marker replace did not, the old open
        # marker necessarily has the pre-finalization artifact count.  Preserve
        # that protocol evidence for reconciliation instead of collapsing it
        # into an unsafe/tampered marker.
        if (
            marker.state == "open"
            and manifest.lifecycle_state == "closed"
            and identity_mismatches == [
                f"artifact_count {marker.artifact_count} != {len(manifest.artifacts)}"
            ]
        ):
            diagnostics.append(
                ObservabilityDiagnostic(
                    code="lifecycle_marker_state_mismatch",
                    message=(
                        f"{LIFECYCLE_MARKER_FILENAME} is the prior open marker "
                        f"while manifest.json is closed ({detail})"
                    ),
                    path=LIFECYCLE_MARKER_FILENAME,
                )
            )
            return LifecycleMarkerVerification(
                present=True,
                state="open",
                matches_manifest=False,
                manifest_sha256_verified=False,
                diagnostic_code="lifecycle_marker_state_mismatch",
                detail=detail,
            )
        diagnostics.append(
            ObservabilityDiagnostic(
                code="lifecycle_marker_identity_mismatch",
                message=f"{LIFECYCLE_MARKER_FILENAME} disagrees with manifest identity: {detail}",
                path=LIFECYCLE_MARKER_FILENAME,
            )
        )
        return LifecycleMarkerVerification(
            present=True,
            state="invalid",
            matches_manifest=False,
            manifest_sha256_verified=False,
            diagnostic_code="lifecycle_marker_identity_mismatch",
            detail=detail,
        )

    manifest_path = _check_no_symlinks_under(bundle_dir, [MANIFEST_FILENAME])
    manifest_bytes = manifest_path.read_bytes()
    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    hash_ok = (
        len(manifest_bytes) == marker.manifest_bytes
        and actual_sha256 == marker.manifest_sha256
    )

    expected_manifest_state = "closed" if marker.state == "finalized" else marker.state
    if manifest.lifecycle_state != expected_manifest_state:
        detail = (
            f"lifecycle marker state is {marker.state!r} "
            f"(expects manifest {expected_manifest_state!r}), "
            f"but manifest.lifecycle_state is {manifest.lifecycle_state!r}"
        )
        diagnostics.append(
            ObservabilityDiagnostic(
                code="lifecycle_marker_state_mismatch",
                message=detail,
                path=LIFECYCLE_MARKER_FILENAME,
            )
        )
        return LifecycleMarkerVerification(
            present=True,
            state=marker.state,
            matches_manifest=False,
            manifest_sha256_verified=hash_ok,
            diagnostic_code="lifecycle_marker_state_mismatch",
            detail=detail,
        )

    if not hash_ok:
        detail = (
            f"manifest.json digest/size ({len(manifest_bytes)} B, {actual_sha256[:12]}...) "
            f"does not match {LIFECYCLE_MARKER_FILENAME} "
            f"({marker.manifest_bytes} B, {marker.manifest_sha256[:12]}...)"
        )
        diagnostics.append(
            ObservabilityDiagnostic(
                code="lifecycle_marker_hash_mismatch",
                message=detail,
                path=LIFECYCLE_MARKER_FILENAME,
            )
        )
        return LifecycleMarkerVerification(
            present=True,
            state=marker.state,
            matches_manifest=False,
            manifest_sha256_verified=False,
            diagnostic_code="lifecycle_marker_hash_mismatch",
            detail=detail,
        )

    return LifecycleMarkerVerification(
        present=True,
        state=marker.state,
        matches_manifest=True,
        manifest_sha256_verified=True,
        diagnostic_code=None,
        detail=None,
    )


def _reconcile_observers(
    inspection: ObservabilityBundleInspection,
    extra_diagnostics: list[ObservabilityDiagnostic],
) -> tuple[ObserverBatchReconciliation, ...]:
    """Reconcile per-observer status counts against recovered sample packets."""

    profile_statuses_by_obs: dict[str, list[ObservabilityProfileStatus]] = {}
    for ps in inspection.profile_statuses:
        profile_statuses_by_obs.setdefault(ps.observer_id, []).append(ps)

    all_observer_ids: set[str] = {
        str(obs["observer_id"]) for obs in inspection.manifest.observers
    }
    all_observer_ids.update(inspection.samples.keys())
    all_observer_ids.update(inspection.statuses.keys())
    all_observer_ids.update(profile_statuses_by_obs.keys())

    torn_by_obs: dict[str, int] = {}
    for diag in inspection.torn_tails:
        if diag.observer_id is not None:
            torn_by_obs[diag.observer_id] = torn_by_obs.get(diag.observer_id, 0) + 1

    interior_by_obs: dict[str, int] = {}
    for diag in inspection.interior_errors:
        if diag.observer_id is not None:
            interior_by_obs[diag.observer_id] = (
                interior_by_obs.get(diag.observer_id, 0) + 1
            )

    reconciled: list[ObserverBatchReconciliation] = []
    for obs_id in sorted(all_observer_ids):
        stream = inspection.samples.get(obs_id, ())
        read_valid = len(stream)
        torn_count = torn_by_obs.get(obs_id, 0)
        interior_count = interior_by_obs.get(obs_id, 0)
        status: ObservabilityStatus | None = inspection.statuses.get(obs_id)
        pstatuses = profile_statuses_by_obs.get(obs_id, [])

        obs_diag_codes: list[str] = []
        if torn_count > 0:
            _append_unique(obs_diag_codes, "torn_tail")
        if interior_count > 0:
            _append_unique(obs_diag_codes, "invalid_interior_record")

        # Profile-only observer (no sample stream and no sample status file).
        if status is None and pstatuses and obs_id not in inspection.samples:
            if any(ps.lifecycle_state == "unsafe" for ps in pstatuses):
                prof_lifecycle = "unsafe"
            elif any(ps.lifecycle_state == "open" for ps in pstatuses):
                prof_lifecycle = "open"
            else:
                prof_lifecycle = "closed"

            comp_vals = [ps.data_completeness for ps in pstatuses]
            if "unknown" in comp_vals:
                prof_comp = "unknown"
            elif "partial" in comp_vals:
                prof_comp = "partial"
            elif all(val == "none" for val in comp_vals):
                prof_comp = "none"
            else:
                prof_comp = "complete"

            prof_trunc: list[str] = []
            for ps in pstatuses:
                for reason in ps.truncation_reasons or ():
                    _append_unique(prof_trunc, reason)
            export_failed = any(
                ps.export_status.lower() in _FAILURE_OUTCOMES for ps in pstatuses
            )
            if export_failed:
                _append_unique(obs_diag_codes, "profile_export_failed")
            reconciled.append(
                ObserverBatchReconciliation(
                    observer_id=obs_id,
                    has_final_status=True,
                    is_profile_only=True,
                    lifecycle_state=prof_lifecycle,
                    data_completeness=prof_comp,
                    batch_status="complete" if prof_comp == "complete" else "partial",
                    attempted=None,
                    written=None,
                    dropped=None,
                    failed=None,
                    pending=None,
                    read_valid_packets=0,
                    torn_tail_count=0,
                    interior_error_count=0,
                    written_vs_read_delta=None,
                    count_mismatch=False,
                    application_outcome="not_supplied",
                    invocation_outcome="failed" if export_failed else "completed",
                    writer_state="not_applicable",
                    flush_confirmed=None,
                    fsync_confirmed=None,
                    helper_cleanup_state="not_applicable",
                    truncation_reasons=tuple(prof_trunc),
                    diagnostic_codes=tuple(obs_diag_codes),
                )
            )
            continue

        # Missing final status: never fabricate zero counts!
        if status is None:
            _append_unique(obs_diag_codes, "missing_final_status")
            reconciled.append(
                ObserverBatchReconciliation(
                    observer_id=obs_id,
                    has_final_status=False,
                    is_profile_only=False,
                    lifecycle_state="open",
                    data_completeness="partial" if read_valid > 0 else "unknown",
                    batch_status="unknown",
                    attempted=None,
                    written=None,
                    dropped=None,
                    failed=None,
                    pending=None,
                    read_valid_packets=read_valid,
                    torn_tail_count=torn_count,
                    interior_error_count=interior_count,
                    written_vs_read_delta=None,
                    count_mismatch=False,
                    application_outcome="unknown",
                    invocation_outcome="unknown",
                    writer_state="unknown",
                    flush_confirmed=None,
                    fsync_confirmed=None,
                    helper_cleanup_state="unknown",
                    truncation_reasons=(),
                    diagnostic_codes=tuple(obs_diag_codes),
                )
            )
            continue

        attempted = int(status.counts["attempted"])
        written = int(status.counts["written"])
        dropped = int(status.counts["dropped"])
        failed = int(status.counts["failed"])
        pending_raw = status.counts.get("pending")
        pending = (
            int(pending_raw)
            if pending_raw is not None
            else (0 if status.lifecycle_state == "closed" else max(0, attempted - (written + dropped + failed)))
        )

        delta = read_valid - written
        count_mismatch = delta != 0
        if count_mismatch:
            _append_unique(obs_diag_codes, "batch_count_mismatch")
            extra_diagnostics.append(
                ObservabilityDiagnostic(
                    code="batch_count_mismatch",
                    message=(
                        f"observer {obs_id!r} status reports written={written} "
                        f"(attempted={attempted}, dropped={dropped}, failed={failed}), "
                        f"but recovered {read_valid} valid sample packet(s) "
                        f"(delta={delta:+d}, torn_tails={torn_count}, "
                        f"interior_errors={interior_count})"
                    ),
                    path=f"status/{obs_id}.json",
                    observer_id=obs_id,
                )
            )

        if count_mismatch or interior_count > 0:
            batch_status = "inconsistent"
        elif status.lifecycle_state == "open" or pending > 0 or torn_count > 0:
            batch_status = "open_pending"
        elif attempted == 0 and written == 0 and read_valid == 0:
            batch_status = "empty"
        elif dropped > 0 or failed > 0 or bool(status.truncation_reasons):
            batch_status = "partial"
        else:
            batch_status = "complete"

        writer_state = str(status.writer.get("state", "unknown"))
        raw_flush = status.writer.get("flush_confirmed")
        flush_confirmed = raw_flush if type(raw_flush) is bool else None
        raw_fsync = status.writer.get("fsync_confirmed")
        fsync_confirmed = raw_fsync if type(raw_fsync) is bool else None
        helper_state = str(status.helper_cleanup.get("state", "unknown"))

        if writer_state in _FAILURE_WRITER_STATES:
            _append_unique(obs_diag_codes, f"writer_{writer_state}")
        if helper_state in _FAILURE_HELPER_STATES:
            _append_unique(obs_diag_codes, f"helper_cleanup_{helper_state}")
        if status.application_outcome.lower() in _FAILURE_OUTCOMES:
            _append_unique(obs_diag_codes, "application_failed")
        if status.invocation_outcome.lower() in _FAILURE_OUTCOMES:
            _append_unique(obs_diag_codes, "invocation_failed")

        reconciled.append(
            ObserverBatchReconciliation(
                observer_id=obs_id,
                has_final_status=True,
                is_profile_only=False,
                lifecycle_state=status.lifecycle_state,
                data_completeness=status.data_completeness,
                batch_status=batch_status,
                attempted=attempted,
                written=written,
                dropped=dropped,
                failed=failed,
                pending=pending,
                read_valid_packets=read_valid,
                torn_tail_count=torn_count,
                interior_error_count=interior_count,
                written_vs_read_delta=delta,
                count_mismatch=count_mismatch,
                application_outcome=status.application_outcome,
                invocation_outcome=status.invocation_outcome,
                writer_state=writer_state,
                flush_confirmed=flush_confirmed,
                fsync_confirmed=fsync_confirmed,
                helper_cleanup_state=helper_state,
                truncation_reasons=status.truncation_reasons,
                diagnostic_codes=tuple(obs_diag_codes),
            )
        )

    return tuple(reconciled)


def _compute_cpu_core_equivalents(
    inspection: ObservabilityBundleInspection,
) -> tuple[CpuCoreEquivalentsSummary, ...]:
    """Compute paired raw CPU core-equivalents strictly from valid consecutive samples."""

    descriptors = inspection.descriptors
    if descriptors is None:
        return ()

    grouped: dict[tuple[str, str], list[ObservabilitySample]] = {}
    for observer_id, stream in sorted(inspection.samples.items()):
        for sample in stream:
            grouped.setdefault((observer_id, sample.source_id), []).append(sample)

    results: list[CpuCoreEquivalentsSummary] = []
    for (observer_id, source_id), source_samples in sorted(grouped.items()):
        ordered = sorted(source_samples, key=lambda item: item.sequence)
        observed_keys: set[str] = set()
        for sample in ordered:
            observed_keys.update(sample.readings.keys())

        matched_pair: tuple[str, str] | None = None
        for user_key, sys_key in _CPU_COUNTER_PAIRS:
            if user_key in observed_keys and sys_key in observed_keys:
                user_desc = descriptors.quantities.get(user_key)
                sys_desc = descriptors.quantities.get(sys_key)
                if (
                    user_desc is not None
                    and sys_desc is not None
                    and user_desc.statistic == "counter"
                    and sys_desc.statistic == "counter"
                    and user_desc.unit == "s"
                    and sys_desc.unit == "s"
                ):
                    matched_pair = (user_key, sys_key)
                    break

        if matched_pair is None:
            continue

        user_key, sys_key = matched_pair
        clocks: set[str] = {sample.clock_domain_id for sample in ordered}
        valid_intervals = 0
        broken_intervals = 0
        user_delta_sum = 0.0
        sys_delta_sum = 0.0
        elapsed_sum = 0.0

        for prev_sample, curr_sample in zip(ordered, ordered[1:]):
            if curr_sample.sequence != prev_sample.sequence + 1:
                broken_intervals += 1
                continue
            if prev_sample.clock_domain_id != curr_sample.clock_domain_id:
                broken_intervals += 1
                continue
            if _subject_generation_key(prev_sample.subject) != _subject_generation_key(
                curr_sample.subject
            ):
                broken_intervals += 1
                continue
            prev_u = prev_sample.readings.get(user_key)
            curr_u = curr_sample.readings.get(user_key)
            prev_s = prev_sample.readings.get(sys_key)
            curr_s = curr_sample.readings.get(sys_key)
            if (
                prev_u is None
                or curr_u is None
                or prev_s is None
                or curr_s is None
                or prev_u.status != "ok"
                or curr_u.status != "ok"
                or prev_s.status != "ok"
                or curr_s.status != "ok"
                or prev_u.value is None
                or curr_u.value is None
                or prev_s.value is None
                or curr_s.value is None
            ):
                broken_intervals += 1
                continue
            du = float(curr_u.value - prev_u.value)
            ds = float(curr_s.value - prev_s.value)
            elapsed_ns = curr_sample.read_end_ns - prev_sample.read_end_ns
            if du < 0.0 or ds < 0.0 or elapsed_ns <= 0:
                broken_intervals += 1
                continue
            valid_intervals += 1
            user_delta_sum += du
            sys_delta_sum += ds
            elapsed_sum += elapsed_ns / 1e9

        if valid_intervals > 0 and elapsed_sum > 0.0:
            total_delta = user_delta_sum + sys_delta_sum
            core_eq: float | None = total_delta / elapsed_sum
            u_out: float | None = user_delta_sum
            s_out: float | None = sys_delta_sum
            t_out: float | None = total_delta
            e_out: float | None = elapsed_sum
        else:
            core_eq = None
            u_out = None
            s_out = None
            t_out = None
            e_out = None

        results.append(
            CpuCoreEquivalentsSummary(
                observer_id=observer_id,
                source_id=source_id,
                clock_domains=tuple(sorted(clocks)),
                valid_interval_count=valid_intervals,
                broken_interval_count=broken_intervals,
                user_cpu_delta_seconds=u_out,
                system_cpu_delta_seconds=s_out,
                total_cpu_delta_seconds=t_out,
                valid_elapsed_seconds=e_out,
                cpu_core_equivalents=core_eq,
                capacity_normalized_utilization=None,
                capacity_denominator_status="unknown",
            )
        )

    return tuple(results)


def _extract_git_commit(manifest: ObservabilityManifest) -> tuple[str, str]:
    """Extract ``(git_commit_text, status)`` from ``manifest.producer.source_commit``."""

    raw = manifest.producer.get("source_commit")
    if isinstance(raw, Mapping):
        status = str(raw.get("status", "unknown"))
        if status == "ok" and isinstance(raw.get("value"), str) and raw["value"].strip():
            return str(raw["value"]), "ok"
        reason = raw.get("reason_code")
        if isinstance(reason, str) and reason.strip():
            return reason, status
        return "not_supplied", status
    if isinstance(raw, str) and raw.strip():
        return raw.strip(), "ok"
    return "not_supplied", "unavailable"


def _extract_external_ref(value: Any) -> str | dict[str, Any]:
    """Preserve explicit ``'not_supplied'`` or return a clean copy of a mapping."""

    if value == "not_supplied" or value is None:
        return "not_supplied"
    if isinstance(value, Mapping):
        if value.get("status") == "unavailable" and value.get("reason_code") == "not_supplied":
            return "not_supplied"
        return dict(value)
    return str(value)


def _extract_device_bindings_and_caveats(
    inspection: ObservabilityBundleInspection,
    cpu_summaries: Sequence[CpuCoreEquivalentsSummary],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...], dict[tuple[str, str, str], tuple[str, ...]]]:
    """Extract linked device bindings, bundle-wide caveats, and per-quantity caveats."""

    device_bindings: list[dict[str, Any]] = []
    seen_bindings: set[str] = set()
    caveats: list[str] = []
    qty_caveats: dict[tuple[str, str, str], list[str]] = {}

    inventory: ObservabilityInventory | None = inspection.inventory
    if inventory is not None:
        for key, value in sorted(inventory.observed.items()):
            if isinstance(value, Mapping) and (
                "physical_uuid" in value
                or "runtime_device" in value
                or "device_uuid" in value
            ):
                binding = {"binding_key": key}
                for field in ("runtime_device", "physical_uuid", "device_uuid", "mig", "evidence"):
                    if field in value:
                        binding[field] = value[field]
                sig = str(sorted(binding.items()))
                if sig not in seen_bindings:
                    seen_bindings.add(sig)
                    device_bindings.append(binding)

        for source_id, meta in sorted(inventory.sources.items()):
            if meta.get("shared_scope_not_exclusive") is True:
                _append_unique(
                    caveats,
                    f"{source_id}: shared cgroup scope is not exclusive per-process entitlement",
                )
            if meta.get("shared_device_not_exclusive") is True:
                _append_unique(
                    caveats,
                    f"{source_id}: whole physical GPU metrics are shared and not per-process attribution",
                )
            reset_status = meta.get("reset_epoch_status")
            if isinstance(reset_status, str) and reset_status != "not_requested":
                _append_unique(
                    caveats,
                    f"{source_id}: allocator peak reset_epoch_status={reset_status}",
                )
            mig_mode = meta.get("mig_mode")
            if isinstance(mig_mode, str) and mig_mode in {
                "enabled",
                "insufficient_permissions",
            }:
                _append_unique(
                    caveats,
                    f"{source_id}: MIG mode {mig_mode!r} renders whole-GPU metrics ambiguous",
                )

    # Inspect sample subjects for device/MIG/shared/cgroup/descendant caveats.
    for observer_id, stream in sorted(inspection.samples.items()):
        for sample in stream:
            subj = sample.subject
            kind = str(subj.get("kind", ""))
            if kind in {"physical_gpu", "logical_device", "process_allocator"}:
                binding_entry: dict[str, Any] = {
                    "source_id": sample.source_id,
                    "subject_kind": kind,
                }
                for key in (
                    "configured_device",
                    "backend",
                    "device_index",
                    "physical_uuid",
                    "uuid",
                    "mig_mode",
                    "mig_ambiguous",
                    "shared_device_not_exclusive",
                    "reset_epoch_status",
                ):
                    if key in subj:
                        binding_entry[key] = subj[key]
                sig = str(sorted(binding_entry.items()))
                if sig not in seen_bindings:
                    seen_bindings.add(sig)
                    device_bindings.append(binding_entry)

            if subj.get("shared_device_not_exclusive") is True:
                _append_unique(
                    caveats,
                    f"{sample.source_id}: shared physical device metrics are not per-process attribution",
                )
            if subj.get("mig_ambiguous") is True:
                _append_unique(
                    caveats,
                    f"{sample.source_id}: MIG subdevice ambiguity detected; whole-GPU metrics unavailable",
                )
            if kind == "cgroup":
                if subj.get("shared_scope_not_exclusive") is True:
                    _append_unique(
                        caveats,
                        f"{sample.source_id}: cgroup scope is shared; memory.current != sum-RSS and memory.peak is cgroup-lifetime",
                    )
                if subj.get("includes_other_tasks") is True:
                    _append_unique(
                        caveats,
                        f"{sample.source_id}: cgroup includes other tasks outside this process",
                    )
            if kind == "descendant_set":
                _append_unique(
                    caveats,
                    f"{sample.source_id}: descendant scan is bounded/non-atomic and sum-RSS double-counts shared pages",
                )

    # Descriptor-level & quantity-summary-level semantic caveats.
    descriptors = inspection.descriptors
    for qsum in inspection.summary.quantities:
        qkey = (qsum.observer_id, qsum.source_id, qsum.quantity_id)
        qlist = qty_caveats.setdefault(qkey, [])
        if descriptors is not None:
            desc = descriptors.quantities.get(qsum.quantity_id)
            if desc is not None:
                for quality_tag in desc.quality:
                    _append_unique(qlist, quality_tag)
        if qsum.statistic == "gauge" and qsum.sampled_max_is_lower_bound:
            _append_unique(qlist, "sampled_max_is_lower_bound")
            _append_unique(
                caveats,
                "sampled gauge maxima are lower bounds on true peak usage",
            )
        if qsum.statistic == "high_water" and qsum.window == "process_lifetime":
            _append_unique(qlist, "process_lifetime_high_water_not_current_rss")
        if qsum.statistic == "high_water" and qsum.window == "allocator_reset_window":
            _append_unique(qlist, "allocator_reset_window_peak")
        if qsum.statistic == "high_water" and qsum.window == "cgroup_lifetime":
            _append_unique(qlist, "cgroup_lifetime_high_water_not_reset_per_run")
        if qsum.statistic == "vendor_window_mean":
            _append_unique(qlist, "vendor_window_not_continuous_integrated")
            _append_unique(
                caveats,
                f"{qsum.source_id}/{qsum.quantity_id}: vendor recent-window busy is not integrated GPU-busy seconds",
            )
        if qsum.broken_interval_count > 0:
            _append_unique(qlist, f"broken_intervals={qsum.broken_interval_count}")
        if qsum.ok_count == 0 and (qsum.unavailable_count > 0 or qsum.error_count > 0):
            _append_unique(qlist, "no_valid_numeric_observations_never_invented_zero")

    if cpu_summaries:
        _append_unique(
            caveats,
            "CPU core-equivalents are raw delta(user+system)/elapsed_s with unknown exclusive capacity denominator",
        )

    return (
        tuple(device_bindings),
        tuple(caveats),
        {key: tuple(val) for key, val in qty_caveats.items()},
    )


def _classify_bundle_reconciliation(
    inspection: ObservabilityBundleInspection,
    marker: LifecycleMarkerVerification,
    observers: tuple[ObserverBatchReconciliation, ...],
) -> tuple[str, str, str, str, str | None]:
    """Derive ``(reconciliation_state, lifecycle_state, data_completeness, finalization, reason)``."""

    manifest = inspection.manifest
    has_recovered_data = any(bool(s) for s in inspection.samples.values()) or bool(
        inspection.profile_statuses
    )

    any_missing_status = any(not obs.has_final_status for obs in observers)
    any_count_mismatch = any(obs.count_mismatch for obs in observers)
    any_open_obs = any(
        obs.has_final_status and obs.lifecycle_state == "open" for obs in observers
    )
    any_unsafe_obs = any(
        obs.has_final_status and obs.lifecycle_state == "unsafe" for obs in observers
    )
    any_failed_outcome = any(
        obs.has_final_status
        and (
            obs.application_outcome.lower() in _FAILURE_OUTCOMES
            or obs.invocation_outcome.lower() in _FAILURE_OUTCOMES
        )
        for obs in observers
    )
    any_writer_failure = any(
        obs.has_final_status
        and (
            obs.writer_state in _FAILURE_WRITER_STATES
            or (obs.flush_confirmed is False and not obs.is_profile_only)
        )
        for obs in observers
    )
    any_helper_failure = any(
        obs.has_final_status and obs.helper_cleanup_state in _FAILURE_HELPER_STATES
        for obs in observers
    )
    has_torn_tails = bool(inspection.torn_tails)
    has_interior_errors = bool(inspection.interior_errors)

    # Finalization is a two-file protocol: the manifest is replaced first and
    # the marker is replaced second.  A process death after the first replace
    # leaves a valid closed manifest beside the previous valid open marker.
    # The state-mismatch diagnostic is the evidence that this is that protocol
    # boundary, rather than an arbitrary marker/hash tamper.
    marker_open_while_manifest_closed = (
        marker.present
        and marker.state == "open"
        and manifest.lifecycle_state == "closed"
        and marker.diagnostic_code == "lifecycle_marker_state_mismatch"
    )
    marker_tampered_or_invalid = marker.present and (
        not marker.matches_manifest or marker.manifest_sha256_verified is False
    ) and not marker_open_while_manifest_closed

    # Independent invalid-marker and unsafe observer evidence takes precedence
    # over an open manifest.  These signals cannot be explained by the benign
    # manifest-before-marker crash window above.
    if marker_tampered_or_invalid:
        return (
            "failed",
            "unsafe",
            "partial" if has_recovered_data else "unknown",
            "unsafe",
            marker.diagnostic_code or "invalid_lifecycle_marker",
        )

    if manifest.lifecycle_state == "unsafe" or any_unsafe_obs or any_helper_failure:
        return (
            "failed",
            "unsafe",
            "partial" if has_recovered_data else "unknown",
            "unsafe",
            "unsafe_lifecycle_state",
        )

    # 1. Open / interrupted execution (manifest still open, mid-finalization
    # crash with open marker, open observer status, or torn JSONL tail).
    if (
        manifest.lifecycle_state == "open"
        or marker_open_while_manifest_closed
        or any_open_obs
        or has_torn_tails
    ):
        reason = (
            marker.diagnostic_code
            if marker_open_while_manifest_closed
            else (
                "torn_tail"
                if has_torn_tails and not any_missing_status
                else (
                    "torn_tail_or_missing_final_status"
                    if has_torn_tails or any_missing_status
                    else "open_lifecycle_state"
                )
            )
        )
        return (
            "open_interrupted",
            "open",
            "partial" if has_recovered_data else "unknown",
            "unknown",
            reason,
        )

    # 2. Closed/unsafe manifest where at least one observer lacks final status.
    if any_missing_status:
        lifecycle = "unsafe" if (manifest.lifecycle_state == "unsafe" or any_unsafe_obs) else "open"
        finalization = "unsafe" if lifecycle == "unsafe" else "unknown"
        return (
            "missing_final_status",
            lifecycle,
            "partial" if has_recovered_data else "unknown",
            finalization,
            "missing_final_status",
        )

    # 3. Failed terminal bundle: failed application/invocation outcome,
    # writer/helper failure,
    # interior corruption, or written-vs-read batch count discrepancy.
    if any_count_mismatch:
        return (
            "failed",
            inspection.derived_lifecycle_state,
            "partial" if has_recovered_data else "unknown",
            "confirmed" if inspection.derived_finalization == "confirmed" else inspection.derived_finalization,
            "batch_count_mismatch",
        )

    if has_interior_errors:
        return (
            "failed",
            inspection.derived_lifecycle_state,
            "partial" if has_recovered_data else "unknown",
            inspection.derived_finalization,
            inspection.derived_reason or "invalid_interior_records",
        )

    if any_writer_failure:
        return (
            "failed",
            inspection.derived_lifecycle_state,
            "partial" if has_recovered_data else "unknown",
            inspection.derived_finalization,
            "writer_error",
        )

    if any_failed_outcome:
        return (
            "failed",
            inspection.derived_lifecycle_state,
            inspection.derived_data_completeness,
            inspection.derived_finalization,
            "failed_run_outcome",
        )

    if inspection.derived_reason in {"undeclared_clock_domain", "missing_descriptors"}:
        return (
            "failed",
            inspection.derived_lifecycle_state,
            inspection.derived_data_completeness,
            inspection.derived_finalization,
            inspection.derived_reason,
        )

    # 4. Clean finalized bundle (note: data_completeness stays independent and
    # may legitimately be "complete", "partial", "none", or "unknown").
    return (
        "clean_final",
        inspection.derived_lifecycle_state,
        inspection.derived_data_completeness,
        inspection.derived_finalization,
        inspection.derived_reason,
    )


def _bound_profile_row(
    row: Mapping[str, Any],
    *,
    max_chars: int,
    max_items: int,
    reasons: list[str],
) -> dict[str, Any]:
    """Bound string and nested collection fields inside one profile summary row."""

    return _bound_json_value(
        row,
        max_chars=max_chars,
        max_items=max_items,
        reasons=reasons,
    )


def _build_bounded_summary_internal(
    *,
    inspection: ObservabilityBundleInspection,
    reconciliation_state: str,
    lifecycle_state: str,
    data_completeness: str,
    finalization: str,
    assessment_reason: str | None,
    lifecycle_marker: LifecycleMarkerVerification,
    observers: tuple[ObserverBatchReconciliation, ...],
    cpu_core_equivalents: tuple[CpuCoreEquivalentsSummary, ...],
    all_diagnostics: tuple[ObservabilityDiagnostic, ...],
    bounds: OperatorSummaryBounds,
) -> BoundedOperatorSummary:
    """Construct a strictly bounded :class:`BoundedOperatorSummary`."""

    truncation_reasons: list[str] = []
    manifest = inspection.manifest
    inventory = inspection.inventory

    git_commit, git_commit_status = _extract_git_commit(manifest)
    observer_ids = tuple(sorted({obs.observer_id for obs in observers}))

    source_ids_set: set[str] = set()
    if inventory is not None:
        source_ids_set.update(inventory.sources.keys())
    for stream in inspection.samples.values():
        for sample in stream:
            source_ids_set.add(sample.source_id)
    for st in inspection.statuses.values():
        source_ids_set.update(st.source_status.keys())
    source_ids = tuple(sorted(source_ids_set))

    device_bindings, raw_caveats, per_qty_caveats = _extract_device_bindings_and_caveats(
        inspection, cpu_core_equivalents
    )

    # Bound sharing & MIG caveats attached to identity link and summary caveats.
    bounded_caveats_list = list(raw_caveats)
    total_caveats = len(bounded_caveats_list)
    if len(bounded_caveats_list) > bounds.max_caveats:
        _append_unique(truncation_reasons, "caveats_truncated")
        bounded_caveats_list = bounded_caveats_list[: bounds.max_caveats]
    bounded_caveats = tuple(
        _truncate_text(
            item,
            max_chars=bounds.max_text_chars,
            reasons=truncation_reasons,
        )
        for item in bounded_caveats_list
    )

    sharing_mig_caveats = tuple(
        item
        for item in bounded_caveats
        if any(
            tok in item.lower()
            for tok in ("shared", "mig", "cgroup", "descendant", "exclusive")
        )
    )

    billed_units = (
        "not_supplied"
        if inventory is None
        else _extract_external_ref(inventory.billed_units)
    )
    identity = ObservabilityIdentityLink(
        invocation_id=manifest.invocation_id,
        run_id=manifest.run_id,
        config_hash=manifest.configuration_digest,
        git_commit=git_commit,
        git_commit_status=git_commit_status,
        inventory_id=manifest.inventory_id,
        descriptor_set_id=manifest.descriptor_set_id,
        observer_ids=observer_ids,
        source_ids=source_ids,
        clock_domains=inspection.summary.clock_domains,
        executor_context=_extract_external_ref(manifest.executor_context),
        scheduler_accounting=_extract_external_ref(manifest.scheduler_accounting),
        billed_units=billed_units,
        device_bindings=device_bindings,
        sharing_and_mig_caveats=sharing_mig_caveats,
    )

    # Bound observers.
    total_observers = len(observers)
    kept_observers = observers
    if len(kept_observers) > bounds.max_observers:
        _append_unique(truncation_reasons, "observers_truncated")
        kept_observers = kept_observers[: bounds.max_observers]

    # Bound quantities while preserving all O2/r5 distinctions.
    all_quantities: tuple[QuantityObservationSummary, ...] = inspection.summary.quantities
    total_quantities = len(all_quantities)
    kept_qsums = all_quantities
    if len(kept_qsums) > bounds.max_quantities:
        _append_unique(truncation_reasons, "quantities_truncated")
        kept_qsums = kept_qsums[: bounds.max_quantities]

    bounded_quantities: list[BoundedOperatorQuantityRow] = []
    for qsum in kept_qsums:
        qkey = (qsum.observer_id, qsum.source_id, qsum.quantity_id)
        raw_qcav = per_qty_caveats.get(qkey, ())
        qcav_slice = list(raw_qcav)
        if len(qcav_slice) > bounds.max_caveats:
            _append_unique(truncation_reasons, "caveats_truncated")
            qcav_slice = qcav_slice[: bounds.max_caveats]
        bounded_qcav = tuple(
            _truncate_text(
                item,
                max_chars=bounds.max_text_chars,
                reasons=truncation_reasons,
            )
            for item in qcav_slice
        )
        bounded_reasons = tuple(
            _truncate_text(
                code,
                max_chars=bounds.max_text_chars,
                reasons=truncation_reasons,
            )
            for code in qsum.reason_codes
        )
        bounded_quantities.append(
            BoundedOperatorQuantityRow(
                observer_id=qsum.observer_id,
                source_id=qsum.source_id,
                quantity_id=qsum.quantity_id,
                unit=qsum.unit,
                statistic=qsum.statistic,
                subject_kind=qsum.subject_kind,
                window=qsum.window,
                clock_domains=qsum.clock_domains,
                sample_count=qsum.sample_count,
                ok_count=qsum.ok_count,
                unavailable_count=qsum.unavailable_count,
                error_count=qsum.error_count,
                reason_codes=bounded_reasons,
                caveats=bounded_qcav,
                sampled_min=qsum.sampled_min,
                sampled_max=qsum.sampled_max,
                sampled_max_is_lower_bound=qsum.sampled_max_is_lower_bound,
                observed_max=qsum.observed_max,
                valid_interval_count=qsum.valid_interval_count,
                broken_interval_count=qsum.broken_interval_count,
                valid_delta_total=qsum.valid_delta_total,
                valid_elapsed_seconds=qsum.valid_elapsed_seconds,
                valid_rate_per_second=qsum.valid_rate_per_second,
                latest_value=qsum.latest_value,
                min_value=qsum.min_value,
                max_value=qsum.max_value,
                observation_min=qsum.observation_min,
                observation_max=qsum.observation_max,
                observation_mean=qsum.observation_mean,
                continuous_window_integrated=qsum.continuous_window_integrated,
            )
        )

    # Bound profile windows and their operator rows.
    summary_by_window: dict[tuple[str, str], ObservabilityProfileSummary] = {
        (ps.observer_id, ps.window_id): ps for ps in inspection.profile_summaries
    }
    all_pstatuses = inspection.profile_statuses
    total_profile_windows = len(all_pstatuses)
    kept_pstatuses = all_pstatuses
    if len(kept_pstatuses) > bounds.max_profile_windows:
        _append_unique(truncation_reasons, "profile_windows_truncated")
        kept_pstatuses = kept_pstatuses[: bounds.max_profile_windows]

    bounded_profiles: list[BoundedOperatorProfileWindow] = []
    for pstatus in kept_pstatuses:
        win_key = (pstatus.observer_id, pstatus.window_id)
        psummary = summary_by_window.get(win_key)
        win_reasons: list[str] = list(pstatus.truncation_reasons or ())
        if psummary is not None:
            for r in psummary.truncation_reasons:
                _append_unique(win_reasons, r)
            raw_rows = list(psummary.rows)
            win_total_rows = psummary.total_rows
            if len(raw_rows) > bounds.max_profile_rows:
                _append_unique(win_reasons, "profile_rows_truncated")
                _append_unique(truncation_reasons, "profile_rows_truncated")
                raw_rows = raw_rows[: bounds.max_profile_rows]
            bounded_rows = tuple(
                _bound_profile_row(
                    row,
                    max_chars=bounds.max_text_chars,
                    max_items=max(bounds.max_profile_rows, 16),
                    reasons=win_reasons,
                )
                for row in raw_rows
            )
            if "text_truncated" in win_reasons:
                _append_unique(truncation_reasons, "text_truncated")
            win_truncated = psummary.truncated or bool(win_reasons) or (
                win_total_rows > len(bounded_rows)
            )
        else:
            win_total_rows = 0
            bounded_rows = ()
            win_truncated = bool(win_reasons)

        bounded_profiles.append(
            BoundedOperatorProfileWindow(
                observer_id=pstatus.observer_id,
                window_id=pstatus.window_id,
                window_kind=pstatus.window_kind,
                lifecycle_state=pstatus.lifecycle_state,
                data_completeness=pstatus.data_completeness,
                exporter=pstatus.exporter,
                export_status=pstatus.export_status,
                guarantee_class=pstatus.guarantee_class,
                requested_activities=pstatus.requested_activities,
                accepted_activities=pstatus.accepted_activities,
                total_rows=win_total_rows,
                rows=bounded_rows,
                truncated=win_truncated,
                truncation_reasons=tuple(win_reasons),
            )
        )

    # Bound diagnostics and their messages.
    total_diagnostics = len(all_diagnostics)
    kept_diags = all_diagnostics
    if len(kept_diags) > bounds.max_diagnostics:
        _append_unique(truncation_reasons, "diagnostics_truncated")
        kept_diags = kept_diags[: bounds.max_diagnostics]

    bounded_diags: list[ObservabilityDiagnostic] = []
    for diag in kept_diags:
        bounded_msg = _truncate_text(
            diag.message,
            max_chars=bounds.max_text_chars,
            reasons=truncation_reasons,
        )
        bounded_diags.append(
            ObservabilityDiagnostic(
                code=diag.code,
                message=bounded_msg,
                path=diag.path,
                line_number=diag.line_number,
                observer_id=diag.observer_id,
                source_id=diag.source_id,
            )
        )

    bounded_marker = LifecycleMarkerVerification(
        present=lifecycle_marker.present,
        state=lifecycle_marker.state,
        matches_manifest=lifecycle_marker.matches_manifest,
        manifest_sha256_verified=lifecycle_marker.manifest_sha256_verified,
        diagnostic_code=lifecycle_marker.diagnostic_code,
        detail=(
            None
            if lifecycle_marker.detail is None
            else _truncate_text(
                lifecycle_marker.detail,
                max_chars=bounds.max_text_chars,
                reasons=truncation_reasons,
            )
        ),
    )

    bounded_reason = (
        None
        if assessment_reason is None
        else _truncate_text(
            assessment_reason,
            max_chars=bounds.max_text_chars,
            reasons=truncation_reasons,
        )
    )

    return BoundedOperatorSummary(
        identity=identity,
        reconciliation_state=reconciliation_state,
        lifecycle_state=lifecycle_state,
        data_completeness=data_completeness,
        finalization=finalization,
        assessment_reason=bounded_reason,
        lifecycle_marker=bounded_marker,
        observers=kept_observers,
        total_observers=total_observers,
        cpu_core_equivalents=cpu_core_equivalents,
        quantities=tuple(bounded_quantities),
        total_quantities=total_quantities,
        profile_windows=tuple(bounded_profiles),
        total_profile_windows=total_profile_windows,
        caveats=bounded_caveats,
        total_caveats=total_caveats,
        diagnostics=tuple(bounded_diags),
        total_diagnostics=total_diagnostics,
        bounds=bounds,
        truncated=bool(truncation_reasons),
        truncation_reasons=tuple(truncation_reasons),
    )


# ---------------------------------------------------------------------------
# Public read-only reconciliation & operator summary entry points
# ---------------------------------------------------------------------------


def reconcile_observability_bundle(
    bundle_dir: str | Path,
    *,
    bounds: OperatorSummaryBounds | None = None,
) -> ReconciledObservabilityBundle:
    """Pure, read-only reconciliation and bounded operator summary for one bundle.

    Performs zero filesystem writes, repairs, or deletions. Validates
    ``manifest.json``, optional ``lifecycle-marker.json``, immutable
    ``inventory.json`` / ``descriptors.json``, sample JSONL streams, observer
    status files, and profile windows, returning an explicit classification
    (``"clean_final"``, ``"failed"``, ``"open_interrupted"``, or
    ``"missing_final_status"``) alongside a bounded operator summary.

    Parameters
    ----------
    bundle_dir : str or pathlib.Path
        Path to one ``observability/<invocation-id>/`` directory.
    bounds : OperatorSummaryBounds or None, optional
        Row, collection, and text bounds applied to the embedded operator
        summary (defaults to :class:`OperatorSummaryBounds()`).

    Returns
    -------
    ReconciledObservabilityBundle
        Reconciled status, batch accounting, CPU core-equivalents, diagnostics,
        and bounded operator summary.
    """

    effective_bounds = OperatorSummaryBounds() if bounds is None else bounds
    if not isinstance(effective_bounds, OperatorSummaryBounds):
        raise TypeError(
            "bounds must be OperatorSummaryBounds or None, "
            f"got {type(effective_bounds).__name__}"
        )

    inspection = inspect_observability_bundle(bundle_dir)
    root = Path(inspection.bundle_dir)

    extra_diagnostics: list[ObservabilityDiagnostic] = []
    marker = _verify_lifecycle_marker(root, inspection.manifest, extra_diagnostics)
    observers = _reconcile_observers(inspection, extra_diagnostics)
    cpu_summaries = _compute_cpu_core_equivalents(inspection)

    (
        reconciliation_state,
        lifecycle_state,
        data_completeness,
        finalization,
        assessment_reason,
    ) = _classify_bundle_reconciliation(inspection, marker, observers)

    all_diagnostics = tuple([*inspection.diagnostics, *extra_diagnostics])
    summary = _build_bounded_summary_internal(
        inspection=inspection,
        reconciliation_state=reconciliation_state,
        lifecycle_state=lifecycle_state,
        data_completeness=data_completeness,
        finalization=finalization,
        assessment_reason=assessment_reason,
        lifecycle_marker=marker,
        observers=observers,
        cpu_core_equivalents=cpu_summaries,
        all_diagnostics=all_diagnostics,
        bounds=effective_bounds,
    )

    return ReconciledObservabilityBundle(
        bundle_dir=inspection.bundle_dir,
        invocation_id=inspection.invocation_id,
        reconciliation_state=reconciliation_state,
        lifecycle_state=lifecycle_state,
        data_completeness=data_completeness,
        finalization=finalization,
        assessment_reason=assessment_reason,
        lifecycle_marker=marker,
        observers=observers,
        cpu_core_equivalents=cpu_summaries,
        diagnostics=all_diagnostics,
        inspection=inspection,
        summary=summary,
    )


def build_bounded_operator_summary(
    reconciled: ReconciledObservabilityBundle,
    *,
    bounds: OperatorSummaryBounds | None = None,
) -> BoundedOperatorSummary:
    """Re-project an already reconciled bundle under custom :class:`OperatorSummaryBounds`."""

    if not isinstance(reconciled, ReconciledObservabilityBundle):
        raise TypeError(
            "reconciled must be a ReconciledObservabilityBundle, "
            f"got {type(reconciled).__name__}"
        )
    effective_bounds = OperatorSummaryBounds() if bounds is None else bounds
    if not isinstance(effective_bounds, OperatorSummaryBounds):
        raise TypeError(
            "bounds must be OperatorSummaryBounds or None, "
            f"got {type(effective_bounds).__name__}"
        )
    return _build_bounded_summary_internal(
        inspection=reconciled.inspection,
        reconciliation_state=reconciled.reconciliation_state,
        lifecycle_state=reconciled.lifecycle_state,
        data_completeness=reconciled.data_completeness,
        finalization=reconciled.finalization,
        assessment_reason=reconciled.assessment_reason,
        lifecycle_marker=reconciled.lifecycle_marker,
        observers=reconciled.observers,
        cpu_core_equivalents=reconciled.cpu_core_equivalents,
        all_diagnostics=reconciled.diagnostics,
        bounds=effective_bounds,
    )


def reconcile_run_observability(
    run_dir: str | Path,
    *,
    relative_dir: str = "observability",
    bounds: OperatorSummaryBounds | None = None,
) -> ReconciledRunObservability:
    """Pure, read-only reconciliation across all invocation bundles in a run directory.

    Absent sidecars (legacy or uncollected runs) return ``status="absent"`` and
    ``data_completeness="unknown"`` without mutating disk or fabricating zero
    utilization. Invocations are never merged across ``invocation_id``.

    Parameters
    ----------
    run_dir : str or pathlib.Path
        TPEN run directory to inspect.
    relative_dir : str, optional
        Run-relative directory holding per-invocation bundles
        (default ``"observability"``).
    bounds : OperatorSummaryBounds or None, optional
        Bounds applied to each invocation summary and ``max_invocations``.

    Returns
    -------
    ReconciledRunObservability
        Run-level reconciliation result with bounded isolated invocations.
    """

    effective_bounds = OperatorSummaryBounds() if bounds is None else bounds
    if not isinstance(effective_bounds, OperatorSummaryBounds):
        raise TypeError(
            "bounds must be OperatorSummaryBounds or None, "
            f"got {type(effective_bounds).__name__}"
        )

    run_inspection = inspect_run_observability(run_dir, relative_dir=relative_dir)
    if run_inspection.status == "absent":
        return ReconciledRunObservability(
            run_dir=run_inspection.run_dir,
            status="absent",
            data_completeness="unknown",
            invocations=(),
            total_invocations=0,
            truncated=False,
            truncation_reasons=(),
        )

    all_reconciled: list[ReconciledObservabilityBundle] = [
        reconcile_observability_bundle(bundle.bundle_dir, bounds=effective_bounds)
        for bundle in run_inspection.invocations
    ]
    total_invocations = len(all_reconciled)
    truncation_reasons: list[str] = []
    kept = all_reconciled
    if len(kept) > effective_bounds.max_invocations:
        _append_unique(truncation_reasons, "invocations_truncated")
        kept = kept[: effective_bounds.max_invocations]

    if not all_reconciled:
        run_completeness = "unknown"
    else:
        comp_values = [item.data_completeness for item in all_reconciled]
        if "unknown" in comp_values:
            run_completeness = "unknown"
        elif "partial" in comp_values:
            run_completeness = "partial"
        elif all(val == "none" for val in comp_values):
            run_completeness = "none"
        elif "none" in comp_values and "complete" in comp_values:
            run_completeness = "partial"
        else:
            run_completeness = "complete"

    return ReconciledRunObservability(
        run_dir=run_inspection.run_dir,
        status="present",
        data_completeness=run_completeness,
        invocations=tuple(kept),
        total_invocations=total_invocations,
        truncated=bool(truncation_reasons),
        truncation_reasons=tuple(truncation_reasons),
    )


def _escape_rendered_line_separators(value: str) -> str:
    """Escape every separator recognized by :meth:`str.splitlines`."""

    escaped: list[str] = []
    for char in value:
        if char == "\n":
            escaped.append(r"\n")
        elif char == "\r":
            escaped.append(r"\r")
        elif char == "\x0b":
            escaped.append(r"\x0b")
        elif char == "\x0c":
            escaped.append(r"\x0c")
        elif char == "\x1c":
            escaped.append(r"\x1c")
        elif char == "\x1d":
            escaped.append(r"\x1d")
        elif char == "\x1e":
            escaped.append(r"\x1e")
        elif char == "\x85":
            escaped.append(r"\x85")
        elif char == "\u2028":
            escaped.append(r"\u2028")
        elif char == "\u2029":
            escaped.append(r"\u2029")
        else:
            escaped.append(char)
    return "".join(escaped)


def render_operator_summary(
    summary: BoundedOperatorSummary,
    *,
    max_lines: int | None = None,
    max_line_chars: int | None = None,
) -> str:
    """Render a strictly line-bounded and width-bounded plain-text operator summary.

    Parameters
    ----------
    summary : BoundedOperatorSummary
        Bounded operator summary to render.
    max_lines : int or None, optional
        Override for maximum rendered lines (defaults to
        ``summary.bounds.max_rendered_lines``).
    max_line_chars : int or None, optional
        Override for maximum characters per line (defaults to
        ``summary.bounds.max_text_chars``).

    Returns
    -------
    str
        Deterministic ASCII-friendly text obeying both line and width caps with
        an explicit truncation notice when capped.
    """

    if not isinstance(summary, BoundedOperatorSummary):
        raise TypeError(
            "summary must be a BoundedOperatorSummary, "
            f"got {type(summary).__name__}"
        )
    line_cap = (
        summary.bounds.max_rendered_lines
        if max_lines is None
        else _require_positive_int(max_lines, "max_lines")
    )
    char_cap = (
        summary.bounds.max_text_chars
        if max_line_chars is None
        else _require_positive_int(max_line_chars, "max_line_chars")
    )

    ident = summary.identity
    raw_lines: list[str] = [
        (
            f"invocation={ident.invocation_id} run={ident.run_id} "
            f"state={summary.reconciliation_state} lifecycle={summary.lifecycle_state} "
            f"completeness={summary.data_completeness} finalization={summary.finalization}"
        ),
        (
            f"config_hash={ident.config_hash} git_commit={ident.git_commit} "
            f"inventory={ident.inventory_id} descriptors={ident.descriptor_set_id}"
        ),
        (
            f"clocks={','.join(ident.clock_domains) or 'unknown'} "
            f"marker={summary.lifecycle_marker.state} "
            f"executor={ident.executor_context if isinstance(ident.executor_context, str) else 'supplied'} "
            f"accounting={ident.scheduler_accounting if isinstance(ident.scheduler_accounting, str) else 'supplied'}"
        ),
    ]
    if summary.assessment_reason is not None:
        raw_lines.append(f"assessment_reason={summary.assessment_reason}")

    for obs in summary.observers:
        counts_desc = (
            f"attempted={obs.attempted} written={obs.written} "
            f"dropped={obs.dropped} failed={obs.failed} pending={obs.pending} "
            f"read={obs.read_valid_packets}"
            if obs.has_final_status and not obs.is_profile_only
            else (
                "profile_only"
                if obs.is_profile_only
                else f"status=missing read={obs.read_valid_packets}"
            )
        )
        raw_lines.append(
            f"observer={obs.observer_id} batch={obs.batch_status} "
            f"outcome={obs.invocation_outcome} {counts_desc}"
        )

    for cpu in summary.cpu_core_equivalents:
        eq_text = (
            "unknown"
            if cpu.cpu_core_equivalents is None
            else f"{cpu.cpu_core_equivalents:.4f}"
        )
        raw_lines.append(
            f"cpu_core_equivalents[{cpu.observer_id}/{cpu.source_id}]={eq_text} "
            f"(valid_intervals={cpu.valid_interval_count}, broken={cpu.broken_interval_count}, "
            f"capacity_denominator={cpu.capacity_denominator_status})"
        )

    for qrow in summary.quantities:
        if qrow.statistic == "gauge":
            val_text = (
                f"min={qrow.sampled_min} max_lower_bound={qrow.sampled_max}"
                if qrow.ok_count > 0
                else "unknown"
            )
        elif qrow.statistic == "high_water":
            val_text = (
                f"observed_max={qrow.observed_max}" if qrow.ok_count > 0 else "unknown"
            )
        elif qrow.statistic == "counter":
            val_text = (
                f"rate={qrow.valid_rate_per_second} delta={qrow.valid_delta_total}"
                if qrow.valid_interval_count > 0
                else "unknown"
            )
        elif qrow.statistic == "capacity":
            val_text = (
                f"latest={qrow.latest_value} min={qrow.min_value} max={qrow.max_value}"
                if qrow.ok_count > 0
                else "unknown"
            )
        else:
            val_text = (
                f"obs_mean={qrow.observation_mean} obs_max={qrow.observation_max} "
                f"integrated={qrow.continuous_window_integrated}"
                if qrow.ok_count > 0
                else "unknown"
            )
        raw_lines.append(
            f"qty[{qrow.observer_id}/{qrow.source_id}/{qrow.quantity_id}] "
            f"stat={qrow.statistic} unit={qrow.unit} window={qrow.window} "
            f"ok={qrow.ok_count}/{qrow.sample_count} {val_text}"
        )

    for caveat in summary.caveats:
        raw_lines.append(f"caveat: {caveat}")
    for diag in summary.diagnostics:
        raw_lines.append(f"diag[{diag.code}]: {diag.message}")
    if summary.truncated:
        raw_lines.append(
            f"truncated=true reasons={','.join(summary.truncation_reasons)}"
        )

    # Apply physical-line normalization before either line or width limits;
    # this affects display text only and leaves structured identity unchanged.
    raw_lines = [_escape_rendered_line_separators(line) for line in raw_lines]

    dummy_reasons: list[str] = []
    if len(raw_lines) > line_cap:
        if line_cap == 1:
            kept_lines = [
                _truncate_text(
                    f"... truncated ({len(raw_lines)} lines > max_lines={line_cap})",
                    max_chars=char_cap,
                    reasons=dummy_reasons,
                )
            ]
        else:
            omitted = len(raw_lines) - (line_cap - 1)
            kept_lines = [
                _truncate_text(line, max_chars=char_cap, reasons=dummy_reasons)
                for line in raw_lines[: line_cap - 1]
            ]
            kept_lines.append(
                _truncate_text(
                    f"... truncated ({omitted} more line(s); max_lines={line_cap})",
                    max_chars=char_cap,
                    reasons=dummy_reasons,
                )
            )
    else:
        kept_lines = [
            _truncate_text(line, max_chars=char_cap, reasons=dummy_reasons)
            for line in raw_lines
        ]

    return "\n".join(kept_lines)


__all__ = [
    "OBSERVABILITY_BATCH_STATUSES",
    "OBSERVABILITY_MARKER_STATES",
    "OBSERVABILITY_OPERATOR_SUMMARY_SCHEMA",
    "OBSERVABILITY_RECONCILIATION_SCHEMA",
    "OBSERVABILITY_RECONCILIATION_STATES",
    "BoundedOperatorProfileWindow",
    "BoundedOperatorQuantityRow",
    "BoundedOperatorSummary",
    "CpuCoreEquivalentsSummary",
    "LifecycleMarkerVerification",
    "ObservabilityIdentityLink",
    "ObserverBatchReconciliation",
    "OperatorSummaryBounds",
    "ReconciledObservabilityBundle",
    "ReconciledRunObservability",
    "build_bounded_operator_summary",
    "reconcile_observability_bundle",
    "reconcile_run_observability",
    "render_operator_summary",
]
