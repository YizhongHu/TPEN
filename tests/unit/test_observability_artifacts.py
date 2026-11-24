"""Unit tests for versioned ``tpen.run-observability/v1`` artifacts and pure inspection."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import tpen.artifacts as artifacts_mod
import tpen.inspection as inspection_mod
import tpen.observability_artifacts as obs_mod
from tpen.artifacts import (
    ArtifactDestination,
    ArtifactManager,
    ArtifactRoutingError,
    ObservabilityArtifactError,
    ObservabilityArtifactRef,
    ObservabilityBundleWriter,
    ObservabilityDescriptorSet,
    ObservabilityInventory,
    ObservabilityManifest,
    ObservabilityPathError,
    ObservabilityProfileStatus,
    ObservabilityProfileSummary,
    ObservabilityReading,
    ObservabilitySample,
    ObservabilitySchemaError,
    ObservabilityStatus,
    QuantityDescriptor,
    RUN_OBSERVABILITY_SCHEMA,
    RunArtifactSink,
)
from tpen.callback.base import claims_conflict
from tpen.checkpoint.manifest import CheckpointManifest
from tpen.durable_append import PartialAppendError, ends_without_newline
from tpen.inspection import (
    ObservabilityBundleInspection,
    RunObservabilityInspection,
    inspect_observability_bundle,
    inspect_run,
    inspect_run_observability,
)
from tpen.observability_artifacts import (
    LEGACY_PROFILE_STATUS_FILENAME,
    PROFILE_STATUS_FILENAME,
    parse_observability_record,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INVOCATION_ID = "00000000-0000-4000-8000-000000000001"
CONFIG_DIGEST = "151ea9c424d3b6c92294acdd31433cf663c6223e1560daf113678cc2e71a2c2f"


# ---------------------------------------------------------------------------
# Canonical r5 Appendix A-D wire fixtures (synthetic design records)
# ---------------------------------------------------------------------------


def _appendix_a_manifest() -> dict[str, Any]:
    return {
        "schema": "tpen.run-observability/v1",
        "kind": "manifest",
        "invocation_id": INVOCATION_ID,
        "lifecycle_state": "closed",
        "data_completeness": "complete",
        "run_id": "core",
        "producer": {
            "name": "proposed callback observability",
            "version": "proposed-contract-r5",
            "source_commit": {
                "status": "unavailable",
                "reason_code": "synthetic_example",
            },
        },
        "configuration_digest": CONFIG_DIGEST,
        "inventory_id": "inventory-example",
        "descriptor_set_id": "descriptors-example",
        "observers": [
            {
                "observer_id": "resources-example",
                "callback_position": 0,
                "callback_type": "proposed ResourceSamples",
                "selected_sources": {
                    "self-rusage": [
                        "cpu_user_s",
                        "cpu_system_s",
                        "rss_high_water_b",
                    ],
                    "linux-memory": ["rss_current_b"],
                    "nvidia-device": [
                        "device_memory_used_b",
                        "device_memory_total_b",
                        "gpu_busy_fraction",
                    ],
                },
            }
        ],
        "limits": {
            "interval_seconds": 10,
            "raw_sample_bytes": 67108864,
            "status_bytes": 262144,
        },
        "executor_context": "not_supplied",
        "scheduler_accounting": "not_supplied",
        "artifacts": [
            {
                "path": "inventory.json",
                "bytes": 3105,
                "sha256": "a147e0324f4a9f60edb1c83edff6a06bf74114cb3969ec2d5ae23e9c21bbed98",
            },
            {
                "path": "descriptors.json",
                "bytes": 2320,
                "sha256": "c8c6ba465ed478d33502db3d307e9d89713e461d38e1570563592fd1e7cc719f",
            },
            {
                "path": "resources/resources-example.jsonl",
                "bytes": 3594,
                "sha256": "3d826f909cfb4165b409481b629f40766be437e6eca25bfde581b5b812f586b0",
            },
            {
                "path": "status/resources-example.json",
                "bytes": 1255,
                "sha256": "8c12af86040be3ed0c2b5c221ed32df67f233e882f62c35f91859c466adc08e5",
            },
        ],
    }


def _appendix_a_inventory(invocation_id: str = INVOCATION_ID) -> dict[str, Any]:
    return {
        "schema": "tpen.run-observability/v1",
        "kind": "inventory",
        "invocation_id": invocation_id,
        "inventory_id": "inventory-example",
        "host": {
            "host_id": "host-example",
            "boot_id": "00000000-0000-4000-8000-000000000002",
            "platform": "linux-example",
        },
        "process": {
            "kind": "process",
            "host_id": "host-example",
            "boot_id": "00000000-0000-4000-8000-000000000002",
            "pid": 1234,
            "start_ticks": 9000,
        },
        "clock_domains": [
            {
                "clock_domain_id": "host-example-boot-monotonic",
                "kind": "host_monotonic_ns",
                "host_id": "host-example",
                "boot_id": "00000000-0000-4000-8000-000000000002",
                "anchor_monotonic_ns": 1000000000000,
                "anchor_read_bracket_ns": [999999990000, 1000000010000],
                "anchor_utc": "2026-11-19T18:00:00+00:00",
                "run_timezone": "America/New_York",
                "clock_tick_hz": 100,
            }
        ],
        "requested": {
            "device": "cuda:0",
            "dtype": "float64",
            "cpu_threads": {
                "status": "unavailable",
                "reason_code": "not_supplied",
            },
        },
        "observed": {
            "cpu_affinity": {"status": "ok", "value": [2, 3, 4, 5]},
            "effective_cpu_quota": {
                "status": "unavailable",
                "reason_code": "not_supplied",
            },
            "device-binding-1": {
                "runtime_device": "cuda:0",
                "physical_uuid": "GPU-example",
                "mig": {"status": "unavailable", "reason_code": "not_applicable"},
                "evidence": "synthetic matched runtime and vendor UUID",
            },
            "software": {
                "python": "3.12-example",
                "torch": "2.12-example",
                "driver": "driver-example",
            },
        },
        "sources": {
            "self-rusage": {
                "producer": "existing ProcessRUsageProbe adapter",
                "source_version": "proposed-contract-r5",
                "execution": "in_process_thread",
                "selected_quantities": [
                    "cpu_user_s",
                    "cpu_system_s",
                    "rss_high_water_b",
                ],
                "capability": "available",
            },
            "linux-memory": {
                "producer": "proposed LinuxProcessMemoryProbe",
                "source_version": "proposed-contract-r5",
                "execution": "helper_procfs_for_exact_target",
                "selected_quantities": ["rss_current_b"],
                "capability": "available",
            },
            "nvidia-device": {
                "producer": "proposed NvidiaSmiDeviceProbe",
                "source_version": "proposed-contract-r5",
                "execution": "supervised_read_only_command",
                "selected_quantities": [
                    "device_memory_used_b",
                    "device_memory_total_b",
                    "gpu_busy_fraction",
                ],
                "capability": "available",
                "tool": {
                    "path": "example-nvidia-smi",
                    "version": "tool-example",
                    "parser_version": "proposed-r5",
                },
            },
        },
        "executor_context": {"status": "unavailable", "reason_code": "not_supplied"},
        "scheduler_allocation": {
            "status": "unavailable",
            "reason_code": "not_supplied",
        },
        "billed_units": {"status": "unavailable", "reason_code": "not_supplied"},
    }


def _appendix_a_descriptors(invocation_id: str = INVOCATION_ID) -> dict[str, Any]:
    return {
        "schema": "tpen.run-observability/v1",
        "kind": "descriptors",
        "invocation_id": invocation_id,
        "descriptor_set_id": "descriptors-example",
        "quantities": {
            "cpu_user_s": {
                "source_id": "self-rusage",
                "unit": "s",
                "statistic": "counter",
                "subject_kind": "process",
                "window": "process_lifetime",
                "native_name": "user_cpu_seconds",
                "native_unit": "s",
                "conversion": "identity",
            },
            "cpu_system_s": {
                "source_id": "self-rusage",
                "unit": "s",
                "statistic": "counter",
                "subject_kind": "process",
                "window": "process_lifetime",
                "native_name": "system_cpu_seconds",
                "native_unit": "s",
                "conversion": "identity",
            },
            "rss_high_water_b": {
                "source_id": "self-rusage",
                "unit": "B",
                "statistic": "high_water",
                "subject_kind": "process",
                "window": "process_lifetime",
                "native_name": "peak_rss_mb",
                "native_unit": "MiB",
                "conversion": "multiply_by_1048576",
            },
            "rss_current_b": {
                "source_id": "linux-memory",
                "unit": "B",
                "statistic": "gauge",
                "subject_kind": "process",
                "window": "read_interval",
                "native_name": "VmRSS",
                "native_unit": "KiB",
                "conversion": "multiply_by_1024",
                "quality": ["approximate_rss"],
            },
            "device_memory_used_b": {
                "source_id": "nvidia-device",
                "unit": "B",
                "statistic": "gauge",
                "subject_kind": "physical_gpu",
                "window": "read_interval",
                "native_name": "memory.used",
                "native_unit": "MiB",
                "conversion": "multiply_by_1048576",
            },
            "device_memory_total_b": {
                "source_id": "nvidia-device",
                "unit": "B",
                "statistic": "capacity",
                "subject_kind": "physical_gpu",
                "window": "inventory_or_read_interval",
                "native_name": "memory.total",
                "native_unit": "MiB",
                "conversion": "multiply_by_1048576",
            },
            "gpu_busy_fraction": {
                "source_id": "nvidia-device",
                "unit": "1",
                "statistic": "vendor_window_mean",
                "subject_kind": "physical_gpu",
                "window": "vendor_recent_window_unknown_bounds",
                "native_name": "utilization.gpu",
                "native_unit": "percent",
                "conversion": "divide_by_100",
            },
        },
    }


def _appendix_a_samples(invocation_id: str = INVOCATION_ID) -> list[dict[str, Any]]:
    proc_subject = {
        "kind": "process",
        "host_id": "host-example",
        "boot_id": "00000000-0000-4000-8000-000000000002",
        "pid": 1234,
        "start_ticks": 9000,
    }
    gpu_subject = {
        "kind": "physical_gpu",
        "uuid": "GPU-example",
        "pci_bus_id": "0000:01:00.0",
        "mapping_evidence": "inventory-example/device-binding-1",
    }
    return [
        {
            "schema": "tpen.run-observability/v1",
            "kind": "sample",
            "invocation_id": invocation_id,
            "observer_id": "resources-example",
            "source_id": "self-rusage",
            "descriptor_set_id": "descriptors-example",
            "sequence": 0,
            "clock_domain_id": "host-example-boot-monotonic",
            "subject": proc_subject,
            "read_start_ns": 1000999990000,
            "read_end_ns": 1001000000000,
            "readings": {
                "cpu_user_s": {"status": "ok", "value": 100.0},
                "cpu_system_s": {"status": "ok", "value": 5.0},
                "rss_high_water_b": {"status": "ok", "value": 5368709120},
            },
        },
        {
            "schema": "tpen.run-observability/v1",
            "kind": "sample",
            "invocation_id": invocation_id,
            "observer_id": "resources-example",
            "source_id": "linux-memory",
            "descriptor_set_id": "descriptors-example",
            "sequence": 0,
            "clock_domain_id": "host-example-boot-monotonic",
            "subject": proc_subject,
            "read_start_ns": 1001009990000,
            "read_end_ns": 1001010000000,
            "readings": {
                "rss_current_b": {"status": "ok", "value": 1073741824},
            },
        },
        {
            "schema": "tpen.run-observability/v1",
            "kind": "sample",
            "invocation_id": invocation_id,
            "observer_id": "resources-example",
            "source_id": "nvidia-device",
            "descriptor_set_id": "descriptors-example",
            "sequence": 0,
            "clock_domain_id": "host-example-boot-monotonic",
            "subject": gpu_subject,
            "read_start_ns": 1001099990000,
            "read_end_ns": 1001100000000,
            "readings": {
                "device_memory_used_b": {"status": "ok", "value": 4294967296},
                "device_memory_total_b": {"status": "ok", "value": 17179869184},
                "gpu_busy_fraction": {"status": "ok", "value": 0.0},
            },
        },
        {
            "schema": "tpen.run-observability/v1",
            "kind": "sample",
            "invocation_id": invocation_id,
            "observer_id": "resources-example",
            "source_id": "self-rusage",
            "descriptor_set_id": "descriptors-example",
            "sequence": 1,
            "clock_domain_id": "host-example-boot-monotonic",
            "subject": proc_subject,
            "read_start_ns": 1010999990000,
            "read_end_ns": 1011000000000,
            "readings": {
                "cpu_user_s": {"status": "ok", "value": 112.0},
                "cpu_system_s": {"status": "ok", "value": 7.0},
                "rss_high_water_b": {"status": "ok", "value": 5368709120},
            },
        },
        {
            "schema": "tpen.run-observability/v1",
            "kind": "sample",
            "invocation_id": invocation_id,
            "observer_id": "resources-example",
            "source_id": "linux-memory",
            "descriptor_set_id": "descriptors-example",
            "sequence": 1,
            "clock_domain_id": "host-example-boot-monotonic",
            "subject": proc_subject,
            "read_start_ns": 1011009990000,
            "read_end_ns": 1011010000000,
            "readings": {
                "rss_current_b": {"status": "ok", "value": 2147483648},
            },
        },
        {
            "schema": "tpen.run-observability/v1",
            "kind": "sample",
            "invocation_id": invocation_id,
            "observer_id": "resources-example",
            "source_id": "nvidia-device",
            "descriptor_set_id": "descriptors-example",
            "sequence": 1,
            "clock_domain_id": "host-example-boot-monotonic",
            "subject": gpu_subject,
            "read_start_ns": 1011099990000,
            "read_end_ns": 1011100000000,
            "readings": {
                "device_memory_used_b": {"status": "ok", "value": 5368709120},
                "device_memory_total_b": {"status": "ok", "value": 17179869184},
                "gpu_busy_fraction": {"status": "ok", "value": 0.6},
            },
        },
    ]


def _appendix_a_status(invocation_id: str = INVOCATION_ID) -> dict[str, Any]:
    return {
        "schema": "tpen.run-observability/v1",
        "kind": "status",
        "invocation_id": invocation_id,
        "observer_id": "resources-example",
        "lifecycle_state": "closed",
        "data_completeness": "complete",
        "source_status": {
            "self-rusage": "available",
            "linux-memory": "available",
            "nvidia-device": "available",
        },
        "coverage": {
            "activation_ns": 1000990000000,
            "acquisition_end_ns": 1011200000000,
            "clock_domain_id": "host-example-boot-monotonic",
            "bootstrap": "unmeasured",
            "requested_interval_seconds": 10,
            "missed_deadlines": 0,
            "source_windows_ns": {
                "self-rusage": [1001000000000, 1011000000000],
                "linux-memory": [1001010000000, 1011010000000],
                "nvidia-device": [1001100000000, 1011100000000],
            },
        },
        "counts": {
            "attempted": 6,
            "written": 6,
            "dropped": 0,
            "failed": 0,
        },
        "truncation_reasons": [],
        "writer": {
            "state": "closed",
            "flush_confirmed": True,
            "fsync_confirmed": False,
        },
        "helper_cleanup": {"state": "confirmed_stopped"},
        "application_outcome": "completed",
        "invocation_outcome": "completed",
        "export_status": "not_requested",
    }


def _appendix_b_sample_and_status() -> tuple[dict[str, Any], dict[str, Any]]:
    sample = copy.deepcopy(_appendix_a_samples()[-1])
    sample["readings"]["gpu_busy_fraction"] = {
        "status": "unavailable",
        "reason_code": "unsupported",
        "detail": "selected runtime tool returned N/A",
    }
    status = copy.deepcopy(_appendix_a_status())
    status["data_completeness"] = "partial"
    status["source_status"]["nvidia-device"] = "partial"
    status["unavailable_fields"] = ["nvidia-device/gpu_busy_fraction"]
    return sample, status


def _appendix_c_status() -> dict[str, Any]:
    status = copy.deepcopy(_appendix_a_status())
    status["data_completeness"] = "partial"
    status["coverage"]["missed_deadlines"] = 1
    status["counts"] = {"attempted": 8, "written": 6, "dropped": 1, "failed": 1}
    status["truncation_reasons"] = ["queue_full"]
    return status


def _appendix_d_profile_status(invocation_id: str = INVOCATION_ID) -> dict[str, Any]:
    return {
        "schema": "tpen.run-observability/v1",
        "kind": "profile_status",
        "invocation_id": invocation_id,
        "observer_id": "profile-example",
        "window_id": "window-0001",
        "window_kind": "elapsed",
        "requested_activities": ["cpu", "configured_device"],
        "accepted_activities": ["cpu"],
        "observed_event_counts": {"cpu": 100, "device": 0},
        "clock_domain_id": "host-example-boot-monotonic",
        "activation_ns": 1001000000000,
        "requested_start_ns": 1001000000000,
        "requested_end_ns": 1031000000000,
        "start_call_bracket_ns": [1001000000000, 1001002000000],
        "stop_call_bracket_ns": [1045000000000, 1045200000000],
        "backend_event_bounds": {
            "status": "unavailable",
            "reason_code": "not_reported",
        },
        "stop_overshoot_bounds_seconds": [14.0, 14.2],
        "longest_checkpoint_gap_seconds": 39.6,
        "lifecycle_state": "closed",
        "data_completeness": "partial",
        "guarantee_class": "cooperative_native_capture",
        "exporter": "TorchOperatorSummary",
        "max_rows": 100,
        "export_status": "complete",
        "activity_diagnostics": ["requested device activity not accepted"],
    }


def _make_manager(
    tmp_path: Path,
    *,
    run_id: str = "run-001",
    routed: bool = False,
) -> ArtifactManager:
    logical_root = tmp_path / "logical"
    if routed:
        physical_root = tmp_path / "physical_leaf"
        physical_root.mkdir(parents=True, exist_ok=True)
        return ArtifactManager(
            logical_root,
            "exp",
            "sec",
            run_id,
            destination=ArtifactDestination(physical_root),
        )
    return ArtifactManager(logical_root, "exp", "sec", run_id)


def _open_manifest(
    invocation_id: str = INVOCATION_ID,
    *,
    observers: list[dict[str, Any]] | None = None,
) -> ObservabilityManifest:
    base = _appendix_a_manifest()
    base["invocation_id"] = invocation_id
    base["lifecycle_state"] = "open"
    base["data_completeness"] = "unknown"
    base["artifacts"] = []
    if observers is not None:
        base["observers"] = observers
    return ObservabilityManifest.from_mapping(base)


def _closed_manifest_template(
    invocation_id: str = INVOCATION_ID,
    *,
    data_completeness: str = "complete",
    observers: list[dict[str, Any]] | None = None,
) -> ObservabilityManifest:
    base = _appendix_a_manifest()
    base["invocation_id"] = invocation_id
    base["lifecycle_state"] = "closed"
    base["data_completeness"] = data_completeness
    base["artifacts"] = []
    if observers is not None:
        base["observers"] = observers
    return ObservabilityManifest.from_mapping(base)


def _write_complete_bundle(
    manager: ArtifactManager,
    *,
    invocation_id: str = INVOCATION_ID,
    sink: RunArtifactSink | None = None,
    samples: list[dict[str, Any]] | None = None,
    status_dict: dict[str, Any] | None = None,
    data_completeness: str = "complete",
    profile_status_dict: dict[str, Any] | None = None,
    profile_summary: ObservabilityProfileSummary | None = None,
) -> tuple[ObservabilityBundleWriter, ObservabilityManifest]:
    observers = list(_appendix_a_manifest()["observers"])
    if profile_status_dict is not None:
        observers.append(
            {
                "observer_id": profile_status_dict["observer_id"],
                "callback_position": 1,
                "callback_type": "proposed TorchProfile",
            }
        )
    writer = ObservabilityBundleWriter(manager, invocation_id, sink=sink)
    writer.write_open_manifest(_open_manifest(invocation_id, observers=observers))
    writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory(invocation_id))
    )
    writer.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors(invocation_id))
    )
    for raw_sample in (
        _appendix_a_samples(invocation_id) if samples is None else samples
    ):
        result = writer.append_sample(ObservabilitySample.from_mapping(raw_sample))
        assert result.written is True
    writer.write_status(
        ObservabilityStatus.from_mapping(
            _appendix_a_status(invocation_id) if status_dict is None else status_dict
        )
    )
    if profile_status_dict is not None:
        writer.write_profile_status(
            ObservabilityProfileStatus.from_mapping(profile_status_dict)
        )
    if profile_summary is not None:
        writer.write_profile_summary(profile_summary)
    closed = writer.finalize_manifest(
        _closed_manifest_template(
            invocation_id,
            data_completeness=data_completeness,
            observers=observers,
        )
    )
    return writer, closed


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ---------------------------------------------------------------------------
# 1. Exact Appendix A-D wire round-trips
# ---------------------------------------------------------------------------


def test_appendix_a_to_d_wire_payloads_round_trip_faithfully() -> None:
    manifest_raw = _appendix_a_manifest()
    inventory_raw = _appendix_a_inventory()
    descriptors_raw = _appendix_a_descriptors()
    samples_raw = _appendix_a_samples()
    status_a_raw = _appendix_a_status()
    sample_b_raw, status_b_raw = _appendix_b_sample_and_status()
    status_c_raw = _appendix_c_status()
    profile_d_raw = _appendix_d_profile_status()

    assert ObservabilityManifest.from_mapping(manifest_raw).to_dict() == manifest_raw
    assert ObservabilityInventory.from_mapping(inventory_raw).to_dict() == inventory_raw
    assert (
        ObservabilityDescriptorSet.from_mapping(descriptors_raw).to_dict()
        == descriptors_raw
    )
    for sample_raw in [*samples_raw, sample_b_raw]:
        assert ObservabilitySample.from_mapping(sample_raw).to_dict() == sample_raw
    for status_raw in (status_a_raw, status_b_raw, status_c_raw):
        assert ObservabilityStatus.from_mapping(status_raw).to_dict() == status_raw
    assert (
        ObservabilityProfileStatus.from_mapping(profile_d_raw).to_dict()
        == profile_d_raw
    )

    summary = ObservabilityProfileSummary(
        invocation_id=INVOCATION_ID,
        observer_id="profile-example",
        window_id="window-0001",
        exporter="TorchOperatorSummary",
        max_rows=10,
        total_rows=2,
        rows=(
            {"operator": "aten::mm", "self_cpu_time_us": 1200.5, "calls": 4},
            {"operator": "aten::add", "self_cpu_time_us": 150.0, "calls": 8},
        ),
        truncated=False,
        truncation_reasons=(),
        source_provenance={"torch_version": "2.12-example"},
    )
    assert (
        ObservabilityProfileSummary.from_mapping(summary.to_dict()).to_dict()
        == summary.to_dict()
    )

    # Generic record dispatcher covers all v1 kinds.
    assert isinstance(parse_observability_record(manifest_raw), ObservabilityManifest)
    assert isinstance(parse_observability_record(inventory_raw), ObservabilityInventory)
    assert isinstance(
        parse_observability_record(descriptors_raw), ObservabilityDescriptorSet
    )
    assert isinstance(parse_observability_record(samples_raw[0]), ObservabilitySample)
    assert isinstance(parse_observability_record(status_a_raw), ObservabilityStatus)
    assert isinstance(
        parse_observability_record(profile_d_raw), ObservabilityProfileStatus
    )
    assert isinstance(
        parse_observability_record(summary.to_dict()), ObservabilityProfileSummary
    )


# ---------------------------------------------------------------------------
# 2. Normal Appendix A writer -> pure reader & reducer round-trip
# ---------------------------------------------------------------------------


def test_normal_appendix_a_writer_and_reader_round_trip(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    writer, closed_manifest = _write_complete_bundle(manager)

    assert closed_manifest.lifecycle_state == "closed"
    assert all(ref.path != "manifest.json" for ref in closed_manifest.artifacts)
    assert all(
        ref.bytes is not None and ref.sha256 is not None
        for ref in closed_manifest.artifacts
    )

    bundle = inspect_observability_bundle(writer.bundle_dir)
    assert isinstance(bundle, ObservabilityBundleInspection)
    assert bundle.invocation_id == INVOCATION_ID
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "complete"
    assert bundle.derived_finalization == "confirmed"
    assert bundle.derived_reason is None
    assert bundle.torn_tails == ()
    assert bundle.interior_errors == ()
    assert len(bundle.samples["resources-example"]) == 6

    summary = bundle.summary
    assert summary.clock_domains == ("host-example-boot-monotonic",)

    # Counter: 14 CPU-seconds over 10 s -> 1.2 user + 0.2 system = 1.4 core-equivalents.
    cpu_user = summary.quantity("self-rusage", "cpu_user_s")
    cpu_sys = summary.quantity("self-rusage", "cpu_system_s")
    assert cpu_user is not None and cpu_sys is not None
    assert cpu_user.statistic == "counter"
    assert cpu_user.valid_interval_count == 1
    assert cpu_user.broken_interval_count == 0
    assert math.isclose(cpu_user.valid_delta_total or 0.0, 12.0)
    assert math.isclose(cpu_user.valid_elapsed_seconds or 0.0, 10.0)
    assert math.isclose(cpu_user.valid_rate_per_second or 0.0, 1.2)
    assert math.isclose(cpu_sys.valid_rate_per_second or 0.0, 0.2)
    assert math.isclose(
        (cpu_user.valid_rate_per_second or 0.0)
        + (cpu_sys.valid_rate_per_second or 0.0),
        1.4,
    )

    # Lifetime high-water stays distinct from sampled gauge max.
    rss_hw = summary.quantity("self-rusage", "rss_high_water_b")
    rss_cur = summary.quantity("linux-memory", "rss_current_b")
    assert rss_hw is not None and rss_cur is not None
    assert rss_hw.statistic == "high_water"
    assert rss_hw.observed_max == 5368709120
    assert rss_hw.sampled_max is None
    assert rss_hw.sampled_max_is_lower_bound is False

    assert rss_cur.statistic == "gauge"
    assert rss_cur.sampled_min == 1073741824
    assert rss_cur.sampled_max == 2147483648
    assert rss_cur.sampled_max_is_lower_bound is True
    assert rss_cur.observed_max is None

    # Capacity and vendor_window_mean stay distinct; vendor window is never integrated.
    dev_total = summary.quantity("nvidia-device", "device_memory_total_b")
    gpu_busy = summary.quantity("nvidia-device", "gpu_busy_fraction")
    assert dev_total is not None and gpu_busy is not None
    assert dev_total.statistic == "capacity"
    assert dev_total.latest_value == 17179869184
    assert dev_total.min_value == 17179869184
    assert dev_total.max_value == 17179869184

    assert gpu_busy.statistic == "vendor_window_mean"
    assert gpu_busy.window == "vendor_recent_window_unknown_bounds"
    assert gpu_busy.observation_min == 0.0
    assert gpu_busy.observation_max == 0.6
    assert math.isclose(gpu_busy.observation_mean or 0.0, 0.3)
    assert gpu_busy.continuous_window_integrated is False

    # Run-level discovery returns the exact bundle.
    run_obs = inspect_run_observability(manager.run_dir)
    assert isinstance(run_obs, RunObservabilityInspection)
    assert run_obs.status == "present"
    assert len(run_obs.invocations) == 1
    assert run_obs.invocations[0].summary.to_dict() == summary.to_dict()


# ---------------------------------------------------------------------------
# 3. Unavailable Appendix B field & zero fabrication guarantee
# ---------------------------------------------------------------------------


def test_appendix_b_unavailable_field_never_fabricates_zero(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    sample_b, status_b = _appendix_b_sample_and_status()
    samples = _appendix_a_samples()
    samples[-1] = sample_b

    writer, _ = _write_complete_bundle(
        manager,
        samples=samples,
        status_dict=status_b,
        data_completeness="partial",
    )
    bundle = inspect_observability_bundle(writer.bundle_dir)
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "partial"
    assert bundle.statuses["resources-example"].unavailable_fields == (
        "nvidia-device/gpu_busy_fraction",
    )

    gpu_busy = bundle.summary.quantity("nvidia-device", "gpu_busy_fraction")
    assert gpu_busy is not None
    assert gpu_busy.ok_count == 1
    assert gpu_busy.unavailable_count == 1
    assert gpu_busy.error_count == 0
    assert gpu_busy.reason_codes == ("unsupported",)
    assert gpu_busy.observation_min == 0.0
    assert gpu_busy.observation_max == 0.0
    assert gpu_busy.observation_mean == 0.0

    # When ALL observations are unavailable, mean/min/max remain None (never 0).
    manager_all_unavail = _make_manager(tmp_path, run_id="run-all-unavail")
    samples_none_ok = copy.deepcopy(samples)
    samples_none_ok[2]["readings"]["gpu_busy_fraction"] = {
        "status": "unavailable",
        "reason_code": "mig_unsupported",
    }
    writer2, _ = _write_complete_bundle(
        manager_all_unavail,
        samples=samples_none_ok,
        status_dict=status_b,
        data_completeness="partial",
    )
    bundle2 = inspect_observability_bundle(writer2.bundle_dir)
    gpu_busy2 = bundle2.summary.quantity("nvidia-device", "gpu_busy_fraction")
    assert gpu_busy2 is not None
    assert gpu_busy2.ok_count == 0
    assert gpu_busy2.unavailable_count == 2
    assert gpu_busy2.reason_codes == ("mig_unsupported", "unsupported")
    assert gpu_busy2.observation_min is None
    assert gpu_busy2.observation_max is None
    assert gpu_busy2.observation_mean is None


# ---------------------------------------------------------------------------
# 4. Partial/capped Appendix C status, independent budgets & bounded fields
# ---------------------------------------------------------------------------


def test_sample_byte_cap_preserves_prior_records_and_status_budget_is_independent(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    samples = [ObservabilitySample.from_mapping(s) for s in _appendix_a_samples()]
    def _line_len(sample: ObservabilitySample) -> int:
        return (
            len(
                json.dumps(
                    sample.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                    ensure_ascii=True,
                ).encode("ascii")
            )
            + 1
        )

    # Allow exactly 2 sample lines, refusing the 3rd BEFORE writing any partial bytes.
    cap_bytes = _line_len(samples[0]) + _line_len(samples[1]) + 10
    assert cap_bytes < _line_len(samples[0]) + _line_len(samples[1]) + _line_len(samples[2])
    sink = RunArtifactSink(
        max_sample_bytes=cap_bytes,
        max_pending_bytes=512,
        max_status_bytes=16_384,
        max_text_chars=24,
        max_list_items=6,
    )
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID, sink=sink)
    writer.write_open_manifest(_open_manifest())
    writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    writer.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )

    # Pending-bytes budget helper enforces cap without mutating disk.
    assert writer.try_reserve_pending("resources-example", 400) is True
    assert writer.pending_bytes("resources-example") == 400
    assert writer.try_reserve_pending("resources-example", 200) is False
    assert "pending_bytes_cap_reached" in writer.observer_truncation_reasons(
        "resources-example"
    )
    writer.release_pending("resources-example", 400)
    assert writer.pending_bytes("resources-example") == 0

    r0 = writer.append_sample(samples[0])
    r1 = writer.append_sample(samples[1])
    r2 = writer.append_sample(samples[2])
    assert r0.written is True and r0.capped is False
    assert r1.written is True and r1.capped is False
    assert r2.written is False and r2.capped is True
    assert r2.reason == "sample_bytes_cap_reached"

    jsonl_path = writer.bundle_dir / "resources" / "resources-example.jsonl"
    raw_bytes = jsonl_path.read_bytes()
    assert len(raw_bytes) <= cap_bytes
    assert raw_bytes.endswith(b"\n")
    assert len(raw_bytes.splitlines()) == 2

    # Separate status budget succeeds even though sample cap is saturated,
    # and bounds overly long diagnostic text/lists while keeping truncation tags.
    status_c = _appendix_c_status()
    status_c["counts"] = {"attempted": 3, "written": 2, "dropped": 1, "failed": 0}
    status_c["diagnostics"] = [
        "x" * 100,
        "second diagnostic item",
        "third item",
        "fourth item",
        "fifth item",
        "sixth item",
        "seventh item exceeding max_list_items",
        "eighth item exceeding max_list_items",
    ]
    status_path = writer.write_status(ObservabilityStatus.from_mapping(status_c))
    written_status = ObservabilityStatus.from_mapping(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert written_status.diagnostics is not None
    assert len(written_status.diagnostics) == 6
    assert len(written_status.diagnostics[0]) == 24
    assert written_status.diagnostics[0].endswith("...")
    assert "sample_bytes_cap_reached" in written_status.truncation_reasons
    assert "list_truncated" in written_status.truncation_reasons
    assert "text_truncated" in written_status.truncation_reasons

    # Finalize and inspect cleanly.
    writer.finalize_manifest(_closed_manifest_template(data_completeness="partial"))
    bundle = inspect_observability_bundle(writer.bundle_dir)
    assert bundle.derived_lifecycle_state == "closed"
    assert bundle.derived_data_completeness == "partial"
    assert len(bundle.samples["resources-example"]) == 2
    assert bundle.torn_tails == ()

    # Explicit max_status_bytes enforcement refuses oversized status writes.
    tiny_status_sink = RunArtifactSink(max_status_bytes=64)
    tiny_writer = ObservabilityBundleWriter(
        _make_manager(tmp_path, run_id="tiny-status"),
        INVOCATION_ID,
        sink=tiny_status_sink,
    )
    with pytest.raises(ObservabilityArtifactError, match="max_status_bytes"):
        tiny_writer.write_status(ObservabilityStatus.from_mapping(_appendix_a_status()))


# ---------------------------------------------------------------------------
# 5. Open / interrupted / torn-tail vs interior corrupt records (strictly read-only)
# ---------------------------------------------------------------------------


def test_open_torn_tail_and_interior_errors_diagnosed_separately_and_read_only(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)
    writer.write_open_manifest(_open_manifest())
    writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    writer.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    samples = _appendix_a_samples()
    writer.append_sample(ObservabilitySample.from_mapping(samples[0]))
    writer.append_sample(ObservabilitySample.from_mapping(samples[1]))

    # Simulate an uncatchable termination leaving an unterminated JSON prefix.
    jsonl_path = writer.bundle_dir / "resources" / "resources-example.jsonl"
    with jsonl_path.open("ab") as handle:
        handle.write(b'{"schema":"tpen.run-observability/v1","kind":"sample","truncated')

    before = _tree_snapshot(writer.bundle_dir)
    bundle = inspect_observability_bundle(writer.bundle_dir)
    assert _tree_snapshot(writer.bundle_dir) == before

    assert bundle.derived_lifecycle_state == "open"
    assert bundle.derived_data_completeness == "partial"
    assert bundle.derived_finalization == "unknown"
    assert bundle.derived_reason == "torn_tail_or_missing_final_status"
    assert len(bundle.samples["resources-example"]) == 2
    assert len(bundle.torn_tails) == 1
    assert bundle.torn_tails[0].code == "torn_tail"
    assert bundle.torn_tails[0].line_number == 3
    assert bundle.interior_errors == ()

    # Now construct a separate bundle with a corrupt terminated interior record
    # followed by a valid terminated sample record (no torn tail).
    manager2 = _make_manager(tmp_path, run_id="interior-corrupt")
    writer2 = ObservabilityBundleWriter(manager2, INVOCATION_ID)
    writer2.write_open_manifest(_open_manifest())
    writer2.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    writer2.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    writer2.append_sample(ObservabilitySample.from_mapping(samples[0]))
    jsonl_path2 = writer2.bundle_dir / "resources" / "resources-example.jsonl"
    with jsonl_path2.open("ab") as handle:
        handle.write(b'{"corrupt_interior": true\n')
    writer2.append_sample(ObservabilitySample.from_mapping(samples[3]))

    before2 = _tree_snapshot(writer2.bundle_dir)
    bundle2 = inspect_observability_bundle(writer2.bundle_dir)
    assert _tree_snapshot(writer2.bundle_dir) == before2

    assert bundle2.torn_tails == ()
    assert len(bundle2.interior_errors) == 1
    assert bundle2.interior_errors[0].code == "invalid_interior_record"
    assert bundle2.interior_errors[0].line_number == 2
    assert len(bundle2.samples["resources-example"]) == 2


def test_counter_reset_and_generation_change_break_intervals(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    base_samples = _appendix_a_samples()
    s0 = copy.deepcopy(base_samples[0])  # cpu_user_s=100.0, pid=1234, start_ticks=9000
    s1 = copy.deepcopy(base_samples[3])  # cpu_user_s=112.0, seq=1
    # Sequence 2: counter reset (cpu_user_s drops to 10.0)
    s2 = copy.deepcopy(s1)
    s2["sequence"] = 2
    s2["read_start_ns"] = 1020999990000
    s2["read_end_ns"] = 1021000000000
    s2["readings"]["cpu_user_s"] = {"status": "ok", "value": 10.0}
    s2["readings"]["cpu_system_s"] = {"status": "ok", "value": 8.0}
    # Sequence 3: generation change (start_ticks changes from 9000 -> 9999)
    s3 = copy.deepcopy(s2)
    s3["sequence"] = 3
    s3["read_start_ns"] = 1030999990000
    s3["read_end_ns"] = 1031000000000
    s3["subject"]["start_ticks"] = 9999
    s3["readings"]["cpu_user_s"] = {"status": "ok", "value": 25.0}
    s3["readings"]["cpu_system_s"] = {"status": "ok", "value": 10.0}

    status = _appendix_a_status()
    status["counts"] = {"attempted": 4, "written": 4, "dropped": 0, "failed": 0}
    writer, _ = _write_complete_bundle(
        manager,
        samples=[s0, s1, s2, s3],
        status_dict=status,
    )
    bundle = inspect_observability_bundle(writer.bundle_dir)
    cpu_user = bundle.summary.quantity("self-rusage", "cpu_user_s")
    assert cpu_user is not None
    # Only s0 -> s1 is valid; s1 -> s2 is counter_reset; s2 -> s3 is generation_change.
    assert cpu_user.valid_interval_count == 1
    assert cpu_user.broken_interval_count == 2
    assert math.isclose(cpu_user.valid_delta_total or 0.0, 12.0)
    diag_codes = {d.code for d in bundle.diagnostics}
    assert "counter_reset" in diag_codes
    assert "generation_change" in diag_codes


# ---------------------------------------------------------------------------
# 6. Profile-linked Appendix D bundle + legacy inspect_run coexistence
# ---------------------------------------------------------------------------


def _write_completed_legacy_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "status.json").write_text(
        json.dumps({"status": "completed"}) + "\n", encoding="utf-8"
    )
    checkpoint = run_dir / "checkpoints" / "step_000005"
    checkpoint.mkdir(parents=True)
    model_bytes = b"model-payload"
    (checkpoint / "model.pt").write_bytes(model_bytes)
    CheckpointManifest(
        schema_version=2,
        kind="tpen.checkpoint",
        next_iteration=5,
        completed_updates=4,
        created_at_unix=1.0,
        files={"model": "model.pt"},
        hashes={"model_sha256": hashlib.sha256(model_bytes).hexdigest()},
        runtime={},
        provenance={},
    ).write(checkpoint / "manifest.json")
    (checkpoint / "COMPLETE").write_text("complete\n", encoding="utf-8")


def test_profile_linked_bundle_and_legacy_inspect_run_coexistence(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    _write_completed_legacy_run(manager.run_dir)

    profile_status = _appendix_d_profile_status()
    rows = tuple(
        {"operator": f"aten::op_{i}", "self_cpu_time_us": float(100 - i)}
        for i in range(6)
    )
    summary_input = ObservabilityProfileSummary(
        invocation_id=INVOCATION_ID,
        observer_id="profile-example",
        window_id="window-0001",
        exporter="TorchOperatorSummary",
        max_rows=10,
        total_rows=6,
        rows=rows,
    )

    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)
    observers = [
        *_appendix_a_manifest()["observers"],
        {
            "observer_id": "profile-example",
            "callback_position": 1,
            "callback_type": "proposed TorchProfile",
        },
    ]
    writer.write_open_manifest(_open_manifest(observers=observers))
    writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    writer.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    for sample in _appendix_a_samples():
        writer.append_sample(ObservabilitySample.from_mapping(sample))
    writer.write_status(ObservabilityStatus.from_mapping(_appendix_a_status()))
    pstatus_path = writer.write_profile_status(
        ObservabilityProfileStatus.from_mapping(profile_status)
    )
    psummary_path = writer.write_profile_summary(summary_input, max_rows=3)
    writer.finalize_manifest(
        _closed_manifest_template(data_completeness="partial", observers=observers)
    )

    # Profile status MUST be written as profile-status.json, NEVER status.json.
    assert pstatus_path.name == PROFILE_STATUS_FILENAME
    assert psummary_path.name == "summary.json"
    assert list(writer.bundle_dir.rglob("status.json")) == []

    # Legacy inspect_run MUST still succeed with verdict == "completed".
    legacy_evidence = inspect_run(manager.run_dir)
    assert legacy_evidence.verdict == "completed"
    assert legacy_evidence.run_status == "completed"
    assert legacy_evidence.refusal is None

    # Pure observability reader inspects resources + profile status + bounded summary.
    run_obs = inspect_run_observability(manager.run_dir)
    assert run_obs.status == "present"
    assert len(run_obs.invocations) == 1
    bundle = run_obs.invocations[0]
    assert len(bundle.profile_statuses) == 1
    assert bundle.profile_statuses[0].window_id == "window-0001"
    assert bundle.profile_statuses[0].stop_overshoot_bounds_seconds == (14.0, 14.2)
    assert len(bundle.profile_summaries) == 1
    ps = bundle.profile_summaries[0]
    assert ps.max_rows == 3
    assert ps.total_rows == 6
    assert len(ps.rows) == 3
    assert ps.truncated is True
    assert "max_rows_exceeded" in ps.truncation_reasons

    # Standalone synthetic fixture using r5 Appendix D legacy profiles/.../status.json
    # is ALSO readable by inspect_observability_bundle.
    standalone = tmp_path / "standalone_appendix_d"
    win_dir = standalone / "profiles" / "profile-example" / "window-0001"
    win_dir.mkdir(parents=True)
    legacy_pstatus_bytes = (
        json.dumps(profile_status, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    (win_dir / LEGACY_PROFILE_STATUS_FILENAME).write_bytes(legacy_pstatus_bytes)
    rel_legacy = f"profiles/profile-example/window-0001/{LEGACY_PROFILE_STATUS_FILENAME}"
    standalone_manifest = ObservabilityManifest(
        invocation_id=INVOCATION_ID,
        lifecycle_state="closed",
        data_completeness="partial",
        run_id="core",
        producer={"name": "synthetic"},
        configuration_digest=CONFIG_DIGEST,
        inventory_id="inventory-example",
        descriptor_set_id="descriptors-example",
        observers=(
            {
                "observer_id": "profile-example",
                "callback_position": 0,
                "callback_type": "proposed TorchProfile",
            },
        ),
        limits={"status_bytes": 262144},
        artifacts=(
            ObservabilityArtifactRef(
                path=rel_legacy,
                bytes=len(legacy_pstatus_bytes),
                sha256=hashlib.sha256(legacy_pstatus_bytes).hexdigest(),
            ),
        ),
    )
    (standalone / "manifest.json").write_text(
        json.dumps(standalone_manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    standalone_bundle = inspect_observability_bundle(standalone)
    assert len(standalone_bundle.profile_statuses) == 1
    assert standalone_bundle.profile_statuses[0].observer_id == "profile-example"


# ---------------------------------------------------------------------------
# 7. Routed containment, ownership claims, symlink & tamper refusals
# ---------------------------------------------------------------------------


def test_routed_destination_containment_and_ownership_claims(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path, routed=True)
    sink = RunArtifactSink(relative_dir="nested/observability")

    # Pure sink performs zero I/O in __init__.
    assert not (manager.run_dir / "nested").exists()
    assert not manager.logical_run_dir.exists()

    claim_a = sink.ownership_claim(manager, INVOCATION_ID, "resources-a")
    claim_a_dup = sink.ownership_claim(manager, INVOCATION_ID, "resources-a")
    claim_b = sink.ownership_claim(manager, INVOCATION_ID, "resources-b")
    assert claim_a.kind == "artifact_writer"
    assert claims_conflict(claim_a, claim_a_dup) is True
    assert claims_conflict(claim_a, claim_b) is False
    assert manager.destination is not None
    assert str(manager.destination.physical_root.resolve()) in claim_a.scope

    writer, _ = _write_complete_bundle(manager, sink=sink)
    assert str(writer.bundle_dir.resolve()).startswith(
        str(manager.destination.physical_root.resolve())
    )
    assert not manager.logical_run_dir.exists()

    inspected = inspect_run_observability(
        manager.run_dir, relative_dir="nested/observability"
    )
    assert inspected.status == "present"
    assert len(inspected.invocations) == 1


@pytest.mark.parametrize(
    "bad_dir",
    [
        "",
        "   ",
        "/abs/path",
        "../escape",
        "obs/../escape",
        ".",
        "obs/./sub",
        "obs\\win",
        "obs\x00null",
        "obs\nline",
        "obs/",
        "~/home",
    ],
)
def test_sink_rejects_unsafe_relative_dir(bad_dir: str) -> None:
    with pytest.raises((ObservabilityPathError, ArtifactRoutingError)):
        RunArtifactSink(relative_dir=bad_dir)


@pytest.mark.parametrize(
    "bad_observer_id",
    ["status", "STATUS", "Status", "../x", "a/b", "a\\b", ".hidden", "-flag", "", "a b"],
)
def test_observer_id_validation_forbids_status_collision_and_escapes(
    bad_observer_id: str,
) -> None:
    sample = _appendix_a_samples()[0]
    sample["observer_id"] = bad_observer_id
    with pytest.raises((ObservabilityPathError, ArtifactRoutingError)):
        ObservabilitySample.from_mapping(sample)


def test_symlink_and_tampered_sha256_and_size_are_refused(tmp_path: Path) -> None:
    # 1. Symlinked observability directory refused by writer and reader.
    manager = _make_manager(tmp_path, run_id="symlink-writer")
    manager.run_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside_dir"
    outside.mkdir()
    (manager.run_dir / "observability").symlink_to(outside)

    sink = RunArtifactSink()
    with pytest.raises((ObservabilityPathError, ArtifactRoutingError)):
        sink.resolve_bundle_dir(manager, INVOCATION_ID)
    with pytest.raises((ObservabilityPathError, ArtifactRoutingError)):
        inspect_run_observability(manager.run_dir)

    # 2. Symlinked artifact file inside bundle refused by reader.
    manager2 = _make_manager(tmp_path, run_id="symlink-file")
    writer2, _ = _write_complete_bundle(manager2)
    secret = tmp_path / "secret.json"
    secret.write_text("{}", encoding="utf-8")
    (writer2.bundle_dir / "stray_link.json").symlink_to(secret)
    with pytest.raises(ObservabilityPathError):
        inspect_observability_bundle(writer2.bundle_dir)

    # 3. Tampered closed bundle bytes / SHA-256 mismatch refused.
    manager3 = _make_manager(tmp_path, run_id="tampered-hash")
    writer3, _ = _write_complete_bundle(manager3)
    inv_path = writer3.bundle_dir / "inventory.json"
    original = inv_path.read_text(encoding="utf-8")
    # Same byte length, different content -> sha256 mismatch.
    tampered = original.replace("host-example", "host-TAMPERD", 1)
    assert len(tampered.encode("utf-8")) == len(original.encode("utf-8"))
    inv_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ObservabilityArtifactError, match="sha256 mismatch"):
        inspect_observability_bundle(writer3.bundle_dir)

    # Byte-size mismatch refused.
    inv_path.write_text(original + " ", encoding="utf-8")
    with pytest.raises(ObservabilityArtifactError, match="byte size mismatch"):
        inspect_observability_bundle(writer3.bundle_dir)


# ---------------------------------------------------------------------------
# 8. Negative schema, write-once, sequence, subject-kind & multi-invocation tests
# ---------------------------------------------------------------------------


def test_negative_schema_validation_rules() -> None:
    # NaN / Infinity / bool rejected in readings.
    for bad_value in (float("nan"), float("inf"), float("-inf"), True, False, None, "1.0"):
        with pytest.raises(ObservabilitySchemaError):
            ObservabilityReading(status="ok", value=bad_value)  # type: ignore[arg-type]

    # "ok" missing value, "unavailable"/"error" carrying value rejected.
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityReading.from_mapping({"status": "ok"})
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityReading.from_mapping(
            {"status": "unavailable", "reason_code": "unsupported", "value": 0}
        )
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityReading.from_mapping(
            {"status": "error", "reason_code": "timeout", "value": 0.0}
        )

    # Untagged None inside sample.readings rejected.
    bad_sample = _appendix_a_samples()[0]
    bad_sample["readings"]["cpu_user_s"] = None
    with pytest.raises(ObservabilitySchemaError):
        ObservabilitySample.from_mapping(bad_sample)

    # Non-mapping untrusted inputs and non-string mapping keys rejected across
    # all boundary deserializers without tripping runtime Typeguard checks.
    for non_mapping in (None, 123, [], {1: "bad"}):
        with pytest.raises(ObservabilitySchemaError):
            parse_observability_record(non_mapping)
        for cls in (
            ObservabilityReading,
            QuantityDescriptor,
            ObservabilityArtifactRef,
            ObservabilityManifest,
            ObservabilityInventory,
            ObservabilityDescriptorSet,
            ObservabilitySample,
            ObservabilityStatus,
            ObservabilityProfileStatus,
            ObservabilityProfileSummary,
        ):
            with pytest.raises(ObservabilitySchemaError):
                cls.from_mapping(non_mapping)

    # Nested non-mapping elements inside manifest.artifacts,
    # descriptors.quantities, profile_status.artifacts, and non-sequence
    # descriptor.quality raise ObservabilitySchemaError cleanly.
    bad_manifest_artifacts = _appendix_a_manifest()
    bad_manifest_artifacts["artifacts"] = [None]
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityManifest.from_mapping(bad_manifest_artifacts)

    bad_descriptors = _appendix_a_descriptors()
    bad_descriptors["quantities"] = {"x": None}
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityDescriptorSet.from_mapping(bad_descriptors)

    for bad_quality in (None, 123, "approximate_rss"):
        bad_qty_desc = copy.deepcopy(_appendix_a_descriptors()["quantities"]["rss_current_b"])
        bad_qty_desc["quality"] = bad_quality
        with pytest.raises(ObservabilitySchemaError):
            QuantityDescriptor.from_mapping(bad_qty_desc)

    bad_profile_artifacts = _appendix_d_profile_status()
    bad_profile_artifacts["artifacts"] = [None]
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityProfileStatus.from_mapping(bad_profile_artifacts)

    # read_end_ns < read_start_ns and negative/bool sequence rejected.
    bad_time = _appendix_a_samples()[0]
    bad_time["read_end_ns"] = bad_time["read_start_ns"] - 1
    with pytest.raises(ObservabilitySchemaError):
        ObservabilitySample.from_mapping(bad_time)

    for bad_seq in (-1, True, False, 1.5):
        bad_seq_sample = _appendix_a_samples()[0]
        bad_seq_sample["sequence"] = bad_seq
        with pytest.raises(ObservabilitySchemaError):
            ObservabilitySample.from_mapping(bad_seq_sample)

    # Unsupported major schema and unknown kind rejected.
    bad_manifest = _appendix_a_manifest()
    bad_manifest["schema"] = "tpen.run-observability/v2"
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityManifest.from_mapping(bad_manifest)

    bad_kind = _appendix_a_manifest()
    bad_kind["kind"] = "unknown_kind"
    with pytest.raises(ObservabilitySchemaError):
        parse_observability_record(bad_kind)

    # Invalid SHA-256 digest rejected.
    bad_digest = _appendix_a_manifest()
    bad_digest["configuration_digest"] = "not-a-sha256"
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityManifest.from_mapping(bad_digest)

    # Manifest referencing manifest.json recursively rejected.
    with pytest.raises(ObservabilityPathError):
        ObservabilityArtifactRef(path="manifest.json", bytes=10, sha256="a" * 64)

    # Closed status count mismatch rejected.
    bad_status = _appendix_a_status()
    bad_status["counts"] = {"attempted": 7, "written": 6, "dropped": 0, "failed": 0}
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityStatus.from_mapping(bad_status)

    # Bool in status counts rejected.
    bool_status = _appendix_a_status()
    bool_status["counts"]["attempted"] = True
    with pytest.raises(ObservabilitySchemaError):
        ObservabilityStatus.from_mapping(bool_status)

    # RunArtifactSink rejects non-positive / non-finite / bool bounds.
    with pytest.raises(ObservabilitySchemaError):
        RunArtifactSink(flush_interval_seconds=float("nan"))
    with pytest.raises(ObservabilitySchemaError):
        RunArtifactSink(flush_interval_seconds=0.0)
    with pytest.raises(ObservabilitySchemaError):
        RunArtifactSink(max_sample_bytes=True)  # type: ignore[arg-type]
    with pytest.raises(ObservabilitySchemaError):
        RunArtifactSink(max_status_bytes=0)


def test_writer_enforces_write_once_sequence_and_descriptor_matching(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)
    writer.write_open_manifest(_open_manifest())
    inv = ObservabilityInventory.from_mapping(_appendix_a_inventory())
    desc = ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())

    writer.write_inventory(inv)
    with pytest.raises(ObservabilityArtifactError, match="write-once"):
        writer.write_inventory(inv)

    writer.write_descriptors(desc)
    with pytest.raises(ObservabilityArtifactError, match="write-once"):
        writer.write_descriptors(desc)

    samples = _appendix_a_samples()
    writer.append_sample(ObservabilitySample.from_mapping(samples[0]))

    # Duplicate / regressed sequence rejected.
    with pytest.raises(ObservabilitySchemaError, match="strictly increase"):
        writer.append_sample(ObservabilitySample.from_mapping(samples[0]))

    # Subject-kind mismatch against descriptors rejected.
    wrong_subject = copy.deepcopy(samples[3])
    wrong_subject["subject"] = {
        "kind": "physical_gpu",
        "uuid": "GPU-example",
    }
    with pytest.raises(ObservabilitySchemaError, match="subject_kind"):
        writer.append_sample(ObservabilitySample.from_mapping(wrong_subject))

    # Unknown quantity rejected.
    unknown_qty = copy.deepcopy(samples[3])
    unknown_qty["readings"]["undeclared_metric"] = {"status": "ok", "value": 1}
    with pytest.raises(ObservabilitySchemaError, match="not declared"):
        writer.append_sample(ObservabilitySample.from_mapping(unknown_qty))

    # Finalize then refuse further writes.
    writer.write_status(ObservabilityStatus.from_mapping(_appendix_a_status()))
    writer.finalize_manifest(_closed_manifest_template())
    with pytest.raises(ObservabilityArtifactError, match="finalized"):
        writer.append_sample(ObservabilitySample.from_mapping(samples[3]))


def test_absent_observability_and_multi_invocation_isolation_and_duplicate_refusal(
    tmp_path: Path,
) -> None:
    # Absent observability/ returns status="absent" without mutating disk.
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    absent = inspect_run_observability(empty_run)
    assert absent == RunObservabilityInspection(
        run_dir=str(empty_run.resolve()), status="absent", invocations=()
    )
    assert not (empty_run / "observability").exists()

    # Two distinct invocations under one run_dir remain strictly isolated.
    manager = _make_manager(tmp_path, run_id="multi-inv")
    inv1 = "00000000-0000-4000-8000-000000000011"
    inv2 = "00000000-0000-4000-8000-000000000022"
    _write_complete_bundle(manager, invocation_id=inv1)
    _write_complete_bundle(manager, invocation_id=inv2)

    inspected = inspect_run_observability(manager.run_dir)
    assert inspected.status == "present"
    assert [b.invocation_id for b in inspected.invocations] == [inv1, inv2]
    assert len(inspected.invocations[0].samples["resources-example"]) == 6
    assert len(inspected.invocations[1].samples["resources-example"]) == 6

    # Duplicate invocation_id across two bundle directories is refused.
    dup_dir = manager.run_dir / "observability" / "duplicate-copy"
    dup_dir.mkdir()
    src_dir = manager.run_dir / "observability" / inv1
    for item in src_dir.rglob("*"):
        rel = item.relative_to(src_dir)
        dest = dup_dir / rel
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(item.read_bytes())
    with pytest.raises(ObservabilityArtifactError, match="duplicate invocation_id"):
        inspect_run_observability(manager.run_dir)


def test_reader_rejects_on_disk_nan_and_unsupported_schema(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path, run_id="nan-disk")
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)
    writer.write_open_manifest(_open_manifest())
    # Inject raw JSON with literal NaN on disk.
    manifest_path = writer.bundle_dir / "manifest.json"
    raw = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        raw.replace('"interval_seconds": 10', '"interval_seconds": NaN'),
        encoding="utf-8",
    )
    with pytest.raises(ObservabilitySchemaError, match="non-finite JSON constant"):
        inspect_observability_bundle(writer.bundle_dir)

    # Unsupported major schema on disk is rejected with ObservabilitySchemaError.
    manifest_path.write_text(
        raw.replace(RUN_OBSERVABILITY_SCHEMA, "tpen.run-observability/v2"),
        encoding="utf-8",
    )
    with pytest.raises(ObservabilitySchemaError, match="unsupported manifest schema"):
        inspect_observability_bundle(writer.bundle_dir)

    # Non-object top-level JSON on disk ([], "str", 123) is rejected with ObservabilitySchemaError.
    for non_object_json in ("[]\n", '"str"\n', "123\n"):
        manifest_path.write_text(non_object_json, encoding="utf-8")
        with pytest.raises(ObservabilitySchemaError, match="must contain a JSON object"):
            inspect_observability_bundle(writer.bundle_dir)

    # Restore valid open manifest and verify a non-object JSON interior line
    # in a sample JSONL stream is isolated as invalid_interior_record rather
    # than crashing with a runtime type error.
    manifest_path.write_text(raw, encoding="utf-8")
    resources_dir = writer.bundle_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    (resources_dir / "resources-example.jsonl").write_text("[]\n", encoding="utf-8")
    bundle = inspect_observability_bundle(writer.bundle_dir)
    assert len(bundle.interior_errors) == 1
    assert bundle.interior_errors[0].code == "invalid_interior_record"
    assert bundle.interior_errors[0].line_number == 1


# ---------------------------------------------------------------------------
# 9. Re-exports & Torch / experiments / tpen_executor import firewall
# ---------------------------------------------------------------------------


def test_reexports_and_import_firewall() -> None:
    assert "RunArtifactSink" in artifacts_mod.__all__
    assert "ObservabilityBundleWriter" in artifacts_mod.__all__
    assert "RunArtifactSink" in dir(artifacts_mod)
    assert artifacts_mod.RUN_OBSERVABILITY_SCHEMA == RUN_OBSERVABILITY_SCHEMA
    assert "inspect_observability_bundle" in inspection_mod.__all__
    assert "inspect_run_observability" in inspection_mod.__all__

    # AST hygiene check on tpen/observability_artifacts.py, tpen/artifacts.py, and tpen/inspection.py.
    for rel, forbid_torch in (
        ("tpen/observability_artifacts.py", True),
        ("tpen/artifacts.py", False),
        ("tpen/inspection.py", True),
    ):
        source_path = REPO_ROOT / rel
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        forbidden_roots = (
            {"torch", "experiments", "tpen_executor"}
            if forbid_torch
            else {"experiments", "tpen_executor"}
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_roots
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                assert node.module.split(".")[0] not in forbidden_roots

    # Fresh subprocess checks: observability_artifacts + artifacts remain 100% torch-free
    # and free of experiments/tpen_executor in both import orders.
    for import_stmt in (
        "import tpen.observability_artifacts, tpen.artifacts",
        "import tpen.artifacts; from tpen.artifacts import RunArtifactSink, ObservabilityBundleWriter",
        "import tpen.observability_artifacts; from tpen.observability_artifacts import inspect_observability_bundle, inspect_run_observability",
    ):
        probe = (
            f"{import_stmt}; import sys; "
            "bad = [m for m in sys.modules if m == 'torch' or m.startswith('torch.') "
            "or m == 'experiments' or m.startswith('experiments.') "
            "or m == 'tpen_executor' or m.startswith('tpen_executor.')]; "
            "assert not bad, f'forbidden modules loaded: {bad}'; "
            "print('firewall-ok')"
        )
        res = subprocess.run(
            [sys.executable, "-B", "-c", probe],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == "firewall-ok"

    # Fresh subprocess checks tpen.inspection remains free of experiments/tpen_executor
    # and has no circular import with tpen.observability_artifacts / tpen.artifacts.
    inspection_probe = (
        "import tpen.inspection, tpen.artifacts, tpen.observability_artifacts, sys; "
        "bad = [m for m in sys.modules if m == 'experiments' or m.startswith('experiments.') "
        "or m == 'tpen_executor' or m.startswith('tpen_executor.')]; "
        "assert not bad, f'forbidden modules loaded: {bad}'; "
        "print('inspection-ok')"
    )
    res_insp = subprocess.run(
        [sys.executable, "-B", "-c", inspection_probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res_insp.returncode == 0, res_insp.stderr
    assert res_insp.stdout.strip() == "inspection-ok"


# ---------------------------------------------------------------------------
# 10. T2 review discriminating & coverage-gap tests (41211cc6)
# ---------------------------------------------------------------------------


def test_counter_missing_operand_breaks_interval(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path, run_id="counter-missing-operand")
    base_s0 = _appendix_a_samples()[0]

    s0 = copy.deepcopy(base_s0)
    s0["sequence"] = 0
    s0["read_start_ns"] = 1_000_999_990_000
    s0["read_end_ns"] = 1_001_000_000_000
    s0["readings"] = {
        "cpu_user_s": {"status": "ok", "value": 100.0},
        "cpu_system_s": {"status": "ok", "value": 5.0},
    }

    s1 = copy.deepcopy(s0)
    s1["sequence"] = 1
    s1["read_start_ns"] = 1_010_999_990_000
    s1["read_end_ns"] = 1_011_000_000_000
    # cpu_user_s omitted: missing operand in the middle sample.
    s1["readings"] = {
        "cpu_system_s": {"status": "ok", "value": 7.0},
    }

    s2 = copy.deepcopy(s0)
    s2["sequence"] = 2
    s2["read_start_ns"] = 1_020_999_990_000
    s2["read_end_ns"] = 1_021_000_000_000
    s2["readings"] = {
        "cpu_user_s": {"status": "ok", "value": 124.0},
        "cpu_system_s": {"status": "ok", "value": 9.0},
    }

    status = _appendix_a_status()
    status["counts"] = {"attempted": 3, "written": 3, "dropped": 0, "failed": 0}
    writer, _ = _write_complete_bundle(
        manager,
        samples=[s0, s1, s2],
        status_dict=status,
        data_completeness="complete",
    )

    bundle = inspect_observability_bundle(writer.bundle_dir)
    cpu_system = bundle.summary.quantity("self-rusage", "cpu_system_s")
    cpu_user = bundle.summary.quantity("self-rusage", "cpu_user_s")
    assert cpu_system is not None and cpu_user is not None

    assert cpu_system.valid_interval_count == 2
    assert cpu_system.broken_interval_count == 0
    assert math.isclose(cpu_system.valid_delta_total or 0.0, 4.0)

    # Missing cpu_user_s at seq=1 must break intervals rather than bridging seq=0 -> seq=2.
    assert cpu_user.valid_interval_count == 0
    assert cpu_user.broken_interval_count >= 1
    assert cpu_user.valid_delta_total is None
    assert cpu_user.valid_rate_per_second is None


def test_counter_sequence_gap_breaks_interval(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path, run_id="counter-seq-gap")
    base_s0 = _appendix_a_samples()[0]

    s0 = copy.deepcopy(base_s0)
    s0["sequence"] = 0
    s0["read_start_ns"] = 1_000_999_990_000
    s0["read_end_ns"] = 1_001_000_000_000
    s0["readings"] = {"cpu_user_s": {"status": "ok", "value": 100.0}}

    # Skip seq=1 entirely to simulate a dropped sample.
    s2 = copy.deepcopy(s0)
    s2["sequence"] = 2
    s2["read_start_ns"] = 1_020_999_990_000
    s2["read_end_ns"] = 1_021_000_000_000
    s2["readings"] = {"cpu_user_s": {"status": "ok", "value": 124.0}}

    status = _appendix_a_status()
    status["data_completeness"] = "partial"
    status["counts"] = {"attempted": 3, "written": 2, "dropped": 1, "failed": 0}
    writer, _ = _write_complete_bundle(
        manager,
        samples=[s0, s2],
        status_dict=status,
        data_completeness="partial",
    )

    bundle = inspect_observability_bundle(writer.bundle_dir)
    cpu_user = bundle.summary.quantity("self-rusage", "cpu_user_s")
    assert cpu_user is not None
    assert cpu_user.valid_interval_count == 0
    assert cpu_user.broken_interval_count == 1
    assert cpu_user.valid_delta_total is None
    assert cpu_user.valid_rate_per_second is None


def test_closed_bundle_missing_descriptors_cannot_claim_complete_and_checks_descriptor_set_id(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path, run_id="missing-descriptors")
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)
    writer.write_open_manifest(_open_manifest())
    writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    # Intentionally omit write_descriptors() and write samples carrying a wrong descriptor_set_id.
    wrong_samples = copy.deepcopy(_appendix_a_samples())
    for raw in wrong_samples:
        raw["descriptor_set_id"] = "wrong-descriptor-set"
        res = writer.append_sample(ObservabilitySample.from_mapping(raw))
        assert res.written is True
    writer.write_status(ObservabilityStatus.from_mapping(_appendix_a_status()))
    closed = writer.finalize_manifest(
        _closed_manifest_template(data_completeness="complete")
    )
    assert not (writer.bundle_dir / "descriptors.json").exists()
    assert closed.descriptor_set_id == "descriptors-example"

    bundle = inspect_observability_bundle(writer.bundle_dir)
    assert bundle.descriptors is None
    assert bundle.derived_data_completeness != "complete"
    assert len(bundle.interior_errors) >= 1
    assert bundle.samples["resources-example"] == ()


def test_reader_diagnoses_undeclared_clock_domain_against_inventory(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path, run_id="undeclared-clock-domain")
    base_samples = _appendix_a_samples()
    s0 = copy.deepcopy(base_samples[0])
    s1 = copy.deepcopy(base_samples[3])
    s0["clock_domain_id"] = "undeclared-foreign-clock"
    s1["clock_domain_id"] = "undeclared-foreign-clock"

    status = _appendix_a_status()
    status["counts"] = {"attempted": 2, "written": 2, "dropped": 0, "failed": 0}
    writer, _ = _write_complete_bundle(
        manager,
        samples=[s0, s1],
        status_dict=status,
        data_completeness="complete",
    )

    try:
        bundle = inspect_observability_bundle(writer.bundle_dir)
    except ObservabilitySchemaError:
        return

    assert bundle.derived_data_completeness != "complete"
    assert bool(bundle.interior_errors) or any(
        "clock" in diag.code for diag in bundle.diagnostics
    )


def test_closed_manifest_cannot_omit_inspected_artifacts_from_integrity_check(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path, run_id="manifest-omits-artifacts")
    writer, _ = _write_complete_bundle(manager)

    # Overwrite manifest.json with a closed complete manifest whose artifacts list is empty.
    empty_closed_manifest = _closed_manifest_template(data_completeness="complete")
    assert empty_closed_manifest.artifacts == ()
    (writer.bundle_dir / "manifest.json").write_text(
        json.dumps(empty_closed_manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        bundle = inspect_observability_bundle(writer.bundle_dir)
    except ObservabilityArtifactError:
        return

    assert bundle.derived_finalization != "confirmed"
    assert bundle.derived_data_completeness != "complete"


def test_deserializers_reject_explicit_untagged_null_on_optional_fields() -> None:
    status_null_clock = copy.deepcopy(_appendix_a_status())
    status_null_clock["coverage"]["clock_domain_id"] = None

    cases: list[tuple[str, Any]] = [
        (
            "ObservabilityReading.detail=None",
            lambda: ObservabilityReading.from_mapping(
                {"status": "unavailable", "reason_code": "timeout", "detail": None}
            ),
        ),
        (
            "ObservabilityArtifactRef.bytes/sha256=None",
            lambda: ObservabilityArtifactRef.from_mapping(
                {"path": "inventory.json", "bytes": None, "sha256": None}
            ),
        ),
        (
            "ObservabilityArtifactRef.kind=None",
            lambda: ObservabilityArtifactRef.from_mapping(
                {"path": "inventory.json", "kind": None}
            ),
        ),
        (
            "ObservabilityStatus.unavailable_fields=None",
            lambda: ObservabilityStatus.from_mapping(
                {**_appendix_a_status(), "unavailable_fields": None}
            ),
        ),
        (
            "ObservabilityStatus.diagnostics=None",
            lambda: ObservabilityStatus.from_mapping(
                {**_appendix_a_status(), "diagnostics": None}
            ),
        ),
        (
            "ObservabilityStatus.coverage.clock_domain_id=None",
            lambda: ObservabilityStatus.from_mapping(status_null_clock),
        ),
        (
            "ObservabilityProfileStatus.max_rows=None",
            lambda: ObservabilityProfileStatus.from_mapping(
                {**_appendix_d_profile_status(), "max_rows": None}
            ),
        ),
        (
            "ObservabilityProfileStatus.artifacts=None",
            lambda: ObservabilityProfileStatus.from_mapping(
                {**_appendix_d_profile_status(), "artifacts": None}
            ),
        ),
        (
            "ObservabilityProfileStatus.truncation_reasons=None",
            lambda: ObservabilityProfileStatus.from_mapping(
                {**_appendix_d_profile_status(), "truncation_reasons": None}
            ),
        ),
        (
            "ObservabilityProfileStatus.stop_overshoot_bounds_seconds=None",
            lambda: ObservabilityProfileStatus.from_mapping(
                {
                    **_appendix_d_profile_status(),
                    "stop_overshoot_bounds_seconds": None,
                }
            ),
        ),
    ]

    accepted: list[str] = []
    for label, call in cases:
        try:
            call()
        except ObservabilitySchemaError:
            continue
        accepted.append(label)

    assert not accepted, (
        f"deserializers accepted explicit untagged null on optional fields: {accepted}"
    )


def test_symlink_escape_refused_in_ownership_claim_and_profile_status_artifacts(
    tmp_path: Path,
) -> None:
    # Sub-case A: symlinked resources/ directory under bundle_dir refused by ownership_claim.
    manager = _make_manager(tmp_path, run_id="symlink-claim", routed=True)
    sink = RunArtifactSink()
    outside_dir = tmp_path / "outside_claim"
    outside_dir.mkdir()
    bundle_dir = sink.resolve_bundle_dir(manager, INVOCATION_ID)
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "resources").symlink_to(outside_dir)

    with pytest.raises((ObservabilityPathError, ArtifactRoutingError)):
        sink.ownership_claim(manager, INVOCATION_ID, "resources-example")

    # Sub-case B: open bundle profile_status referencing symlinked staging/trace.json refused.
    manager2 = _make_manager(tmp_path, run_id="symlink-profile-status")
    writer = ObservabilityBundleWriter(manager2, INVOCATION_ID)
    observers = [
        {
            "observer_id": "profile-example",
            "callback_position": 0,
            "callback_type": "proposed TorchProfile",
        }
    ]
    writer.write_open_manifest(_open_manifest(observers=observers))
    outside_profile = tmp_path / "outside_profile"
    outside_profile.mkdir()
    (outside_profile / "trace.json").write_text("{}", encoding="utf-8")

    pstatus_raw = _appendix_d_profile_status()
    pstatus_raw["artifacts"] = [
        {"path": "profiles/profile-example/window-0001/staging/trace.json"}
    ]
    writer.write_profile_status(ObservabilityProfileStatus.from_mapping(pstatus_raw))
    win_dir = writer.bundle_dir / "profiles" / "profile-example" / "window-0001"
    (win_dir / "staging").symlink_to(outside_profile)

    with pytest.raises(ObservabilityPathError):
        inspect_observability_bundle(writer.bundle_dir)


def test_bounded_truncation_preserves_schema_validity_and_bounds_provenance_and_row_keys(
    tmp_path: Path,
) -> None:
    # Sub-case A: leading-whitespace diagnostic truncated at max_text_chars=3 must remain valid.
    sink_a = RunArtifactSink(max_text_chars=3, max_list_items=4)
    writer_a = ObservabilityBundleWriter(
        _make_manager(tmp_path, run_id="bounded-text-small"),
        INVOCATION_ID,
        sink=sink_a,
    )
    writer_a.write_open_manifest(_open_manifest())

    status_raw = _appendix_a_status()
    status_raw["diagnostics"] = ["   timeout reading nvidia-smi"]
    status_path = writer_a.write_status(ObservabilityStatus.from_mapping(status_raw))
    parsed_status = ObservabilityStatus.from_mapping(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert parsed_status.diagnostics is not None
    assert len(parsed_status.diagnostics) == 1
    assert parsed_status.diagnostics[0].strip() != ""
    assert len(parsed_status.diagnostics[0]) <= 3
    assert "text_truncated" in parsed_status.truncation_reasons

    pstatus_raw = _appendix_d_profile_status()
    pstatus_raw["activity_diagnostics"] = [
        "   requested device activity not accepted"
    ]
    pstatus_path = writer_a.write_profile_status(
        ObservabilityProfileStatus.from_mapping(pstatus_raw)
    )
    parsed_pstatus = ObservabilityProfileStatus.from_mapping(
        json.loads(pstatus_path.read_text(encoding="utf-8"))
    )
    assert len(parsed_pstatus.activity_diagnostics) == 1
    assert parsed_pstatus.activity_diagnostics[0].strip() != ""
    assert len(parsed_pstatus.activity_diagnostics[0]) <= 3
    assert parsed_pstatus.truncation_reasons is not None
    assert "text_truncated" in parsed_pstatus.truncation_reasons

    # Sub-case B: profile summary must bound source_provenance and preserve distinct row keys.
    sink_b = RunArtifactSink(max_text_chars=16, max_list_items=4)
    writer_b = ObservabilityBundleWriter(
        _make_manager(tmp_path, run_id="bounded-summary-prov"),
        INVOCATION_ID,
        sink=sink_b,
    )
    writer_b.write_open_manifest(_open_manifest())
    summary = ObservabilityProfileSummary(
        invocation_id=INVOCATION_ID,
        observer_id="profile-example",
        window_id="window-0001",
        exporter="TorchOperatorSummary",
        max_rows=10,
        total_rows=1,
        rows=({"self_cuda_time_total_us": 10.0, "self_cuda_time_total_pct": 0.5},),
        source_provenance={"long_note": "z" * 200, "items": list(range(20))},
    )
    summary_path = writer_b.write_profile_summary(summary)
    parsed_summary = ObservabilityProfileSummary.from_mapping(
        json.loads(summary_path.read_text(encoding="utf-8"))
    )
    assert len(parsed_summary.source_provenance["long_note"]) <= 16
    assert len(parsed_summary.source_provenance["items"]) <= 4
    assert len(parsed_summary.rows[0]) == 2


def test_first_sample_partial_append_error_is_honest_and_tracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _make_manager(tmp_path, run_id="partial-append-first")
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)
    writer.write_open_manifest(_open_manifest())
    writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    writer.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )

    def _torn_append(path: Path, record: str) -> None:
        with path.open("ab") as handle:
            handle.write(b'{"partial":"torn')
        raise PartialAppendError("short write")

    monkeypatch.setattr(obs_mod, "append_record", _torn_append)
    sample0 = ObservabilitySample.from_mapping(_appendix_a_samples()[0])
    with pytest.raises(PartialAppendError, match="short write"):
        writer.append_sample(sample0)

    status = _appendix_a_status()
    status["lifecycle_state"] = "unsafe"
    status["data_completeness"] = "unknown"
    status["counts"] = {"attempted": 1, "written": 0, "dropped": 0, "failed": 1}
    writer.write_status(ObservabilityStatus.from_mapping(status))

    unsafe_raw = _appendix_a_manifest()
    unsafe_raw["lifecycle_state"] = "unsafe"
    unsafe_raw["data_completeness"] = "unknown"
    unsafe_raw["artifacts"] = []
    closed = writer.finalize_manifest(ObservabilityManifest.from_mapping(unsafe_raw))
    assert "resources/resources-example.jsonl" in {
        ref.path for ref in closed.artifacts
    }


def test_torn_tail_non_utf8_and_pre_append_repair_newline_byte_cap(
    tmp_path: Path,
) -> None:
    # 1. Truncated non-UTF-8 torn tail is isolated cleanly without mutating disk.
    manager = _make_manager(tmp_path, run_id="torn-tail-non-utf8")
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)
    writer.write_open_manifest(_open_manifest())
    writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    writer.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    samples = [ObservabilitySample.from_mapping(s) for s in _appendix_a_samples()]
    assert writer.append_sample(samples[0]).written is True

    jsonl_path = writer.bundle_dir / "resources" / "resources-example.jsonl"
    with jsonl_path.open("ab") as handle:
        handle.write(b'{"schema":"tpen.run-observability/v1","bad":"\xc3')

    before = _tree_snapshot(writer.bundle_dir)
    bundle = inspect_observability_bundle(writer.bundle_dir)
    assert _tree_snapshot(writer.bundle_dir) == before
    assert len(bundle.samples["resources-example"]) == 1
    assert len(bundle.torn_tails) == 1
    assert bundle.torn_tails[0].code == "torn_tail"
    assert bundle.torn_tails[0].line_number == 2
    assert bundle.interior_errors == ()

    # 2. Pre-append byte-cap accounts for the 1-byte repair newline (prefix_bytes=1).
    line_len = len(
        json.dumps(
            samples[0].to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("ascii")
    )
    unterminated_prefix = b'{"torn":"bytes"'
    assert len(unterminated_prefix) == 15

    # Tight cap (15 + 1 + line_len): missing 1 byte for trailing newline -> refused.
    tight_sink = RunArtifactSink(max_sample_bytes=15 + 1 + line_len)
    tight_writer = ObservabilityBundleWriter(
        _make_manager(tmp_path, run_id="repair-cap-tight"),
        INVOCATION_ID,
        sink=tight_sink,
    )
    tight_writer.write_open_manifest(_open_manifest())
    tight_writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    tight_writer.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    tight_jsonl = tight_writer.bundle_dir / "resources" / "resources-example.jsonl"
    tight_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tight_jsonl.write_bytes(unterminated_prefix)
    assert ends_without_newline(tight_jsonl) is True

    res_tight = tight_writer.append_sample(samples[0])
    assert res_tight.written is False
    assert res_tight.capped is True
    assert len(tight_jsonl.read_bytes()) == 15

    # Exact cap (15 + 1 + line_len + 1): succeeds and terminates with newline.
    exact_sink = RunArtifactSink(max_sample_bytes=15 + 1 + line_len + 1)
    exact_writer = ObservabilityBundleWriter(
        _make_manager(tmp_path, run_id="repair-cap-exact"),
        INVOCATION_ID,
        sink=exact_sink,
    )
    exact_writer.write_open_manifest(_open_manifest())
    exact_writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    exact_writer.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    exact_jsonl = exact_writer.bundle_dir / "resources" / "resources-example.jsonl"
    exact_jsonl.parent.mkdir(parents=True, exist_ok=True)
    exact_jsonl.write_bytes(unterminated_prefix)
    assert ends_without_newline(exact_jsonl) is True

    res_exact = exact_writer.append_sample(samples[0])
    assert res_exact.written is True
    assert res_exact.capped is False
    assert len(exact_jsonl.read_bytes()) == 15 + 1 + line_len + 1
    assert ends_without_newline(exact_jsonl) is False


def test_atomic_write_cleans_temp_file_on_error_and_preserves_prior_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _make_manager(tmp_path, run_id="atomic-write-error")
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)
    manifest_path = writer.write_open_manifest(_open_manifest())
    writer.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    writer.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    writer.write_status(ObservabilityStatus.from_mapping(_appendix_a_status()))
    original_bytes = manifest_path.read_bytes()

    def _fail_replace(src: Any, dst: Any) -> None:
        raise OSError("simulated ENOSPC")

    monkeypatch.setattr(obs_mod.os, "replace", _fail_replace)
    with pytest.raises(OSError, match="simulated ENOSPC"):
        writer.finalize_manifest(_closed_manifest_template())

    assert manifest_path.read_bytes() == original_bytes
    assert writer.finalized is False
    assert list(writer.bundle_dir.rglob("*.tmp")) == []


def test_counter_cross_clock_domain_breaks_interval_and_unsafe_lifecycle_inspection(
    tmp_path: Path,
) -> None:
    # Bundle 1: cross-clock-domain counter samples break interval with clock_domain_mismatch.
    manager1 = _make_manager(tmp_path, run_id="cross-clock-domain")
    inv1 = _appendix_a_inventory()
    dom_a = copy.deepcopy(inv1["clock_domains"][0])
    dom_a["clock_domain_id"] = "clock-a"
    dom_b = copy.deepcopy(inv1["clock_domains"][0])
    dom_b["clock_domain_id"] = "clock-b"
    inv1["clock_domains"] = [dom_a, dom_b]

    base_samples = _appendix_a_samples()
    s0 = copy.deepcopy(base_samples[0])
    s0["clock_domain_id"] = "clock-a"
    s1 = copy.deepcopy(base_samples[3])
    s1["clock_domain_id"] = "clock-b"

    status1 = _appendix_a_status()
    status1["coverage"]["clock_domain_id"] = "clock-a"
    status1["counts"] = {"attempted": 2, "written": 2, "dropped": 0, "failed": 0}

    writer1 = ObservabilityBundleWriter(manager1, INVOCATION_ID)
    writer1.write_open_manifest(_open_manifest())
    writer1.write_inventory(ObservabilityInventory.from_mapping(inv1))
    writer1.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    assert writer1.append_sample(ObservabilitySample.from_mapping(s0)).written is True
    assert writer1.append_sample(ObservabilitySample.from_mapping(s1)).written is True
    writer1.write_status(ObservabilityStatus.from_mapping(status1))
    writer1.finalize_manifest(_closed_manifest_template(data_completeness="complete"))

    bundle1 = inspect_observability_bundle(writer1.bundle_dir)
    cpu_user = bundle1.summary.quantity("self-rusage", "cpu_user_s")
    assert cpu_user is not None
    assert cpu_user.valid_interval_count == 0
    assert cpu_user.broken_interval_count == 1
    assert cpu_user.valid_rate_per_second is None
    assert "clock_domain_mismatch" in {diag.code for diag in bundle1.diagnostics}

    # Bundle 2a: unsafe lifecycle with recovered sample -> derived_data_completeness == "partial".
    manager2 = _make_manager(tmp_path, run_id="unsafe-with-samples")
    writer2 = ObservabilityBundleWriter(manager2, INVOCATION_ID)
    writer2.write_open_manifest(_open_manifest())
    writer2.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    writer2.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    assert (
        writer2.append_sample(ObservabilitySample.from_mapping(base_samples[0])).written
        is True
    )
    status2 = _appendix_a_status()
    status2["lifecycle_state"] = "unsafe"
    status2["data_completeness"] = "partial"
    status2["counts"] = {"attempted": 1, "written": 1, "dropped": 0, "failed": 0}
    writer2.write_status(ObservabilityStatus.from_mapping(status2))
    unsafe_manifest_2 = _appendix_a_manifest()
    unsafe_manifest_2["lifecycle_state"] = "unsafe"
    unsafe_manifest_2["data_completeness"] = "partial"
    unsafe_manifest_2["artifacts"] = []
    writer2.finalize_manifest(ObservabilityManifest.from_mapping(unsafe_manifest_2))

    bundle2 = inspect_observability_bundle(writer2.bundle_dir)
    assert bundle2.derived_lifecycle_state == "unsafe"
    assert bundle2.derived_finalization == "unsafe"
    assert bundle2.derived_reason == "unsafe_lifecycle_state"
    assert bundle2.derived_data_completeness == "partial"

    # Bundle 2b: unsafe lifecycle with 0 samples and no profile statuses -> "unknown".
    manager3 = _make_manager(tmp_path, run_id="unsafe-no-data")
    writer3 = ObservabilityBundleWriter(manager3, INVOCATION_ID)
    writer3.write_open_manifest(_open_manifest())
    writer3.write_inventory(
        ObservabilityInventory.from_mapping(_appendix_a_inventory())
    )
    writer3.write_descriptors(
        ObservabilityDescriptorSet.from_mapping(_appendix_a_descriptors())
    )
    status3 = _appendix_a_status()
    status3["lifecycle_state"] = "unsafe"
    status3["data_completeness"] = "unknown"
    status3["counts"] = {"attempted": 0, "written": 0, "dropped": 0, "failed": 0}
    writer3.write_status(ObservabilityStatus.from_mapping(status3))
    unsafe_manifest_3 = _appendix_a_manifest()
    unsafe_manifest_3["lifecycle_state"] = "unsafe"
    unsafe_manifest_3["data_completeness"] = "unknown"
    unsafe_manifest_3["artifacts"] = []
    writer3.finalize_manifest(ObservabilityManifest.from_mapping(unsafe_manifest_3))

    bundle3 = inspect_observability_bundle(writer3.bundle_dir)
    assert bundle3.derived_lifecycle_state == "unsafe"
    assert bundle3.derived_finalization == "unsafe"
    assert bundle3.derived_reason == "unsafe_lifecycle_state"
    assert bundle3.derived_data_completeness == "unknown"
