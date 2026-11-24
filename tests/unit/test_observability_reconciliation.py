"""Unit and integration tests for O7 observability recovery boundaries and reconciliation."""

from __future__ import annotations

import ast
import copy
from datetime import UTC
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

from omegaconf import OmegaConf
import pytest
from typeguard import TypeCheckError, suppress_type_checks

import tpen.inspection as inspection_mod
import tpen.observability_artifacts as obs_mod
import tpen.observability_reconciliation as rec_mod
from tpen.accelerator import (
    AcceleratorIdentity,
    AcceleratorKind,
    AllocatorCurrentUsage,
    AllocatorUsage,
    DeviceEventTimer,
    NvidiaDeviceCommandResult,
    NvidiaSmiDeviceProbe,
    TorchAllocatorCurrentProbe,
    TorchAllocatorPeakProbe,
)
from tpen.artifacts import (
    ArtifactManager,
    RunClock,
    RunContext,
    RunMetadata,
)
from tpen.callback.resource_samples import (
    AllocatorSource,
    NvidiaDeviceSource,
    RUsageSource,
    ResourceSamples,
    _ResourceSamplesHandle,
    _ResourceSamplesScope,
    _ResourceSamplesSession,
)
from tpen.callback.resource_usage import ResourceUsage
from tpen.checkpoint.profiles import (
    NATIVE_RECOVERY_PROFILE,
    OBSERVATIONAL_CALLBACK_TARGETS,
    SUPPORTED_CALLBACK_TARGETS,
    check_profile,
)
from tpen.checkpoint.restore import restore_checkpoint
from tpen.checkpoint.save import save_checkpoint
from tpen.execution.native_profile import (
    UnsupportedMutableStateError,
    validate_native_recovery_profile,
)
from tpen.inspection import (
    EVAL_RUN_EVIDENCE_SCHEMA,
    LIFECYCLE_MARKER_FILENAME,
    OBSERVABILITY_BATCH_STATUSES,
    OBSERVABILITY_MARKER_STATES,
    OBSERVABILITY_OPERATOR_SUMMARY_SCHEMA,
    OBSERVABILITY_RECONCILIATION_SCHEMA,
    OBSERVABILITY_RECONCILIATION_STATES,
    BoundedOperatorSummary,
    ObservabilityLifecycleMarker,
    OperatorSummaryBounds,
    ReconciledObservabilityBundle,
    ReconciledRunObservability,
    build_bounded_operator_summary,
    inspect_eval_run,
    inspect_run,
    reconcile_observability_bundle,
    reconcile_run_observability,
    render_operator_summary,
)
from tpen.observability_artifacts import (
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
    parse_observability_record,
)
from tpen.process_resources import (
    LinuxCgroupV2Probe,
    LinuxProcessMemoryProbe,
    LinuxProcessTreeProbe,
    ProcessRUsageProbe,
    ProcessResourceBaseline,
)
from tpen.run_events import RunCompleted, RunFailed

REPO_ROOT = Path(__file__).resolve().parents[2]
INVOCATION_ID = "00000000-0000-4000-8000-000000000701"
INVOCATION_ID_2 = "00000000-0000-4000-8000-000000000702"
CONFIG_DIGEST = "151ea9c424d3b6c92294acdd31433cf663c6223e1560daf113678cc2e71a2c2f"


# ---------------------------------------------------------------------------
# Helpers & deterministic fixtures
# ---------------------------------------------------------------------------


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _make_manager(tmp_path: Path, run_id: str = "o7-run") -> ArtifactManager:
    return ArtifactManager(
        root=tmp_path / "outputs",
        experiment="obs-o7",
        sector="unit",
        run_id=run_id,
        layout="flat",
    )


def _make_context(
    tmp_path: Path,
    *,
    run_id: str = "o7-live-run",
    invocation_id: str | None = None,
    device: str = "cpu",
) -> RunContext:
    manager = _make_manager(tmp_path, run_id=run_id)
    metadata = RunMetadata(
        run_id=run_id,
        run_name=run_id,
        timestamp="2026-11-24T12:00:00+00:00",
        timezone="UTC",
        git_branch="test-branch",
        dirty_worktree=False,
        command="pytest",
        config_path="o7.yaml",
        resolved_config_path=str(manager.path("resolved_config.yaml")),
        run_dir=str(manager.run_dir),
        device=device,
        dtype="float64",
        git_commit="abc123def456",
    )
    cfg = OmegaConf.create(
        {
            "experiment": {"name": "obs-o7", "sector": "unit", "run_name": run_id},
            "run": {"root": str(tmp_path / "outputs"), "run_id": run_id},
            "runtime": {"seed": 7, "device": device, "dtype": "float64"},
        }
    )
    kwargs: dict[str, Any] = {
        "cfg": cfg,
        "source_cfg": OmegaConf.create({}),
        "metadata": metadata,
        "artifact_manager": manager,
        "clock": RunClock(timezone="UTC", tzinfo=UTC),
        "callbacks": [],
        "loggers": [],
    }
    if invocation_id is not None:
        kwargs["invocation_id"] = invocation_id
    return RunContext(**kwargs)


def _base_manifest(
    *,
    invocation_id: str = INVOCATION_ID,
    lifecycle_state: str = "open",
    data_completeness: str = "unknown",
    observers: list[dict[str, Any]] | None = None,
    executor_context: Any = "not_supplied",
    scheduler_accounting: Any = "not_supplied",
) -> ObservabilityManifest:
    obs_list = (
        observers
        if observers is not None
        else [
            {
                "observer_id": "resources-main",
                "callback_position": 0,
                "callback_type": "tpen.callback.resource_samples.ResourceSamples",
            }
        ]
    )
    return ObservabilityManifest(
        invocation_id=invocation_id,
        lifecycle_state=lifecycle_state,
        data_completeness=data_completeness,
        run_id="o7-run",
        producer={
            "name": "tpen.callback.resource_samples.ResourceSamples",
            "version": "1",
            "source_commit": {"status": "ok", "value": "deadbeef1234"},
        },
        configuration_digest=CONFIG_DIGEST,
        inventory_id="inv-o7",
        descriptor_set_id="desc-o7",
        observers=tuple(obs_list),
        limits={"interval_seconds": 10, "raw_sample_bytes": 67108864, "status_bytes": 262144},
        executor_context=executor_context,
        scheduler_accounting=scheduler_accounting,
        artifacts=(),
    )


def _base_inventory(invocation_id: str = INVOCATION_ID) -> ObservabilityInventory:
    return ObservabilityInventory(
        invocation_id=invocation_id,
        inventory_id="inv-o7",
        host={"host_id": "node-o7", "boot_id": "boot-o7", "platform": "linux"},
        process={"kind": "process", "host_id": "node-o7", "pid": 4242, "start_ticks": 100},
        clock_domains=(
            {
                "clock_domain_id": "node-o7-monotonic-ns",
                "kind": "host_monotonic_ns",
                "host_id": "node-o7",
                "anchor_monotonic_ns": 1_000_000_000_000,
                "anchor_read_bracket_ns": [999_999_990_000, 1_000_000_010_000],
                "anchor_utc": "2026-11-24T12:00:00+00:00",
                "run_timezone": "UTC",
            },
        ),
        requested={"device": "cuda:0", "dtype": "float64"},
        observed={
            "device-binding-0": {
                "runtime_device": "cuda:0",
                "physical_uuid": "GPU-11111111-2222-3333-4444-555555555555",
                "mig": {"status": "unavailable", "reason_code": "not_applicable"},
            },
            "software": {"python": "3.12"},
        },
        sources={
            "self-rusage": {"execution": "in_process_thread", "capability": "available"},
            "linux-memory": {"execution": "in_process_thread", "capability": "available"},
            "torch-allocator": {
                "execution": "in_process_cooperative",
                "capability": "available",
                "reset_epoch_status": "reset_at_activation",
            },
            "nvidia-smi-device": {
                "execution": "supervised_subprocess",
                "capability": "available",
                "shared_device_not_exclusive": True,
            },
            "linux-cgroup-v2": {
                "execution": "in_process_thread",
                "capability": "available",
                "shared_scope_not_exclusive": True,
            },
        },
    )


def _base_descriptors(invocation_id: str = INVOCATION_ID) -> ObservabilityDescriptorSet:
    return ObservabilityDescriptorSet(
        invocation_id=invocation_id,
        descriptor_set_id="desc-o7",
        quantities={
            "cpu_user_s": QuantityDescriptor(
                source_id="self-rusage",
                unit="s",
                statistic="counter",
                subject_kind="process",
                window="process_lifetime",
                native_name="ru_utime",
                native_unit="s",
                conversion="identity",
            ),
            "cpu_system_s": QuantityDescriptor(
                source_id="self-rusage",
                unit="s",
                statistic="counter",
                subject_kind="process",
                window="process_lifetime",
                native_name="ru_stime",
                native_unit="s",
                conversion="identity",
            ),
            "rss_high_water_b": QuantityDescriptor(
                source_id="self-rusage",
                unit="B",
                statistic="high_water",
                subject_kind="process",
                window="process_lifetime",
                native_name="ru_maxrss",
                native_unit="KiB",
                conversion="multiply_by_1024",
            ),
            "rss_current_b": QuantityDescriptor(
                source_id="linux-memory",
                unit="B",
                statistic="gauge",
                subject_kind="process",
                window="read_interval",
                native_name="VmRSS",
                native_unit="KiB",
                conversion="multiply_by_1024",
                quality=("approximate_rss",),
            ),
            "allocator.current.allocated_bytes": QuantityDescriptor(
                source_id="torch-allocator",
                unit="B",
                statistic="gauge",
                subject_kind="process_allocator",
                window="read_interval",
                native_name="memory_allocated",
                native_unit="B",
                conversion="identity",
            ),
            "allocator.peak.allocated_bytes": QuantityDescriptor(
                source_id="torch-allocator",
                unit="B",
                statistic="high_water",
                subject_kind="process_allocator",
                window="allocator_reset_window",
                native_name="max_memory_allocated",
                native_unit="B",
                conversion="identity",
            ),
            "device_memory_used_b": QuantityDescriptor(
                source_id="nvidia-smi-device",
                unit="B",
                statistic="gauge",
                subject_kind="physical_gpu",
                window="read_interval",
                native_name="memory.used",
                native_unit="MiB",
                conversion="multiply_by_1048576",
            ),
            "device_memory_total_b": QuantityDescriptor(
                source_id="nvidia-smi-device",
                unit="B",
                statistic="capacity",
                subject_kind="physical_gpu",
                window="inventory_or_read_interval",
                native_name="memory.total",
                native_unit="MiB",
                conversion="multiply_by_1048576",
            ),
            "gpu_busy_fraction": QuantityDescriptor(
                source_id="nvidia-smi-device",
                unit="1",
                statistic="vendor_window_mean",
                subject_kind="physical_gpu",
                window="vendor_recent_window_unknown_bounds",
                native_name="utilization.gpu",
                native_unit="percent",
                conversion="divide_by_100",
            ),
            "cgroup.memory.peak_bytes": QuantityDescriptor(
                source_id="linux-cgroup-v2",
                unit="B",
                statistic="high_water",
                subject_kind="cgroup",
                window="cgroup_lifetime",
                native_name="memory.peak",
                native_unit="B",
                conversion="identity",
            ),
        },
    )


def _sample_packets(
    *,
    invocation_id: str = INVOCATION_ID,
    observer_id: str = "resources-main",
    busy_unavailable: bool = False,
) -> list[ObservabilitySample]:
    proc_subj = {"kind": "process", "host_id": "node-o7", "pid": 4242, "start_ticks": 100}
    alloc_subj = {
        "kind": "process_allocator",
        "configured_device": "cuda:0",
        "backend": "cuda",
        "device_index": 0,
        "reset_epoch_status": "reset_at_activation",
    }
    gpu_subj = {
        "kind": "physical_gpu",
        "physical_uuid": "GPU-11111111-2222-3333-4444-555555555555",
        "mig_mode": "enabled" if busy_unavailable else "disabled",
        "mig_ambiguous": busy_unavailable,
        "shared_device_not_exclusive": True,
    }
    cg_subj = {
        "kind": "cgroup",
        "cgroup_version": 2,
        "cgroup_path": "/slurm/job_1",
        "shared_scope_not_exclusive": True,
        "includes_other_tasks": True,
    }
    t0 = 1_001_000_000_000
    t1 = 1_011_000_000_000
    busy_r0 = (
        ObservabilityReading(
            status="unavailable",
            reason_code="mig_subdevice_ambiguous",
            detail="MIG enabled on GPU",
        )
        if busy_unavailable
        else ObservabilityReading(status="ok", value=0.6)
    )
    busy_r1 = (
        ObservabilityReading(
            status="unavailable",
            reason_code="mig_subdevice_ambiguous",
            detail="MIG enabled on GPU",
        )
        if busy_unavailable
        else ObservabilityReading(status="ok", value=0.8)
    )
    return [
        ObservabilitySample(
            invocation_id=invocation_id,
            observer_id=observer_id,
            source_id="self-rusage",
            descriptor_set_id="desc-o7",
            sequence=0,
            clock_domain_id="node-o7-monotonic-ns",
            subject=proc_subj,
            read_start_ns=t0,
            read_end_ns=t0,
            readings={
                "cpu_user_s": ObservabilityReading(status="ok", value=10.0),
                "cpu_system_s": ObservabilityReading(status="ok", value=1.0),
                "rss_high_water_b": ObservabilityReading(status="ok", value=500_000_000),
            },
        ),
        ObservabilitySample(
            invocation_id=invocation_id,
            observer_id=observer_id,
            source_id="self-rusage",
            descriptor_set_id="desc-o7",
            sequence=1,
            clock_domain_id="node-o7-monotonic-ns",
            subject=proc_subj,
            read_start_ns=t1,
            read_end_ns=t1,
            readings={
                "cpu_user_s": ObservabilityReading(status="ok", value=22.0),
                "cpu_system_s": ObservabilityReading(status="ok", value=3.0),
                "rss_high_water_b": ObservabilityReading(status="ok", value=600_000_000),
            },
        ),
        ObservabilitySample(
            invocation_id=invocation_id,
            observer_id=observer_id,
            source_id="linux-memory",
            descriptor_set_id="desc-o7",
            sequence=0,
            clock_domain_id="node-o7-monotonic-ns",
            subject=proc_subj,
            read_start_ns=t0,
            read_end_ns=t0 + 1000,
            readings={"rss_current_b": ObservabilityReading(status="ok", value=300_000_000)},
        ),
        ObservabilitySample(
            invocation_id=invocation_id,
            observer_id=observer_id,
            source_id="torch-allocator",
            descriptor_set_id="desc-o7",
            sequence=0,
            clock_domain_id="node-o7-monotonic-ns",
            subject=alloc_subj,
            read_start_ns=t0,
            read_end_ns=t0 + 2000,
            readings={
                "allocator.current.allocated_bytes": ObservabilityReading(status="ok", value=1024),
                "allocator.peak.allocated_bytes": ObservabilityReading(status="ok", value=4096),
            },
        ),
        ObservabilitySample(
            invocation_id=invocation_id,
            observer_id=observer_id,
            source_id="nvidia-smi-device",
            descriptor_set_id="desc-o7",
            sequence=0,
            clock_domain_id="node-o7-monotonic-ns",
            subject=gpu_subj,
            read_start_ns=t0,
            read_end_ns=t0 + 5000,
            readings={
                "device_memory_used_b": ObservabilityReading(status="ok", value=1_048_576),
                "device_memory_total_b": ObservabilityReading(status="ok", value=85_899_345_920),
                "gpu_busy_fraction": busy_r0,
            },
        ),
        ObservabilitySample(
            invocation_id=invocation_id,
            observer_id=observer_id,
            source_id="nvidia-smi-device",
            descriptor_set_id="desc-o7",
            sequence=1,
            clock_domain_id="node-o7-monotonic-ns",
            subject=gpu_subj,
            read_start_ns=t1,
            read_end_ns=t1 + 5000,
            readings={
                "device_memory_used_b": ObservabilityReading(status="ok", value=2_097_152),
                "device_memory_total_b": ObservabilityReading(status="ok", value=85_899_345_920),
                "gpu_busy_fraction": busy_r1,
            },
        ),
        ObservabilitySample(
            invocation_id=invocation_id,
            observer_id=observer_id,
            source_id="linux-cgroup-v2",
            descriptor_set_id="desc-o7",
            sequence=0,
            clock_domain_id="node-o7-monotonic-ns",
            subject=cg_subj,
            read_start_ns=t0,
            read_end_ns=t0 + 3000,
            readings={
                "cgroup.memory.peak_bytes": ObservabilityReading(status="ok", value=900_000_000),
            },
        ),
    ]


def _make_status(
    *,
    invocation_id: str = INVOCATION_ID,
    observer_id: str = "resources-main",
    lifecycle_state: str = "closed",
    data_completeness: str = "complete",
    counts: dict[str, int] | None = None,
    application_outcome: str = "completed",
    invocation_outcome: str = "completed",
    writer: dict[str, Any] | None = None,
    helper_cleanup: dict[str, Any] | None = None,
    truncation_reasons: tuple[str, ...] = (),
) -> ObservabilityStatus:
    return ObservabilityStatus(
        invocation_id=invocation_id,
        observer_id=observer_id,
        lifecycle_state=lifecycle_state,
        data_completeness=data_completeness,
        source_status={
            "self-rusage": "available",
            "linux-memory": "available",
            "torch-allocator": "available",
            "nvidia-smi-device": "available",
            "linux-cgroup-v2": "available",
        },
        coverage={
            "activation_ns": 1_000_990_000_000,
            "acquisition_end_ns": 1_011_200_000_000,
            "clock_domain_id": "node-o7-monotonic-ns",
            "requested_interval_seconds": 10,
            "missed_deadlines": 0,
        },
        counts=(
            counts
            if counts is not None
            else {"attempted": 7, "written": 7, "dropped": 0, "failed": 0}
        ),
        truncation_reasons=truncation_reasons,
        writer=(
            writer
            if writer is not None
            else {"state": "closed", "flush_confirmed": True, "fsync_confirmed": False}
        ),
        helper_cleanup=(
            helper_cleanup
            if helper_cleanup is not None
            else {"state": "confirmed_stopped"}
        ),
        application_outcome=application_outcome,
        invocation_outcome=invocation_outcome,
        export_status="not_requested",
    )


def _write_clean_bundle(
    tmp_path: Path,
    *,
    run_id: str = "clean-run",
    invocation_id: str = INVOCATION_ID,
    busy_unavailable: bool = False,
    counts: dict[str, int] | None = None,
    data_completeness: str = "complete",
    application_outcome: str = "completed",
    invocation_outcome: str = "completed",
    lifecycle_state: str = "closed",
    writer_info: dict[str, Any] | None = None,
    helper_cleanup: dict[str, Any] | None = None,
    truncation_reasons: tuple[str, ...] = (),
) -> tuple[ArtifactManager, ObservabilityBundleWriter]:
    manager = _make_manager(tmp_path, run_id=run_id)
    writer = ObservabilityBundleWriter(manager, invocation_id)
    writer.write_open_manifest(_base_manifest(invocation_id=invocation_id))
    writer.write_inventory(_base_inventory(invocation_id=invocation_id))
    writer.write_descriptors(_base_descriptors(invocation_id=invocation_id))
    for pkt in _sample_packets(
        invocation_id=invocation_id, busy_unavailable=busy_unavailable
    ):
        assert writer.append_sample(pkt).written is True
    writer.write_status(
        _make_status(
            invocation_id=invocation_id,
            lifecycle_state=lifecycle_state,
            data_completeness=data_completeness,
            counts=counts,
            application_outcome=application_outcome,
            invocation_outcome=invocation_outcome,
            writer=writer_info,
            helper_cleanup=helper_cleanup,
            truncation_reasons=truncation_reasons,
        )
    )
    writer.finalize_manifest(
        _base_manifest(
            invocation_id=invocation_id,
            lifecycle_state=lifecycle_state,
            data_completeness=data_completeness,
        )
    )
    return manager, writer


# ---------------------------------------------------------------------------
# 1. Torch-free, psutil-free, and core import boundary tests
# ---------------------------------------------------------------------------


def test_o7_reconciliation_module_is_torch_psutil_experiments_executor_free(
    tmp_path: Path,
) -> None:
    """Importing O7 modules must never pull forbidden packages.

    ``tpen.inspection`` is a public compatibility facade whose pre-O7
    trajectory-record import eagerly loads the evaluation bundle and Torch.
    O7's contract is therefore limited to the new reconciliation/artifact and
    resource-sample modules; changing the existing inspection import boundary
    belongs to the evaluation owner.
    """

    source = (REPO_ROOT / "tpen" / "observability_reconciliation.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    forbidden = {"torch", "psutil", "experiments", "tpen_executor"}
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in forbidden:
                    found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in forbidden:
                found.append(node.module)
    assert found == []

    # Dynamic subprocess test with poison traps for torch and psutil.
    (tmp_path / "torch.py").write_text('raise AssertionError("torch imported")\n')
    (tmp_path / "psutil.py").write_text('raise AssertionError("psutil imported")\n')
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(REPO_ROOT)])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, tpen.observability_reconciliation, "
                "tpen.observability_artifacts, "
                "tpen.callback.resource_samples; "
                "assert 'torch' not in sys.modules; "
                "assert 'psutil' not in sys.modules; "
                "print('OK')"
            ),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.strip() == "OK"


# ---------------------------------------------------------------------------
# 2. Additive open/finalized lifecycle marker contract & crash boundaries
# ---------------------------------------------------------------------------


def test_lifecycle_marker_open_finalized_roundtrip_and_tamper_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _make_manager(tmp_path, run_id="marker-lifecycle")
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)

    # Open manifest creates open lifecycle-marker.json linked to manifest sha256.
    writer.write_open_manifest(_base_manifest())
    marker_path = writer.bundle_dir / LIFECYCLE_MARKER_FILENAME
    assert marker_path.is_file()
    raw_open_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    parsed_open = parse_observability_record(raw_open_marker)
    assert isinstance(parsed_open, ObservabilityLifecycleMarker)
    assert parsed_open.state == "open"
    assert parsed_open.manifest_lifecycle_state == "open"
    assert parsed_open.to_dict() == raw_open_marker

    # Manifest artifacts must refuse referencing lifecycle-marker.json recursively.
    with pytest.raises(ObservabilityPathError, match="lifecycle-marker.json"):
        ObservabilityArtifactRef(path=LIFECYCLE_MARKER_FILENAME)

    # Marker state must agree internally with manifest_lifecycle_state.
    with pytest.raises(ObservabilitySchemaError, match="requires manifest_lifecycle_state"):
        ObservabilityLifecycleMarker(
            invocation_id=INVOCATION_ID,
            state="finalized",
            manifest_lifecycle_state="open",
            manifest_data_completeness="unknown",
            manifest_sha256="a" * 64,
            manifest_bytes=10,
            run_id="r",
            configuration_digest=CONFIG_DIGEST,
            inventory_id="inv",
            descriptor_set_id="desc",
        )

    writer.write_inventory(_base_inventory())
    writer.write_descriptors(_base_descriptors())
    for pkt in _sample_packets():
        assert writer.append_sample(pkt).written is True
    writer.write_status(_make_status())

    # Simulate crash right after atomic manifest.json replace, before marker replace.
    real_replace = obs_mod.os.replace
    replace_calls = {"count": 0}

    def _fail_second_replace(src: Any, dst: Any) -> None:
        replace_calls["count"] += 1
        if replace_calls["count"] == 2:
            raise OSError("simulated crash before marker replace")
        real_replace(src, dst)

    monkeypatch.setattr(obs_mod.os, "replace", _fail_second_replace)
    with pytest.raises(OSError, match="simulated crash before marker replace"):
        writer.finalize_manifest(
            _base_manifest(lifecycle_state="closed", data_completeness="complete")
        )
    monkeypatch.undo()
    assert writer.finalized is False

    # Reconciler detects closed manifest with stale open marker -> open_interrupted!
    snap_before = _tree_snapshot(writer.bundle_dir)
    mid_crash = reconcile_observability_bundle(writer.bundle_dir)
    assert _tree_snapshot(writer.bundle_dir) == snap_before
    assert mid_crash.reconciliation_state == "open_interrupted"
    assert mid_crash.lifecycle_state == "open"
    assert mid_crash.finalization == "unknown"
    assert mid_crash.data_completeness == "partial"
    assert mid_crash.lifecycle_marker.present is True
    assert mid_crash.lifecycle_marker.state == "open"
    assert mid_crash.lifecycle_marker.matches_manifest is False
    assert mid_crash.lifecycle_marker.diagnostic_code == "lifecycle_marker_state_mismatch"

    # Now finish cleanly -> finalized marker matches manifest hash & bytes.
    finalized = writer.finalize_manifest(
        _base_manifest(lifecycle_state="closed", data_completeness="complete")
    )
    assert all(ref.path != LIFECYCLE_MARKER_FILENAME for ref in finalized.artifacts)
    clean = reconcile_observability_bundle(writer.bundle_dir)
    assert clean.reconciliation_state == "clean_final"
    assert clean.lifecycle_marker.present is True
    assert clean.lifecycle_marker.state == "finalized"
    assert clean.lifecycle_marker.matches_manifest is True
    assert clean.lifecycle_marker.manifest_sha256_verified is True

    # Tampering manifest.json after finalization triggers hash mismatch -> failed/unsafe.
    manifest_path = writer.bundle_dir / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(manifest_text + "\n", encoding="utf-8")
    tampered = reconcile_observability_bundle(writer.bundle_dir)
    assert tampered.reconciliation_state == "failed"
    assert tampered.lifecycle_state == "unsafe"
    assert tampered.finalization == "unsafe"
    assert tampered.lifecycle_marker.manifest_sha256_verified is False
    assert tampered.lifecycle_marker.diagnostic_code == "lifecycle_marker_hash_mismatch"

    # Restore valid manifest and test O2 backward compatibility (absent marker).
    manifest_path.write_text(manifest_text, encoding="utf-8")
    marker_path.unlink()
    o2_compat = reconcile_observability_bundle(writer.bundle_dir)
    assert o2_compat.reconciliation_state == "clean_final"
    assert o2_compat.lifecycle_marker.present is False
    assert o2_compat.lifecycle_marker.state == "absent"
    assert o2_compat.lifecycle_marker.matches_manifest is True
    assert o2_compat.lifecycle_marker.manifest_sha256_verified is None

    # Unsupported major schema in lifecycle-marker.json fails loudly.
    bad_marker = dict(raw_open_marker)
    bad_marker["schema"] = "tpen.run-observability/v2"
    marker_path.write_text(json.dumps(bad_marker), encoding="utf-8")
    with pytest.raises(ObservabilitySchemaError, match="unsupported lifecycle_marker schema"):
        reconcile_observability_bundle(writer.bundle_dir)


# ---------------------------------------------------------------------------
# 3. Four required states + incomplete-batch discrepancies + absent/legacy
# ---------------------------------------------------------------------------


def test_four_required_states_and_incomplete_batch_interpretation(
    tmp_path: Path,
) -> None:
    # --- State 1a: clean_final (complete) ---
    _, clean_writer = _write_clean_bundle(tmp_path, run_id="state-clean")
    snap = _tree_snapshot(clean_writer.bundle_dir)
    clean = reconcile_observability_bundle(clean_writer.bundle_dir)
    assert _tree_snapshot(clean_writer.bundle_dir) == snap
    assert clean.reconciliation_state == "clean_final"
    assert clean.lifecycle_state == "closed"
    assert clean.data_completeness == "complete"
    assert clean.finalization == "confirmed"
    obs = clean.observer("resources-main")
    assert obs is not None
    assert obs.batch_status == "complete"
    assert obs.count_mismatch is False
    assert obs.attempted == 7 and obs.written == 7 and obs.read_valid_packets == 7
    assert obs.dropped == 0 and obs.failed == 0 and obs.pending == 0

    # Paired CPU core-equivalents: (12 + 2) / 10 = 1.4 > 1.0, unknown denominator.
    assert len(clean.cpu_core_equivalents) == 1
    cpu_eq = clean.cpu_core_equivalents[0]
    assert math.isclose(cpu_eq.cpu_core_equivalents or 0.0, 1.4)
    assert cpu_eq.capacity_normalized_utilization is None
    assert cpu_eq.capacity_denominator_status == "unknown"

    # Distinct metric checks in bounded summary.
    summary = clean.summary
    rss_gauge = summary.quantity("linux-memory", "rss_current_b")
    rss_hw = summary.quantity("self-rusage", "rss_high_water_b")
    alloc_cur = summary.quantity("torch-allocator", "allocator.current.allocated_bytes")
    alloc_pk = summary.quantity("torch-allocator", "allocator.peak.allocated_bytes")
    gpu_busy = summary.quantity("nvidia-smi-device", "gpu_busy_fraction")
    cg_pk = summary.quantity("linux-cgroup-v2", "cgroup.memory.peak_bytes")
    assert rss_gauge is not None and rss_gauge.sampled_max_is_lower_bound is True
    assert rss_hw is not None and rss_hw.statistic == "high_water" and rss_hw.observed_max == 600_000_000
    assert alloc_cur is not None and alloc_cur.statistic == "gauge"
    assert alloc_pk is not None and alloc_pk.window == "allocator_reset_window"
    assert gpu_busy is not None and gpu_busy.continuous_window_integrated is False
    assert cg_pk is not None and cg_pk.window == "cgroup_lifetime"

    # --- State 1b: Appendix-C style partial clean_final (dropped=1, failed=1, written==read) ---
    _, partial_writer = _write_clean_bundle(
        tmp_path,
        run_id="state-clean-partial",
        counts={"attempted": 9, "written": 7, "dropped": 1, "failed": 1},
        data_completeness="partial",
        truncation_reasons=("queue_full",),
    )
    clean_partial = reconcile_observability_bundle(partial_writer.bundle_dir)
    assert clean_partial.reconciliation_state == "clean_final"
    assert clean_partial.data_completeness == "partial"
    partial_obs = clean_partial.observer("resources-main")
    assert partial_obs is not None
    assert partial_obs.batch_status == "partial"
    assert partial_obs.count_mismatch is False
    assert partial_obs.dropped == 1 and partial_obs.failed == 1

    # --- State 2a: failed run outcome ---
    _, failed_writer = _write_clean_bundle(
        tmp_path,
        run_id="state-failed-outcome",
        application_outcome="failed",
        invocation_outcome="failed",
    )
    failed_bundle = reconcile_observability_bundle(failed_writer.bundle_dir)
    assert failed_bundle.reconciliation_state == "failed"
    assert failed_bundle.assessment_reason == "failed_run_outcome"

    # --- State 2b: unsafe lifecycle / helper timeout ---
    _, unsafe_writer = _write_clean_bundle(
        tmp_path,
        run_id="state-failed-unsafe",
        lifecycle_state="unsafe",
        data_completeness="partial",
        writer_info={"state": "unsafe", "flush_confirmed": False, "fsync_confirmed": False},
        helper_cleanup={"state": "unconfirmed_timeout"},
    )
    unsafe_bundle = reconcile_observability_bundle(unsafe_writer.bundle_dir)
    assert unsafe_bundle.reconciliation_state == "failed"
    assert unsafe_bundle.lifecycle_state == "unsafe"
    assert unsafe_bundle.finalization == "unsafe"
    assert unsafe_bundle.data_completeness == "partial"

    # --- State 2c: incomplete-batch count discrepancy (written=9 vs 7 read packets) ---
    _, mismatch_writer = _write_clean_bundle(
        tmp_path,
        run_id="state-batch-mismatch",
        counts={"attempted": 9, "written": 9, "dropped": 0, "failed": 0},
        data_completeness="complete",
    )
    mismatch_bundle = reconcile_observability_bundle(mismatch_writer.bundle_dir)
    assert mismatch_bundle.reconciliation_state == "failed"
    assert mismatch_bundle.assessment_reason == "batch_count_mismatch"
    assert mismatch_bundle.data_completeness == "partial"
    mismatch_obs = mismatch_bundle.observer("resources-main")
    assert mismatch_obs is not None
    assert mismatch_obs.count_mismatch is True
    assert mismatch_obs.written == 9 and mismatch_obs.read_valid_packets == 7
    assert mismatch_obs.written_vs_read_delta == -2
    assert mismatch_obs.batch_status == "inconsistent"
    assert "batch_count_mismatch" in {d.code for d in mismatch_bundle.diagnostics}

    # --- State 3: open_interrupted with torn tail AND separate interior error ---
    open_mgr = _make_manager(tmp_path, run_id="state-open-interrupted")
    open_writer = ObservabilityBundleWriter(open_mgr, INVOCATION_ID)
    open_writer.write_open_manifest(_base_manifest())
    open_writer.write_inventory(_base_inventory())
    open_writer.write_descriptors(_base_descriptors())
    pkts = _sample_packets()
    assert open_writer.append_sample(pkts[0]).written is True
    jsonl_path = open_writer.bundle_dir / "resources" / "resources-main.jsonl"
    # Inject one corrupt interior JSON line, then another valid packet, then a torn tail.
    with jsonl_path.open("ab") as handle:
        handle.write(b'{"corrupt_interior": true}\n')
    assert open_writer.append_sample(pkts[1]).written is True
    with jsonl_path.open("ab") as handle:
        handle.write(b'{"schema":"tpen.run-observability/v1","truncated":\xc3')

    open_snap = _tree_snapshot(open_writer.bundle_dir)
    open_bundle = reconcile_observability_bundle(open_writer.bundle_dir)
    assert _tree_snapshot(open_writer.bundle_dir) == open_snap
    assert open_bundle.reconciliation_state == "open_interrupted"
    assert open_bundle.lifecycle_state == "open"
    assert open_bundle.finalization == "unknown"
    assert open_bundle.data_completeness == "partial"
    assert len(open_bundle.inspection.torn_tails) == 1
    assert len(open_bundle.inspection.interior_errors) == 1
    open_obs = open_bundle.observer("resources-main")
    assert open_obs is not None
    assert open_obs.has_final_status is False
    assert open_obs.attempted is None and open_obs.written is None
    assert open_obs.dropped is None and open_obs.failed is None and open_obs.pending is None
    assert open_obs.read_valid_packets == 2
    assert open_obs.torn_tail_count == 1 and open_obs.interior_error_count == 1
    assert open_obs.batch_status == "unknown"

    # --- State 4: missing_final_status (closed manifest, packets exist, status absent) ---
    missing_mgr = _make_manager(tmp_path, run_id="state-missing-final-status")
    missing_writer = ObservabilityBundleWriter(missing_mgr, INVOCATION_ID)
    missing_writer.write_open_manifest(_base_manifest())
    missing_writer.write_inventory(_base_inventory())
    missing_writer.write_descriptors(_base_descriptors())
    assert missing_writer.append_sample(pkts[0]).written is True
    assert missing_writer.append_sample(pkts[2]).written is True
    missing_writer.finalize_manifest(
        _base_manifest(lifecycle_state="closed", data_completeness="partial")
    )
    missing_snap = _tree_snapshot(missing_writer.bundle_dir)
    missing_bundle = reconcile_observability_bundle(missing_writer.bundle_dir)
    assert _tree_snapshot(missing_writer.bundle_dir) == missing_snap
    assert missing_bundle.reconciliation_state == "missing_final_status"
    assert missing_bundle.lifecycle_state == "open"
    assert missing_bundle.finalization == "unknown"
    assert missing_bundle.data_completeness == "partial"
    missing_obs = missing_bundle.observer("resources-main")
    assert missing_obs is not None
    assert missing_obs.has_final_status is False
    # Never fabricate 0 for unknown counts!
    assert missing_obs.attempted is None
    assert missing_obs.written is None
    assert missing_obs.dropped is None
    assert missing_obs.failed is None
    assert missing_obs.pending is None
    assert missing_obs.written_vs_read_delta is None
    assert missing_obs.read_valid_packets == 2
    assert missing_obs.batch_status == "unknown"

    # --- Absent / legacy run ---
    legacy_dir = tmp_path / "legacy-run"
    legacy_dir.mkdir()
    legacy_snap = _tree_snapshot(legacy_dir)
    legacy = reconcile_run_observability(legacy_dir)
    assert _tree_snapshot(legacy_dir) == legacy_snap
    assert isinstance(legacy, ReconciledRunObservability)
    assert legacy.status == "absent"
    assert legacy.data_completeness == "unknown"
    assert legacy.invocations == ()
    assert legacy.total_invocations == 0


def test_unavailable_readings_never_invent_zero_while_genuine_zero_preserved(
    tmp_path: Path,
) -> None:
    _, writer = _write_clean_bundle(
        tmp_path,
        run_id="mig-unavailable",
        busy_unavailable=True,
        data_completeness="partial",
    )
    reconciled = reconcile_observability_bundle(writer.bundle_dir)
    busy = reconciled.summary.quantity("nvidia-smi-device", "gpu_busy_fraction")
    assert busy is not None
    assert busy.ok_count == 0
    assert busy.unavailable_count == 2
    assert busy.observation_min is None
    assert busy.observation_max is None
    assert busy.observation_mean is None
    assert busy.continuous_window_integrated is False
    assert "mig_subdevice_ambiguous" in busy.reason_codes
    assert any("MIG" in cav for cav in reconciled.summary.caveats)

    # Genuine zero counter delta and zero GPU busy stay 0.0 (not None).
    zero_mgr = _make_manager(tmp_path, run_id="genuine-zero")
    zero_writer = ObservabilityBundleWriter(zero_mgr, INVOCATION_ID)
    zero_writer.write_open_manifest(_base_manifest())
    zero_writer.write_inventory(_base_inventory())
    zero_writer.write_descriptors(_base_descriptors())
    proc_subj = {"kind": "process", "host_id": "node-o7", "pid": 4242, "start_ticks": 100}
    zero_writer.append_sample(
        ObservabilitySample(
            invocation_id=INVOCATION_ID,
            observer_id="resources-main",
            source_id="self-rusage",
            descriptor_set_id="desc-o7",
            sequence=0,
            clock_domain_id="node-o7-monotonic-ns",
            subject=proc_subj,
            read_start_ns=10_000_000_000,
            read_end_ns=10_000_000_000,
            readings={
                "cpu_user_s": ObservabilityReading(status="ok", value=5.0),
                "cpu_system_s": ObservabilityReading(status="ok", value=1.0),
            },
        )
    )
    zero_writer.append_sample(
        ObservabilitySample(
            invocation_id=INVOCATION_ID,
            observer_id="resources-main",
            source_id="self-rusage",
            descriptor_set_id="desc-o7",
            sequence=1,
            clock_domain_id="node-o7-monotonic-ns",
            subject=proc_subj,
            read_start_ns=20_000_000_000,
            read_end_ns=20_000_000_000,
            readings={
                "cpu_user_s": ObservabilityReading(status="ok", value=5.0),
                "cpu_system_s": ObservabilityReading(status="ok", value=1.0),
            },
        )
    )
    zero_writer.write_status(
        _make_status(counts={"attempted": 2, "written": 2, "dropped": 0, "failed": 0})
    )
    zero_writer.finalize_manifest(
        _base_manifest(lifecycle_state="closed", data_completeness="complete")
    )
    zero_rec = reconcile_observability_bundle(zero_writer.bundle_dir)
    assert len(zero_rec.cpu_core_equivalents) == 1
    assert zero_rec.cpu_core_equivalents[0].cpu_core_equivalents == 0.0


# ---------------------------------------------------------------------------
# 4. Bounded operator summaries, identity linkage, and Typeguard checks
# ---------------------------------------------------------------------------


def test_bounded_operator_summary_enforces_row_text_line_and_invocation_bounds(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path, run_id="bounded-summary")

    # Write two isolated invocations into the same run_dir to test max_invocations too.
    for inv_id in (INVOCATION_ID, INVOCATION_ID_2):
        writer = ObservabilityBundleWriter(manager, inv_id)
        writer.write_open_manifest(
            _base_manifest(
                invocation_id=inv_id,
                observers=[
                    {"observer_id": "resources-main", "callback_position": 0},
                    {"observer_id": "profile-main", "callback_position": 1},
                ],
            )
        )
        writer.write_inventory(_base_inventory(invocation_id=inv_id))
        writer.write_descriptors(_base_descriptors(invocation_id=inv_id))
        for pkt in _sample_packets(invocation_id=inv_id):
            assert writer.append_sample(pkt).written is True
        writer.write_status(_make_status(invocation_id=inv_id))
        writer.write_profile_status(
            ObservabilityProfileStatus(
                invocation_id=inv_id,
                observer_id="profile-main",
                window_id="window-0001",
                window_kind="operation",
                requested_activities=("cpu",),
                accepted_activities=("cpu",),
                observed_event_counts={"cpu": 5},
                clock_domain_id="node-o7-monotonic-ns",
                activation_ns=1_000_000_000_000,
                lifecycle_state="closed",
                data_completeness="complete",
                guarantee_class="cooperative_native_capture",
                exporter="TorchOperatorSummary",
                export_status="complete",
                max_rows=10,
            )
        )
        writer.write_profile_summary(
            ObservabilityProfileSummary(
                invocation_id=inv_id,
                observer_id="profile-main",
                window_id="window-0001",
                exporter="TorchOperatorSummary",
                max_rows=10,
                total_rows=4,
                rows=(
                    {"operator": "aten::very_long_operator_name_alpha_00001", "calls": 10},
                    {"operator": "aten::very_long_operator_name_beta_00002", "calls": 8},
                    {"operator": "aten::very_long_operator_name_gamma_00003", "calls": 4},
                    {"operator": "aten::very_long_operator_name_delta_00004", "calls": 2},
                ),
                truncated=False,
                truncation_reasons=(),
            )
        )
        writer.finalize_manifest(
            _base_manifest(
                invocation_id=inv_id,
                lifecycle_state="closed",
                data_completeness="complete",
                observers=[
                    {"observer_id": "resources-main", "callback_position": 0},
                    {"observer_id": "profile-main", "callback_position": 1},
                ],
            )
        )

    tight = OperatorSummaryBounds(
        max_quantities=3,
        max_observers=1,
        max_diagnostics=2,
        max_caveats=2,
        max_profile_windows=1,
        max_profile_rows=2,
        max_text_chars=24,
        max_rendered_lines=6,
        max_invocations=1,
    )
    run_rec = reconcile_run_observability(manager.run_dir, bounds=tight)
    assert run_rec.status == "present"
    assert run_rec.total_invocations == 2
    assert len(run_rec.invocations) == 1
    assert run_rec.truncated is True
    assert "invocations_truncated" in run_rec.truncation_reasons

    bundle = run_rec.invocations[0]
    summary = bundle.summary
    assert isinstance(summary, BoundedOperatorSummary)
    assert summary.schema == OBSERVABILITY_OPERATOR_SUMMARY_SCHEMA
    assert summary.truncated is True
    assert "quantities_truncated" in summary.truncation_reasons
    assert "observers_truncated" in summary.truncation_reasons
    assert "caveats_truncated" in summary.truncation_reasons
    assert "profile_rows_truncated" in summary.truncation_reasons
    assert "text_truncated" in summary.truncation_reasons

    # Strict collection and text bounds.
    assert len(summary.quantities) == 3 and summary.total_quantities == 10
    assert len(summary.observers) == 1 and summary.total_observers == 2
    assert len(summary.caveats) == 2 and summary.total_caveats > 2
    assert all(len(c) <= 24 for c in summary.caveats)
    assert len(summary.profile_windows) == 1
    assert len(summary.profile_windows[0].rows) == 2
    assert summary.profile_windows[0].total_rows == 4
    for row in summary.profile_windows[0].rows:
        for val in row.values():
            if isinstance(val, str):
                assert len(val) <= 24

    # Complete identity linkage preserved intact (not corrupted by text cap).
    ident = summary.identity
    assert ident.invocation_id == INVOCATION_ID
    assert ident.run_id == "o7-run"
    assert ident.config_hash == CONFIG_DIGEST
    assert ident.git_commit == "deadbeef1234"
    assert ident.inventory_id == "inv-o7"
    assert ident.descriptor_set_id == "desc-o7"
    assert set(ident.observer_ids) == {"resources-main", "profile-main"}
    assert "self-rusage" in ident.source_ids
    assert ident.clock_domains == ("node-o7-monotonic-ns",)
    assert ident.executor_context == "not_supplied"
    assert ident.scheduler_accounting == "not_supplied"
    assert ident.billed_units == "not_supplied"
    assert len(ident.device_bindings) >= 1

    # Re-projecting via build_bounded_operator_summary with wider bounds recovers all rows.
    wide = build_bounded_operator_summary(bundle, bounds=OperatorSummaryBounds())
    assert len(wide.quantities) == wide.total_quantities == 10
    assert len(wide.observers) == wide.total_observers == 2

    # Rendered plain text obeys both line count and per-line character limits.
    rendered = render_operator_summary(summary, max_lines=5, max_line_chars=30)
    lines = rendered.splitlines()
    assert len(lines) == 5
    assert all(len(line) <= 30 for line in lines)
    assert "truncated" in lines[-1]

    # Strict JSON-safe serialization round-trip.
    encoded = json.dumps(run_rec.to_dict(), allow_nan=False)
    assert INVOCATION_ID in encoded


def test_o7_public_api_input_validation_is_typeguard_safe() -> None:
    for bad_val in (0, -1, True, "5"):
        with pytest.raises((TypeError, ValueError, TypeCheckError)):
            OperatorSummaryBounds(max_quantities=bad_val)  # type: ignore[arg-type]

    with suppress_type_checks():
        with pytest.raises(TypeError):
            OperatorSummaryBounds(max_quantities=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            OperatorSummaryBounds(max_quantities=0)
        with pytest.raises(TypeError):
            build_bounded_operator_summary("not-a-bundle")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            render_operator_summary("not-a-summary")  # type: ignore[arg-type]

    assert OBSERVABILITY_RECONCILIATION_SCHEMA == "tpen.run-observability-reconciliation/v1"
    assert OBSERVABILITY_RECONCILIATION_STATES == {
        "clean_final",
        "failed",
        "open_interrupted",
        "missing_final_status",
    }
    assert "inconsistent" in OBSERVABILITY_BATCH_STATUSES
    assert "finalized" in OBSERVABILITY_MARKER_STATES


# ---------------------------------------------------------------------------
# 5. Real producer-to-consumer integration: ResourceSamples -> O7 reconciler
# ---------------------------------------------------------------------------


class _FakeRUsageProbe:
    def __init__(self) -> None:
        self._calls = 0

    def read(self) -> ProcessResourceBaseline:
        self._calls += 1
        return ProcessResourceBaseline(
            user_cpu_seconds=1.0 + 0.5 * self._calls,
            system_cpu_seconds=0.2 + 0.1 * self._calls,
            read_block_operations=self._calls,
            write_block_operations=0,
            voluntary_context_switches=self._calls,
            involuntary_context_switches=0,
            peak_rss_mb=128.0,
        )


class _FakeAllocatorCurrentProbe:
    def __init__(self, device: Any) -> None:
        self._identity = AcceleratorIdentity(
            kind=AcceleratorKind.CUDA,
            index=0,
            uuid="GPU-11111111-2222-3333-4444-555555555555",
        )

    def identity(self) -> AcceleratorIdentity:
        return self._identity

    def available(self) -> bool:
        return True

    def read(self) -> AllocatorCurrentUsage:
        return AllocatorCurrentUsage(
            identity=self._identity,
            allocated_bytes=2048,
            reserved_bytes=4096,
            device_count=1,
        )


class _FakeAllocatorPeakProbe:
    def __init__(self, device: Any) -> None:
        self._identity = AcceleratorIdentity(
            kind=AcceleratorKind.CUDA,
            index=0,
            uuid="GPU-11111111-2222-3333-4444-555555555555",
        )

    def identity(self) -> AcceleratorIdentity:
        return self._identity

    def available(self) -> bool:
        return True

    def reset(self) -> AcceleratorIdentity:
        return self._identity

    def read(self) -> AllocatorUsage:
        return AllocatorUsage(
            identity=self._identity,
            allocated_mb=4.0,
            reserved_mb=8.0,
            device_count=1,
        )


def test_real_resource_samples_producer_to_reconciler_clean_and_failed(
    tmp_path: Path,
) -> None:
    gpu_uuid = "GPU-11111111-2222-3333-4444-555555555555"
    gpu_identity = AcceleratorIdentity(kind=AcceleratorKind.CUDA, index=0, uuid=gpu_uuid)

    def _fake_nvidia_runner(argv: Any, **kwargs: Any) -> NvidiaDeviceCommandResult:
        del argv, kwargs
        return NvidiaDeviceCommandResult(
            returncode=0,
            stdout=f"{gpu_uuid}, 2026/11/24 12:00:00.000, 1024, 40960, 55, Disabled\n",
        )

    nvidia_probe = NvidiaSmiDeviceProbe(
        identity=gpu_identity, command_runner=_fake_nvidia_runner
    )

    # 1. Clean completed run through real ResourceSamples -> O7 reconciler.
    ctx_clean = _make_context(tmp_path, run_id="producer-clean", device="cuda:0")
    cb_clean = ResourceSamples(
        observer_id="obs-live",
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=True, probe=_FakeRUsageProbe()),
        allocator=AllocatorSource(
            enabled=True,
            device="cuda:0",
            reset_peaks=True,
            current_probe_factory=_FakeAllocatorCurrentProbe,
            peak_probe_factory=_FakeAllocatorPeakProbe,
        ),
        nvidia_device=NvidiaDeviceSource(
            enabled=True,
            device="cuda:0",
            probe=nvidia_probe,
        ),
    )
    with cb_clean.run_scope(ctx_clean) as handle:
        assert handle is not None
        handle.checkpoint()
        # Direct typed RunCompleted delivery via callback API.
        from tpen.events import Occurrence

        cb_clean.handle_occurrence(
            Occurrence(event=RunCompleted(status="completed"), count=1),
            ctx_clean,
        )

    run_clean = reconcile_run_observability(ctx_clean.run_dir)
    assert run_clean.status == "present"
    assert len(run_clean.invocations) == 1
    rec_clean = run_clean.invocations[0]
    assert rec_clean.reconciliation_state == "clean_final"
    assert rec_clean.lifecycle_marker.present is True
    assert rec_clean.lifecycle_marker.state == "finalized"
    assert rec_clean.lifecycle_marker.manifest_sha256_verified is True
    assert rec_clean.summary.identity.invocation_id == ctx_clean.invocation_id
    assert rec_clean.summary.identity.git_commit == "abc123def456"
    assert rec_clean.summary.quantity("nvidia-smi-device", "device.busy_fraction") is not None
    assert (
        rec_clean.summary.quantity("nvidia-smi-device", "device.busy_fraction").continuous_window_integrated  # type: ignore[union-attr]
        is False
    )

    # 2. Failed run through real ResourceSamples -> O7 reconciler classifies "failed".
    ctx_fail = _make_context(tmp_path, run_id="producer-failed", device="cpu")
    cb_fail = ResourceSamples(
        observer_id="obs-fail",
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=True, probe=_FakeRUsageProbe()),
    )
    with pytest.raises(RuntimeError, match="boom"):
        with cb_fail.run_scope(ctx_fail):
            from tpen.events import Occurrence

            cb_fail.handle_occurrence(
                Occurrence(
                    event=RunFailed(error_type="RuntimeError", error_message="boom"),
                    count=1,
                ),
                ctx_fail,
            )
            raise RuntimeError("boom")

    rec_fail = reconcile_run_observability(ctx_fail.run_dir).invocations[0]
    assert rec_fail.reconciliation_state == "failed"
    assert rec_fail.lifecycle_marker.present is True
    assert rec_fail.lifecycle_marker.state == "finalized"
    assert rec_fail.lifecycle_marker.manifest_sha256_verified is True
    assert rec_fail.observer("obs-fail") is not None
    assert rec_fail.observer("obs-fail").invocation_outcome == "failed"  # type: ignore[union-attr]


def test_inspect_run_and_inspect_eval_run_schemas_remain_unchanged_with_o7_bundles(
    tmp_path: Path,
) -> None:
    """Coexisting O7 bundles and lifecycle-marker.json never perturb closed RunEvidence."""

    manager, _ = _write_clean_bundle(tmp_path, run_id="eval-with-o7")
    run_dir = manager.run_dir
    (run_dir / "status.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )

    ev = inspect_run(run_dir)
    assert ev.verdict == "failed"
    assert ev.refusal is None
    assert set(ev.to_dict()) == {
        "schema",
        "run_dir",
        "verdict",
        "run_status",
        "checkpoint",
        "refusal",
    }

    eval_ev = inspect_eval_run(run_dir)
    assert eval_ev.schema == EVAL_RUN_EVIDENCE_SCHEMA
    assert eval_ev.verdict == "failed"
    assert eval_ev.refusal is None
    assert list(run_dir.rglob("status.json")) == [run_dir / "status.json"]


# ---------------------------------------------------------------------------
# 6. Non-serialization & checkpoint/recovery isolation boundary tests
# ---------------------------------------------------------------------------


def test_live_observability_objects_refuse_pickle_deepcopy_and_torch_save(
    tmp_path: Path,
) -> None:
    """Live handles, writers, probes, baselines, and sessions cannot be serialized."""

    manager = _make_manager(tmp_path, run_id="pickle-refusal")
    ctx = _make_context(tmp_path, run_id="pickle-refusal-ctx")
    writer = ObservabilityBundleWriter(manager, INVOCATION_ID)
    cb = ResourceSamples(
        observer_id="obs-refuse",
        rusage=RUsageSource(enabled=True, probe=_FakeRUsageProbe()),
    )
    scope = _ResourceSamplesScope(cb, ctx)
    session = _ResourceSamplesSession(cb, ctx)
    handle = _ResourceSamplesHandle(session)

    rusage_probe = ProcessRUsageProbe()
    rusage_baseline = ProcessResourceBaseline(
        user_cpu_seconds=1.0,
        system_cpu_seconds=0.5,
        read_block_operations=0,
        write_block_operations=0,
        voluntary_context_switches=1,
        involuntary_context_switches=0,
        peak_rss_mb=64.0,
    )
    linux_mem_probe = LinuxProcessMemoryProbe()
    linux_tree_probe = LinuxProcessTreeProbe()
    linux_cg_probe = LinuxCgroupV2Probe()
    nvidia_probe = NvidiaSmiDeviceProbe()
    legacy_usage = ResourceUsage()
    event_timer = DeviceEventTimer(_module=object(), _event_factory=object())

    live_objects: list[Any] = [
        writer,
        cb,
        scope,
        session,
        handle,
        rusage_probe,
        rusage_baseline,
        linux_mem_probe,
        linux_tree_probe,
        linux_cg_probe,
        nvidia_probe,
        legacy_usage,
        event_timer,
    ]

    # Add real Torch allocator probes on CPU when torch is installed.
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None:
        live_objects.append(TorchAllocatorPeakProbe("cpu"))
        live_objects.append(TorchAllocatorCurrentProbe("cpu"))

    for obj in live_objects:
        with pytest.raises(TypeError, match="cannot be serialized or checkpointed"):
            obj.__getstate__()
        with pytest.raises(TypeError, match="cannot be serialized or checkpointed"):
            obj.__reduce__()
        for proto in (0, pickle.HIGHEST_PROTOCOL):
            with pytest.raises(TypeError, match="cannot be serialized or checkpointed"):
                pickle.dumps(obj, protocol=proto)
        with pytest.raises(TypeError, match="cannot be serialized or checkpointed"):
            copy.deepcopy(obj)
        if torch is not None:
            with pytest.raises(TypeError, match="cannot be serialized or checkpointed"):
                torch.save(obj, BytesIO())


def test_native_recovery_admission_remains_fail_closed_for_resource_samples() -> None:
    """O7 must never weaken fail-closed train_resume admission in checkpoint/profiles.py."""

    assert "tpen.callback.ResourceSamples" not in OBSERVATIONAL_CALLBACK_TARGETS
    assert "tpen.callback.ResourceSamples" not in SUPPORTED_CALLBACK_TARGETS
    assert "tpen.callback.resource_samples.ResourceSamples" not in SUPPORTED_CALLBACK_TARGETS

    for target in (
        "tpen.callback.ResourceSamples",
        "tpen.callback.resource_samples.ResourceSamples",
    ):
        cfg = {
            "run": {"dir": "/tmp/tpen-o7-profile-test"},
            "runtime": {"device": "cpu", "seed": 7},
            "callbacks": [
                {"_target_": "tpen.callback.Status"},
                {"_target_": target},
            ],
            "loggers": [{"_target_": "tpen.logging.JSONL"}],
        }
        check = check_profile(cfg, NATIVE_RECOVERY_PROFILE)
        assert check.verdict == "refused"
        assert check.refusal is not None
        assert check.refusal["code"] == "unsupported-callback"
        with pytest.raises(UnsupportedMutableStateError):
            validate_native_recovery_profile(cfg)


def test_checkpoint_save_and_restore_cannot_reactivate_observability_and_isolates_invocations(
    tmp_path: Path,
) -> None:
    """Real checkpoint save/restore carries zero observability state and isolates invocations."""

    torch = pytest.importorskip("torch")

    class _ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))

    class _ToyTrainer:
        def __init__(self, next_iteration: int = 0, completed_updates: int = 0) -> None:
            self.next_iteration = next_iteration
            self.completed_updates = completed_updates

        def state_dict(self) -> dict[str, Any]:
            return {
                "next_iteration": self.next_iteration,
                "completed_updates": self.completed_updates,
            }

        def load_state_dict(self, state: dict[str, Any]) -> None:
            self.next_iteration = int(state["next_iteration"])
            self.completed_updates = int(state["completed_updates"])

    class _ToySampler:
        def __init__(self, cursor: int = 0) -> None:
            self.cursor = cursor

        def mcmc_state_dict(self) -> dict[str, Any]:
            return {"cursor": self.cursor}

        def load_mcmc_state_dict(self, state: dict[str, Any], device: Any = None) -> None:
            del device
            self.cursor = int(state["cursor"])

    # --- Invocation 1: active ResourceSamples while saving a train_resume checkpoint ---
    ctx1 = _make_context(tmp_path, run_id="ckpt-isolation-run", device="cpu")
    model1 = _ToyModel()
    opt1 = torch.optim.SGD(model1.parameters(), lr=0.1)
    trainer1 = _ToyTrainer(next_iteration=5, completed_updates=5)
    sampler1 = _ToySampler(cursor=11)
    cb1 = ResourceSamples(
        observer_id="obs-inv1",
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=True, probe=_FakeRUsageProbe()),
    )

    ckpt_root = ctx1.run_dir / "checkpoints"
    with cb1.run_scope(ctx1):
        ckpt_dir = save_checkpoint(
            output_dir=ckpt_root,
            next_iteration=5,
            completed_updates=5,
            model=model1,
            context=ctx1,
            optimizer=opt1,
            trainer=trainer1,
            sampler=sampler1,
        )

    # Confirm zero observability state leaked into checkpoint manifest/trainer files.
    manifest_raw = (ckpt_dir / "manifest.json").read_text(encoding="utf-8")
    assert "observability" not in manifest_raw.lower()
    assert "obs-inv1" not in manifest_raw
    assert ctx1.invocation_id not in manifest_raw

    # --- Invocation 2: restore checkpoint WITHOUT observability -> zero reactivation ---
    ctx2 = _make_context(tmp_path, run_id="ckpt-restored-no-obs", device="cpu")
    assert ctx2.invocation_id != ctx1.invocation_id
    model2 = _ToyModel()
    opt2 = torch.optim.SGD(model2.parameters(), lr=0.1)
    trainer2 = _ToyTrainer(next_iteration=0, completed_updates=0)
    sampler2 = _ToySampler(cursor=0)

    report = restore_checkpoint(
        load={"mode": "train_resume", "path": str(ckpt_dir)},
        model=model2,
        context=ctx2,
        optimizer=opt2,
        trainer=trainer2,
        sampler=sampler2,
    )
    assert report.mode == "train_resume"
    assert report.next_iteration == 5
    assert trainer2.next_iteration == 5 and trainer2.completed_updates == 5
    assert sampler2.cursor == 11
    # No observability sidecar was created or reactivated by restore_checkpoint.
    assert not (ctx2.run_dir / "observability").exists()
    assert reconcile_run_observability(ctx2.run_dir).status == "absent"

    # --- Invocation 3: restore in same run_dir with fresh invocation_id stays isolated ---
    ctx3 = _make_context(tmp_path, run_id="ckpt-isolation-run", device="cpu")
    assert ctx3.invocation_id != ctx1.invocation_id
    model3 = _ToyModel()
    opt3 = torch.optim.SGD(model3.parameters(), lr=0.1)
    cb3 = ResourceSamples(
        observer_id="obs-inv3",
        interval_seconds=60.0,
        rusage=RUsageSource(enabled=True, probe=_FakeRUsageProbe()),
    )
    with cb3.run_scope(ctx3):
        restore_checkpoint(
            load={"mode": "train_resume", "path": str(ckpt_dir)},
            model=model3,
            context=ctx3,
            optimizer=opt3,
            trainer=_ToyTrainer(),
            sampler=_ToySampler(),
        )

    reconciled_multi = reconcile_run_observability(ctx1.run_dir)
    assert reconciled_multi.status == "present"
    assert reconciled_multi.total_invocations == 2
    ids = {inv.invocation_id for inv in reconciled_multi.invocations}
    assert ids == {ctx1.invocation_id, ctx3.invocation_id}
    assert all(inv.reconciliation_state == "clean_final" for inv in reconciled_multi.invocations)
    # Neither invocation's observer leaked into the other invocation.
    by_id = {inv.invocation_id: inv for inv in reconciled_multi.invocations}
    assert by_id[ctx1.invocation_id].summary.identity.observer_ids == ("obs-inv1",)
    assert by_id[ctx3.invocation_id].summary.identity.observer_ids == ("obs-inv3",)
