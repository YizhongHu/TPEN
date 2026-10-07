"""Tests for explicit distributed execution identity and profile artifacts."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest
from typeguard import TypeCheckError, suppress_type_checks

from tpen.accelerator import AcceleratorIdentity, AcceleratorKind, AllocatorUsage
from tpen.distributed import (
    ExecutionTopology,
    ProfileRecord,
    ProfileScope,
    RankLocalJSONLWriter,
    ScalarMetric,
    execution_topology_from_facts,
    project_scalars,
)
from tpen.process_resources import ProcessResourceResult, ResourceUnavailable


def _topology(rank: int | None, *, device: str = "cuda:3") -> ExecutionTopology:
    return ExecutionTopology(
        global_rank=rank,
        global_size=2,
        local_rank=0 if rank == 0 else 1 if rank == 1 else None,
        local_size=2,
        node_rank=0,
        node_size=1,
        host="node-a",
        pid=1000 + (rank or 0),
        device=device,
    )


def _process(*, unavailable: bool = False) -> ProcessResourceResult:
    value = ResourceUnavailable("getrusage failed") if unavailable else 1
    return ProcessResourceResult(
        user_cpu_seconds=value,
        system_cpu_seconds=value,
        read_block_operations=value,
        write_block_operations=value,
        voluntary_context_switches=value,
        involuntary_context_switches=value,
        peak_rss_mb=value,
    )


def test_topology_preserves_explicit_ranks_and_device_without_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOCAL_RANK", "99")
    topology = _topology(1)
    assert topology.global_rank == 1
    assert topology.local_rank == 1
    assert topology.device == "cuda:3"
    assert topology.rank_path_component == "rank-00001"


def test_missing_global_rank_stays_explicit_and_cannot_make_path() -> None:
    topology = _topology(None)
    assert topology.global_rank is None
    with pytest.raises(ValueError, match="global rank is unavailable"):
        _ = topology.rank_path_component


def test_execution_topology_from_facts_normalizes_mapping_and_preserves_type() -> None:
    facts = {
        "global_rank": 1,
        "global_size": 2,
        "local_rank": 1,
        "local_size": 2,
        "node_rank": 0,
        "node_size": 1,
        "host": "node-b",
        "pid": 1001,
        "device": "cuda:3",
        "job_id": "job-1",
        "device_identity": {"kind": "cuda", "index": 3, "uuid": "GPU-3"},
    }

    topology = execution_topology_from_facts(facts)

    assert topology.global_rank == 1
    assert topology.local_rank == 1
    assert topology.host == "node-b"
    assert topology.job_id == "job-1"
    assert topology.device_identity == AcceleratorIdentity(
        AcceleratorKind.CUDA, 3, "GPU-3"
    )
    assert execution_topology_from_facts(topology) is topology


def test_execution_topology_from_facts_rejects_boundary_shape_errors() -> None:
    facts = {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": "node-a",
        "pid": 1000,
        "device": "cpu",
    }

    unsupported = dict(facts, launcher_job_id="not-a-topology-fact")
    with pytest.raises(ValueError, match="unsupported production runner topology"):
        execution_topology_from_facts(unsupported)

    missing = dict(facts)
    missing.pop("host")
    with pytest.raises(ValueError, match="missing required facts"):
        execution_topology_from_facts(missing)

    malformed = dict(facts, device_identity={"kind": "not-an-accelerator"})
    with pytest.raises(ValueError, match="device_identity is malformed"):
        execution_topology_from_facts(malformed)

    with pytest.raises((TypeError, TypeCheckError), match=r"(facts|ExecutionTopology or mapping)"):
        execution_topology_from_facts(object())
    with suppress_type_checks():
        with pytest.raises(TypeError, match="ExecutionTopology or mapping"):
            execution_topology_from_facts(object())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("global_size", 0, "global_size must be positive"),
        ("global_rank", "0", "global_rank must be an int"),
        ("global_rank", True, "global_rank must be an int"),
    ],
)
def test_execution_topology_from_facts_rejects_invalid_values(
    field: str, value: object, message: str
) -> None:
    facts = {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": "node-a",
        "pid": 1000,
        "device": "cpu",
    }
    facts[field] = value

    with pytest.raises(ValueError, match=message):
        execution_topology_from_facts(facts)


@pytest.mark.parametrize(
    ("rank_field", "rank_value"),
    [("global_rank", False), ("global_rank", 0.5), ("global_size", True)],
)
def test_execution_topology_rejects_non_integer_rank_and_size_types(
    rank_field: str, rank_value: object
) -> None:
    values = {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": "node-a",
        "pid": 1000,
        "device": "cuda:3",
    }
    values[rank_field] = rank_value

    with pytest.raises((TypeError, ValueError)):
        ExecutionTopology(**values)


def test_two_ranks_sharing_one_device_write_distinct_jsonl_files(tmp_path) -> None:
    first = _topology(0)
    second = _topology(1)
    def record(topology: ExecutionTopology) -> ProfileRecord:
        return ProfileRecord(ProfileScope.PROCESS, 2.5, topology, process=_process())
    RankLocalJSONLWriter(tmp_path, first).write(record(first))
    RankLocalJSONLWriter(tmp_path, second).write(record(second))

    first_path = tmp_path / "profiles" / "rank-00000" / "resources.jsonl"
    second_path = tmp_path / "profiles" / "rank-00001" / "resources.jsonl"
    assert first_path != second_path
    assert json.loads(first_path.read_text().splitlines()[0])["device"] == "cuda:3"
    assert json.loads(second_path.read_text().splitlines()[0])["global_rank"] == 1


def test_writer_persists_exact_device_identity(tmp_path) -> None:
    base = _topology(0)
    topology = ExecutionTopology(
        global_rank=base.global_rank,
        global_size=base.global_size,
        local_rank=base.local_rank,
        local_size=base.local_size,
        node_rank=base.node_rank,
        node_size=base.node_size,
        host=base.host,
        pid=base.pid,
        device=base.device,
        device_identity=AcceleratorIdentity(AcceleratorKind.CUDA, 3, "GPU-3"),
    )
    assert topology.device_identity is not None
    record = ProfileRecord(
        ProfileScope.DEVICE,
        1.0,
        topology,
        device=AllocatorUsage(
            identity=topology.device_identity,
            allocated_mb=1.0,
            reserved_mb=2.0,
            device_count=4,
        ),
    )
    writer = RankLocalJSONLWriter(tmp_path, topology)
    writer.write(record)
    payload = json.loads(writer.path.read_text())
    assert payload["device_identity"] == {"kind": "cuda", "index": 3, "uuid": "GPU-3"}


def test_profile_record_rejects_device_identity_mismatch_before_serialization() -> None:
    topology = replace(
        _topology(0),
        device_identity=AcceleratorIdentity(AcceleratorKind.CUDA, 3, "GPU-3"),
    )
    usage = AllocatorUsage(
        identity=AcceleratorIdentity(AcceleratorKind.CUDA, 1, "GPU-1"),
        allocated_mb=1.0,
        reserved_mb=2.0,
        device_count=4,
    )

    with pytest.raises(ValueError, match="device identity"):
        ProfileRecord(ProfileScope.DEVICE, 1.0, topology, device=usage)


def test_writer_degrades_unserializable_field_to_parseable_error_record(tmp_path) -> None:
    topology = replace(_topology(0), host=object())
    record = ProfileRecord(ProfileScope.PROCESS, 1.0, topology, process=_process())
    writer = RankLocalJSONLWriter(tmp_path, topology)

    writer.write(record)

    payload = json.loads(writer.path.read_text())
    assert payload["scope"] == "process"
    assert payload["global_rank"] == 0
    assert payload["serialization_error"] == {"fields": ["host"]}


@pytest.mark.parametrize("monotonic_time", [math.nan, math.inf, -math.inf])
def test_profile_record_rejects_nonfinite_monotonic_time(monotonic_time: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ProfileRecord(ProfileScope.PROCESS, monotonic_time, _topology(0), process=_process())


def test_projector_emits_deterministic_scalar_flags_without_reason_text() -> None:
    record = ProfileRecord(ProfileScope.PROCESS, 1.0, _topology(0), process=_process(unavailable=True))
    assert project_scalars(record) == (
        # The reason is typed evidence, never a free-text metric value.
        ScalarMetric("user_cpu_seconds_unavailable", True),
        ScalarMetric("system_cpu_seconds_unavailable", True),
        ScalarMetric("read_block_operations_unavailable", True),
        ScalarMetric("write_block_operations_unavailable", True),
        ScalarMetric("voluntary_context_switches_unavailable", True),
        ScalarMetric("involuntary_context_switches_unavailable", True),
        ScalarMetric("peak_rss_mb_unavailable", True),
    )


def test_projector_accepts_device_readings() -> None:
    usage = AllocatorUsage(
        identity=AcceleratorIdentity(AcceleratorKind.CUDA, 3, "GPU-3"),
        allocated_mb=2.0,
        reserved_mb=4.0,
        device_count=2,
    )
    record = ProfileRecord(ProfileScope.DEVICE, 3.0, _topology(0), device=usage)
    assert [(item.key, item.value) for item in project_scalars(record)] == [
        ("allocated_mb", 2.0),
        ("reserved_mb", 4.0),
        ("device_count", 2),
    ]


def test_node_and_job_records_require_explicit_aggregate_data() -> None:
    with pytest.raises(ValueError, match="explicit aggregate"):
        ProfileRecord(ProfileScope.NODE, 1.0, _topology(0))
    with pytest.raises(ValueError, match="explicit aggregate"):
        ProfileRecord(ProfileScope.JOB, 1.0, _topology(0))


def test_profile_record_rejects_a_non_profile_scope() -> None:
    with pytest.raises((TypeError, ValueError)):
        ProfileRecord(
            scope="process",
            monotonic_time=1.0,
            topology=_topology(0),
            process=_process(),
        )


def test_profile_record_rejects_both_process_and_device_payloads() -> None:
    topology = _topology(0)
    device = AllocatorUsage(
        identity=AcceleratorIdentity(AcceleratorKind.CUDA, 3, "GPU-3"),
        allocated_mb=1.0,
        reserved_mb=2.0,
        device_count=1,
    )

    with pytest.raises((TypeError, ValueError)):
        ProfileRecord(
            scope=ProfileScope.PROCESS,
            monotonic_time=1.0,
            topology=topology,
            process=_process(),
            device=device,
        )


def test_projector_marks_missing_device_count_explicitly() -> None:
    usage = AllocatorUsage(
        identity=AcceleratorIdentity(AcceleratorKind.CUDA, 3, "GPU-3"),
        allocated_mb=2.0,
        reserved_mb=4.0,
        device_count=None,
    )
    record = ProfileRecord(ProfileScope.DEVICE, 3.0, _topology(0), device=usage)
    assert [(item.key, item.value) for item in project_scalars(record)] == [
        ("allocated_mb", 2.0),
        ("reserved_mb", 4.0),
        ("device_count_unavailable", True),
    ]
