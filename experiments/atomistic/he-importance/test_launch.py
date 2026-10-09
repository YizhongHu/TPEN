"""Controls for the production HI launch seam."""

from __future__ import annotations

from collections import UserList, UserString, deque
from collections.abc import Buffer, Mapping, Sequence
from importlib import import_module
import inspect
import os
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest
from omegaconf import OmegaConf

from tpen.artifacts import RunContext, RunResult
from tpen.distributed import AcceleratorIdentity, AcceleratorKind, ExecutionTopology
from tpen.run import run_from_config
from tpen.runner import Runner


stage_coordinate = import_module("experiments.atomistic.he-importance.stage_coordinate")
launch = import_module("experiments.atomistic.he-importance.launch")

_EXECUTION_FACT_KEY_PROBES = (
    "global_rank",
    "global_size",
    "local_rank",
    "local_size",
    "node_rank",
    "node_size",
    "host",
    "pid",
    "device",
    "job_id",
    "device_identity",
    "rank",
    "world_size",
    "launcher",
    "launcher_job_id",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "hostname",
    "host_name",
    "process_id",
    "processid",
    "device_id",
    "deviceid",
    "cuda_visible_devices",
    "visible_devices",
    "num_nodes",
    "node_count",
    "num_gpus",
    "gpu_id",
    "master_addr",
    "master_port",
    "SLURM_NTASKS",
    "PMI_RANK",
    "PMIX_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "MPI_LOCALRANKID",
    "pbs_nodenum",
    "pbs_tasknum",
    "PBS_NODENUM",
    "PBS_TASKNUM",
)


def _cell(tmp_path: Path, *, identity: dict[str, object] | None = None) -> object:
    scientific_identity = {"architecture": "control"}
    if identity:
        scientific_identity.update(identity)
    return stage_coordinate.materialize_stage(
        "O1",
        [
            {
                "scientific_identity": scientific_identity,
                "payload": {"updates": 50_000},
                "topology": {},
            }
        ],
        stage_coordinate.OptimizerCell("adam", "available"),
        tmp_path,
    )[0]


def _topology(*, host: str = "test-host", device: str = "cuda:0") -> ExecutionTopology:
    return ExecutionTopology(
        global_rank=0,
        global_size=1,
        local_rank=0,
        local_size=1,
        node_rank=0,
        node_size=1,
        host=host,
        pid=os.getpid(),
        device=device,
        job_id="test-job",
    )


def test_launch_populates_topology_under_designated_key(tmp_path: Path) -> None:
    cell = _cell(tmp_path)

    plan = launch.prepare_train_launch(cell, _topology())

    assert plan.cell.manifest["topology"]["global_rank"] == 0
    assert plan.cell.manifest["topology"]["global_size"] == 1
    assert plan.cell.manifest["topology"]["device"] == "cuda:0"
    assert "global_rank" not in plan.cell.manifest["scientific_identity"]
    assert "global_rank" not in plan.cell.manifest["payload"]
    assert launch.launch_train(cell, _topology(), runner=lambda _: 0) == 0


def test_launch_snapshots_duck_topology_for_manifest_and_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)

    class ChangingTopology:
        def __init__(self) -> None:
            self.source = _topology()
            self.host_reads = 0

        @property
        def host(self) -> str:
            self.host_reads += 1
            return "manifest-host" if self.host_reads == 1 else "runner-host"

        def __getattr__(self, name: str) -> object:
            return getattr(self.source, name)

    topology = ChangingTopology()
    received: list[object] = []
    rebound_cells: list[object] = []
    original_populate = launch.populate_execution_topology

    def recording_populate(source: object, facts: object) -> object:
        rebound = original_populate(source, facts)
        rebound_cells.append(rebound)
        return rebound

    monkeypatch.setattr(launch, "populate_execution_topology", recording_populate)

    def recording_runner(config: object, *, topology: object) -> int:
        del config
        received.append(topology)
        return 0

    assert launch.launch_train(cell, topology, runner=recording_runner) == 0
    assert topology.host_reads == 1
    assert rebound_cells[0].manifest["topology"]["host"] == "manifest-host"
    assert received == [{"global_rank": 0, "global_size": 1, "local_rank": 0,
                         "local_size": 1, "node_rank": 0, "node_size": 1,
                         "host": "manifest-host", "pid": os.getpid(),
                         "device": "cuda:0", "job_id": "test-job",
                         "device_identity": None}]


def test_launch_detaches_duck_nested_topology_before_validation_and_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)

    class DuckTopology:
        def __init__(self) -> None:
            self.source = _topology()
            self.live_host = {"name": "manifest-host"}
            self.host_reads = 0

        @property
        def host(self) -> dict[str, str]:
            self.host_reads += 1
            return self.live_host

        def __getattr__(self, name: str) -> object:
            return getattr(self.source, name)

    topology = DuckTopology()
    received: list[object] = []
    rebound_cells: list[object] = []
    original_populate = launch.populate_execution_topology

    def recording_populate(source: object, facts: object) -> object:
        rebound = original_populate(source, facts)
        rebound_cells.append(rebound)
        return rebound

    monkeypatch.setattr(launch, "populate_execution_topology", recording_populate)

    def recording_runner(config: object, *, topology: object) -> int:
        del config
        topology["host"]["name"] = "runner-snapshot"
        received.append(topology)
        return 0

    assert launch.launch_train(cell, topology, runner=recording_runner) == 0
    assert topology.host_reads == 1
    assert topology.live_host == {"name": "manifest-host"}
    assert rebound_cells[0].manifest["topology"]["host"] == {"name": "manifest-host"}
    assert received == [{"global_rank": 0, "global_size": 1, "local_rank": 0,
                         "local_size": 1, "node_rank": 0, "node_size": 1,
                         "host": {"name": "runner-snapshot"}, "pid": os.getpid(),
                         "device": "cuda:0", "job_id": "test-job",
                         "device_identity": None}]


def test_launch_detaches_nested_topology_facts_before_validation_and_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    facts = launch.execution_topology_facts(_topology())

    class ChangingIdentity(dict[str, object]):
        reads = 0

        def items(self):
            type(self).reads += 1
            kind = "cpu" if type(self).reads == 1 else "invalid-runner-kind"
            return [("kind", kind), ("index", None), ("uuid", None)]

    identity = ChangingIdentity()
    facts["device_identity"] = identity
    received: list[object] = []
    rebound_cells: list[object] = []
    original_populate = launch.populate_execution_topology

    def recording_populate(source: object, topology: object) -> object:
        rebound = original_populate(source, topology)
        rebound_cells.append(rebound)
        return rebound

    monkeypatch.setattr(launch, "populate_execution_topology", recording_populate)

    def recording_runner(config: object, *, topology: object) -> int:
        del config
        received.append(topology)
        return 0

    assert launch.launch_train(cell, facts, runner=recording_runner) == 0
    assert identity.reads == 1
    assert rebound_cells[0].manifest["topology"]["device_identity"]["kind"] == "cpu"
    assert received[0]["device_identity"]["kind"] == "cpu"


def test_execution_topology_facts_rejects_non_topology_object(tmp_path: Path) -> None:
    cell = _cell(tmp_path)

    with pytest.raises(
        launch.LaunchValidationError,
        match="topology must be an execution topology or mapping",
    ):
        launch.prepare_train_launch(cell, object())


def test_launch_refuses_missing_or_empty_topology(tmp_path: Path) -> None:
    cell = _cell(tmp_path)

    with pytest.raises(launch.LaunchValidationError, match="populated"):
        launch.launch_train(cell, None, runner=lambda _: pytest.fail("runner was called"))
    with pytest.raises(launch.LaunchValidationError, match="populated"):
        launch.launch_train(cell, {}, runner=lambda _: pytest.fail("runner was called"))
    with pytest.raises(launch.LaunchValidationError, match="populated"):
        launch.launch_train(cell, None)


def test_launch_refuses_execution_facts_outside_topology(tmp_path: Path) -> None:
    cell = _cell(tmp_path, identity={"global_rank": 0})

    with pytest.raises(launch.LaunchValidationError, match="under manifest.topology"):
        launch.prepare_train_launch(cell, _topology())

    with pytest.raises(launch.LaunchValidationError, match="under manifest.topology"):
        launch.launch_train(cell, _topology(), runner=lambda _: pytest.fail("runner was called"))


def test_launch_topology_does_not_change_content_hash(tmp_path: Path) -> None:
    cell = _cell(tmp_path)

    first = launch.prepare_train_launch(cell, _topology(host="host-a", device="cuda:0"))
    second = launch.prepare_train_launch(cell, _topology(host="host-b", device="cuda:1"))

    assert first.cell.content_hash == cell.content_hash == second.cell.content_hash
    assert first.cell.output_path == cell.output_path == second.cell.output_path
    assert first.cell.manifest["topology"] != second.cell.manifest["topology"]


def test_launch_calls_injected_runner_with_l1_resolved_config(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    received: list[object] = []
    expected = OmegaConf.to_container(
        import_module("experiments.atomistic.he-importance.train_config").resolve_train_config(cell),
        resolve=True,
    )

    def recording_runner(config: object) -> int:
        received.append(config)
        return 0

    assert launch.launch_train(cell, _topology(), runner=recording_runner) == 0
    assert len(received) == 1
    assert OmegaConf.to_container(received[0], resolve=True) == expected
    assert inspect.signature(launch.launch_train).parameters["runner"].default is run_from_config


@pytest.mark.parametrize(
    "shape",
    [
        lambda key: {key: 0},
        lambda key: {"nested": {key: 0}},
        lambda key: {"nested": [{key: 0}]},
    ],
)
@pytest.mark.parametrize(
    "key", _EXECUTION_FACT_KEY_PROBES,
)
def test_public_launch_rejects_execution_fact_matrix(
    tmp_path: Path, shape: object, key: str
) -> None:
    cell = _cell(tmp_path, identity=shape(key))

    with pytest.raises(launch.LaunchValidationError, match="under manifest.topology"):
        launch.launch_train(cell, _topology(), runner=lambda _: pytest.fail("runner was called"))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("matrix_rank", 4),
        ("low_rank", 8),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        lambda key, value: {key: value},
        lambda key, value: {"nested": {key: value}},
        lambda key, value: {"nested": [{key: value}]},
    ],
)
def test_public_launch_accepts_undeclared_scientific_names(
    tmp_path: Path, key: str, value: object, shape: object
) -> None:
    cell = _cell(tmp_path, identity=shape(key, value))
    calls: list[object] = []

    assert launch.launch_train(cell, _topology(), runner=lambda config: calls.append(config) or 0) == 0
    assert len(calls) == 1


@pytest.mark.parametrize(
    "key",
    [
        "CUDA_DEVICE_ID",
        "RANK_ID",
        "process_count",
        "PBS_JOBID",
        "NVIDIA_VISIBLE_DEVICES",
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        lambda key: {key: "sentinel"},
        lambda key: {"nested": {key: "sentinel"}},
        lambda key: {"nested": [{key: "sentinel"}]},
    ],
)
def test_public_launch_rejects_new_execution_fact_names(
    tmp_path: Path, key: str, shape: object
) -> None:
    cell = _cell(tmp_path, identity=shape(key))

    with pytest.raises(launch.LaunchValidationError, match="under manifest.topology"):
        launch.launch_train(cell, _topology(), runner=lambda _: pytest.fail("runner was called"))


def test_typed_execution_topology_serializes_every_distinct_field() -> None:
    identity = AcceleratorIdentity(
        kind=AcceleratorKind("cuda"), index=19, uuid="distinct-uuid"
    )
    topology = ExecutionTopology(
        global_rank=11,
        global_size=12,
        local_rank=13,
        local_size=14,
        node_rank=15,
        node_size=16,
        host="distinct-host",
        pid=17,
        device="distinct-device",
        job_id="distinct-job",
        device_identity=identity,
    )

    assert launch.execution_topology_facts(topology) == {
        "global_rank": 11,
        "global_size": 12,
        "local_rank": 13,
        "local_size": 14,
        "node_rank": 15,
        "node_size": 16,
        "host": "distinct-host",
        "pid": 17,
        "device": "distinct-device",
        "job_id": "distinct-job",
        "device_identity": {"kind": "cuda", "index": 19, "uuid": "distinct-uuid"},
    }


def test_typed_serializer_covers_current_topology_schema() -> None:
    assert set(launch.execution_topology_facts(_topology())) == {
        field.name for field in fields(ExecutionTopology)
    }
    assert {"kind", "index", "uuid"} == {
        field.name for field in fields(AcceleratorIdentity)
    }


def test_non_string_identity_key_is_refused_at_materialization(tmp_path: Path) -> None:
    """L1 refuses a non-string identity key before launch ever sees a cell.

    Manifests are deep-frozen, so a post-hoc mutation of ``cell.manifest``
    cannot exercise launch's ``isinstance(key, str)`` backstop; that branch
    is only reachable if this upstream refusal ever stopped enforcing it.
    This test is a MESSAGE-PIN, not mechanism coverage: under the mutant that
    removes L1's string-key check, L1 still refuses via a second, deeper
    guard (``MaterializationError: identity values must be finite JSON
    data``), so this test only reddens through its ``match=``. The invariant
    is double-guarded, and route coverage for the deeper guard comes from
    L1's own ``test_canonical_hash_preserves_the_typed_non_string_key_failure``.
    If this test goes red, L1 stopped rejecting non-string keys with this
    specific message and the launch-side backstop needs re-examination for
    whether it became load-bearing.
    """
    with pytest.raises(
        stage_coordinate.MaterializationError, match="identity mapping keys must be strings"
    ):
        stage_coordinate.materialize_stage(
            "O1",
            [
                {
                    "scientific_identity": {"architecture": "control", 1: "scientific-value"},
                    "payload": {"updates": 50_000},
                    "topology": {},
                }
            ],
            stage_coordinate.OptimizerCell("adam", "available"),
            tmp_path,
        )


def test_execution_fact_detector_rejects_lie_about_lower() -> None:
    class Sneaky(str):
        def lower(self) -> str:
            return "benign_name"

    with pytest.raises(launch.LaunchValidationError, match="global_rank"):
        launch._reject_execution_facts_outside_topology(
            {"scientific_identity": {Sneaky("global_rank"): 1}, "payload": {}, "topology": {}}
        )


def test_execution_fact_detector_rejects_plain_string_subclass() -> None:
    class Plain(str):
        pass

    with pytest.raises(launch.LaunchValidationError, match="global_rank"):
        launch._reject_execution_facts_outside_topology(
            {"scientific_identity": {Plain("global_rank"): 1}, "payload": {}, "topology": {}}
        )


def test_identity_non_mapping_error_is_specific(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    facts = launch.execution_topology_facts(_topology())
    facts["device_identity"] = "not-a-mapping"

    with pytest.raises(launch.LaunchValidationError, match="must be a mapping or null"):
        launch.launch_train(cell, facts)


def test_identity_rejects_unsupported_nested_keys(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    facts = launch.execution_topology_facts(_topology())
    facts["device_identity"] = {
        "kind": "cuda", "index": 0, "uuid": None, "pci_bus_id": "sentinel"
    }

    with pytest.raises(launch.LaunchValidationError, match="pci_bus_id"):
        launch.launch_train(cell, facts)


def test_identity_unknown_keys_follow_runner_strictness_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    facts = launch.execution_topology_facts(_topology())
    facts["device_identity"] = {
        "kind": "cuda", "index": 0, "uuid": None, "caller_note": "keep"
    }

    # The config-only route is caller-open: nested unknown keys are admitted.
    assert launch.launch_train(cell, facts, runner=lambda _: 0) == 0

    # The TPEN crossing remains strict: the same nested unknown key is rejected.
    runner = _install_default_runner_witness(monkeypatch, [])
    with pytest.raises(launch.LaunchValidationError, match="caller_note"):
        launch.launch_train(cell, facts, runner=runner)


def test_explicit_topology_runner_receives_topology(tmp_path: Path) -> None:
    cell = _cell(tmp_path)

    received: list[object] = []

    def wrapper(config: object, *, topology: object) -> int:
        del config
        received.append(topology)
        return 0

    assert launch.launch_train(cell, _topology(), runner=wrapper) == 0
    # Amendment `contract-amendment-injected-runner-mapping-2026-10-08`
    # replaces the predecessor's typed `.host` assertion with the mapping contract.
    assert isinstance(received[0], dict) and received[0]["host"] == "test-host"


def test_kwargs_runner_receives_topology(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    received: list[object] = []

    def wrapper(config: object, **kwargs: object) -> int:
        del config
        received.append(kwargs["topology"])
        return 0

    assert launch.launch_train(cell, _topology(), runner=wrapper) == 0
    # Amendment `contract-amendment-injected-runner-mapping-2026-10-08`
    # replaces the predecessor's typed `.job_id` assertion with the mapping contract.
    assert isinstance(received[0], dict) and received[0]["job_id"] == "test-job"


def test_plain_injected_runner_keeps_config_only_contract(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    received: list[object] = []

    def recording_runner(config: object) -> int:
        received.append(config)
        return 0

    assert launch.launch_train(cell, _topology(), runner=recording_runner) == 0
    assert len(received) == 1


@pytest.mark.parametrize("key", ["WORLD_SIZE", "hostname", "process_id", "cuda_visible_devices"])
def test_public_launch_accepts_execution_aliases_inside_topology(tmp_path: Path, key: str) -> None:
    cell = _cell(tmp_path)
    facts = launch.execution_topology_facts(_topology())
    facts[key] = "inside"

    assert launch.launch_train(cell, facts, runner=lambda _: 0) == 0


def test_default_runner_receives_supplied_topology_at_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    topology = _topology()
    received_contexts: list[object] = []
    consumer_topologies: list[object] = []

    import tpen.run as production_run

    class FakeContext(RunContext):
        def __init__(self, received_topology: object) -> None:
            self.cfg = None
            self.loggers: tuple[object, ...] = ()
            self.metadata = SimpleNamespace(status="initialized")
            self.topology = received_topology

        def emit(self, _: object) -> None:
            return None

    class RecordingRunner(Runner):
        def run(self, context: object) -> RunResult:
            received_contexts.append(context)
            return RunResult(status="completed")

    def consumer(cfg: object, **kwargs: object) -> FakeContext:
        del cfg
        consumer_topologies.append(kwargs.get("topology"))
        return FakeContext(kwargs.get("topology"))

    monkeypatch.setattr(production_run, "prepare_run_context", consumer)
    monkeypatch.setattr(production_run, "_seed_runtime_rngs", lambda _: None)
    monkeypatch.setattr(production_run, "_instantiate_runner", lambda _: RecordingRunner())

    assert launch.launch_train(cell, topology, runner=production_run.run_from_config) == 0
    assert consumer_topologies == [topology]
    assert len(received_contexts) == 1
    assert received_contexts[0].topology.job_id == "test-job"
    assert received_contexts[0].topology.host == "test-host"


def test_default_runner_topology_is_per_call_not_process_global(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    ordinary_config = launch.prepare_train_launch(cell, _topology()).config
    entered = threading.Event()
    release = threading.Event()
    seen: list[object] = []

    import tpen.run as production_run

    class FakeContext(RunContext):
        def __init__(self, cfg: object, topology: object) -> None:
            self.cfg = cfg
            self.loggers: tuple[object, ...] = ()
            self.metadata = SimpleNamespace(status="initialized")
            self.topology = topology

        def emit(self, _: object) -> None:
            return None

    class RecordingRunner(Runner):
        def run(self, _: object) -> RunResult:
            return RunResult(status="completed")

    def consumer(cfg: object, **kwargs: object) -> FakeContext:
        topology = kwargs.get("topology")
        seen.append(topology)
        if topology is not None:
            entered.set()
            assert release.wait(5)
        return FakeContext(cfg, topology)

    monkeypatch.setattr(production_run, "prepare_run_context", consumer)
    monkeypatch.setattr(production_run, "_seed_runtime_rngs", lambda _: None)
    monkeypatch.setattr(production_run, "_instantiate_runner", lambda _: RecordingRunner())

    result: list[int] = []
    thread = threading.Thread(
        target=lambda: result.append(
            launch.launch_train(
                cell,
                _topology(host="host-a"),
                runner=production_run.run_from_config,
            )
        )
    )
    thread.start()
    assert entered.wait(5)

    # An ordinary core call in the overlap must not inherit the launch's
    # topology, even while the launch is still in its consumer.
    assert production_run.run_from_config(ordinary_config) == 0
    release.set()
    thread.join(5)

    assert not thread.is_alive()
    assert result == [0]
    assert any(value is None for value in seen)
    assert seen[0].host == "host-a"


def _install_default_runner_witness(
    monkeypatch: pytest.MonkeyPatch, seen: list[object]
) -> object:
    import tpen.run as production_run

    class FakeContext(RunContext):
        def __init__(self, cfg: object, topology: object) -> None:
            self.cfg = cfg
            self.loggers: tuple[object, ...] = ()
            self.metadata = SimpleNamespace(status="initialized")
            self.topology = topology

        def emit(self, _: object) -> None:
            return None

    class RecordingRunner(Runner):
        def run(self, _: object) -> RunResult:
            return RunResult(status="completed")

    def consumer(cfg: object, **kwargs: object) -> FakeContext:
        topology = kwargs.get("topology")
        seen.append(topology)
        return FakeContext(cfg, topology)

    monkeypatch.setattr(production_run, "prepare_run_context", consumer)
    monkeypatch.setattr(production_run, "_seed_runtime_rngs", lambda _: None)
    monkeypatch.setattr(production_run, "_instantiate_runner", lambda _: RecordingRunner())
    return production_run.run_from_config


def test_default_runner_accepts_canonical_mapping_at_consumer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    seen: list[object] = []
    runner = _install_default_runner_witness(monkeypatch, seen)
    facts = launch.execution_topology_facts(_topology())

    assert launch.launch_train(cell, facts, runner=runner) == 0
    assert len(seen) == 1
    assert seen[0].job_id == "test-job"


def test_default_runner_refuses_unsupported_mapping_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    facts = launch.execution_topology_facts(_topology())
    facts.update(
        launcher_job_id="EXTRA-SCHEDULER-JOB",
        cuda_visible_devices="EXTRA-DEVICE-VISIBILITY",
    )
    runner = _install_default_runner_witness(monkeypatch, [])

    with pytest.raises(launch.LaunchValidationError, match="unsupported production runner"):
        launch.launch_train(cell, facts, runner=runner)


def test_default_mapping_required_and_identity_guards_have_both_polarities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)
    facts = launch.execution_topology_facts(_topology())
    facts["device_identity"] = {"kind": "cuda", "index": 0, "uuid": "GPU-0"}
    seen: list[object] = []
    runner = _install_default_runner_witness(monkeypatch, seen)

    assert launch.launch_train(cell, facts, runner=runner) == 0
    assert seen[0].device_identity.uuid == "GPU-0"

    missing = launch.execution_topology_facts(_topology())
    missing.pop("host")
    with pytest.raises(launch.LaunchValidationError, match="missing required facts"):
        launch.launch_train(cell, missing, runner=runner)

    malformed = launch.execution_topology_facts(_topology())
    malformed["device_identity"] = {"kind": "not-an-accelerator"}
    with pytest.raises(launch.LaunchValidationError, match="device_identity is malformed"):
        launch.launch_train(cell, malformed, runner=runner)


@pytest.mark.parametrize(("result", "expected"), [(0, 0), (7, 7), (-1, -1)])
def test_exit_code_accepts_integer_polarity(result: int, expected: int) -> None:
    assert launch._exit_code(result) == expected


def test_exit_code_rejects_all_non_integer_result_shapes() -> None:
    class DetailedResult(RunResult):
        pass

    ForgedResult = type(
        "RunResult",
        (),
        {"__module__": "tpen.artifacts", "status": "completed"},
    )
    rejected = (
        RunResult(status="completed"),
        RunResult(status="failed"),
        DetailedResult(status="completed"),
        DetailedResult(status="failed"),
        SimpleNamespace(status="completed"),
        ForgedResult(),
        "0",
        True,
    )
    for result in rejected:
        with pytest.raises(launch.LaunchValidationError, match="runner must return an int"):
            launch._exit_code(result)


def test_exit_code_rejects_non_result_non_int() -> None:
    with pytest.raises(launch.LaunchValidationError, match="runner must return an int"):
        launch._exit_code("0")


def test_source_shape_guard_has_positive_and_negative_arms(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    assert launch._source_cell(cell) is cell
    with pytest.raises(launch.LaunchValidationError, match="MaterializedCell"):
        launch._source_cell(object())


class _FlipMapping(dict[str, object]):
    def __init__(self, values: dict[str, object], flip_after: int) -> None:
        super().__init__(values)
        self.flip_after = flip_after
        self.reads = 0

    def items(self):
        self.reads += 1
        values = dict(super().items())
        if self.reads > self.flip_after:
            values["global_rank"] = 1
        return values.items()


def test_populate_validates_the_frozen_manifest_across_flip_schedules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = _cell(tmp_path)
    events: list[str] = []
    original_with = launch._STAGE_API.with_execution_topology
    original_reject = launch._reject_execution_facts_outside_topology

    def recording_with(cell: object, topology: Mapping[str, object]) -> object:
        events.append("freeze")
        bound = original_with(cell, topology)
        if isinstance(cell.manifest, dict):
            identity = cell.manifest["scientific_identity"]
            if isinstance(identity, dict):
                identity["post_freeze_marker"] = True
        return bound

    def recording_reject(manifest: Mapping[str, object]) -> None:
        events.append("reject")
        original_reject(manifest)

    monkeypatch.setattr(launch._STAGE_API, "with_execution_topology", recording_with)
    monkeypatch.setattr(launch, "_reject_execution_facts_outside_topology", recording_reject)
    for flip_after in range(0, 8):
        scientific_identity = _FlipMapping(
            dict(clean.manifest["scientific_identity"]), flip_after
        )
        manifest = dict(clean.manifest)
        manifest["scientific_identity"] = scientific_identity
        candidate = replace(clean, manifest=manifest)
        try:
            bound = launch.populate_execution_topology(candidate, {"host": "node"})
        except launch.LaunchValidationError:
            continue
        assert events[-2:] == ["freeze", "reject"]
        assert "global_rank" not in bound.manifest["scientific_identity"]
        assert "post_freeze_marker" not in bound.manifest["scientific_identity"]

    ordinary = launch.populate_execution_topology(clean, {"host": "node"})
    assert ordinary.manifest["topology"]["host"] == "node"


class _MroHidingMeta(type):
    @property
    def __mro__(cls) -> tuple[type, ...]:
        del cls
        return (object,)


class _MroHiddenBuffer(dict[str, object], metaclass=_MroHidingMeta):
    def __buffer__(self, flags: int) -> memoryview:
        del flags
        return memoryview(b"payload")


def test_buffer_detection_reads_the_real_mro_descriptor() -> None:
    value = _MroHiddenBuffer(name="node")
    assert memoryview(value).tobytes() == b"payload"
    with pytest.raises(launch.LaunchValidationError, match="binary buffer refused"):
        launch._detach_topology_facts(value)

    class HonestBuffer(dict[str, object]):
        def __buffer__(self, flags: int) -> memoryview:
            del flags
            return memoryview(b"payload")

    with pytest.raises(launch.LaunchValidationError, match="binary buffer refused"):
        launch._detach_topology_facts(HonestBuffer(name="node"))


def test_launch_propagates_success_and_handled_failure_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cell = _cell(tmp_path)

    assert launch.launch_train(cell, _topology(), runner=lambda _: 0) == 0

    import tpen.run as production_run

    class FakeContext(RunContext):
        def __init__(self) -> None:
            self.cfg = None
            self.loggers: tuple[object, ...] = ()
            self.metadata = SimpleNamespace(status="initialized")
            self.events: list[object] = []

        def emit(self, event: object) -> None:
            self.events.append(event)

    class HandledFailureRunner(Runner):
        def run(self, _: object) -> RunResult:
            return RunResult(status="failed")

    context = FakeContext()

    monkeypatch.setattr(production_run, "validate_hi_train_config", lambda _: None)
    monkeypatch.setattr(production_run, "_seed_runtime_rngs", lambda _: None)
    monkeypatch.setattr(
        production_run,
        "prepare_run_context",
        lambda *args, **kwargs: context,
    )
    monkeypatch.setattr(
        production_run,
        "_instantiate_runner",
        lambda _: HandledFailureRunner(),
    )

    assert launch.launch_train(cell, _topology(), runner=production_run.run_from_config) == 1
    assert context.metadata.status == "failed"


@pytest.mark.parametrize("status", ["completed", "not-a-run-result"])
def test_exit_code_rejects_unrelated_status_object(status: str) -> None:
    class UnrelatedResult:
        pass

    result = UnrelatedResult()
    result.status = status

    with pytest.raises(launch.LaunchValidationError, match="runner must return an int"):
        launch._exit_code(result)


def test_exit_code_rejects_unknown_runresult_status() -> None:
    with pytest.raises(launch.LaunchValidationError, match="runner must return an int"):
        launch._exit_code(RunResult(status="unknown"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("global_size", 0, "global_size must be positive"),
        ("global_rank", "0", "global_rank must be an int"),
        ("global_rank", True, "global_rank must be an int"),
        ("local_rank", 1, r"local_rank must be in \[0, 1\)"),
    ],
)
def test_launch_rejects_invalid_topology_values_before_injected_runner(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    cell = _cell(tmp_path)
    facts = launch.execution_topology_facts(_topology())
    facts[field] = value

    with pytest.raises(launch.LaunchValidationError, match=message):
        launch.launch_train(cell, facts, runner=lambda _: pytest.fail("runner was called"))


@pytest.mark.parametrize("result", [True])
def test_exit_code_rejects_boolean(result: object) -> None:
    with pytest.raises(launch.LaunchValidationError, match="runner must return an int"):
        launch._exit_code(result)


def test_topology_detacher_closes_nested_sequence_schema() -> None:
    user_list = UserList(["before"])
    queue = deque(["before"])
    facts = launch.execution_topology_facts(
        {"host": {"nested": [user_list, (queue,)]}}
    )

    user_list[0] = "after"
    queue.append("after")

    assert facts == {"host": {"nested": [["before"], [["before"]]]}}
    assert type(facts["host"]["nested"]) is list
    assert type(facts["host"]["nested"][0]) is list
    assert type(facts["host"]["nested"][1]) is list


def test_duck_topology_detacher_closes_nested_sequence_schema() -> None:
    live = UserList([deque(["before"])])

    class DuckTopology:
        global_rank = 0
        global_size = 1
        local_rank = 0
        local_size = 1
        node_rank = 0
        node_size = 1
        pid = 1
        device = "cpu"
        job_id = "job"
        device_identity = None

        @property
        def host(self) -> UserList[object]:
            return live

    facts = launch.execution_topology_facts(DuckTopology())
    live[0].append("after")

    assert facts["host"] == [["before"]]
    assert type(facts["host"]) is list
    assert type(facts["host"][0]) is list


def test_topology_detacher_rejects_unadmitted_object() -> None:
    with pytest.raises(
        launch.LaunchValidationError,
        match="unsupported topology fact value type: object",
    ):
        launch.execution_topology_facts({"host": object()})


class _BinaryMapping(bytes, Mapping):
    def __iter__(self):
        return iter(("name",))

    def __len__(self):
        return 1

    def __getitem__(self, key):
        if key != "name":
            raise KeyError(key)
        return "node"


class _TypeErrorBufferSequence(Sequence):
    def __init__(self):
        self._values = [1, 2]

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]

    def __buffer__(self, flags):
        del flags
        raise TypeError("export unavailable")


def _late_buffer_mapping() -> Mapping[str, object]:
    class LateMapping(dict):
        pass

    value = LateMapping(name="node")
    assert not isinstance(value, Buffer)
    LateMapping.__buffer__ = lambda self, flags: memoryview(b"node")
    LateMapping.__release_buffer__ = lambda self, view: None
    assert memoryview(value).tobytes() == b"node"
    assert not isinstance(value, Buffer)
    return value


def test_launch_entry_guard_rejects_cache_stale_buffer() -> None:
    with pytest.raises(launch.LaunchValidationError, match="binary buffer refused"):
        launch.execution_topology_facts(_late_buffer_mapping())


def test_launch_detacher_rejects_cache_stale_buffer() -> None:
    with pytest.raises(launch.LaunchValidationError, match="binary buffer refused"):
        launch._detach_topology_facts(_late_buffer_mapping())


@pytest.mark.parametrize(
    "value",
    [b"node1", bytearray(b"node1"), memoryview(b"node1"), _BinaryMapping(b"node1"), _TypeErrorBufferSequence()],
)
@pytest.mark.parametrize("nested", [False, True])
def test_topology_detacher_rejects_all_buffer_providers(
    value: object, nested: bool
) -> None:
    candidate = {"host": value} if nested else value
    with pytest.raises(launch.LaunchValidationError, match="binary buffer refused"):
        launch.execution_topology_facts(candidate)


class _StringKey(str):
    pass


@pytest.mark.parametrize("key", [UserString("host"), _StringKey("host"), 1])
def test_topology_detacher_rejects_non_exact_string_keys(key: object) -> None:
    with pytest.raises(launch.LaunchValidationError, match="mapping keys must be exact str"):
        launch.execution_topology_facts({key: "node"})

    assert launch.execution_topology_facts({"host": "node"}) == {"host": "node"}


def test_populate_execution_topology_detaches_direct_public_input(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    live = UserList(["before"])

    rebound = launch.populate_execution_topology(
        cell, {"host": {"nested": [live]}}
    )
    live[0] = "after"

    assert dict(rebound.manifest["topology"]["host"]) == {
        "nested": (("before",),)
    }


class _ScalarSpoofMeta(type):
    def __eq__(cls, other: object) -> bool:
        return other is str

    __hash__ = type.__hash__


class _ScalarSpoof(dict[str, object], metaclass=_ScalarSpoofMeta):
    pass


def test_detachers_use_exact_scalar_identity_and_copy_spoofed_mapping() -> None:
    value = _ScalarSpoof(name="before")
    facts = launch.execution_topology_facts({"host": value})

    value["name"] = "after"
    assert facts == {"host": {"name": "before"}}
    assert facts["host"] is not value


class _IdentityBuffer(bytearray):
    kind = SimpleNamespace(value="cpu")
    index = None
    uuid = None


class _KindBuffer(bytearray):
    value = "cpu"


def test_duck_projection_classifies_identity_and_kind_before_projection() -> None:
    class DuckTopology:
        global_rank = 0
        global_size = 1
        local_rank = 0
        local_size = 1
        node_rank = 0
        node_size = 1
        host = "node"
        pid = 1
        device = "cpu"
        job_id = None

        def __init__(self, identity: object) -> None:
            self.device_identity = identity

    with pytest.raises(launch.LaunchValidationError, match="binary buffer refused"):
        launch.execution_topology_facts(DuckTopology(_IdentityBuffer(b"identity")))

    class KindIdentity:
        kind = _KindBuffer(b"kind")
        index = None
        uuid = None

    with pytest.raises(launch.LaunchValidationError, match="binary buffer refused"):
        launch.execution_topology_facts(DuckTopology(KindIdentity()))

    ordinary = launch.execution_topology_facts(DuckTopology(SimpleNamespace(
        kind=SimpleNamespace(value="cpu"), index=None, uuid=None
    )))
    assert ordinary["device_identity"] == {"kind": "cpu", "index": None, "uuid": None}


class _SkipRoot(str):
    def __ne__(self, other: object) -> bool:
        return False


def test_root_topology_exemption_uses_base_string_comparison() -> None:
    manifest = {
        _SkipRoot("scientific_identity"): {"global_rank": 1},
        "topology": {},
    }
    with pytest.raises(launch.LaunchValidationError, match="execution fact"):
        launch._reject_execution_facts_outside_topology(manifest)

    launch._reject_execution_facts_outside_topology(
        {"scientific_identity": {"architecture": "control"}, "topology": {}}
    )


def test_signature_parameter_routing_uses_base_string_comparison(
    tmp_path: Path,
) -> None:
    cell = _cell(tmp_path)
    received: list[object] = []

    class LyingName(str):
        def __eq__(self, other: object) -> bool:
            return False

        __hash__ = str.__hash__

    def runner(config: object, *, topology: object) -> int:
        del config
        received.append(topology)
        return 0

    runner.__signature__ = inspect.Signature(
        [inspect.Parameter(LyingName("topology"), inspect.Parameter.KEYWORD_ONLY)]
    )
    facts = {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": "node",
        "pid": 1,
        "device": "cpu",
    }
    assert launch.launch_train(cell, facts, runner=runner) == 0
    assert received == [facts]


def test_prepare_train_launch_owns_manifest_before_downstream_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_cell = _cell(tmp_path)
    clean_manifest = dict(clean_cell.manifest)
    clean_manifest["topology"] = {"host": "test-host"}
    dirty_manifest = dict(clean_manifest)
    dirty_identity = dict(clean_manifest["scientific_identity"])
    dirty_identity["global_rank"] = 1
    dirty_manifest["scientific_identity"] = dirty_identity

    @stage_coordinate.dataclass(frozen=True)
    class DivergentCell:
        manifest: Mapping[str, object]
        content_hash: str
        output_path: Path
        seed_streams: Mapping[str, int]
        clean: Mapping[str, object]
        dirty: Mapping[str, object]
        reads: list[int]
        flip_after: int

        def __getattribute__(self, name: str) -> object:
            if name == "manifest":
                reads = object.__getattribute__(self, "reads")
                reads[0] += 1
                limit = object.__getattribute__(self, "flip_after")
                chosen = "clean" if reads[0] <= limit else "dirty"
                return object.__getattribute__(self, chosen)
            if name == "content_hash":
                reads = object.__getattribute__(self, "reads")
                limit = object.__getattribute__(self, "flip_after")
                chosen = "clean" if reads[0] <= limit else "dirty"
                value = object.__getattribute__(self, chosen)
                return stage_coordinate.content_hash(value)
            return object.__getattribute__(self, name)

    original_resolve = launch._TRAIN_CONFIG.resolve_train_config
    downstream_cells: list[object] = []

    def recording_resolve(source: object) -> object:
        downstream_cells.append(source)
        return original_resolve(source)

    monkeypatch.setattr(launch._TRAIN_CONFIG, "resolve_train_config", recording_resolve)

    # _source_cell and binding perform the initial reads; flip after the
    # detector's read so the old caller-typed return diverges downstream.
    for flip_after in range(6, 10):
        candidate = DivergentCell(
            manifest=clean_manifest,
            content_hash="unused-field",
            output_path=clean_cell.output_path,
            seed_streams=clean_cell.seed_streams,
            clean=clean_manifest,
            dirty=dirty_manifest,
            reads=[0],
            flip_after=flip_after,
        )
        plan = launch.prepare_train_launch(candidate, _topology())
        assert isinstance(plan.cell, launch._OwnedLaunchCell)
        assert isinstance(downstream_cells[-1], launch._OwnedLaunchCell)
        assert plan.cell.manifest["scientific_identity"] == clean_manifest["scientific_identity"]
        assert plan.topology["host"] == "test-host"

    ordinary = launch.prepare_train_launch(clean_cell, _topology())
    assert isinstance(ordinary.cell, launch._OwnedLaunchCell)


def test_content_hash_is_exact_and_binds_both_public_launch_entries(
    tmp_path: Path,
) -> None:
    row_a = _cell(tmp_path / "a", identity={"architecture": "a"})
    row_b = _cell(tmp_path / "b", identity={"architecture": "b"})
    seen_run_ids: list[str] = []

    assert launch.prepare_train_launch(row_b, _topology()).config.run.run_id == row_b.content_hash

    def runner(config: object) -> int:
        seen_run_ids.append(str(config.run.run_id))
        return 0

    assert launch.launch_train(row_b, _topology(), runner=runner) == 0
    assert seen_run_ids == [row_b.content_hash]

    class LyingHash(str):
        def __ne__(self, other: object) -> bool:
            del other
            return False

        def __eq__(self, other: object) -> bool:
            del other
            return True

        __hash__ = str.__hash__

    class LyingNonStringHash:
        def __ne__(self, other: object) -> bool:
            del other
            return False

        def __eq__(self, other: object) -> bool:
            del other
            return True

        def __str__(self) -> str:
            return row_a.content_hash

    mismatches = (
        (row_a.content_hash, "cell content hash does not bind"),
        (LyingHash(row_a.content_hash), "exact str"),
        (LyingNonStringHash(), "exact str"),
    )
    for held, message in mismatches:
        forged = replace(row_b, content_hash=held)
        with pytest.raises(launch.LaunchValidationError, match=message):
            launch.prepare_train_launch(forged, _topology())
        with pytest.raises(launch.LaunchValidationError, match=message):
            launch.launch_train(forged, _topology(), runner=lambda _: 0)
