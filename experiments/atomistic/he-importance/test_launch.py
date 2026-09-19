"""Controls for the production HI launch seam."""

from __future__ import annotations

from importlib import import_module
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from tpen.artifacts import RunContext, RunResult
from tpen.distributed import ExecutionTopology
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


def test_launch_refuses_missing_or_empty_topology(tmp_path: Path) -> None:
    cell = _cell(tmp_path)

    with pytest.raises(launch.LaunchValidationError, match="populated"):
        launch.launch_train(cell, None, runner=lambda _: pytest.fail("runner was called"))
    with pytest.raises(launch.LaunchValidationError, match="populated"):
        launch.populate_execution_topology(cell, {})


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


@pytest.mark.parametrize(
    ("result", "expected"),
    [(RunResult(status="completed"), 0), (RunResult(status="failed"), 1), (0, 0), (7, 7)],
)
def test_exit_code_direct_runresult_and_int_polarity(result: object, expected: int) -> None:
    assert launch._exit_code(result) == expected


def test_exit_code_rejects_non_result_non_int() -> None:
    with pytest.raises(launch.LaunchValidationError, match="int or RunResult"):
        launch._exit_code("0")


def test_source_shape_guard_has_positive_and_negative_arms(tmp_path: Path) -> None:
    cell = _cell(tmp_path)
    assert launch._source_cell(cell) is cell
    with pytest.raises(launch.LaunchValidationError, match="MaterializedCell"):
        launch._source_cell(object())


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
