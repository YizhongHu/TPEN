"""Torch-dependent tests for the Stage-Q facts-to-core route.

The tests keep the real ``launch_train`` -> ``run_from_config`` ->
``prepare_run_context`` path and replace only final runner construction.  They
therefore exercise topology conversion and context consumption without model
initialization or training updates.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import socket
import sys

import pytest


torch = pytest.importorskip("torch")
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    pytest.skip("Stage-Q core-route tests require CUDA", allow_module_level=True)

from tpen.run import run_from_config

# Keep the experiment boundary sanctioned: recover the defining module from
# the imported public entry point instead of adding tpen.* imports here.
tpen_run = sys.modules[run_from_config.__module__]
RunResult = tpen_run.RunResult
Runner = tpen_run.Runner


stage = importlib.import_module("experiments.atomistic.he-importance.stage_coordinate")
optimizer_grid = importlib.import_module("experiments.atomistic.he-importance.optimizer_grid")
runner = importlib.import_module("experiments.atomistic.he-importance.run_stage_q")


def _independent_expected_facts() -> dict[str, object]:
    """Freeze direct torch/OS observations before invoking the subject."""

    properties = torch.cuda.get_device_properties(0)
    return {
        "global_rank": 0,
        "global_size": 1,
        "local_rank": 0,
        "local_size": 1,
        "node_rank": 0,
        "node_size": 1,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "device": "cuda:0",
        "job_id": os.environ.get("PBS_JOBID") or os.environ.get("PBS_JOB_ID"),
        "device_identity": {
            "kind": "cuda",
            "index": 0,
            "uuid": str(properties.uuid),
        },
    }


class _FinalRunnerStub(Runner):
    def __init__(self, status: str) -> None:
        self._status = status

    def run(self, context: object) -> RunResult:
        del context
        return RunResult(status=self._status)


def _install_core_spies(
    expected_facts: dict[str, object],
    status: str,
    observed: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spy on the real topology conversion and final runner construction."""

    del expected_facts
    real_topology_factory = tpen_run.execution_topology_from_facts

    def topology_factory(facts: object) -> object:
        observed["topology_factory_calls"] = int(observed.get("topology_factory_calls", 0)) + 1
        observed["facts_at_core_boundary"] = facts
        converted = real_topology_factory(facts)
        observed["typed_topology"] = converted
        return converted

    def instantiate(context: object) -> Runner:
        observed["runner_constructor_calls"] = int(observed.get("runner_constructor_calls", 0)) + 1
        observed["context_run_dir"] = context.run_dir
        observed["context_topology"] = context.topology
        return _FinalRunnerStub(status)

    monkeypatch.setattr(tpen_run, "execution_topology_from_facts", topology_factory)
    monkeypatch.setattr(tpen_run, "_instantiate_runner", instantiate)


def _args(tmp_path: Path) -> object:
    return runner.build_parser().parse_args(
        ["--seed-label", "810001", "--output-root", str(tmp_path)]
    )


def test_r3_real_run_context_and_callback_paths_complete_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_facts = _independent_expected_facts()
    observed: dict[str, object] = {}
    _install_core_spies(expected_facts, "completed", observed, monkeypatch)

    # This calls the real launch_train -> real run_from_config -> real
    # prepare_run_context chain.  The only fake is the final runner object.
    args = _args(tmp_path)
    expected_topology_type = tpen_run.ExecutionTopology
    result = runner.execute(args)
    assert observed["topology_factory_calls"] == 1
    assert observed["runner_constructor_calls"] == 1
    assert observed["facts_at_core_boundary"] == expected_facts
    assert observed["facts_at_core_boundary"] is not expected_facts
    assert observed["typed_topology"] is not observed["facts_at_core_boundary"]
    assert type(observed["typed_topology"]) is expected_topology_type
    topology = observed["typed_topology"]
    for field in (
        "global_rank", "global_size", "local_rank", "local_size", "node_rank",
        "node_size", "host", "pid", "device", "job_id",
    ):
        assert getattr(topology, field) == expected_facts[field]
    identity = topology.device_identity
    expected_identity = expected_facts["device_identity"]
    assert type(identity).__name__ == "AcceleratorIdentity"
    assert identity.kind.value == expected_identity["kind"]
    assert identity.index == expected_identity["index"]
    assert identity.uuid == expected_identity["uuid"]
    assert observed["context_topology"] is observed["typed_topology"]
    context_run_dir = Path(observed["context_run_dir"])
    assert context_run_dir.is_dir()
    materialized = stage.materialize_intended_configurations(
        tmp_path, availability=optimizer_grid.roster_availability(), stages=("Q",)
    )
    selected = next(
        cell
        for cell in materialized
        if cell.manifest["seed_identity"]["label"] == 810001
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
    assert result == 0
    # The resolver freezes callback output_path at the cell path while the
    # ArtifactManager owns the flat context path.  Both are observed without
    # assuming they are identical; completion follows the shipped callback.
    assert context_run_dir == tmp_path / selected.content_hash
    assert (selected.output_path / "resolved_config.yaml").is_file()
    assert (selected.output_path / "COMPLETE").is_file()


def test_r3_failed_core_result_returns_one_without_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_facts = _independent_expected_facts()
    observed: dict[str, object] = {}
    _install_core_spies(expected_facts, "failed", observed, monkeypatch)

    result = runner.execute(_args(tmp_path))
    assert result == 1
    assert observed["topology_factory_calls"] == 1
    assert observed["runner_constructor_calls"] == 1
    assert observed["facts_at_core_boundary"] == expected_facts
    materialized = stage.materialize_intended_configurations(
        tmp_path, availability=optimizer_grid.roster_availability(), stages=("Q",)
    )
    selected = next(
        cell
        for cell in materialized
        if cell.manifest["seed_identity"]["label"] == 810001
        and cell.manifest["scientific_identity"]["context"]["name"] == "O-raw"
        and cell.manifest["scientific_identity"]["optimizer"]["method"] == "adam"
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["lr"] == 0.005
        and cell.manifest["scientific_identity"]["optimizer"]["levels"]["beta2"] == 0.999
    )
    assert not (selected.output_path / "COMPLETE").exists()
