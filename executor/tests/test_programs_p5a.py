"""P5a multi-slot and control/science binding acceptance probes."""

from __future__ import annotations

import pytest

from tpen_executor.configuration import StagedHydra
from tpen_executor.programs import MaterializedProgram, WorkerProgram


def test_materialized_program_is_frozen_slotted_and_has_immutable_slots() -> None:
    """Materialized slots cannot be replaced, extended, or mutated in place."""

    record = MaterializedProgram({"payload": object()})
    with pytest.raises(AttributeError):
        record.slots = {}  # type: ignore[misc]
    with pytest.raises(AttributeError):
        record.extra = "not allowed"  # type: ignore[attr-defined]
    with pytest.raises((TypeError, AttributeError)):
        record.slots["payload"] = None  # type: ignore[index]


def test_packaged_plan_shape_materializes_without_submitter_construction() -> None:
    """The plan-shaped config has independent allocation and worker slots."""

    staged = StagedHydra().compose_config()
    assert staged.program is not None
    assert set(staged.programs) == {"program", "program.dispatcher.worker"}
    env = {
        "paths": {
            "control_python": "/prepared/shared/bin/python",
            "control_setup": "/prepared/shared/control.sh",
            "science_python": "/prepared/science/bin/python",
            "science_setup": "/prepared/science/setup.sh",
        }
    }
    allocation = staged.program.materialize(env)
    assert isinstance(allocation, MaterializedProgram)
    assert set(allocation) == {"environment", "dispatcher"}
    assert allocation.environment.python == "/prepared/shared/bin/python"
    assert isinstance(allocation.dispatcher.worker, WorkerProgram)
    with pytest.raises((TypeError, AttributeError)):
        allocation.slots["environment"] = None  # type: ignore[index]

    worker = staged.programs["program.dispatcher.worker"].materialize(env)
    assert set(worker) == {"environment", "process", "runner", "evidence"}
    assert worker.environment.python == "/prepared/science/bin/python"


def test_same_prepared_environment_can_bind_control_and_science() -> None:
    """Separate slots may intentionally carry equal prepared descriptors."""

    staged = StagedHydra().compose_config()
    paths = {
        "paths": {
            "control_python": "/prepared/shared/bin/python",
            "control_setup": "/prepared/shared/setup.sh",
            "science_python": "/prepared/shared/bin/python",
            "science_setup": "/prepared/shared/setup.sh",
        }
    }
    allocation = staged.program.materialize(paths)  # type: ignore[union-attr]
    worker = staged.programs["program.dispatcher.worker"].materialize(paths)
    assert allocation.environment == worker.environment
    assert allocation.environment is not worker.environment

    distinct = dict(paths)
    distinct["paths"] = dict(paths["paths"])
    distinct["paths"]["science_python"] = "/prepared/science/bin/python"
    distinct_worker = staged.programs["program.dispatcher.worker"].materialize(distinct)
    assert allocation.environment != distinct_worker.environment
