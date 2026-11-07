"""Tests for the attempt-free dispatch seam.

These tests are deliberately torch-free: the dispatch seam is pure planning and
identity plumbing, so nothing here should need a model, a device, or a GPU.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from pathlib import Path
from typing import Sequence

import pytest

from experiments.toolkit.dispatch import (
    CONTEXT_ENV_KEY_PATTERN,
    DEFAULT_RUNTIME,
    SCHEMA_VERSION_V2,
    AllocationContext,
    DispatchExecutor,
    DispatchRecord,
    DispatchSpec,
    LogicalTaskSpec,
    StagePlanV2,
    admit_tasks,
    logical_task_id_from_parts,
    mint_admission_id,
    read_dispatch_records,
    write_dispatch_records,
)
from experiments.toolkit.jsonio import read_json, read_jsonl, write_json, write_jsonl
from experiments.toolkit.resources import ResourceSpec
from experiments.toolkit.specs import STAGE_MANIFEST, TASKS_JSONL, CompletionSpec


def _logical_task(
    tmp_path: Path,
    *,
    logical_task_id: str = "opaque-a",
    run_id: str = "run-a",
    stage: str = "train",
) -> LogicalTaskSpec:
    """Return a minimal but fully populated logical task rooted at ``tmp_path``."""

    result_dir = tmp_path / "results" / run_id
    return LogicalTaskSpec(
        logical_task_id=logical_task_id,
        stage=stage,
        run_id=run_id,
        # `{python}` is science intent, not a resolved interpreter path; admission
        # is what turns intent into argv.
        command=("{python}", "-m", "tpen.cli", "--run", run_id),
        result_dir=str(result_dir),
        inputs=("configs/he.yaml",),
        outputs=(str(result_dir / "checkpoint"),),
        logs=(str(result_dir / "row_status.json"),),
        params={"seed": 7},
        resources=ResourceSpec(profile="cuda", device="cuda", gpus=1),
        dependencies=(),
        completion=CompletionSpec(policy="status_completed", status_path=str(result_dir / "status.json")),
        resume={"from": "latest"},
        metadata={"note": "unit test row"},
    )


def _plan(tmp_path: Path, tasks: Sequence[LogicalTaskSpec]) -> StagePlanV2:
    """Return a stage plan wrapping ``tasks``."""

    return StagePlanV2(
        study="he-cutover",
        stage="train",
        plan_id="plan-0001",
        results_root=str(tmp_path / "results"),
        tasks=tuple(tasks),
    )


# --------------------------------------------------------------------------
# StagePlanV2
# --------------------------------------------------------------------------


def test_stage_plan_v2_round_trips_through_disk(tmp_path: Path) -> None:
    """A written plan reads back field-for-field identical."""

    plan = _plan(tmp_path, [_logical_task(tmp_path), _logical_task(tmp_path, logical_task_id="opaque-b", run_id="run-b")])
    directory = plan.write(tmp_path / "plan")

    restored = StagePlanV2.read(directory)

    assert restored == plan
    assert restored.tasks[0].command == ("{python}", "-m", "tpen.cli", "--run", "run-a")
    assert restored.tasks[1].logical_task_id == "opaque-b"
    assert restored.schema_version == SCHEMA_VERSION_V2


def test_stage_plan_v2_is_attempt_free_on_disk(tmp_path: Path) -> None:
    """Neither the manifest nor any task row carries an attempt identity."""

    plan = _plan(tmp_path, [_logical_task(tmp_path)])
    directory = plan.write(tmp_path / "plan")

    manifest = read_json(directory / STAGE_MANIFEST)
    rows = read_jsonl(directory / TASKS_JSONL)

    assert "attempt_id" not in manifest
    assert all("attempt_id" not in row for row in rows)
    assert not hasattr(plan, "attempt_id")
    assert not hasattr(plan.tasks[0], "attempt_id")


def test_stage_plan_v2_read_checks_schema_before_parsing_rows(tmp_path: Path) -> None:
    """A wrong schema tag is reported even when the rows are also unparseable.

    The rows below are missing ``logical_task_id``, so parsing them would raise
    ``missing required field: logical_task_id``. Seeing that message instead of
    the schema message would prove the reader parsed rows before checking the
    schema.
    """

    directory = tmp_path / "plan"
    write_json(
        directory / STAGE_MANIFEST,
        {
            "schema_version": "experiment-toolkit/v1",
            "study": "he-cutover",
            "stage": "train",
            "plan_id": "plan-0001",
            "results_root": str(tmp_path / "results"),
            "tasks_path": TASKS_JSONL,
            "n_tasks": 1,
        },
    )
    write_jsonl(directory / TASKS_JSONL, [{"stage": "train", "run_id": "run-a"}])

    with pytest.raises(ValueError) as excinfo:
        StagePlanV2.read(directory)

    message = str(excinfo.value)
    assert "experiment-toolkit/v1" in message
    assert "refusing to parse task rows" in message
    assert "logical_task_id" not in message


def test_stage_plan_v2_rejects_duplicate_logical_task_id(tmp_path: Path) -> None:
    """Two rows sharing one opaque id are rejected at validation."""

    duplicate = _plan(
        tmp_path,
        [_logical_task(tmp_path), _logical_task(tmp_path, run_id="run-b")],
    )

    with pytest.raises(ValueError, match="duplicate logical_task_id"):
        duplicate.validate()


def test_stage_plan_v2_rejects_foreign_schema_version(tmp_path: Path) -> None:
    """An in-memory plan tagged with another schema never validates."""

    plan = dataclasses.replace(
        _plan(tmp_path, [_logical_task(tmp_path)]),
        schema_version="experiment-toolkit/v1",
    )

    with pytest.raises(ValueError, match="does not match 'experiment-toolkit/v2'"):
        plan.validate()


# --------------------------------------------------------------------------
# Logical identity opacity
# --------------------------------------------------------------------------


def test_logical_task_id_is_opaque_and_never_re_derived(tmp_path: Path) -> None:
    """An id unrelated to the convention validates and round-trips unchanged."""

    conventional = logical_task_id_from_parts(stage="train", run_id="run-a", plan_id="plan-0001")
    arbitrary = "01HZZZ-not-derived-from-anything"
    assert arbitrary != conventional

    task = dataclasses.replace(_logical_task(tmp_path), logical_task_id=arbitrary)
    plan = _plan(tmp_path, [task])
    directory = plan.write(tmp_path / "plan")

    assert StagePlanV2.read(directory).tasks[0].logical_task_id == arbitrary


def test_dispatch_module_never_calls_the_convention_helper() -> None:
    """``logical_task_id_from_parts`` has zero call sites inside the module.

    This is a source-level audit of the seam's own text, not a runtime probe of
    any data structure: it walks the module's abstract syntax tree looking for
    call expressions naming the helper. Re-deriving an opaque id is the exact
    coupling this slice exists to remove, so a call site must fail the suite.
    """

    from experiments.toolkit import dispatch as dispatch_module

    tree = ast.parse(inspect.getsource(dispatch_module))
    call_sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "logical_task_id_from_parts")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "logical_task_id_from_parts")
        )
    ]

    assert call_sites == [], (
        "logical_task_id_from_parts is a planner convention and must have no call sites "
        f"in dispatch.py; found {len(call_sites)} at lines "
        f"{[node.lineno for node in call_sites]}"
    )


# --------------------------------------------------------------------------
# admit_tasks
# --------------------------------------------------------------------------


def _admit(tmp_path: Path, tasks, *, admission_id: str = "adm-1", runtimes=None, **kwargs):
    """Admit ``tasks`` from a plan containing them, with one argv per task."""

    plan = kwargs.pop("plan", None) or _plan(tmp_path, tasks)
    runtimes = runtimes or {DEFAULT_RUNTIME: [["python", "-m", "tpen.cli", task.run_id] for task in tasks]}
    return admit_tasks(
        plan,
        tasks,
        admission_id=admission_id,
        argv_by_runtime=runtimes,
        cwd=str(tmp_path),
        environment={"TPEN_SMOKE": "1"},
        **kwargs,
    )


def test_admit_tasks_mints_one_dispatch_per_task_and_runtime(tmp_path: Path) -> None:
    """The cross product is emitted in task order, then runtime order."""

    tasks = [
        _logical_task(tmp_path),
        _logical_task(tmp_path, logical_task_id="opaque-b", run_id="run-b"),
    ]
    dispatches = _admit(
        tmp_path,
        tasks,
        runtimes={
            "tpen-cu126": [["python", "a"], ["python", "b"]],
            "tpen-cpu": [["python", "a-cpu"], ["python", "b-cpu"]],
        },
    )

    assert len(dispatches) == 4
    assert [(d.logical_task_id, d.runtime) for d in dispatches] == [
        ("opaque-a", "tpen-cu126"),
        ("opaque-a", "tpen-cpu"),
        ("opaque-b", "tpen-cu126"),
        ("opaque-b", "tpen-cpu"),
    ]
    assert dispatches[0].argv == ("python", "a")
    assert dispatches[3].argv == ("python", "b-cpu")


def test_admit_tasks_attempt_id_omits_a_single_unnamed_runtime(tmp_path: Path) -> None:
    """One ``DEFAULT_RUNTIME`` adds no information, so it is left out."""

    task = _logical_task(tmp_path)
    (dispatch,) = _admit(tmp_path, [task], admission_id="adm-1")

    assert dispatch.attempt_id == "adm-1:opaque-a"
    assert dispatch.runtime == DEFAULT_RUNTIME


def test_admit_tasks_attempt_id_includes_a_named_runtime(tmp_path: Path) -> None:
    """A named runtime always appears, even when it is the only one."""

    task = _logical_task(tmp_path)
    (dispatch,) = _admit(
        tmp_path,
        [task],
        admission_id="adm-1",
        runtimes={"tpen-cu126": [["python", "a"]]},
    )

    assert dispatch.attempt_id == "adm-1:opaque-a:tpen-cu126"


def test_admit_tasks_attempt_id_disambiguates_multiple_runtimes(tmp_path: Path) -> None:
    """Two runtimes for one task must not collide on attempt identity."""

    task = _logical_task(tmp_path)
    dispatches = _admit(
        tmp_path,
        [task],
        admission_id="adm-1",
        runtimes={"tpen-cu126": [["python", "a"]], "tpen-cpu": [["python", "b"]]},
    )

    attempt_ids = [dispatch.attempt_id for dispatch in dispatches]
    assert attempt_ids == ["adm-1:opaque-a:tpen-cu126", "adm-1:opaque-a:tpen-cpu"]
    assert len(set(attempt_ids)) == 2


def test_admit_tasks_is_deterministic_for_one_admission_id(tmp_path: Path) -> None:
    """Re-admitting under the same admission id reproduces the same attempts."""

    task = _logical_task(tmp_path)
    first = _admit(tmp_path, [task], admission_id="adm-1")
    second = _admit(tmp_path, [task], admission_id="adm-1")

    assert [d.attempt_id for d in first] == [d.attempt_id for d in second]


def test_retry_uses_a_fresh_admission_id_and_fresh_attempts(tmp_path: Path) -> None:
    """A retry is a new admission, so every attempt identity changes."""

    task = _logical_task(tmp_path)
    (first,) = _admit(tmp_path, [task], admission_id="adm-1")
    (retry,) = _admit(tmp_path, [task], admission_id="adm-2")

    assert first.logical_task_id == retry.logical_task_id
    assert first.admission_id != retry.admission_id
    assert first.attempt_id != retry.attempt_id


def test_admission_ref_is_provenance_not_identity(tmp_path: Path) -> None:
    """Sharing an ``admission_ref`` does not make two admissions the same."""

    task = _logical_task(tmp_path)
    (first,) = _admit(tmp_path, [task], admission_id="adm-1", admission_ref="slurm-999")
    (second,) = _admit(tmp_path, [task], admission_id="adm-2", admission_ref="slurm-999")

    assert first.metadata["admission_ref"] == second.metadata["admission_ref"] == "slurm-999"
    assert first.attempt_id != second.attempt_id


def test_admit_tasks_rejects_a_task_outside_the_plan(tmp_path: Path) -> None:
    """Admission may only admit rows the plan actually contains."""

    planned = _logical_task(tmp_path)
    intruder = _logical_task(tmp_path, logical_task_id="opaque-intruder", run_id="run-x")
    plan = _plan(tmp_path, [planned])

    with pytest.raises(ValueError, match="is not in stage plan"):
        admit_tasks(
            plan,
            [intruder],
            admission_id="adm-1",
            argv_by_runtime={DEFAULT_RUNTIME: [["python", "x"]]},
            cwd=str(tmp_path),
            environment={},
        )


def test_admit_tasks_rejects_an_altered_copy_of_a_planned_task(tmp_path: Path) -> None:
    """Admission requires the actual plan row, not merely its opaque id."""

    planned = _logical_task(tmp_path)
    altered = dataclasses.replace(planned, result_dir=str(tmp_path / "other-results"))
    plan = _plan(tmp_path, [planned])

    with pytest.raises(ValueError, match="does not match its stage-plan row"):
        admit_tasks(
            plan,
            [altered],
            admission_id="adm-1",
            argv_by_runtime={DEFAULT_RUNTIME: [["python", "x"]]},
            cwd=str(tmp_path),
            environment={},
        )


def test_admit_tasks_rejects_a_duplicate_admitted_task(tmp_path: Path) -> None:
    """One admission may not admit the same logical row twice."""

    task = _logical_task(tmp_path)
    plan = _plan(tmp_path, [task])

    with pytest.raises(ValueError, match="duplicate logical_task_id in admitted tasks"):
        admit_tasks(
            plan,
            [task, task],
            admission_id="adm-1",
            argv_by_runtime={DEFAULT_RUNTIME: [["python", "a"], ["python", "a"]]},
            cwd=str(tmp_path),
            environment={},
        )


def test_admit_tasks_rejects_misaligned_argv(tmp_path: Path) -> None:
    """Each runtime must supply exactly one argv per admitted task."""

    tasks = [
        _logical_task(tmp_path),
        _logical_task(tmp_path, logical_task_id="opaque-b", run_id="run-b"),
    ]
    plan = _plan(tmp_path, tasks)

    with pytest.raises(ValueError, match="does not match tasks length 2"):
        admit_tasks(
            plan,
            tasks,
            admission_id="adm-1",
            argv_by_runtime={"tpen-cu126": [["python", "a"]]},
            cwd=str(tmp_path),
            environment={},
        )


def test_admit_tasks_carries_planned_completion_and_status_path(tmp_path: Path) -> None:
    """Admission copies the planned predicate and row status path verbatim."""

    task = _logical_task(tmp_path)
    (dispatch,) = _admit(tmp_path, [task])

    assert dispatch.completion == task.completion
    assert dispatch.row_status_path == task.logs[0]
    assert dispatch.result_dir == task.result_dir
    assert dispatch.environment == {"TPEN_SMOKE": "1"}
    assert dispatch.metadata["plan_id"] == "plan-0001"


# --------------------------------------------------------------------------
# DispatchSpec
# --------------------------------------------------------------------------


def test_dispatch_spec_is_frozen_and_argv_is_an_immutable_tuple(tmp_path: Path) -> None:
    """A minted dispatch cannot be edited in place, and neither can its argv."""

    (dispatch,) = _admit(tmp_path, [_logical_task(tmp_path)])

    assert isinstance(dispatch.argv, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        dispatch.argv = ("python", "something-else")  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        dispatch.attempt_id = "forged"  # type: ignore[misc]
    with pytest.raises(TypeError):
        dispatch.argv[0] = "sh"  # type: ignore[index]


def test_dispatch_spec_validation_is_non_empty_checks_only(tmp_path: Path) -> None:
    """An opaque, non-conventional attempt id is accepted; a blank one is not."""

    (dispatch,) = _admit(tmp_path, [_logical_task(tmp_path)])

    # Nothing re-derives attempt_id from admission_id + logical_task_id.
    dataclasses.replace(dispatch, attempt_id="totally-unrelated-attempt").validate()

    with pytest.raises(ValueError, match="attempt_id must be a non-empty string"):
        dataclasses.replace(dispatch, attempt_id="  ").validate()
    with pytest.raises(ValueError, match="argv must be non-empty"):
        dataclasses.replace(dispatch, argv=()).validate()


# --------------------------------------------------------------------------
# DispatchRecord
# --------------------------------------------------------------------------


def test_dispatch_record_requires_submitted_command_to_equal_argv(tmp_path: Path) -> None:
    """Executors may not mutate argv, and the receipt enforces it."""

    (dispatch,) = _admit(tmp_path, [_logical_task(tmp_path)])

    record = DispatchRecord.accepted(
        dispatch,
        backend="parsl",
        launcher_job_id="job-1",
        submitted_command=dispatch.argv,
    )
    assert record.submitted_command == dispatch.argv

    with pytest.raises(ValueError, match="does not equal dispatch argv"):
        DispatchRecord.accepted(
            dispatch,
            backend="parsl",
            launcher_job_id="job-1",
            submitted_command=tuple(dispatch.argv) + ("--sneaky-extra-flag",),
        )


def test_dispatch_records_round_trip_through_jsonl(tmp_path: Path) -> None:
    """Records survive a write/read cycle and land in the expected file."""

    tasks = [
        _logical_task(tmp_path),
        _logical_task(tmp_path, logical_task_id="opaque-b", run_id="run-b"),
    ]
    dispatches = _admit(tmp_path, tasks)
    records = tuple(
        DispatchRecord.accepted(
            dispatch,
            backend="parsl",
            launcher_job_id=f"job-{index}",
            submitted_command=dispatch.argv,
            metadata={"pool": "attach"},
        )
        for index, dispatch in enumerate(dispatches)
    )

    path = write_dispatch_records(tmp_path / "out", records)

    assert path.name == "dispatch_records.jsonl"
    assert read_dispatch_records(tmp_path / "out") == records
    first_row = json.loads(path.read_text().splitlines()[0])
    assert first_row["submitted_command"] == list(dispatches[0].argv)
    assert first_row["attempt_id"] == dispatches[0].attempt_id


# --------------------------------------------------------------------------
# mint_admission_id
# --------------------------------------------------------------------------


def test_mint_admission_id_is_fresh_every_call() -> None:
    """Two mints never collide; a retry therefore always gets a new scope."""

    minted = {mint_admission_id("he-cutover") for _ in range(8)}

    assert len(minted) == 8


def test_mint_admission_id_sanitizes_the_label() -> None:
    """Unsafe label characters become ``-`` so the id is file-name safe."""

    minted = mint_admission_id("he cutover/smoke:1")

    assert minted.startswith("he-cutover-smoke-1-")
    assert all(character.isalnum() or character in "._-" for character in minted)


def test_mint_admission_id_rejects_a_blank_label() -> None:
    """A blank label would produce an unattributable admission scope."""

    with pytest.raises(ValueError, match="label must be a non-empty string"):
        mint_admission_id("   ")


# --------------------------------------------------------------------------
# AllocationContext
# --------------------------------------------------------------------------


def test_allocation_context_resolves_deadline_from_named_variable() -> None:
    """The named facility variable is consulted before the Slurm fallback."""

    context = AllocationContext(
        allocation_id="alloc-1",
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=("0", "1"),
        deadline_env_var="PBS_JOB_END_TIME",
        deadline_guard_min=5,
    ).validate()

    resolved = context.deadline_unix(
        {"PBS_JOB_END_TIME": "1700000000", "SLURM_JOB_END_TIME": "1600000000"}
    )

    assert resolved == 1700000000.0


def test_allocation_context_explicit_deadline_wins() -> None:
    """An explicit deadline outranks every environment source."""

    context = AllocationContext(
        allocation_id="alloc-1",
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=("0",),
        deadline="1800000000",
        deadline_env_var="PBS_JOB_END_TIME",
    ).validate()

    assert context.deadline_unix({"PBS_JOB_END_TIME": "1700000000"}) == 1800000000.0


def test_allocation_context_empty_visibility_values_inherits_and_round_trips() -> None:
    """An empty binding leaves visibility ownership with the scheduler."""

    context = AllocationContext(
        allocation_id="alloc-1",
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=(),
    ).validate()

    serialized = json.loads(json.dumps(context.to_dict()))
    round_tripped = AllocationContext(
        **{**serialized, "visibility_values": tuple(serialized["visibility_values"])}
    ).validate()

    assert serialized["visibility_values"] == []
    assert round_tripped == context


def test_allocation_context_defaults_nodes_per_block_to_none_and_omits_it() -> None:
    """The optional node requirement is absent from legacy payloads."""

    context = AllocationContext(
        allocation_id="alloc-1",
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=(),
    ).validate()

    assert context.nodes_per_block is None
    assert "nodes_per_block" not in context.to_dict()


def test_allocation_context_nodes_per_block_round_trips() -> None:
    """A declared node requirement survives JSON-compatible serialization."""

    context = AllocationContext(
        allocation_id="alloc-1",
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=(),
        nodes_per_block=2,
    ).validate()

    serialized = json.loads(json.dumps(context.to_dict()))
    round_tripped = AllocationContext(
        **{**serialized, "visibility_values": tuple(serialized["visibility_values"])}
    ).validate()

    assert serialized["nodes_per_block"] == 2
    assert round_tripped == context


@pytest.mark.parametrize("nodes_per_block", [0, -1])
def test_allocation_context_rejects_non_positive_nodes_per_block(nodes_per_block: int) -> None:
    """Node blocks must request at least one node."""

    with pytest.raises(ValueError, match="nodes_per_block"):
        AllocationContext(
            allocation_id="alloc-1",
            visibility_variable="CUDA_VISIBLE_DEVICES",
            visibility_values=(),
            nodes_per_block=nodes_per_block,
        ).validate()


def test_allocation_context_legacy_serialization_is_byte_identical() -> None:
    """Legacy serialization remains an explicit, byte-for-byte payload."""

    context = AllocationContext(
        allocation_id="alloc-1",
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=(),
    ).validate()

    actual = json.dumps(context.to_dict(), separators=(",", ":")).encode()
    expected = (
        b'{"allocation_id":"alloc-1","visibility_variable":"CUDA_VISIBLE_DEVICES",'
        b'"visibility_values":[],"run_root":null,"deadline":null,'
        b'"deadline_env_var":null,"deadline_guard_min":1,"environment":{}}'
    )

    assert actual == expected


def test_allocation_context_rejects_blank_visibility_entry() -> None:
    """Worker-specific bindings cannot contain an empty visibility value."""

    with pytest.raises(ValueError, match="visibility_values"):
        AllocationContext(
            allocation_id="alloc-1",
            visibility_variable="CUDA_VISIBLE_DEVICES",
            visibility_values=("0", " "),
        ).validate()


def test_allocation_context_rejects_undeclared_environment_key() -> None:
    """Context declarations cannot widen the executor observation namespace."""

    with pytest.raises(ValueError, match="context-environment-key"):
        AllocationContext(
            allocation_id="alloc-1",
            visibility_variable="CUDA_VISIBLE_DEVICES",
            visibility_values=(),
            environment={"P10_CONTEXT_VALUE": "unsafe"},
        ).validate()


def test_toolkit_and_executor_context_key_patterns_match() -> None:
    """The decoupled toolkit and executor must enforce the same namespace."""

    from tpen_executor.runners import CONTEXT_ENV_KEY_PATTERN as executor_pattern

    assert CONTEXT_ENV_KEY_PATTERN == executor_pattern


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allocation_id": " "}, "allocation_id"),
        ({"visibility_variable": " "}, "visibility_variable"),
        ({"deadline_guard_min": -1}, "deadline_guard_min"),
    ],
)
def test_allocation_context_keeps_other_validation_rules(
    kwargs: dict[str, str | int], message: str
) -> None:
    """The inherit correction does not relax unrelated attach validation."""

    with pytest.raises(ValueError, match=message):
        AllocationContext(
            **{
                "allocation_id": "alloc-1",
                "visibility_variable": "CUDA_VISIBLE_DEVICES",
                "visibility_values": ("0",),
                **kwargs,
            }
        ).validate()


# --------------------------------------------------------------------------
# DispatchExecutor protocol
# --------------------------------------------------------------------------


class _RecordingExecutor:
    """Minimal conforming executor: it submits argv verbatim and reports it."""

    def dispatch(
        self,
        dispatches: Sequence[DispatchSpec],
        *,
        context: AllocationContext | None,
    ) -> Sequence[DispatchRecord]:
        """Accept every dispatch without planning, retrying, or editing argv."""

        self.context = context
        return tuple(
            DispatchRecord.accepted(
                dispatch,
                backend="recording",
                launcher_job_id=f"job-{index}",
                submitted_command=dispatch.argv,
            )
            for index, dispatch in enumerate(dispatches)
        )


def test_dispatch_executor_protocol_accepts_a_conforming_backend(tmp_path: Path) -> None:
    """A backend implementing only ``dispatch`` satisfies the seam contract."""

    executor = _RecordingExecutor()
    assert isinstance(executor, DispatchExecutor)

    dispatches = _admit(tmp_path, [_logical_task(tmp_path)])
    context = AllocationContext(
        allocation_id="alloc-1",
        visibility_variable="CUDA_VISIBLE_DEVICES",
        visibility_values=("0",),
    ).validate()

    records = executor.dispatch(dispatches, context=context)

    assert [record.attempt_id for record in records] == [d.attempt_id for d in dispatches]
    assert records[0].submitted_command == dispatches[0].argv
    assert executor.context is context


def test_dispatch_executor_protocol_rejects_a_non_conforming_object() -> None:
    """An object without ``dispatch`` is not a ``DispatchExecutor``."""

    class _NotAnExecutor:
        def submit(self) -> None:
            """Wrong method name on purpose."""

    assert not isinstance(_NotAnExecutor(), DispatchExecutor)
