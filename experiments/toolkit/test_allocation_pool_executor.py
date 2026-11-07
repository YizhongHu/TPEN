"""Measured tests for the scheduler-neutral allocation-pool executor."""

from __future__ import annotations

import json
import multiprocessing
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Sequence
from types import SimpleNamespace

import pytest

from experiments.toolkit import (
    AllocationPoolExecutor,
    CompletionSpec,
    ResourceSpec,
    StagePlan,
    SubmissionRequest,
    TaskSpec,
    task_id_from_parts,
)
import experiments.toolkit.executors as executors_module
from experiments.toolkit.jsonio import read_json
from experiments.toolkit.task_state import claim_row_for_pass, pass_claim_path


_WORKER_SCRIPT = r"""
import json
import os
import sys
import time
from pathlib import Path

mode = sys.argv[1]

if mode == "run":
    trace_path = Path(sys.argv[2])
    completion_path = Path(sys.argv[3])
    delay = float(sys.argv[4])
    returncode = int(sys.argv[5])
    launcher_status_path = Path(sys.argv[6]) if len(sys.argv) > 6 else None
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    observed_status = None
    if launcher_status_path is not None:
        observed_status = json.loads(launcher_status_path.read_text())["status"]
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "start",
                    "time": time.time(),
                    "launcher_status": observed_status,
                }
            )
            + "\n"
        )
    time.sleep(delay)
    if returncode == 0:
        completion_path.touch()
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"kind": "end", "time": time.time()}) + "\n")
    raise SystemExit(returncode)

if mode == "visibility":
    ready_dir = Path(sys.argv[2])
    marker = ready_dir / sys.argv[3]
    output_path = Path(sys.argv[4])
    variable = sys.argv[5]
    ready_dir.mkdir(parents=True, exist_ok=True)
    marker.touch()
    timeout = time.time() + 10.0
    while len(tuple(ready_dir.glob("*.ready"))) < 2:
        if time.time() >= timeout:
            raise SystemExit(9)
        time.sleep(0.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(os.environ.get(variable, "<missing>"))
    raise SystemExit(0)

raise SystemExit(10)
"""


def _write_worker_script(tmp_path: Path) -> Path:
    script = tmp_path / "pool_worker.py"
    script.write_text(_WORKER_SCRIPT)
    return script


def _task(
    tmp_path: Path,
    run_id: str,
    command: Sequence[str],
    *,
    completion_path: Path | None = None,
) -> TaskSpec:
    stage = "01_train"
    attempt_id = "A1"
    completion = (
        CompletionSpec(policy="file_exists", output_paths=(str(completion_path),))
        if completion_path is not None
        else CompletionSpec(policy="none")
    )
    return TaskSpec(
        task_id=task_id_from_parts(stage=stage, run_id=run_id, attempt_id=attempt_id),
        stage=stage,
        attempt_id=attempt_id,
        run_id=run_id,
        command=tuple(str(part) for part in command),
        result_dir=str(tmp_path / "results" / stage / run_id / attempt_id),
        logs=(str(tmp_path / "launcher" / f"{run_id}.json"),),
        resources=ResourceSpec(profile="gpu", device="gpu", threads=1, gpus=1),
        completion=completion,
    )


def _plan(tmp_path: Path, tasks: Sequence[TaskSpec]) -> StagePlan:
    return StagePlan(
        study="allocation-pool-test",
        stage="01_train",
        attempt_id="A1",
        results_root=str(tmp_path / "results"),
        tasks=tuple(tasks),
    )


def _request(tasks: Sequence[TaskSpec]) -> SubmissionRequest:
    return SubmissionRequest(
        command_sets={"allocation": [task.command for task in tasks]},
        submitted_commands=[task.command for task in tasks],
    )


def _trace(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _status_task(
    tmp_path: Path,
    run_id: str,
    command: Sequence[str],
    status_path: Path,
) -> TaskSpec:
    task = _task(tmp_path, run_id, command)
    return TaskSpec(
        task_id=task.task_id,
        stage=task.stage,
        attempt_id=task.attempt_id,
        run_id=task.run_id,
        command=task.command,
        result_dir=task.result_dir,
        logs=task.logs,
        resources=task.resources,
        completion=CompletionSpec(policy="status_completed", status_path=str(status_path)),
    )


def _single_worker_executor(tmp_path: Path, pass_id: str) -> AllocationPoolExecutor:
    return AllocationPoolExecutor(
        pass_id=pass_id,
        n_workers=1,
        visibility_variable="TPEN_CTX_VISIBLE_DEVICE",
        visibility_values=("worker-0",),
        run_root=tmp_path / "pool",
        deadline_guard_min=0,
    )


def _attempt_statuses(tmp_path: Path) -> list[dict[str, object]]:
    return [
        read_json(path)
        for path in sorted((tmp_path / "pool" / "_allocation_pool").glob("*/attempt*/status.json"))
    ]


def test_zero_exit_requires_completed_status(tmp_path: Path) -> None:
    status_path = tmp_path / "run" / "status.json"
    command = (
        sys.executable,
        "-c",
        "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
        "p.parent.mkdir(parents=True); p.write_text(json.dumps({'status':'completed'}))",
        str(status_path),
    )
    task = _status_task(tmp_path, "completed", command, status_path)

    records = _single_worker_executor(tmp_path, "completed-pass").submit(
        _plan(tmp_path, (task,)), (task,), _request((task,))
    )

    assert len(records) == 1
    assert read_json(task.logs[0])["status"] == "success"
    assert _attempt_statuses(tmp_path)[0]["status"] == "success"


@pytest.mark.parametrize(
    ("status_payload", "run_id"),
    [
        (None, "missing"),
        ('{"status":"failed"}', "not-completed"),
    ],
)
def test_zero_exit_without_completed_status_fails_with_receipts(
    tmp_path: Path,
    status_payload: str | None,
    run_id: str,
) -> None:
    status_path = tmp_path / run_id / "status.json"
    if status_payload is None:
        command = (sys.executable, "-c", "raise SystemExit(0)")
    else:
        command = (
            sys.executable,
            "-c",
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "p.parent.mkdir(parents=True); p.write_text(sys.argv[2])",
            str(status_path),
            status_payload,
        )
    task = _status_task(tmp_path, run_id, command, status_path)

    with pytest.raises(RuntimeError, match="completion predicate 'status_completed'.*not satisfied"):
        _single_worker_executor(tmp_path, "pass-a").submit(
            _plan(tmp_path, (task,)), (task,), _request((task,))
        )

    attempt = _attempt_statuses(tmp_path)[0]
    launcher = read_json(task.logs[0])
    assert attempt["status"] == launcher["status"] == "failed"
    assert attempt["returncode"] == launcher["returncode"] == 0
    assert "not satisfied" in str(attempt["completion_error"])
    assert attempt["completion_error"] == launcher["completion_error"]


def test_zero_exit_missing_status_is_retryable_on_new_pass(tmp_path: Path) -> None:
    status_path = tmp_path / "retry" / "status.json"
    task = _status_task(
        tmp_path,
        "retry",
        (sys.executable, "-c", "raise SystemExit(0)"),
        status_path,
    )
    plan = _plan(tmp_path, (task,))
    request = _request((task,))

    for pass_id in ("pass-a", "pass-b"):
        with pytest.raises(RuntimeError, match="completion predicate"):
            _single_worker_executor(tmp_path, pass_id).submit(plan, (task,), request)

    assert len(_attempt_statuses(tmp_path)) == 2


def test_completion_failure_drains_unrelated_pending_tasks(tmp_path: Path) -> None:
    later_marker = tmp_path / "later-ran"
    missing = _status_task(
        tmp_path,
        "missing-first",
        (sys.executable, "-c", "raise SystemExit(0)"),
        tmp_path / "missing-first" / "status.json",
    )
    later_status = tmp_path / "later" / "status.json"
    later = _status_task(
        tmp_path,
        "later",
        (
            sys.executable,
            "-c",
            "import json,pathlib,sys; marker=pathlib.Path(sys.argv[1]); "
            "status=pathlib.Path(sys.argv[2]); marker.touch(); "
            "status.parent.mkdir(parents=True); "
            "status.write_text(json.dumps({'status':'completed'}))",
            str(later_marker),
            str(later_status),
        ),
        later_status,
    )
    tasks = (missing, later)

    with pytest.raises(RuntimeError, match="missing-first"):
        _single_worker_executor(tmp_path, "drain-pass").submit(
            _plan(tmp_path, tasks), tasks, _request(tasks)
        )

    assert later_marker.is_file()
    assert read_json(missing.logs[0])["status"] == "failed"
    assert read_json(later.logs[0])["status"] == "success"
    assert {status["status"] for status in _attempt_statuses(tmp_path)} == {
        "failed",
        "success",
    }


def test_fatal_worker_error_wins_over_completion_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    completion_ready = threading.Event()
    stop_observed = threading.Event()
    real_event = threading.Event

    class TrackingStopEvent:
        def __init__(self) -> None:
            self._event = real_event()

        def is_set(self) -> bool:
            return self._event.is_set()

        def set(self) -> None:
            self._event.set()
            stop_observed.set()

    monkeypatch.setattr(
        executors_module,
        "threading",
        SimpleNamespace(
            Event=TrackingStopEvent,
            Thread=threading.Thread,
            Lock=threading.Lock,
        ),
    )
    original_execute = AllocationPoolExecutor._execute_task

    def controlled_execute(self, **kwargs):
        task = kwargs["task"]
        if task.run_id == "completion":
            try:
                return original_execute(self, **kwargs)
            except RuntimeError:
                completion_ready.set()
                assert stop_observed.wait(timeout=5)
                raise
        record = original_execute(self, **kwargs)
        if task.run_id == "fatal":
            assert completion_ready.wait(timeout=5)
            raise ValueError("fatal worker error")
        return record

    monkeypatch.setattr(AllocationPoolExecutor, "_execute_task", controlled_execute)
    completion = _status_task(
        tmp_path,
        "completion",
        (sys.executable, "-c", "raise SystemExit(0)"),
        tmp_path / "completion" / "status.json",
    )
    fatal = _task(
        tmp_path,
        "fatal",
        (sys.executable, "-c", "raise SystemExit(9)"),
    )
    queued_marker = tmp_path / "queued-ran"
    queued = _task(
        tmp_path,
        "queued",
        (
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).touch()",
            str(queued_marker),
        ),
    )
    tasks = (completion, fatal, queued)
    executor = AllocationPoolExecutor(
        pass_id="fatal-priority-pass",
        n_workers=2,
        visibility_variable="TPEN_CTX_VISIBLE_DEVICE",
        visibility_values=("worker-0", "worker-1"),
        run_root=tmp_path / "pool",
        deadline_guard_min=0,
    )

    with pytest.raises(ValueError, match="fatal worker error"):
        executor.submit(_plan(tmp_path, tasks), tasks, _request(tasks))

    assert read_json(completion.logs[0])["status"] == "failed"
    assert read_json(completion.logs[0])["completion_error"]
    assert read_json(fatal.logs[0])["status"] == "failed"
    assert read_json(fatal.logs[0])["returncode"] == 9
    assert not queued_marker.exists()
    assert not pass_claim_path(
        tmp_path / "pool",
        "fatal-priority-pass",
        queued.task_id,
    ).exists()


def test_nonzero_exit_remains_failed(tmp_path: Path) -> None:
    status_path = tmp_path / "nonzero" / "status.json"
    task = _status_task(
        tmp_path,
        "nonzero",
        (sys.executable, "-c", "raise SystemExit(7)"),
        status_path,
    )

    records = _single_worker_executor(tmp_path, "nonzero-pass").submit(
        _plan(tmp_path, (task,)), (task,), _request((task,))
    )

    assert len(records) == 1
    attempt = _attempt_statuses(tmp_path)[0]
    assert attempt["status"] == "failed"
    assert attempt["returncode"] == 7
    assert "completion_error" not in attempt


def test_completed_task_is_skipped_before_claiming(tmp_path: Path) -> None:
    status_path = tmp_path / "already-completed" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text('{"status":"completed"}')
    task = _status_task(
        tmp_path,
        "already-completed",
        ("/bin/false",),
        status_path,
    )

    records = _single_worker_executor(tmp_path, "skip-pass").submit(
        _plan(tmp_path, (task,)), (task,), _request((task,))
    )

    assert records == ()
    assert not pass_claim_path(tmp_path / "pool", "skip-pass", task.task_id).exists()
    assert not Path(task.logs[0]).exists()


def _run_pool_process(
    plan: StagePlan,
    tasks: tuple[TaskSpec, ...],
    request: SubmissionRequest,
    run_root: str,
) -> None:
    """Run one executor process for the cross-process claim race."""

    AllocationPoolExecutor(
        pass_id="process-race",
        n_workers=1,
        visibility_variable="TPEN_CTX_VISIBLE_DEVICE",
        visibility_values=("worker-0",),
        run_root=run_root,
        deadline_guard_min=0,
    ).submit(plan, tasks, request)


def test_pool_claims_once_across_real_processes(tmp_path: Path) -> None:
    marker = tmp_path / "marker.txt"
    task = _task(
        tmp_path,
        "run-a",
        (
            sys.executable,
            "-c",
            "from pathlib import Path; import time; p=Path(__import__('sys').argv[1]); "
            "p.open('a').write('claimed\\n'); time.sleep(0.1)",
            str(marker),
        ),
    )
    plan = _plan(tmp_path, (task,))
    request = _request((task,))
    run_root = tmp_path / "pool"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_run_pool_process,
            args=(plan, (task,), request, str(run_root)),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0

    assert marker.read_text(encoding="utf-8") == "claimed\n"
    assert pass_claim_path(run_root, "process-race", task.task_id).is_dir()


def test_pool_refills_workers_and_retries_only_failed_rows(tmp_path: Path) -> None:
    script = _write_worker_script(tmp_path)
    definitions = (
        ("slow", 0.8, 0),
        ("broken", 0.05, 3),
        ("refill", 0.05, 0),
        ("last", 0.05, 0),
        ("tail", 0.05, 0),
    )
    traces = {name: tmp_path / "traces" / f"{name}.jsonl" for name, _, _ in definitions}
    completions = {name: tmp_path / "completed" / name for name, _, _ in definitions}
    tasks = tuple(
        _task(
            tmp_path,
            name,
            (
                sys.executable,
                str(script),
                "run",
                str(traces[name]),
                str(completions[name]),
                str(delay),
                str(returncode),
                str(tmp_path / "launcher" / f"{name}.json"),
            ),
            completion_path=completions[name],
        )
        for name, delay, returncode in definitions
    )
    plan = _plan(tmp_path, tasks)
    run_root = tmp_path / "pool"

    first = AllocationPoolExecutor(
        pass_id="pass-a",
        n_workers=2,
        visibility_variable="TPEN_CTX_VISIBLE_DEVICE",
        visibility_values=("worker-0", "worker-1"),
        run_root=run_root,
        deadline_guard_min=0,
    ).submit(plan, tasks, _request(tasks))

    assert len(first) == len(tasks)
    assert all(
        len([row for row in _trace(traces[name]) if row["kind"] == "start"]) == 1
        for name, _, _ in definitions
    )
    assert all(
        next(row for row in _trace(traces[name]) if row["kind"] == "start")[
            "launcher_status"
        ]
        == "running"
        for name, _, _ in definitions
    )
    slow_end = next(row["time"] for row in _trace(traces["slow"]) if row["kind"] == "end")
    refill_start = next(
        row["time"] for row in _trace(traces["refill"]) if row["kind"] == "start"
    )
    assert float(refill_start) < float(slow_end)
    assert all(record.backend == "allocation_pool" for record in first)
    assert all(
        record.claim_path == str(pass_claim_path(run_root, "pass-a", record.task_id))
        for record in first
    )
    broken_first = next(record for record in first if record.run_id == "broken")
    assert read_json(str(broken_first.status_path))["returncode"] == 3
    assert read_json(str(broken_first.status_path))["status"] == "failed"
    for task in tasks:
        launcher_status = read_json(task.logs[0])
        assert launcher_status["status"] == ("failed" if task.run_id == "broken" else "success")
        assert launcher_status["command"] == shlex.join(task.command)

    second = AllocationPoolExecutor(
        pass_id="pass-b",
        n_workers=2,
        visibility_variable="TPEN_CTX_VISIBLE_DEVICE",
        visibility_values=("worker-0", "worker-1"),
        run_root=run_root,
        deadline_guard_min=0,
    ).submit(plan, tasks, _request(tasks))

    assert [record.run_id for record in second] == ["broken"]
    assert len([row for row in _trace(traces["broken"]) if row["kind"] == "start"]) == 2
    for name in ("slow", "refill", "last", "tail"):
        assert len([row for row in _trace(traces[name]) if row["kind"] == "start"]) == 1
    broken_attempt_dir = Path(str(second[0].metadata["attempt_dir"]))
    assert sorted(path.name for path in broken_attempt_dir.parent.glob("attempt*")) == [
        "attempt1",
        "attempt2",
    ]


@pytest.mark.parametrize("variable", ["CUDA_VISIBLE_DEVICES", "ZE_AFFINITY_MASK"])
def test_pool_binds_distinct_worker_visibility_values(tmp_path: Path, variable: str) -> None:
    script = _write_worker_script(tmp_path)
    ready_dir = tmp_path / "ready"
    outputs = {name: tmp_path / "visibility" / name for name in ("run-a", "run-b")}
    tasks = tuple(
        _task(
            tmp_path,
            name,
            (
                sys.executable,
                str(script),
                "visibility",
                str(ready_dir),
                f"{name}.ready",
                str(outputs[name]),
                variable,
            ),
        )
        for name in outputs
    )

    records = AllocationPoolExecutor(
        pass_id=f"pass-{variable}",
        n_workers=2,
        visibility_variable=variable,
        visibility_values=("device-a", "device-b"),
        run_root=tmp_path / "pool",
        deadline_guard_min=0,
        environment={variable: "must-be-overridden"},
    ).submit(_plan(tmp_path, tasks), tasks, _request(tasks))

    assert len(records) == 2
    assert {path.read_text() for path in outputs.values()} == {"device-a", "device-b"}
    assert {record.metadata["visibility_variable"] for record in records} == {variable}


def test_deadline_stops_new_claims_without_killing_running_tasks(tmp_path: Path) -> None:
    script = _write_worker_script(tmp_path)
    tasks = tuple(
        _task(
            tmp_path,
            f"run-{index}",
            (
                sys.executable,
                str(script),
                "run",
                str(tmp_path / "traces" / f"run-{index}.jsonl"),
                str(tmp_path / "completed" / f"run-{index}"),
                "5.0",
                "0",
                str(tmp_path / "launcher" / f"run-{index}.json"),
            ),
            completion_path=tmp_path / "completed" / f"run-{index}",
        )
        for index in range(4)
    )
    run_root = tmp_path / "pool"

    records = AllocationPoolExecutor(
        pass_id="deadline-pass",
        n_workers=2,
        visibility_variable="TPEN_CTX_VISIBLE_DEVICE",
        visibility_values=("worker-0", "worker-1"),
        run_root=run_root,
        deadline=time.time() + 64.0,
        deadline_guard_min=1,
    ).submit(_plan(tmp_path, tasks), tasks, _request(tasks))

    assert [record.run_id for record in records] == ["run-0", "run-1"]
    assert all((tmp_path / "completed" / f"run-{index}").is_file() for index in (0, 1))
    assert all(not (tmp_path / "traces" / f"run-{index}.jsonl").exists() for index in (2, 3))
    assert all(
        not pass_claim_path(run_root, "deadline-pass", task.task_id).exists()
        for task in tasks[2:]
    )


def test_pool_runs_task_command_verbatim_as_argv(tmp_path: Path) -> None:
    task = _task(tmp_path, "run-a", ("/usr/bin/true",))
    request = SubmissionRequest(
        command_sets={"ignored": [("/usr/bin/false",)]},
        submitted_commands=[("/usr/bin/false",)],
    )

    records = AllocationPoolExecutor(
        pass_id="verbatim-pass",
        n_workers=1,
        visibility_variable="TPEN_CTX_VISIBLE_DEVICE",
        visibility_values=("worker-0",),
        run_root=tmp_path / "pool",
        deadline_guard_min=0,
    ).submit(_plan(tmp_path, (task,)), (task,), request)

    assert records[0].submitted_command == ("/usr/bin/true",)
    assert read_json(str(records[0].status_path))["returncode"] == 0


def test_pool_does_not_execute_a_row_already_claimed_in_the_pass(tmp_path: Path) -> None:
    task = _task(tmp_path, "run-a", ("/bin/false",))
    run_root = tmp_path / "pool"
    assert claim_row_for_pass(run_root, "shared-pass", task.task_id) is True

    records = AllocationPoolExecutor(
        pass_id="shared-pass",
        n_workers=1,
        visibility_variable="TPEN_CTX_VISIBLE_DEVICE",
        visibility_values=("worker-0",),
        run_root=run_root,
        deadline_guard_min=0,
    ).submit(_plan(tmp_path, (task,)), (task,), _request((task,)))

    assert records == ()
    assert not Path(task.logs[0]).exists()


@pytest.mark.parametrize(
    ("n_workers", "visibility_values", "deadline_guard_min", "message"),
    [
        (0, (), 0, "n_workers must be positive"),
        (2, ("only-one",), 0, "length must equal n_workers"),
        (2, "ab", 0, "must be a sequence, not a string"),
        (1, ("worker-0",), -1, "deadline_guard_min must be non-negative"),
    ],
)
def test_pool_rejects_invalid_worker_configuration(
    tmp_path: Path,
    n_workers: int,
    visibility_values: Sequence[str],
    deadline_guard_min: int,
    message: str,
) -> None:
    task = _task(tmp_path, "run-a", ("/bin/true",))
    executor = AllocationPoolExecutor(
        pass_id="invalid-pass",
        n_workers=n_workers,
        visibility_variable="TPEN_CTX_VISIBLE_DEVICE",
        visibility_values=visibility_values,
        run_root=tmp_path / "pool",
        deadline_guard_min=deadline_guard_min,
    )

    with pytest.raises(ValueError, match=message):
        executor.submit(_plan(tmp_path, (task,)), (task,), _request((task,)))
