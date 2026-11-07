"""S1 bounded attached allocation-slot pool acceptance tests.

The packet uses only executor records and a tiny child-side ``tpen`` stub.
No experiment or science package is imported by these tests.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap
import threading
import time
from types import SimpleNamespace

import pytest

from tpen_executor import composition
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch.parsl import ParslDispatcher, _ChildProcessHandle
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import AllocationSlot, BindingError
from tpen_executor.results import RunManifest
from tpen_executor.runners import EvidenceUnavailable, TPENEvidence
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun, ConfiguredRunTable
from tpen_executor.workers import ProcessOutcome, SubprocessWorker


HAS_PARSL = importlib.util.find_spec("parsl") is not None
PARSL_REQUIRED = pytest.mark.skipif(not HAS_PARSL, reason="requires the approved executor parsl extra")


def _group(*task_ids: str) -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    members = tuple(TaskSpec(task_id, task_id, program, digest(task_id), 1, "steps") for task_id in task_ids)
    return ReadyGroup("s1-group", members, digest({"plan": "s1"}))


def _slots(count: int = 2) -> list[dict[str, object]]:
    return [
        AllocationSlot(
            f"slot-{index}",
            "node-0",
            (str(index),),
            (index,),
            (("OMP_NUM_THREADS", index + 1),),
            (("S1_SLOT_MARKER", f"slot-{index}"),),
        ).to_dict()
        for index in range(count)
    ]


def _allocation(tmp_path: Path, *, slots: object | None = None, binding: bool = False) -> ConfiguredAllocation:
    facts: dict[str, object] = {}
    if slots is not None:
        facts["slots"] = {"schema": "tpen-executor/allocation-slots-v1", "slots": slots}
    if binding:
        facts["binding"] = {"environment": {"CUDA_VISIBLE_DEVICES": "0"}, "node": "localhost"}
    return ConfiguredAllocation(AllocationId("s1-allocation"), BackendJobId("s1-job"), facts)


def _runs(tmp_path: Path, *task_ids: str, modes: dict[str, str] | None = None) -> dict[str, ConfiguredRun]:
    modes = {} if modes is None else modes
    result: dict[str, ConfiguredRun] = {}
    for task_id in task_ids:
        config = tmp_path / f"{task_id}.mode"
        config.write_text(modes.get(task_id, "ok"), encoding="utf-8")
        result[task_id] = ConfiguredRun(task_id, str(config), (f"payload.task={task_id}",))
    return result


def _stub_package(tmp_path: Path) -> Path:
    package = tmp_path / "stub" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        textwrap.dedent(
            """
            import json
            import os
            import sys
            import time
            from pathlib import Path

            def main(argv):
                config = Path(argv[argv.index('--config') + 1])
                mode = config.read_text(encoding='utf-8')
                values = dict(item.split('=', 1) for item in argv if '=' in item)
                core = Path(values['run.root']) / values['run.run_id']
                start = time.time()
                def wait_for(path_value):
                    if not path_value:
                        return
                    deadline = time.monotonic() + 10.0
                    path = Path(path_value)
                    while not path.exists():
                        if time.monotonic() >= deadline:
                            raise RuntimeError(f'event gate timed out: {path}')
                        time.sleep(0.01)

                if mode == 'slow':
                    Path(os.environ['S1_T2_SLOW_STARTED']).touch()
                    wait_for(os.environ['S1_T2_THIRD_STARTED'])
                elif mode == 'fast':
                    wait_for(os.environ['S1_T2_SLOW_STARTED'])
                elif mode == 'third':
                    Path(os.environ['S1_T2_THIRD_STARTED']).touch()
                elif mode == 'timeout':
                    time.sleep(4.0)
                if mode == 'fail':
                    return 7
                if mode == 'refuse':
                    core.mkdir(parents=True, exist_ok=True)
                    (core / 's1-observation.json').write_text(json.dumps({'start': start, 'end': time.time(), 'mode': mode, 'pid': os.getpid(), 'env': dict(os.environ)}), encoding='utf-8')
                    return 0
                core.mkdir(parents=True, exist_ok=True)
                (core / 's1-observation.json').write_text(json.dumps({'start': start, 'end': time.time(), 'mode': mode, 'pid': os.getpid(), 'env': dict(os.environ)}), encoding='utf-8')
                (core / 'evidence.json').write_text(json.dumps({
                    'schema': 'tpen.run-evidence/v1',
                    'run_dir': str(core),
                    'verdict': 'completed',
                    'run_status': 'completed',
                    'checkpoint': {'checkpoint_dir': str(core / 'checkpoint'), 'schema_version': 1, 'next_iteration': 1, 'completed_updates': 1},
                    'refusal': None,
                }), encoding='utf-8')
                return 0

            if __name__ == '__main__':
                raise SystemExit(main(sys.argv[1:]))
            """
        ),
        encoding="utf-8",
    )
    (package / "inspection.py").write_text(
        textwrap.dedent(
            """
            import json
            import sys
            from pathlib import Path
            print(Path(sys.argv[1], 'evidence.json').read_text(encoding='utf-8'))
            """
        ),
        encoding="utf-8",
    )
    return package.parent


def _worker(tmp_path: Path, package: Path, *, extra_env: dict[str, str] | None = None) -> SimpleNamespace:
    environment = PreparedPythonEnvironment(
        sys.executable,
        env={"PYTHONPATH": str(package), **({} if extra_env is None else extra_env)},
    )
    return SimpleNamespace(
        materialize=lambda env: {
            "environment": environment,
            "process": SubprocessWorker(),
            "runner": object(),
            "evidence": TPENEvidence(),
        }
    )


def _observations(manifest: RunManifest) -> dict[str, dict[str, object]]:
    return {
        attempt.task_id: json.loads(Path(attempt.start_receipt.output_dir, "core", "s1-observation.json").read_text(encoding="utf-8"))
        for attempt in manifest.attempts
        if attempt.start_receipt is not None and Path(attempt.start_receipt.output_dir, "core", "s1-observation.json").is_file()
    }


def _dispatcher(tmp_path: Path, package: Path, *, extra_env: dict[str, str] | None = None) -> ParslDispatcher:
    return ParslDispatcher(_worker(tmp_path, package, extra_env=extra_env))


def _composition_inputs(tmp_path: Path, facts: dict[str, object]) -> dict[str, Path]:
    """Build public-admission records for the pre-store binding gate."""

    group = _group("t:0")
    config = tmp_path / "config.yaml"
    config.write_text("ok", encoding="utf-8")
    worker = tmp_path / "worker.yaml"
    worker.write_text(
        textwrap.dedent(
            f"""
            worker:
              _target_: tpen_executor.programs.WorkerProgram
              _recursive_: false
              environment:
                _target_: tpen_executor.environments.PreparedPythonEnvironment
                _recursive_: false
                python: {json.dumps(sys.executable)}
              process:
                _target_: tpen_executor.workers.SubprocessWorker
                _recursive_: false
              runner:
                _target_: tpen_executor.runners.TPENCommand
                _recursive_: false
                config: {json.dumps(str(config))}
              evidence:
                _target_: tpen_executor.runners.TPENEvidence
                _recursive_: false
            """
        ),
        encoding="utf-8",
    )
    paths = {
        "input": tmp_path / "executor_input.json",
        "runs": tmp_path / "configured-runs.json",
        "allocation": tmp_path / "allocation.json",
        "worker": worker,
        "store": tmp_path / "store",
    }
    paths["input"].write_bytes(group.encode())
    paths["runs"].write_bytes(ConfiguredRunTable(group.identity, (ConfiguredRun("t:0", str(config)),)).encode())
    paths["allocation"].write_bytes(
        _allocation(tmp_path, slots=facts.get("slots") if "slots" in facts else None).encode()
        if "slots" in facts
        else _allocation(tmp_path).encode()
    )
    return paths


@PARSL_REQUIRED
@pytest.mark.parametrize(
    "facts",
    [
        {},
        {"slots": []},
    ],
    ids=["missing-binding", "empty-slots"],
)
def test_h1_parsl_binding_refusal_precedes_store_reservation(tmp_path: Path, facts: dict[str, object]) -> None:
    """H1 kills late-admission-after-store reservation."""

    paths = _composition_inputs(tmp_path, facts)
    with pytest.raises(composition.CLIRefusal) as observed:
        composition.admit(
            paths["input"],
            paths["store"],
            paths["allocation"],
            paths["worker"],
            paths["runs"],
            dispatcher_name="parsl",
        )
    assert observed.value.code == "dispatcher-admission"
    assert not paths["store"].exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-slot-id",
        "duplicate-device",
        "overlap-cpu",
        "bad-thread",
        "bad-env",
        "ambiguous-node",
        "multi-node-geometry",
        "empty-slots",
    ],
)
@PARSL_REQUIRED
def test_t1_strict_slot_admission_rejects_before_begin(tmp_path: Path, mutation: str) -> None:
    """T1 kills malformed, overlapping, multi-node, and late-admission mutants."""

    slots = _slots(2)
    if mutation == "duplicate-slot-id":
        slots[1]["slot_id"] = slots[0]["slot_id"]
    elif mutation == "duplicate-device":
        slots[1]["devices"] = ["0"]
    elif mutation == "overlap-cpu":
        slots[1]["cpus"] = [0]
    elif mutation == "bad-thread":
        slots[0]["threads"] = {"OMP_NUM_THREADS": 0}
    elif mutation == "bad-env":
        slots[0]["environment"] = {"OMP_NUM_THREADS": "not-an-int"}
    elif mutation == "ambiguous-node":
        slots[0]["node"] = ""
    elif mutation == "multi-node-geometry":
        slots[1]["node"] = "node-1"
    else:
        slots = []
    allocation = _allocation(tmp_path, slots=slots)
    with pytest.raises(BindingError):
        _dispatcher(tmp_path, _stub_package(tmp_path)).run(
            _group("malformed"),
            _runs(tmp_path, "malformed"),
            AttachedAllocation(allocation, DirectoryStore(tmp_path / "store")),
        )
    assert not list((tmp_path / "store" / "groups").glob("**/start.json"))


@PARSL_REQUIRED
def test_t2_completion_order_refills_a_quiescent_slot(tmp_path: Path) -> None:
    """T2 kills serial-wait and submission-order-refill mutants."""

    package = _stub_package(tmp_path)
    group = _group("slow", "fast", "third")
    slow_started = tmp_path / "slow.started"
    third_started = tmp_path / "third.started"
    lifecycle = AttachedAllocation(_allocation(tmp_path, slots=_slots(2)), DirectoryStore(tmp_path / "store"))
    manifest = _dispatcher(
        tmp_path,
        package,
        extra_env={
            "S1_T2_SLOW_STARTED": str(slow_started),
            "S1_T2_THIRD_STARTED": str(third_started),
        },
    ).run(
        group,
        _runs(tmp_path, "slow", "fast", "third", modes={"slow": "slow", "fast": "fast", "third": "third"}),
        lifecycle,
    )
    observed = _observations(manifest)
    assert slow_started.is_file() and third_started.is_file()
    assert observed["slow"]["start"] < observed["fast"]["end"]
    assert observed["third"]["start"] >= observed["fast"]["end"]
    assert observed["third"]["start"] < observed["slow"]["end"]


@PARSL_REQUIRED
def test_t3_future_completion_does_not_release_before_child_quiescence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T3 kills cap+1 and release-before-reap mutants."""

    package = _stub_package(tmp_path)
    dispatcher = _dispatcher(tmp_path, package)
    lifecycle = AttachedAllocation(_allocation(tmp_path, slots=_slots(2)), DirectoryStore(tmp_path / "store"))
    release = threading.Event()
    submitted_event = threading.Event()
    submitted: list[tuple[str, bool]] = []

    class Future:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id
            self.released = False

        def done(self) -> bool:
            return True

        def result(self, timeout: float | None = None) -> ProcessOutcome:
            del timeout
            assert release.wait(2), "controller released a slot before child quiescence"
            return ProcessOutcome((sys.executable,), 0, "", "", False)

    def submit(materialized, environment, args, **kwargs):
        del materialized, environment, kwargs
        task_id = args[-4].split("=", 1)[-1]
        submitted.append((task_id, release.is_set()))
        if len(submitted) == 2:
            submitted_event.set()
        return Future(task_id), SimpleNamespace(request_kill=lambda: None)

    monkeypatch.setattr(ParslDispatcher, "_submit_process", lambda self, materialized, environment, args, **kwargs: submit(materialized, environment, args, **kwargs))
    evidence = SimpleNamespace(read=lambda *args, **kwargs: (_ for _ in ()).throw(EvidenceUnavailable("gated")))
    object.__setattr__(dispatcher, "worker", SimpleNamespace(materialize=lambda env: {"environment": PreparedPythonEnvironment(sys.executable), "process": SubprocessWorker(), "runner": object(), "evidence": evidence}))
    runner = threading.Thread(
        target=dispatcher.run,
        args=(group := _group("one", "two", "three"), _runs(tmp_path, "one", "two", "three"), lifecycle),
        daemon=True,
    )
    runner.start()
    assert submitted_event.wait(2)
    assert submitted == [("one", False), ("two", False)]
    release.set()
    runner.join(2)
    assert submitted == [("one", False), ("two", False), ("three", True)]


@PARSL_REQUIRED
def test_t4_each_child_gets_only_its_slot_overlay_and_root(tmp_path: Path) -> None:
    """T4 kills shared/wrong environment, shared output, and process affinity mutants."""

    package = _stub_package(tmp_path)
    parent_keys = ("TPEN_SLOT_ID", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "S1_SLOT_MARKER")
    original = {key: os.environ.get(key) for key in parent_keys}
    manifest = _dispatcher(tmp_path, package).run(
        _group("left", "right"),
        _runs(tmp_path, "left", "right"),
        AttachedAllocation(_allocation(tmp_path, slots=_slots(2)), DirectoryStore(tmp_path / "store")),
    )
    assert {key: os.environ.get(key) for key in parent_keys} == original
    observed = _observations(manifest)
    assert observed["left"]["env"]["TPEN_SLOT_ID"] == "slot-0"  # type: ignore[index]
    assert observed["right"]["env"]["TPEN_SLOT_ID"] == "slot-1"  # type: ignore[index]
    assert observed["left"]["env"]["CUDA_VISIBLE_DEVICES"] == "0"  # type: ignore[index]
    assert observed["right"]["env"]["CUDA_VISIBLE_DEVICES"] == "1"  # type: ignore[index]
    assert observed["left"]["env"]["OMP_NUM_THREADS"] == "1"  # type: ignore[index]
    assert observed["right"]["env"]["OMP_NUM_THREADS"] == "2"  # type: ignore[index]
    assert observed["left"]["env"]["S1_SLOT_MARKER"] == "slot-0"  # type: ignore[index]
    assert observed["right"]["env"]["S1_SLOT_MARKER"] == "slot-1"  # type: ignore[index]
    assert observed["left"]["env"]["TPEN_SLOT_CPUS"] != observed["right"]["env"]["TPEN_SLOT_CPUS"]  # type: ignore[index]
    roots = [attempt.start_receipt.output_dir for attempt in manifest.attempts]  # type: ignore[union-attr]
    assert len(roots) == len(set(roots)) == 2


class _RecordingStore(DirectoryStore):
    """Record lifecycle writer identities for the controller-only gate."""

    __slots__ = ()
    writers: list[tuple[str, int]] = []

    def begin_attempt(self, *args, **kwargs):
        type(self).writers.append(("begin", threading.get_ident()))
        return super().begin_attempt(*args, **kwargs)

    def finish_attempt(self, *args, **kwargs):
        type(self).writers.append(("finish", threading.get_ident()))
        return super().finish_attempt(*args, **kwargs)

    def accept_segment(self, *args, **kwargs):
        type(self).writers.append(("accept", threading.get_ident()))
        return super().accept_segment(*args, **kwargs)


@PARSL_REQUIRED
def test_t5_controller_is_the_only_lifecycle_writer_and_accepts_once(tmp_path: Path) -> None:
    """T5 kills leaf-write, dropped/double-finish, and accept-before-evidence mutants."""

    package = _stub_package(tmp_path)
    group = _group("ok", "fail", "refuse")
    _RecordingStore.writers.clear()
    store = _RecordingStore(tmp_path / "store")
    lifecycle = AttachedAllocation(_allocation(tmp_path, slots=_slots(2)), store)
    manifest = _dispatcher(tmp_path, package).run(
        group,
        _runs(tmp_path, "ok", "fail", "refuse", modes={"fail": "fail", "refuse": "refuse"}),
        lifecycle,
    )
    attempts = {attempt.task_id: attempt for attempt in manifest.attempts}
    assert {task_id: attempts[task_id].outcome_receipt.status for task_id in attempts} == {"ok": "completed", "fail": "failed", "refuse": "completed"}  # type: ignore[union-attr]
    assert {segment.task_id for segment in manifest.segments} == {"ok"}
    for attempt in attempts.values():
        assert Path(attempt.start_receipt.output_dir, "start.json").is_file()  # type: ignore[union-attr]
        assert Path(attempt.start_receipt.output_dir, "outcome.json").is_file()  # type: ignore[union-attr]
    controller = threading.get_ident()
    kinds = [kind for kind, _ in _RecordingStore.writers]
    assert kinds.count("begin") == 3 and kinds.count("finish") == 3 and kinds.count("accept") == 1
    assert {ident for _, ident in _RecordingStore.writers} == {controller}


@PARSL_REQUIRED
def test_t6_fail_timeout_and_evidence_refusal_free_only_quiescent_slots(tmp_path: Path) -> None:
    """T6 kills abort-group, cross-member overwrite, and failure-as-complete mutants."""

    package = _stub_package(tmp_path)
    group = _group("fail", "timeout", "refuse", "ok")
    store = DirectoryStore(tmp_path / "store")
    manifest = _dispatcher(tmp_path, package).run(
        group,
        _runs(tmp_path, "fail", "timeout", "refuse", "ok", modes={"fail": "fail", "timeout": "timeout", "refuse": "refuse"}),
        AttachedAllocation(_allocation(tmp_path, slots=_slots(2)), store),
        timeout=2.0,
    )
    attempts = {attempt.task_id: attempt for attempt in manifest.attempts}
    assert attempts["fail"].outcome_receipt.status == "failed"  # type: ignore[union-attr]
    assert attempts["timeout"].outcome_receipt.status == "timed_out"  # type: ignore[union-attr]
    assert attempts["refuse"].outcome_receipt.status == "completed"  # type: ignore[union-attr]
    assert attempts["ok"].outcome_receipt.status == "completed"  # type: ignore[union-attr]
    assert {segment.task_id for segment in manifest.segments} == {"ok"}
    assert all(Path(attempt.start_receipt.output_dir, "outcome.json").is_file() for attempt in attempts.values())  # type: ignore[union-attr]


@PARSL_REQUIRED
def test_t7_cancel_closes_admission_kills_active_and_never_starts_queued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T7 kills swallowed-cancel, post-cancel-refill, and dropped-finish mutants."""

    package = _stub_package(tmp_path)
    dispatcher = _dispatcher(tmp_path, package)
    group = _group("one", "two", "queued")
    lifecycle = AttachedAllocation(_allocation(tmp_path, slots=_slots(2)), DirectoryStore(tmp_path / "store"))
    started: list[str] = []
    killed: list[str] = []

    class Future:
        def __init__(self, task_id: str) -> None:
            self.task_id = task_id
            self.killed = False

        def done(self) -> bool:
            return self.killed

        def result(self, timeout: float | None = None) -> ProcessOutcome:
            del timeout
            return ProcessOutcome((sys.executable,), 137, "", "", False)

    def submit(materialized, environment, args, **kwargs):
        del materialized, environment, kwargs
        task_id = args[-4].split("=", 1)[-1]
        future = Future(task_id)
        started.append(task_id)
        return future, SimpleNamespace(request_kill=lambda future=future: (setattr(future, "killed", True), killed.append(task_id)))

    monkeypatch.setattr(ParslDispatcher, "_submit_process", lambda self, materialized, environment, args, **kwargs: submit(materialized, environment, args, **kwargs))
    first_poll = {"value": True}

    def interrupting_sleep(seconds: float) -> None:
        if first_poll["value"]:
            first_poll["value"] = False
            raise KeyboardInterrupt("S1 cancellation")
        time.sleep(seconds)

    monkeypatch.setattr("tpen_executor.dispatch.parsl._poll_sleep", interrupting_sleep)
    with pytest.raises(KeyboardInterrupt, match="S1 cancellation"):
        dispatcher.run(group, _runs(tmp_path, "one", "two", "queued"), lifecycle)
    assert started == ["one", "two"]
    assert sorted(killed) == ["one", "two"]
    assert not (tmp_path / "store" / "groups" / group.identity / "queued").exists()
    manifest = lifecycle.store.manifest(group)
    assert len(manifest.attempts) == 2
    assert all(attempt.outcome_receipt.status == "failed" for attempt in manifest.attempts)  # type: ignore[union-attr]


@PARSL_REQUIRED
def test_t7_repeated_cancel_interrupts_still_reap_every_active_future(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """M1 kills cancellation cleanup that exits after its second interrupt."""

    dispatcher = _dispatcher(tmp_path, _stub_package(tmp_path))
    lifecycle = AttachedAllocation(_allocation(tmp_path, slots=_slots(2)), DirectoryStore(tmp_path / "store"))
    state = {"polls": 0}

    class Future:
        reaped = False

        def done(self) -> bool:
            return self.reaped

        def result(self, timeout: float | None = None) -> ProcessOutcome:
            del timeout
            assert self.reaped
            return ProcessOutcome((sys.executable,), 137, "", "", False)

    future = Future()

    def submit(self, materialized, environment, args, **kwargs):
        del self
        del materialized, environment, args, kwargs
        return future, SimpleNamespace(request_kill=lambda: None)

    def interrupting_sleep(seconds: float) -> None:
        del seconds
        state["polls"] += 1
        if state["polls"] <= 2:
            raise KeyboardInterrupt(f"S1 cancellation {state['polls']}")
        future.reaped = True

    monkeypatch.setattr(ParslDispatcher, "_submit_process", submit)
    monkeypatch.setattr("tpen_executor.dispatch.parsl._poll_sleep", interrupting_sleep)
    with pytest.raises(KeyboardInterrupt, match="S1 cancellation 1"):
        dispatcher.run(_group("active"), _runs(tmp_path, "active"), lifecycle)
    assert future.reaped


@PARSL_REQUIRED
def test_t7_cancel_before_spawn_finishes_only_the_begun_fence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T7 kills post-cancel refill and dropped terminal-fence mutants."""

    dispatcher = _dispatcher(tmp_path, _stub_package(tmp_path))
    group = _group("first", "queued")
    lifecycle = AttachedAllocation(_allocation(tmp_path, slots=_slots(1)), DirectoryStore(tmp_path / "store"))

    def interrupted_submit(self, *args, **kwargs):
        del self, args, kwargs
        raise KeyboardInterrupt("cancel before child spawn")

    monkeypatch.setattr(ParslDispatcher, "_execute_process", interrupted_submit)
    with pytest.raises(KeyboardInterrupt, match="cancel before child spawn"):
        dispatcher.run(group, _runs(tmp_path, "first", "queued"), lifecycle)
    manifest = lifecycle.store.manifest(group)
    assert [attempt.task_id for attempt in manifest.attempts] == ["first"]
    assert manifest.attempts[0].outcome_receipt.status == "failed"  # type: ignore[union-attr]


def test_t7_cancel_during_spawn_kills_a_late_attached_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """T7 kills a pre-spawn cancellation race in the child-group handle."""

    killed: list[object] = []
    monkeypatch.setattr(_ChildProcessHandle, "_kill_process_group", staticmethod(killed.append))
    handle = _ChildProcessHandle()
    process = SimpleNamespace(pid=123)
    handle.request_kill()
    handle.attach(process)
    assert killed == [process]


@PARSL_REQUIRED
def test_t8_one_explicit_slot_remains_serial_and_preserves_binding_refusal(tmp_path: Path) -> None:
    """T8 kills changed-default and missing-binding mutants."""

    package = _stub_package(tmp_path)
    group = _group("one", "two")
    lifecycle = AttachedAllocation(_allocation(tmp_path, slots=_slots(1)), DirectoryStore(tmp_path / "store"))
    manifest = _dispatcher(tmp_path, package).run(group, _runs(tmp_path, "one", "two"), lifecycle)
    observed = _observations(manifest)
    assert observed["one"]["end"] <= observed["two"]["start"]
    missing = AttachedAllocation(_allocation(tmp_path, slots=None), DirectoryStore(tmp_path / "missing"))
    with pytest.raises(BindingError, match="no worker binding"):
        _dispatcher(tmp_path, package).run(_group("missing"), _runs(tmp_path, "missing"), missing)
