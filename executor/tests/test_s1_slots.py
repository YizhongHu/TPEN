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

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch.parsl import ParslDispatcher
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import AllocationSlot, BindingError, slots_from_allocation
from tpen_executor.results import RunManifest
from tpen_executor.runners import EvidenceUnavailable, TPENEvidence
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
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
            f"node-{index}",
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
                if mode == 'slow':
                    time.sleep(0.60)
                elif mode == 'fast':
                    time.sleep(0.12)
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


def _worker(tmp_path: Path, package: Path) -> SimpleNamespace:
    environment = PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})
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


def _dispatcher(tmp_path: Path, package: Path) -> ParslDispatcher:
    return ParslDispatcher(_worker(tmp_path, package))


@pytest.mark.parametrize(
    "mutation",
    ["duplicate-slot-id", "duplicate-device", "overlap-cpu", "bad-thread", "bad-env", "ambiguous-node", "empty-slots"],
)
def test_t1_strict_slot_admission_rejects_before_begin(tmp_path: Path, mutation: str) -> None:
    """T1 kills duplicate-slot, overlap-allowed, and late-admission mutants."""

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
    else:
        slots = []
    allocation = _allocation(tmp_path, slots=slots)
    with pytest.raises(BindingError):
        slots_from_allocation(allocation)
    assert not (tmp_path / "store" / "groups").exists()


@PARSL_REQUIRED
def test_t2_completion_order_refills_a_quiescent_slot(tmp_path: Path) -> None:
    """T2 kills serial-wait and submission-order-refill mutants."""

    package = _stub_package(tmp_path)
    group = _group("fast", "slow", "third")
    lifecycle = AttachedAllocation(_allocation(tmp_path, slots=_slots(2)), DirectoryStore(tmp_path / "store"))
    manifest = _dispatcher(tmp_path, package).run(
        group,
        _runs(tmp_path, "fast", "slow", "third", modes={"fast": "fast", "slow": "slow"}),
        lifecycle,
    )
    observed = _observations(manifest)
    assert observed["fast"]["start"] < observed["slow"]["end"]
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
    submitted: list[str] = []

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
        submitted.append(task_id)
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
    assert submitted == ["one", "two"]
    release.set()
    runner.join(2)
    assert submitted == ["one", "two", "three"]


@PARSL_REQUIRED
def test_t4_each_child_gets_only_its_slot_overlay_and_root(tmp_path: Path) -> None:
    """T4 kills shared/wrong environment, shared output, and process affinity mutants."""

    package = _stub_package(tmp_path)
    original = os.environ.get("S1_SLOT_MARKER")
    manifest = _dispatcher(tmp_path, package).run(
        _group("left", "right"),
        _runs(tmp_path, "left", "right"),
        AttachedAllocation(_allocation(tmp_path, slots=_slots(2)), DirectoryStore(tmp_path / "store")),
    )
    if original is None:
        assert "S1_SLOT_MARKER" not in os.environ
    else:
        assert os.environ["S1_SLOT_MARKER"] == original
    observed = _observations(manifest)
    assert observed["left"]["env"]["TPEN_SLOT_ID"] == "slot-0"  # type: ignore[index]
    assert observed["right"]["env"]["TPEN_SLOT_ID"] == "slot-1"  # type: ignore[index]
    assert observed["left"]["env"]["TPEN_SLOT_CPUS"] != observed["right"]["env"]["TPEN_SLOT_CPUS"]  # type: ignore[index]
    roots = [attempt.start_receipt.output_dir for attempt in manifest.attempts]  # type: ignore[union-attr]
    assert len(roots) == len(set(roots)) == 2


@PARSL_REQUIRED
def test_t5_controller_is_the_only_lifecycle_writer_and_accepts_once(tmp_path: Path) -> None:
    """T5 kills leaf-write, dropped/double-finish, and accept-before-evidence mutants."""

    package = _stub_package(tmp_path)
    group = _group("ok", "fail", "refuse")
    store = DirectoryStore(tmp_path / "store")
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
        timeout=0.15,
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
