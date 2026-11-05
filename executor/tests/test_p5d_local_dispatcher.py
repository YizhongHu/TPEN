"""P5d attached/local dispatcher acceptance probes.

These tests are intended for the executor test environment on the authorized
cluster. They deliberately use a tiny fake ``tpen`` package for child calls;
the executor itself never imports the science package.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch import LocalDispatcher
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.programs import AllocationProgram, MaterializedProgram, WorkerProgram
from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.workers import ProcessOutcome


def _group(*task_ids: str) -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    members = tuple(TaskSpec(task_id, task_id, program, digest(task_id), 1, "steps") for task_id in task_ids)
    return ReadyGroup("group", members, digest({"plan": 1}))


def _allocation() -> ConfiguredAllocation:
    return ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})


def _runs(tmp_path: Path, *task_ids: str) -> dict[str, ConfiguredRun]:
    return {task_id: ConfiguredRun(task_id, str(tmp_path / f"{task_id}.yaml")) for task_id in task_ids}


def _worker_spec(environment: PreparedPythonEnvironment) -> WorkerProgram:
    raw = {
        "_target_": "tpen_executor.programs.WorkerProgram",
        "_recursive_": False,
        "environment": {
            "_target_": "tpen_executor.environments.PreparedPythonEnvironment",
            "_recursive_": False,
            "python": environment.python,
            "env": dict(environment.env),
        },
        "process": {"_target_": "tpen_executor.workers.SubprocessWorker", "_recursive_": False},
        "runner": {
            "_target_": "tpen_executor.runners.TPENCommand",
            "_recursive_": False,
            "config": "/prepared/config.yaml",
        },
        "evidence": {"_target_": "tpen_executor.runners.TPENEvidence", "_recursive_": False},
    }
    return WorkerProgram(ProgramSpecRef(raw, digest(raw)))


def _evidence(run_dir: str) -> CoreRunEvidence:
    return CoreRunEvidence.from_mapping(
        {
            "schema": "tpen.run-evidence/v1",
            "run_dir": run_dir,
            "verdict": "completed",
            "run_status": "completed",
            "checkpoint": {
                "checkpoint_dir": f"{run_dir}/checkpoint",
                "schema_version": "v1",
                "next_iteration": 1,
                "completed_updates": 1,
            },
            "refusal": None,
        }
    )


def _make_stub_package(tmp_path: Path) -> Path:
    package = tmp_path / "pkg" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        textwrap.dedent(
            """
            import json
            import sys
            import time
            from pathlib import Path

            def main(argv):
                config = Path(argv[argv.index("--config") + 1])
                mode = config.read_text(encoding="utf-8")
                overrides = dict(item.split("=", 1) for item in argv if "=" in item)
                run_dir = Path(overrides["run.root"]) / overrides["run.run_id"]
                if mode == "sleep":
                    time.sleep(2)
                if mode == "fail":
                    return 3
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "evidence.json").write_text(json.dumps({
                    "schema": "tpen.run-evidence/v1",
                    "run_dir": str(run_dir),
                    "verdict": "completed",
                    "run_status": "completed",
                    "checkpoint": {"checkpoint_dir": str(run_dir / "checkpoint"), "schema_version": "v1", "next_iteration": 1, "completed_updates": 1},
                    "refusal": None,
                }), encoding="utf-8")
                return 0

            if __name__ == "__main__":
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

            print(json.dumps(json.loads(Path(sys.argv[1], "evidence.json").read_text(encoding="utf-8"))))
            """
        ),
        encoding="utf-8",
    )
    return package.parent


def test_p5d_three_members_publish_independent_statuses_and_evidence(tmp_path: Path) -> None:
    package = _make_stub_package(tmp_path)
    configs = {}
    for task_id, mode in (("ok", "ok"), ("bad", "fail"), ("late", "sleep")):
        config = tmp_path / f"{task_id}.yaml"
        config.write_text(mode, encoding="utf-8")
        configs[task_id] = ConfiguredRun(task_id, str(config))
    group = _group("ok", "bad", "late")
    store = DirectoryStore(tmp_path / "store")
    dispatcher = LocalDispatcher(_worker_spec(PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})))
    manifest = dispatcher.run(group, configs, AttachedAllocation(_allocation(), store), timeout=0.1)
    attempts = {item.task_id: item for item in manifest.attempts}
    assert attempts["ok"].outcome_receipt.status == "completed"  # type: ignore[union-attr]
    assert attempts["bad"].outcome_receipt.status == "failed"  # type: ignore[union-attr]
    assert attempts["late"].outcome_receipt.status == "timed_out"  # type: ignore[union-attr]
    assert len({item.attempt_id for item in manifest.attempts}) == 3
    assert {item.start_receipt.group_digest for item in manifest.attempts} == {group.identity}
    assert attempts["ok"].outcome_receipt.result_digest is not None  # type: ignore[union-attr]


@pytest.mark.parametrize("mutation", ["missing", "extra", "id-mismatch", "scalar"], ids=["missing-task", "extra-task", "id-mismatch", "scalar-runs"])
def test_p5d_admission_rejects_each_mapping_mutation_before_store_write(tmp_path: Path, mutation: str) -> None:
    group = _group("a", "b")
    runs: object = _runs(tmp_path, "a", "b")
    if mutation == "missing":
        runs = _runs(tmp_path, "a")
    elif mutation == "extra":
        runs = {**_runs(tmp_path, "a", "b"), "c": ConfiguredRun("c", str(tmp_path / "c.yaml"))}
    elif mutation == "id-mismatch":
        runs = {"a": ConfiguredRun("other", str(tmp_path / "a.yaml")), "b": _runs(tmp_path, "b")["b"]}
    elif mutation == "scalar":
        runs = "not-a-mapping"
    dispatcher = LocalDispatcher(SimpleNamespace(materialize=lambda env: {}))
    store = DirectoryStore(tmp_path / "store")
    with pytest.raises((TypeError, ValueError)):
        dispatcher.run(group, runs, AttachedAllocation(_allocation(), store))  # type: ignore[arg-type]
    assert not (tmp_path / "store").exists()


def test_p5d_child_receives_exact_command_and_fenced_run_overrides(tmp_path: Path) -> None:
    calls: list[tuple[object, ...]] = []

    class Process:
        def run(self, environment, args, **kwargs):
            calls.append(tuple(args))
            return ProcessOutcome(tuple(args), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)

    class Evidence:
        def read(self, *args, **kwargs):
            return _evidence(kwargs["run_dir"] if "run_dir" in kwargs else args[2])

    worker = SimpleNamespace(
        materialize=lambda env: {
            "environment": object(),
            "process": Process(),
            "runner": object(),
            "evidence": Evidence(),
        }
    )
    group = _group("a")
    configured = _runs(tmp_path, "a")["a"]
    store = DirectoryStore(tmp_path / "store")
    LocalDispatcher(worker).run(group, {"a": configured}, AttachedAllocation(_allocation(), store))
    assert calls[0][: len(configured.command().args)] == configured.command().args
    assert calls[0][-3].startswith("run.root=")
    assert calls[0][-2:] == ("run.layout=flat", "run.run_id=core")
    assert str(store.root) in calls[0][-3]


def test_p5d_unexpected_member_exception_finishes_and_continues(tmp_path: Path) -> None:
    finished: list[str] = []

    class Process:
        def run(self, environment, args, **kwargs):
            return ProcessOutcome(tuple(args), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)

    class Evidence:
        calls = 0

        def read(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("injected evidence failure")
            return _evidence(args[2])

    class Lifecycle(AttachedAllocation):
        def finish(self, fence, status, finished_at_utc, result_digest=None):
            finished.append(fence.task_id)
            return super().finish(fence, status, finished_at_utc, result_digest)

    worker = SimpleNamespace(materialize=lambda env: {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()})
    group = _group("a", "b")
    store = DirectoryStore(tmp_path / "store")
    LocalDispatcher(worker).run(group, _runs(tmp_path, "a", "b"), Lifecycle(_allocation(), store))
    assert finished == ["a", "b"]
    assert [item.outcome_receipt.status for item in store.manifest(group).attempts] == ["launch_failed", "completed"]  # type: ignore[union-attr]


def test_p5d_evidence_unavailable_keeps_process_status_and_digest_empty(tmp_path: Path) -> None:
    class Process:
        def run(self, environment, args, **kwargs):
            return ProcessOutcome(tuple(args), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)

    class Evidence:
        def read(self, *args, **kwargs):
            raise EvidenceUnavailable("no evidence")

    worker = SimpleNamespace(materialize=lambda env: {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()})
    group = _group("a")
    store = DirectoryStore(tmp_path / "store")
    LocalDispatcher(worker).run(group, _runs(tmp_path, "a"), AttachedAllocation(_allocation(), store))
    outcome = store.manifest(group).attempts[0].outcome_receipt
    assert outcome is not None and outcome.status == "completed" and outcome.result_digest is None


def test_p5d_packaged_config_materializes_local_dispatcher() -> None:
    from tpen_executor.configuration import StagedHydra

    staged = StagedHydra().compose_config()
    result = staged.program.materialize({"paths": {"control_python": sys.executable, "control_setup": "/none", "science_python": sys.executable, "science_setup": "/none"}})  # type: ignore[union-attr]
    assert isinstance(result, MaterializedProgram)
    assert isinstance(result.dispatcher, LocalDispatcher)


def test_p5d_allocation_without_dispatcher_is_typed_error() -> None:
    raw = {
        "_target_": "tpen_executor.programs.AllocationProgram",
        "_recursive_": False,
        "environment": {"_target_": "tpen_executor.environments.PreparedPythonEnvironment", "_recursive_": False, "python": sys.executable},
    }
    with pytest.raises(TypeError, match="dispatcher"):
        AllocationProgram(ProgramSpecRef(raw, digest(raw))).materialize({})


def test_p5d_dispatcher_without_run_is_typed_error() -> None:
    raw = {
        "_target_": "tpen_executor.programs.AllocationProgram",
        "_recursive_": False,
        "environment": {"_target_": "tpen_executor.environments.PreparedPythonEnvironment", "_recursive_": False, "python": sys.executable},
        "dispatcher": {"_target_": "tpen_executor.programs.DummyDispatcher", "_recursive_": False},
    }
    with pytest.raises(TypeError, match="run"):
        AllocationProgram(ProgramSpecRef(raw, digest(raw))).materialize({})


def test_p5d_reentrancy_guard_is_runtime_typed_error(tmp_path: Path) -> None:
    dispatcher: LocalDispatcher

    class Process:
        def run(self, environment, args, **kwargs):
            return ProcessOutcome(tuple(args), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)

    class Evidence:
        def read(self, *args, **kwargs):
            with pytest.raises(RuntimeError, match="reentrant"):
                dispatcher.run(_group("nested"), _runs(tmp_path, "nested"), lifecycle)
            raise EvidenceUnavailable("outer evidence unavailable")

    worker = SimpleNamespace(materialize=lambda env: {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()})
    group = _group("a")
    store = DirectoryStore(tmp_path / "store")
    lifecycle = AttachedAllocation(_allocation(), store)
    dispatcher = LocalDispatcher(worker)
    dispatcher.run(group, _runs(tmp_path, "a"), lifecycle)


def test_p5d_runtime_write_trap_confines_parent_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = _make_stub_package(tmp_path)
    store_root = tmp_path / "store"
    configs = {}
    for task_id in ("a", "b"):
        config = package / f"{task_id}.yaml"
        config.write_text("ok", encoding="utf-8")
        configs[task_id] = ConfiguredRun(task_id, str(config))

    store_lexical = os.path.abspath(os.path.normpath(os.fspath(store_root)))
    package_lexical = os.path.abspath(os.path.normpath(os.fspath(package)))
    before_entries = set(tmp_path.iterdir())
    writes: list[tuple[str, str]] = []
    in_trap = False

    def record(kind: str, path: object) -> None:
        nonlocal in_trap
        if in_trap:
            raise AssertionError("write trap re-entered")
        in_trap = True
        try:
            lexical = os.path.abspath(os.path.normpath(os.fspath(path)))
            writes.append((kind, lexical))
        finally:
            in_trap = False

    original_open = os.open
    original_mkstemp = tempfile.mkstemp

    def trapped_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if flags & write_flags:
            record("open", path)
        return original_open(path, flags, *args, **kwargs)

    def trapped_single(kind: str, original):
        def wrapper(path: object, *args: object, **kwargs: object):
            record(kind, path)
            return original(path, *args, **kwargs)

        return wrapper

    def trapped_pair(kind: str, original):
        def wrapper(source: object, destination: object, *args: object, **kwargs: object):
            record(kind, source)
            record(kind, destination)
            return original(source, destination, *args, **kwargs)

        return wrapper

    def trapped_mkstemp(suffix: str = "", prefix: str = "tmp", dir: str | os.PathLike[str] | None = None, text: bool = False):
        if dir is not None:
            record("mkstemp", dir)
        return original_mkstemp(suffix=suffix, prefix=prefix, dir=dir, text=text)

    with monkeypatch.context() as patch:
        patch.setattr(os, "open", trapped_open)
        for name in ("mkdir", "makedirs", "unlink"):
            patch.setattr(os, name, trapped_single(name, getattr(os, name)))
        for name in ("replace", "link"):
            patch.setattr(os, name, trapped_pair(name, getattr(os, name)))
        patch.setattr(tempfile, "mkstemp", trapped_mkstemp)
        dispatcher = LocalDispatcher(_worker_spec(PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})))
        dispatcher.run(_group("a", "b"), configs, AttachedAllocation(_allocation(), DirectoryStore(store_root)), timeout=1)

    allowed = (store_lexical, package_lexical)
    assert writes
    assert all(any(path == root or path.startswith(root + os.sep) for root in allowed) for _, path in writes)

    # Child-process writes are not traced in-process; the post-run census closes that gap.
    new_entries = set(tmp_path.iterdir()) - before_entries
    assert all(
        any(
            os.path.abspath(os.path.normpath(os.fspath(path))) == root
            or os.path.abspath(os.path.normpath(os.fspath(path))).startswith(root + os.sep)
            for root in allowed
        )
        for path in new_entries
    )
