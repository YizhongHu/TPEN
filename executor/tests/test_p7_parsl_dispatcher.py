"""P7 attached Parsl dispatcher contract and mutation probes.

The Parsl cases use its one-machine ThreadPool provider.  They are skipped in
the base executor environment, while the local conformance arm remains useful
without the optional extra.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch import LocalDispatcher, ParslDispatcher, ParslUnavailableError
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.programs import WorkerProgram
from tpen_executor.results import RunManifest
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.workers import ProcessOutcome


HAS_PARSL = importlib.util.find_spec("parsl") is not None
PARSL_REQUIRED = pytest.mark.skipif(not HAS_PARSL, reason="requires the approved executor parsl extra")


def _group(*task_ids: str) -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    members = tuple(TaskSpec(task_id, task_id, program, digest(task_id), 1, "steps") for task_id in task_ids)
    return ReadyGroup("p7-group", members, digest({"plan": "p7"}))


def _allocation(*, bound: bool = True, allocation_id: str = "p7-allocation") -> ConfiguredAllocation:
    facts = {}
    if bound:
        facts = {
            "binding": {
                "environment": {"CUDA_VISIBLE_DEVICES": "0"},
                "worker_id": "p7-worker-0",
                "node": "localhost",
            }
        }
    return ConfiguredAllocation(AllocationId(allocation_id), BackendJobId(f"job-{allocation_id}"), facts)


def _runs(tmp_path: Path, *task_ids: str, modes: dict[str, str] | None = None) -> dict[str, ConfiguredRun]:
    modes = {} if modes is None else modes
    result = {}
    for task_id in task_ids:
        config = tmp_path / f"{task_id}.cfg"
        config.write_text(modes.get(task_id, "ok"), encoding="utf-8")
        result[task_id] = ConfiguredRun(task_id, str(config), (f"payload.task={task_id}",))
    return result


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
        "runner": {"_target_": "tpen_executor.runners.TPENCommand", "_recursive_": False, "config": "/p7/config"},
        "evidence": {"_target_": "tpen_executor.runners.TPENEvidence", "_recursive_": False},
    }
    return WorkerProgram(ProgramSpecRef(raw, digest(raw)))


def _make_stub_package(tmp_path: Path) -> Path:
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
                config = Path(argv[argv.index("--config") + 1])
                mode = config.read_text(encoding="utf-8")
                overrides = dict(item.split("=", 1) for item in argv if "=" in item)
                run_dir = Path(overrides["run.root"]) / overrides["run.run_id"]
                if mode == "sleep":
                    time.sleep(4)
                if mode == "fail":
                    return 7
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "argv-and-binding.json").write_text(json.dumps({
                    "argv": argv,
                    "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "cwd": os.getcwd(),
                }), encoding="utf-8")
                (run_dir / "evidence.json").write_text(json.dumps({
                    "schema": "tpen.run-evidence/v1",
                    "run_dir": str(run_dir),
                    "verdict": "completed",
                    "run_status": "completed",
                    "checkpoint": {"checkpoint_dir": str(run_dir / "checkpoint"), "schema_version": 1, "next_iteration": 1, "completed_updates": 1},
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


def _dispatcher(dispatcher_type, worker: WorkerProgram):
    if dispatcher_type is ParslDispatcher:
        return dispatcher_type(worker)
    return dispatcher_type(worker)


@pytest.mark.parametrize("dispatcher_type", [LocalDispatcher, ParslDispatcher], ids=["local", "parsl"])
def test_p7_a_shared_contract_conformance_status_evidence_and_segments(tmp_path: Path, dispatcher_type) -> None:
    """A: both backends publish fenced statuses, evidence, and segments identically."""
    if dispatcher_type is ParslDispatcher and not HAS_PARSL:
        pytest.skip("requires the approved executor parsl extra")
    package = _make_stub_package(tmp_path)
    group = _group("ok", "bad", "late")
    runs = _runs(tmp_path, "ok", "bad", "late", modes={"bad": "fail", "late": "sleep"})
    store = DirectoryStore(tmp_path / dispatcher_type.__name__)
    dispatcher = _dispatcher(
        dispatcher_type,
        _worker_spec(PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})),
    )
    manifest = dispatcher.run(group, runs, AttachedAllocation(_allocation(), store), timeout=1)
    attempts = {item.task_id: item for item in manifest.attempts}
    assert attempts["ok"].outcome_receipt.status == "completed"  # type: ignore[union-attr]
    assert attempts["bad"].outcome_receipt.status == "failed"  # type: ignore[union-attr]
    assert attempts["late"].outcome_receipt.status == "timed_out"  # type: ignore[union-attr]
    assert attempts["ok"].outcome_receipt.result_digest is not None  # type: ignore[union-attr]
    assert {segment.task_id for segment in manifest.segments} == {"ok"}
    assert len({item.attempt_id for item in manifest.attempts}) == 3


@pytest.mark.parametrize("dispatcher_type", [LocalDispatcher, ParslDispatcher], ids=["local", "parsl"])
def test_p7_b_exact_argv_payload_and_fresh_output_namespace(tmp_path: Path, dispatcher_type) -> None:
    """B: command payload and all run paths stay inside each fresh fence."""
    if dispatcher_type is ParslDispatcher and not HAS_PARSL:
        pytest.skip("requires the approved executor parsl extra")
    package = _make_stub_package(tmp_path)
    group = _group("one", "two")
    runs = _runs(tmp_path, "one", "two")
    store = DirectoryStore(tmp_path / dispatcher_type.__name__)
    dispatcher = _dispatcher(dispatcher_type, _worker_spec(PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})))
    manifest = dispatcher.run(group, runs, AttachedAllocation(_allocation(), store))
    attempts = {item.task_id: item for item in manifest.attempts}
    for task_id, configured in runs.items():
        output_dir = Path(attempts[task_id].start_receipt.output_dir)  # type: ignore[union-attr]
        observed = json.loads((output_dir / "core" / "argv-and-binding.json").read_text(encoding="utf-8"))
        assert observed["argv"][: len(configured.command().args)] == list(configured.command().args)
        assert observed["argv"][-3:] == [f"run.root={output_dir}", "run.layout=flat", "run.run_id=core"]
        assert observed["cwd"] == str(output_dir)
    assert attempts["one"].start_receipt.output_dir != attempts["two"].start_receipt.output_dir  # type: ignore[union-attr]


@PARSL_REQUIRED
def test_p7_c_parsl_binding_is_applied_to_one_local_provider_worker(tmp_path: Path) -> None:
    """C: explicit allocation binding reaches the local Parsl worker."""
    package = _make_stub_package(tmp_path)
    group = _group("bound")
    store = DirectoryStore(tmp_path / "store")
    dispatcher = ParslDispatcher(_worker_spec(PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})))
    manifest = dispatcher.run(group, _runs(tmp_path, "bound"), AttachedAllocation(_allocation(), store))
    output_dir = Path(manifest.attempts[0].start_receipt.output_dir)  # type: ignore[union-attr]
    observed = json.loads((output_dir / "core" / "argv-and-binding.json").read_text(encoding="utf-8"))
    assert observed["cuda"] == "0"


def test_p7_c_missing_binding_is_named_rejection_before_first_fence(tmp_path: Path) -> None:
    """C rejection: Parsl binding absence is refused before admission writes."""
    if not HAS_PARSL:
        pytest.skip("requires the approved executor parsl extra")
    dispatcher = ParslDispatcher(_worker_spec(PreparedPythonEnvironment(sys.executable)))
    store = DirectoryStore(tmp_path / "store")
    with pytest.raises(ValueError, match="no worker binding"):
        dispatcher.run(_group("unbound"), _runs(tmp_path, "unbound"), AttachedAllocation(_allocation(bound=False), store))
    assert not list((tmp_path / "store" / "groups").glob("**/start.json"))


def test_p7_c_invalid_binding_is_typed_refusal() -> None:
    """C rejection: an empty visibility binding cannot become worker env."""
    from tpen_executor.placement import BindingError, WorkerBinding

    with pytest.raises(BindingError, match="non-empty"):
        WorkerBinding.from_mapping({"environment": {"CUDA_VISIBLE_DEVICES": ""}})


def test_p7_d_child_rc_not_framework_success_controls_status(tmp_path: Path) -> None:
    """D mutant control: a framework-successful call with rc 7 remains failed."""
    calls = []

    class Process:
        def run(self, environment, args, **kwargs):
            calls.append(tuple(args))
            return ProcessOutcome(tuple(args), 7, kwargs["stdout_path"], kwargs["stderr_path"], False)

    class Evidence:
        def read(self, *args, **kwargs):
            raise AssertionError("evidence must not promote a nonzero child")

    worker = SimpleNamespace(materialize=lambda env: {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()})
    store = DirectoryStore(tmp_path / "store")
    manifest = LocalDispatcher(worker).run(_group("rc7"), _runs(tmp_path, "rc7"), AttachedAllocation(_allocation(), store))
    assert manifest.attempts[0].outcome_receipt.status == "failed"  # type: ignore[union-attr]
    assert calls


def test_p7_e_no_extra_local_import_and_typed_parsl_error() -> None:
    """E no-extra arm: LocalDispatcher stays usable without importing Parsl."""
    if HAS_PARSL:
        pytest.skip("no-extra arm is only meaningful when parsl is absent")
    from tpen_executor.dispatch import parsl as parsl_module

    assert "parsl" not in sys.modules
    assert "parsl" not in parsl_module.__dict__
    worker = SimpleNamespace(materialize=lambda env: {})
    with pytest.raises(ParslUnavailableError, match="optional executor extra"):
        ParslDispatcher(worker)


def test_p7_f_worker_binding_is_frozen_and_hash_consistent() -> None:
    from tpen_executor.placement import WorkerBinding

    binding = WorkerBinding.from_mapping({"environment": {"CUDA_VISIBLE_DEVICES": "0"}, "worker_id": "w0"})
    assert binding == WorkerBinding.from_mapping({"environment": {"CUDA_VISIBLE_DEVICES": "0"}, "worker_id": "w0"})
    assert hash(binding) == hash(WorkerBinding.from_mapping({"environment": {"CUDA_VISIBLE_DEVICES": "0"}, "worker_id": "w0"}))


def test_p7_f_binding_preserves_slot_worker_identity_from_allocation_facts() -> None:
    from tpen_executor.placement import binding_from_allocation

    allocation = ConfiguredAllocation(
        AllocationId("identity-allocation"),
        BackendJobId("identity-job"),
        {"accelerator_visibility": ["2"], "nodes": ["node-7"]},
    )
    binding = binding_from_allocation(allocation)
    assert binding.worker_id is None
    assert binding.node == "node-7"
    assert binding.env["CUDA_VISIBLE_DEVICES"] == "2"


def test_p7_g_mutant_reused_dfk_is_rejected_by_fresh_lifecycle(tmp_path: Path) -> None:
    """G mutant: each Parsl run owns one fresh DFK and cleanup call."""
    if not HAS_PARSL:
        pytest.skip("requires the approved executor parsl extra")
    package = _make_stub_package(tmp_path)
    dispatcher = ParslDispatcher(_worker_spec(PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})))
    first_store = DirectoryStore(tmp_path / "first")
    second_store = DirectoryStore(tmp_path / "second")
    dispatcher.run(_group("first"), _runs(tmp_path, "first"), AttachedAllocation(_allocation(allocation_id="first"), first_store))
    dispatcher.run(_group("second"), _runs(tmp_path, "second"), AttachedAllocation(_allocation(allocation_id="second"), second_store))
    assert (first_store.root / "groups").exists()
    assert (second_store.root / "groups").exists()


@PARSL_REQUIRED
def test_p7_b_dfk_cleanup_after_cancellation_allows_next_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B: cancellation cleanup is complete before a later run starts."""
    package = _make_stub_package(tmp_path)
    dispatcher = ParslDispatcher(_worker_spec(PreparedPythonEnvironment(sys.executable, env={"PYTHONPATH": str(package)})))
    cancelled = {"seen": False}

    def raise_cancel(*args, **kwargs):
        cancelled["seen"] = True
        raise KeyboardInterrupt("P7 cancellation probe")

    import tpen_executor.dispatch.parsl as parsl_module

    monkeypatch.setattr(parsl_module, "_run_parsl_child", raise_cancel)
    first_store = DirectoryStore(tmp_path / "cancelled")
    with pytest.raises(KeyboardInterrupt, match="P7 cancellation probe"):
        dispatcher.run(_group("cancelled"), _runs(tmp_path, "cancelled"), AttachedAllocation(_allocation(allocation_id="cancelled"), first_store))
    assert cancelled["seen"]
    assert first_store.manifest(_group("cancelled")).attempts[0].outcome_receipt.status == "failed"  # type: ignore[union-attr]

    monkeypatch.undo()
    second_store = DirectoryStore(tmp_path / "after-cancel")
    dispatcher.run(_group("after"), _runs(tmp_path, "after"), AttachedAllocation(_allocation(allocation_id="after"), second_store))
    assert second_store.manifest(_group("after")).attempts[0].outcome_receipt.status == "completed"  # type: ignore[union-attr]


def test_p7_g_mutant_dropped_evidence_cannot_hide_missing_receipt(tmp_path: Path) -> None:
    """G mutant: evidence absence preserves status and leaves digest empty."""
    class Process:
        def run(self, environment, args, **kwargs):
            return ProcessOutcome(tuple(args), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)

    class Evidence:
        def read(self, *args, **kwargs):
            from tpen_executor.runners import EvidenceUnavailable

            raise EvidenceUnavailable("missing receipt")

    worker = SimpleNamespace(materialize=lambda env: {"environment": object(), "process": Process(), "runner": object(), "evidence": Evidence()})
    manifest = LocalDispatcher(worker).run(_group("missing"), _runs(tmp_path, "missing"), AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "store")))
    receipt = manifest.attempts[0].outcome_receipt
    assert receipt is not None and receipt.status == "completed" and receipt.result_digest is None


def test_p7_g_mutant_shared_output_path_is_fenced(tmp_path: Path) -> None:
    """G mutant: two members cannot share a dispatcher-owned attempt path."""
    group = _group("left", "right")
    store = DirectoryStore(tmp_path / "store")
    dispatcher = LocalDispatcher(SimpleNamespace(materialize=lambda env: {
        "environment": object(),
        "process": SimpleNamespace(run=lambda environment, args, **kwargs: ProcessOutcome(tuple(args), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)),
        "runner": object(),
        "evidence": SimpleNamespace(read=lambda *args, **kwargs: (_ for _ in ()).throw(Exception("no evidence"))),
    }))
    manifest = dispatcher.run(group, _runs(tmp_path, "left", "right"), AttachedAllocation(_allocation(), store))
    output_dirs = [attempt.start_receipt.output_dir for attempt in manifest.attempts]  # type: ignore[union-attr]
    assert len(output_dirs) == len(set(output_dirs)) == 2

