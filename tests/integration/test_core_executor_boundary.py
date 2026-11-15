"""CX-05 two-leaf core/executor round-trip coverage."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from tpen.artifacts import ArtifactDestination
from tpen.callback.planned_yield import PlannedYield
from tpen.checkpoint import (
    COMPLETE_MARKER,
    CheckpointRef,
    iter_valid_publication_receipts,
    publication_receipt_path,
    publication_catalog_path,
    read_latest,
    read_publications,
)
from tpen.checkpoint import CheckpointRestored
from tpen.execution.recovery import (
    OperationalRestoreRequest,
    ScienceIntent,
    SuccessorAdmissionResult,
    select_successor_admission,
)
from tpen.inspection import inspect_run
from tpen.run import PLANNED_YIELD_EXIT_CODE, run_from_config
from tpen.training.events import TrainingIterationCompleted


_FIXTURE = Path("tests/integration/artifacts/training/vmc_smoke.yaml")
_TASK_ID = "cx05"
_SUCCESSOR_ATTEMPT_ID = "cx05-successor"


_EXECUTOR_CHILD = r'''
import builtins
import json
import sys
from pathlib import Path

_real_import = builtins.__import__
_tpen_imports = []


def _block_tpen_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tpen" or name.startswith("tpen."):
        _tpen_imports.append(name)
        raise ModuleNotFoundError("executor child must not import tpen")
    return _real_import(name, globals, locals, fromlist, level)


builtins.__import__ = _block_tpen_import


def _emit(value):
    if _tpen_imports:
        raise RuntimeError(f"executor child imported tpen: {_tpen_imports!r}")
    if not isinstance(value, dict):
        raise TypeError("executor child result must be a mapping")
    value["_executor_boundary"] = {"tpen_imports": list(_tpen_imports)}
    print(json.dumps(value, sort_keys=True))

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, AttemptId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.runners import CoreRunEvidence
from tpen_executor.storage import AttemptFence, DirectoryStore, FenceRejected

if any(name == "tpen" or name.startswith("tpen.") for name in sys.modules):
    raise RuntimeError("executor child started with tpen already imported")


def group():
    program = ProgramSpecRef({}, digest({}))
    member = TaskSpec("cx05", "cx05", program, digest("cx05"), 3, "steps")
    return ReadyGroup("cx05", (member,), digest("cx05-group"))


def allocation():
    return ConfiguredAllocation(AllocationId("cx05-allocation"), BackendJobId("cx05-job"), {})


def fence_from(payload):
    return AttemptFence(
        str(Path(payload["store_root"]).resolve()),
        payload["group_digest"],
        payload["task_id"],
        payload["allocation_id"],
        AttemptId(payload["attempt_id"]),
        payload["output_dir"],
    )


action = sys.argv[1]
payload = json.loads(sys.argv[2])
store = DirectoryStore(payload["store_root"])
ready_group = group()

if action == "init":
    fence = store.begin_attempt(ready_group, "cx05", allocation(), "2026-11-14T00:00:00Z")
    core_dir = Path(fence.output_dir) / "core"
    core_dir.mkdir()
    _emit({
        "store_root": str(Path(store.root).resolve()),
        "group_digest": fence.group_digest,
        "task_id": fence.task_id,
        "allocation_id": fence.allocation_id,
        "attempt_id": fence.attempt_id.value,
        "output_dir": fence.output_dir,
        "core_dir": str(core_dir),
    })
elif action == "record-core":
    fence = fence_from(payload)
    evidence = CoreRunEvidence.from_mapping(
        json.loads(Path(payload["evidence_path"]).read_text(encoding="utf-8"))
    )
    store.finish_attempt(fence, "failed", "2026-11-14T00:01:00Z")
    record = store.record_core_outcome(fence, evidence)
    _emit({"record": record.to_dict()})
elif action == "build-request":
    fence = fence_from(payload)
    try:
        request = store.build_successor_admission_request(
            fence,
            AttemptId(payload["successor_attempt_id"]),
            payload["science_input_digest"],
            payload["target_iterations"],
        )
    except FenceRejected as error:
        _emit({
            "error": {"type": type(error).__name__, "message": str(error)},
        })
    else:
        _emit(request)
elif action == "record-admission":
    fence = fence_from(payload)
    admission = store.record_successor_admission(
        fence, payload["request"], payload["core_acceptance_digest"]
    )
    ledger_path = (
        Path(fence.output_dir).parent
        / "successor-admissions"
        / fence.attempt_id.value
        / "successor-admission.json"
    )
    manifest = store.manifest(ready_group)
    _emit({
        "admission": admission.to_dict(),
        "ledger": json.loads(ledger_path.read_text(encoding="utf-8")),
        "manifest": manifest.to_dict(),
    })
else:
    raise SystemExit(f"unknown child action: {action}")
'''


def _cfg(root: Path, run_id: str, *, max_steps: int = 3) -> DictConfig:
    """Build the reduced real VMC configuration used by both leaves.

    Parameters
    ----------
    root : pathlib.Path
        Logical artifact root.
    run_id : str
        Stable logical run identity shared by the two leaves.
    max_steps : int, optional
        Fixed total horizon for the run.

    Returns
    -------
    omegaconf.DictConfig
        A resolved-by-the-entrypoint training configuration.
    """

    cfg = OmegaConf.load(_FIXTURE)
    cfg.run.root = str(root)
    cfg.run.run_id = run_id
    cfg.trainer.max_steps = max_steps
    cfg.sampler.n_walkers = 8
    cfg.sampler.burn_in = 1
    cfg.sampler.n_steps = 1
    for callback in cfg.callbacks:
        if callback.get("_target_") == "tpen.callback.Checkpoint":
            callback.schedule.every_n = 1
    return cfg


def _child_payload(fence: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the JSON-safe identity facts returned by the child initializer."""

    return {
        "store_root": str(fence["store_root"]),
        "group_digest": str(fence["group_digest"]),
        "task_id": str(fence["task_id"]),
        "allocation_id": str(fence["allocation_id"]),
        "attempt_id": str(fence["attempt_id"]),
        "output_dir": str(fence["output_dir"]),
    }


def _executor_environment() -> dict[str, str]:
    """Build an executor-only environment without Python startup overrides."""

    environment = os.environ.copy()
    for variable in (
        "PYTHONHOME",
        "PYTHONOPTIMIZE",
        "PYTHONNOUSERSITE",
        "PYTHONUSERBASE",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        "PYTHONWARNINGS",
    ):
        environment.pop(variable, None)
    executor_src = Path(__file__).resolve().parents[2] / "executor" / "src"
    environment["PYTHONPATH"] = str(executor_src)
    return environment


def _executor_call(tmp_path: Path, action: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run one executor-only child operation and decode its JSON result.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Per-test root used as the child working directory.
    action : str
        Child operation name.
    payload : mapping
        JSON-safe operation input.

    Returns
    -------
    dict
        JSON result emitted by the child.
    """

    try:
        completed = subprocess.run(
            [sys.executable, "-c", _EXECUTOR_CHILD, action, json.dumps(dict(payload), sort_keys=True)],
            cwd=tmp_path,
            env=_executor_environment(),
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"executor child timed out ({action}, timeout=120s):\n"
            f"stdout={error.stdout}\nstderr={error.stderr}"
        ) from error
    if completed.returncode != 0:
        raise AssertionError(
            f"executor child failed ({action}, rc={completed.returncode}):\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    if completed.stderr:
        raise AssertionError(f"executor child wrote stderr ({action}): {completed.stderr}")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise AssertionError(f"executor child result is not a mapping ({action}): {result!r}")
    boundary = result.get("_executor_boundary")
    if not isinstance(boundary, dict) or boundary.get("tpen_imports") != []:
        raise AssertionError(
            f"executor child crossed tpen import boundary ({action}): {boundary!r}"
        )
    result.pop("_executor_boundary", None)
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    """Encode a JSON mapping with the executor's canonical separators."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _restore_request_bytes(request: OperationalRestoreRequest) -> bytes:
    """Encode every closed field consumed by an operational restore request."""

    return _canonical_json({
        "schema": request.SCHEMA,
        "decision": request.decision.to_dict(),
        "ref": request.ref.to_dict(),
        "target_iterations": request.target_iterations,
        "next_iteration": request.next_iteration,
        "completed_updates": request.completed_updates,
        "load": request.load,
    })


def _content_tree_hash(root: Path) -> str:
    """Hash sorted relative entries, modes, and file bytes under one leaf."""

    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in entries:
        relative = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_dir():
            entry_type = "directory"
        elif path.is_file():
            entry_type = "file"
        else:
            entry_type = "other"
        digest.update(_canonical_json({"name": relative, "type": entry_type, "mode": mode}))
        if entry_type == "file":
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _guarded_successor_start(
    admission: SuccessorAdmissionResult | None,
    cfg: DictConfig,
    destination: ArtifactDestination,
    sentinel: Path,
    core_calls: list[str],
    runtime_callbacks: tuple[object, ...] = (),
) -> int | None:
    """Start a successor only after a typed accepted restore request exists."""

    if admission is None or admission.restore_request is None:
        return None
    sentinel.write_text("accepted restore request passed\n", encoding="utf-8")
    core_calls.append("successor-run_from_config")
    return run_from_config(
        cfg,
        restore_request=admission.restore_request,
        artifact_destination=destination,
        runtime_callbacks=runtime_callbacks,
        raise_exceptions=True,
    )


def _assert_refused_without_successor(
    admission: SuccessorAdmissionResult | None,
    code: str | None,
    cfg: DictConfig,
    destination: ArtifactDestination,
    successor_root: Path,
    sentinel: Path,
    core_calls: list[str],
    admission_ledger: Path,
) -> None:
    """Route one refusal through the guarded launch and prove no side effect."""

    if admission_ledger.exists():
        raise AssertionError("node=refusal-no-successor-admission-ledger-before")
    if admission is not None:
        if admission.restore_request is not None:
            raise AssertionError("node=refusal-no-restore-request")
        if admission.refusal is None:
            raise AssertionError("node=refusal-has-typed-refusal")
        if admission.refusal.code.value != code:
            raise AssertionError(f"node={code}-refusal: {admission.refusal.code.value!r}")
    elif code is not None:
        raise AssertionError("node=refusal-missing-result")
    started = _guarded_successor_start(
        admission,
        cfg,
        destination,
        sentinel,
        core_calls,
    )
    if started is not None:
        raise AssertionError("node=refusal-no-successor-return")
    if sentinel.exists():
        raise AssertionError("node=refusal-no-sentinel")
    if successor_root.exists():
        raise AssertionError("node=refusal-no-physical-leaf")
    if core_calls != []:
        raise AssertionError("node=refusal-no-successor-core-call")
    if admission_ledger.exists():
        raise AssertionError("node=refusal-no-successor-admission-ledger-after")


class _RestoreReadback:
    """Typed callback readback for the restored predecessor coordinate."""

    def __init__(self) -> None:
        self.events: list[CheckpointRestored] = []

    def handle_occurrence(self, occurrence: object, context: object) -> None:
        if isinstance(occurrence.event, CheckpointRestored):
            self.events.append(occurrence.event)


class _TrainingReadback:
    """Count training updates observed by the refused-yield control."""

    def __init__(self) -> None:
        self.events: list[TrainingIterationCompleted] = []

    def handle_occurrence(self, occurrence: object, context: object) -> None:
        if isinstance(occurrence.event, TrainingIterationCompleted):
            self.events.append(occurrence.event)


@pytest.mark.integration
def test_core_executor_two_leaf_round_trip_and_refusal_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise one planned-yield leaf, admission ledger, and restored successor.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned temporary root for every logical, physical, and executor
        artifact written by this test.
    monkeypatch : pytest.MonkeyPatch
        Test-local admission of the sink-free runtime readback callbacks.
    """

    logical_root = tmp_path / "logical"
    executor_root = tmp_path / "executor-store"
    successor_root = tmp_path / "successor-physical"
    successor_sentinel = tmp_path / "successor-started.json"
    successor_core_calls: list[str] = []

    import tpen.run as run_module

    qualified_readback = f"{_RestoreReadback.__module__}.{_RestoreReadback.__qualname__}"
    qualified_training_readback = (
        f"{_TrainingReadback.__module__}.{_TrainingReadback.__qualname__}"
    )
    monkeypatch.setattr(
        run_module,
        "_SINK_FREE_TYPES",
        run_module._SINK_FREE_TYPES | {qualified_readback, qualified_training_readback},
    )

    # Flow 1: begin the executor fence and place core evidence exactly beneath it.
    fence_facts = _executor_call(tmp_path, "init", {"store_root": str(executor_root)})
    predecessor_core = Path(fence_facts["core_dir"])
    predecessor_destination = ArtifactDestination(predecessor_core)
    admission_ledger = (
        Path(fence_facts["output_dir"]).parent
        / "successor-admissions"
        / fence_facts["attempt_id"]
        / "successor-admission.json"
    )

    # Flow 2: run the real predecessor through the public planned-yield boundary.
    predecessor_cfg = _cfg(logical_root, _TASK_ID)
    predecessor_code = run_from_config(
        predecessor_cfg,
        artifact_destination=predecessor_destination,
        runtime_callbacks=(PlannedYield(1, 3),),
        raise_exceptions=True,
    )
    assert predecessor_code == PLANNED_YIELD_EXIT_CODE
    predecessor = inspect_run(predecessor_core)
    assert predecessor.schema == "tpen.run-evidence/v3"
    assert predecessor.outcome == "planned_yield"
    assert predecessor.verdict == "incomplete"
    assert predecessor.run_status == "planned_yield"
    assert predecessor.continuation_eligible is True
    assert predecessor.checkpoint_ref is not None
    assert predecessor.checkpoint_ref["profile"] == "train_resume"
    assert predecessor.checkpoint is not None

    # Flow 3: prove the checkpoint is committed and all three publication artifacts exist.
    checkpoint_dir = Path(predecessor.checkpoint_ref["checkpoint_dir"])
    checkpoint_root = checkpoint_dir.parent
    assert (checkpoint_dir / COMPLETE_MARKER).is_file()
    ref = CheckpointRef.from_directory(checkpoint_dir)
    assert ref.content_id == predecessor.checkpoint_ref["content_id"]
    publications = read_publications(publication_catalog_path(checkpoint_root))
    assert any(
        row.content_id == ref.content_id and row.checkpoint_dir.resolve() == checkpoint_dir.resolve()
        for row in publications
    )
    latest = read_latest(checkpoint_root)
    assert checkpoint_root / latest["checkpoint_dir"] == checkpoint_dir
    receipts = tuple(iter_valid_publication_receipts(publication_receipt_path(checkpoint_root)))
    assert any(row["summary"]["content_id"] == ref.content_id for row in receipts)

    # Flow 4: let the executor child decode root evidence and persist a typed outcome only.
    evidence_path = predecessor_core / "root-evidence.json"
    evidence_path.write_bytes(_canonical_json(predecessor.to_dict()))
    executor_identity = _child_payload(fence_facts)
    record_result = _executor_call(
        tmp_path,
        "record-core",
        {**executor_identity, "evidence_path": str(evidence_path)},
    )
    core_outcome_path = Path(fence_facts["output_dir"]) / "core-outcome.json"
    assert core_outcome_path.is_file()
    assert core_outcome_path.read_bytes() == _canonical_json(record_result["record"])
    assert record_result["record"]["evidence"]["outcome"] == "planned_yield"
    assert record_result["record"]["continuation_eligible"] is True
    assert not (Path(fence_facts["output_dir"]) / "segment.json").exists()

    # Flow 5: build the opaque executor request, then have core accept it by identity.
    intent = ScienceIntent({"fixture": "cx05-two-leaf"}, target_iterations=3)
    request = _executor_call(
        tmp_path,
        "build-request",
        {
            **executor_identity,
            "successor_attempt_id": _SUCCESSOR_ATTEMPT_ID,
            "science_input_digest": intent.science_input_digest,
            "target_iterations": 3,
        },
    )
    accepted = select_successor_admission(intent, request)
    assert accepted.accepted
    assert accepted.restore_request is not None
    restore_request = accepted.restore_request
    restore_digest = hashlib.sha256(_restore_request_bytes(restore_request)).hexdigest()
    request_digest = hashlib.sha256(_canonical_json(request)).hexdigest()
    successor_cfg = _cfg(logical_root, _TASK_ID)
    assert not admission_ledger.exists()

    # Control 1: missing core outcome is refused before a successor can start.
    missing_core_bytes = core_outcome_path.read_bytes()
    core_outcome_path.unlink()
    try:
        missing_result = _executor_call(tmp_path, "build-request", {
            **executor_identity,
            "successor_attempt_id": _SUCCESSOR_ATTEMPT_ID,
            "science_input_digest": intent.science_input_digest,
            "target_iterations": 3,
        })
    finally:
        core_outcome_path.write_bytes(missing_core_bytes)
    assert missing_result == {
        "error": {
            "type": "FenceRejected",
            "message": "core outcome record is missing",
        }
    }
    _assert_refused_without_successor(
        None,
        None,
        successor_cfg,
        ArtifactDestination(successor_root),
        successor_root,
        successor_sentinel,
        successor_core_calls,
        admission_ledger,
    )

    # Control 2: stale checkpoint bytes are refused by core and then restored exactly.
    payload_file = next(
        path for path in sorted(checkpoint_dir.rglob("*"))
        if path.is_file() and path.name not in {COMPLETE_MARKER, "manifest.json"}
    )
    payload_bytes = payload_file.read_bytes()
    payload_file.write_bytes(bytes([payload_bytes[0] ^ 1]) + payload_bytes[1:])
    try:
        stale = select_successor_admission(intent, request)
    finally:
        payload_file.write_bytes(payload_bytes)
    _assert_refused_without_successor(
        stale,
        "stale-ref",
        successor_cfg,
        ArtifactDestination(successor_root),
        successor_root,
        successor_sentinel,
        successor_core_calls,
        admission_ledger,
    )

    # Control 3: remove only the publication receipt while latest and catalog remain.
    receipt_path = publication_receipt_path(checkpoint_root)
    receipt_bytes = receipt_path.read_bytes()
    receipt_path.write_bytes(b"")
    try:
        unpublished = select_successor_admission(intent, request)
    finally:
        receipt_path.write_bytes(receipt_bytes)
    _assert_refused_without_successor(
        unpublished,
        "missing-publication",
        successor_cfg,
        ArtifactDestination(successor_root),
        successor_root,
        successor_sentinel,
        successor_core_calls,
        admission_ledger,
    )

    # Control 4: a model-only profile cannot be promoted to a successor restore.
    model_only_request = json.loads(json.dumps(request))
    model_only_request["core_evidence"]["checkpoint_ref"]["profile"] = "model_only"
    model_only = select_successor_admission(intent, model_only_request)
    _assert_refused_without_successor(
        model_only,
        "model-only",
        successor_cfg,
        ArtifactDestination(successor_root),
        successor_root,
        successor_sentinel,
        successor_core_calls,
        admission_ledger,
    )

    # Control 5: a real pre-fit refused yield is typed as non-eligible evidence.
    refused_root = tmp_path / "refused-physical"
    refused_cfg = _cfg(logical_root, _TASK_ID)
    training_readback = _TrainingReadback()
    refused_code = run_from_config(
        refused_cfg,
        artifact_destination=ArtifactDestination(refused_root),
        runtime_callbacks=(PlannedYield(3, 3), training_readback),
        raise_exceptions=True,
    )
    # _status_to_exit_code maps every refusal to 1; planned yield is the
    # distinct PLANNED_YIELD_EXIT_CODE (2).
    assert refused_code == 1
    assert refused_code != PLANNED_YIELD_EXIT_CODE
    refused = inspect_run(refused_root)
    assert refused.schema == "tpen.run-evidence/v3"
    assert refused.outcome == "refused"
    assert refused.run_status == "refused"
    assert refused.refusal is None
    assert refused.continuation_eligible is False
    assert refused.checkpoint is None
    assert training_readback.events == []
    assert not any(
        path.is_dir() for path in (refused_root / "checkpoints").glob("step_*")
    )
    refused_request = json.loads(json.dumps(request))
    refused_request["core_evidence"] = refused.to_dict()
    refused_admission = select_successor_admission(intent, refused_request)
    _assert_refused_without_successor(
        refused_admission,
        "non-eligible-outcome",
        successor_cfg,
        ArtifactDestination(successor_root),
        successor_root,
        successor_sentinel,
        successor_core_calls,
        admission_ledger,
    )

    # Flow 6: record exactly the accepted request and read the write-once ledger back.
    admission_result = _executor_call(
        tmp_path,
        "record-admission",
        {
            **executor_identity,
            "request": request,
            "core_acceptance_digest": restore_digest,
        },
    )
    ledger = admission_result["ledger"]
    assert ledger["predecessor"]["content_id"] == predecessor.checkpoint_ref["content_id"], "node=admission-ledger-content"
    assert ledger["successor"] == {"task_id": _TASK_ID, "attempt_id": _SUCCESSOR_ATTEMPT_ID}, "node=admission-ledger-successor"
    assert ledger["request_digest"] == request_digest, "node=admission-ledger-request-digest"
    assert ledger["core_acceptance_digest"] == restore_digest, "node=admission-ledger-core-digest"
    assert admission_result["manifest"]["segments"] == [], "node=admission-no-accepted-segment"
    predecessor_attempt = next(
        attempt
        for attempt in admission_result["manifest"]["attempts"]
        if attempt["attempt_id"] == executor_identity["attempt_id"]
    )
    assert predecessor_attempt["outcome_receipt"]["status"] == "failed"
    assert predecessor_attempt["successor_admission"]["predecessor"]["content_id"] == ref.content_id
    assert predecessor_attempt["successor_admission"]["successor"] == {
        "task_id": _TASK_ID,
        "attempt_id": _SUCCESSOR_ATTEMPT_ID,
    }

    # Flow 7: bracket the successor with the predecessor physical tree hash.
    predecessor_tree = _content_tree_hash(predecessor_core)
    restore_readback = _RestoreReadback()
    successor_code = _guarded_successor_start(
        accepted,
        successor_cfg,
        ArtifactDestination(successor_root),
        successor_sentinel,
        successor_core_calls,
        (restore_readback,),
    )
    assert successor_code == 0
    assert successor_core_calls == ["successor-run_from_config"]
    assert len(restore_readback.events) == 1, "node=restore-request-event"
    report = restore_readback.events[0].report
    assert report.next_iteration == predecessor.checkpoint["next_iteration"], "node=restore-request-progress-coordinate"
    assert report.completed_updates == predecessor.checkpoint["completed_updates"], "node=restore-request-progress-coordinate"

    # Flow 8: inspect completion, independent counters, and unchanged predecessor bytes.
    successor = inspect_run(successor_root)
    assert successor.outcome == "completed"
    assert successor.verdict == "completed"
    assert successor.checkpoint is not None
    assert successor.checkpoint["next_iteration"] >= predecessor.checkpoint["next_iteration"], "node=successor-counter-monotonicity"
    assert successor.checkpoint["completed_updates"] >= predecessor.checkpoint["completed_updates"], "node=successor-counter-monotonicity"
    assert successor.checkpoint["next_iteration"] == 3
    assert successor.checkpoint["completed_updates"] == 3
    successor_checkpoint_dir = Path(successor.checkpoint_ref["checkpoint_dir"])
    successor_checkpoint_root = successor_checkpoint_dir.parent
    assert (successor_checkpoint_dir / COMPLETE_MARKER).is_file()
    successor_ref = CheckpointRef.from_directory(successor_checkpoint_dir)
    assert successor_ref.content_id == successor.checkpoint_ref["content_id"]
    successor_publications = read_publications(
        publication_catalog_path(successor_checkpoint_root)
    )
    assert any(
        row.content_id == successor_ref.content_id
        and row.checkpoint_dir.resolve() == successor_checkpoint_dir.resolve()
        for row in successor_publications
    )
    successor_latest = read_latest(successor_checkpoint_root)
    assert successor_checkpoint_root / successor_latest["checkpoint_dir"] == successor_checkpoint_dir
    successor_receipts = tuple(
        iter_valid_publication_receipts(publication_receipt_path(successor_checkpoint_root))
    )
    assert any(row["summary"]["content_id"] == successor_ref.content_id for row in successor_receipts)
    assert _content_tree_hash(predecessor_core) == predecessor_tree, "node=physical-leaf-tree-fence"

    # Flow 9: prove C is byte-identical while both physical leaves remain separate.
    predecessor_resolved = predecessor_core / "resolved_config.yaml"
    successor_resolved = successor_root / "resolved_config.yaml"
    assert predecessor_resolved.read_bytes() == successor_resolved.read_bytes(), "node=logical-config-identity"
    assert predecessor.resolved_config_identity == successor.resolved_config_identity, "node=logical-config-identity"
    predecessor_resolved_cfg = OmegaConf.load(predecessor_resolved)
    successor_resolved_cfg = OmegaConf.load(successor_resolved)
    assert predecessor_resolved_cfg.trainer.max_steps == 3
    assert successor_resolved_cfg.trainer.max_steps == 3
    logical_run_dir = predecessor_resolved_cfg.run.dir
    assert isinstance(logical_run_dir, str) and logical_run_dir
    assert logical_run_dir == successor_resolved_cfg.run.dir
    logical_run_path = Path(logical_run_dir)
    assert not logical_run_path.exists()
    assert logical_run_path != predecessor_core
    assert logical_run_path != successor_root
    assert Path(predecessor.run_dir).is_relative_to(predecessor_core)
    assert Path(successor.run_dir).is_relative_to(successor_root)
    assert Path(predecessor.checkpoint["checkpoint_dir"]).is_relative_to(predecessor_core)
    assert Path(successor.checkpoint["checkpoint_dir"]).is_relative_to(successor_root)
    assert predecessor_core != successor_root
    assert successor_sentinel.is_file()
