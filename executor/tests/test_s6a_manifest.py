"""S6A-A manifest reconstruction for v1 and v2 ready groups."""

from __future__ import annotations

from pathlib import Path

import pytest

from tpen_executor._codec import canonical_bytes
from tpen_executor.results import StartReceipt
from tpen_executor.storage import DirectoryStore

from test_b1a_b_publication import _fixture, _publish
from test_p8_manifest import _evidence


MODULE = Path(__file__).parents[1] / "src" / "tpen_executor" / "storage.py"
MODULE_NAME = "executor/src/tpen_executor/storage.py"
MUTANTS = (
    (
        "v2-manifest-drops-admissions",
        MODULE_NAME,
        "admissions.get((attempt.task_id, attempt.attempt_id)),",
        "None,",
        "test_v2_manifest_rebuilds_attempts_segments_and_admissions",
    ),
    (
        "v2-manifest-group-id-projection",
        MODULE_NAME,
        "manifest_group_id = group.group_id.value",
        "manifest_group_id = group.identity",
        "test_v2_manifest_rebuilds_attempts_segments_and_admissions",
    ),
    (
        "v1-routed-through-v2-branch",
        MODULE_NAME,
        "if isinstance(group, ReadyGroupV2):\n            manifest_group_id = group.group_id.value",
        "if True:\n            manifest_group_id = group.group_id.value",
        "test_v1_manifest_path_keeps_legacy_encoding",
    ),
    (
        "v2-manifest-skips-admission-context",
        MODULE_NAME,
        """                if isinstance(group, ReadyGroupV2):
                    binding = admission.successor_binding
                    member = next(member for member in group.members if member.task_id == task_id)
                    if (
                        binding is None
                        or binding.allocation_attempt.group_id != group.group_id
                        or binding.group_digest != group.identity
                        or binding.lineage_id != member.lineage_id
                    ):
                        raise ValueError(\"successor admission binding is foreign to v2 group\")
""",
        """                if False:
                    pass
""",
        "test_review_s6aa_manifest_checks_admission_context",
    ),
)


def test_v2_manifest_rebuilds_attempts_segments_and_admissions(tmp_path: Path) -> None:
    """Rebuild v2 attempts, accepted segments, and successor admissions."""

    store, group, _lease, _inputs, _bindings, _successors, fences = _fixture(
        tmp_path, ("successor", "completed"), remote=False
    )
    _publish((store, group, _lease, _inputs, _bindings, _successors, fences))

    completed = fences["completed"]
    core_dir = Path(completed.output_dir) / "core"
    checkpoint_dir = core_dir / "checkpoint" / "step_000001"
    evidence = _evidence(str(core_dir), checkpoint_dir=str(checkpoint_dir))
    accepted = store.accept_segment(completed, evidence)

    manifest = DirectoryStore(store.root).manifest(group)

    assert manifest.group_id == group.group_id.value
    assert manifest.group_digest == group.identity
    assert {attempt.task_id for attempt in manifest.attempts} == {"successor", "completed"}
    successor = next(attempt for attempt in manifest.attempts if attempt.task_id == "successor")
    assert successor.successor_admission is not None
    assert manifest.segments == (accepted,)


def test_v2_manifest_rejects_crossed_receipt_identity(tmp_path: Path) -> None:
    """A receipt carrying another v2 group's digest is never projected."""

    store, group, _lease, _inputs, _bindings, _successors, fences = _fixture(
        tmp_path, ("completed",), remote=False
    )
    start_path = Path(fences["completed"].output_dir) / "start.json"
    start = StartReceipt.decode(start_path.read_bytes())
    crossed = dict(start.to_dict())
    crossed["group_digest"] = "f" * 64
    start_path.write_bytes(canonical_bytes(crossed))

    with pytest.raises((ValueError, TypeError)):
        store.manifest(group)


def test_v1_manifest_path_keeps_legacy_encoding(tmp_path: Path) -> None:
    """The v1 path still emits the existing RunManifest wire shape."""

    from test_p8_manifest import _allocation, _group
    from tpen_executor.results import RunManifest

    group = _group()
    store = DirectoryStore(tmp_path / "store")
    fence = store.begin_attempt(group, "task-a", _allocation(), "2026-11-05T00:00:00Z")
    store.finish_attempt(fence, "completed", "2026-11-05T00:01:00Z")

    manifest = store.manifest(group)
    assert manifest.group_id == group.group_id
    assert manifest.group_digest == group.identity
    assert [attempt.task_id for attempt in manifest.attempts] == ["task-a"]
    assert RunManifest.decode(manifest.encode()) == manifest
    assert manifest.schema == "tpen-executor/run-manifest-v2"


def test_s6a_mutant_table_targets_unique_compilable_killers() -> None:
    """Each bounded S6A mutant has one source anchor and a named killer."""

    source = MODULE.read_text(encoding="utf-8")
    names = {
        name
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    }
    import test_review_s6aa_manifest as reviewer_tests
    names.update(
        name
        for name, value in vars(reviewer_tests).items()
        if name.startswith("test_") and callable(value)
    )
    anchors = [row[2] for row in MUTANTS]
    assert len(anchors) == len(set(anchors))
    for _name, filename, before, after, killer in MUTANTS:
        assert filename == MODULE_NAME
        assert source.count(before) == 1
        assert killer in names
        compile(source.replace(before, after, 1), filename, "exec")
