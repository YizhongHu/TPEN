"""Independent round-1 review tests for E1 checkpoint-recovery contract.

Reviewed head: codex/execution/e1-recovery @ c8570c211774aea98c015c8199373516babe4fc1.
Polarity is stated per test relative to that exact head; a change to
``tpen/execution/recovery.py`` after this review invalidates the polarity
claims below.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from tpen.checkpoint.payload import TrainResume
from tpen.checkpoint.reference import CheckpointRef
from tpen.checkpoint.restore import _load_config
from tpen.execution import (
    CheckpointEvidence,
    RecoveryOutcome,
    RecoveryPhase,
    RefusalCode,
    capture_science_intent,
    select_recovery,
)


def _manifest(
    step: int,
    *,
    completed_updates: int = 3,
    payload: dict[str, object] | None = None,
    files: dict[str, str] | None = None,
    schema_version: int = 2,
    kind: str = "tpen.checkpoint",
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "kind": kind,
        "next_iteration": step,
        "completed_updates": completed_updates,
        "created_at_unix": 1.0,
        "files": files or {"model": "model.pt"},
        "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "e1-review-r1", "git_sha": "test"},
        **({} if payload is None else {"payload": payload}),
    }


def _checkpoint(
    root: Path,
    *,
    step: int = 5,
    completed_updates: int = 3,
    payload: dict[str, object] | None = None,
    full_state: bool = True,
    complete: bool = True,
) -> Path:
    path = root / f"step_{step:06d}"
    path.mkdir(parents=True)
    files = {"model": "model.pt"}
    if full_state:
        files.update(
            {
                "optimizer": "optimizer.pt",
                "trainer": "trainer.json",
                "sampler": "sampler.pt",
                "rng": "rng.pt",
            }
        )
    for name in files.values():
        (path / name).write_bytes(b"payload")
    (path / "trainer.json").write_text(
        json.dumps({"next_iteration": step, "completed_updates": completed_updates}),
        encoding="utf-8",
    )
    (path / "manifest.json").write_text(
        json.dumps(
            _manifest(
                step,
                completed_updates=completed_updates,
                payload=payload,
                files=files,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if complete:
        (path / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return path


def _intent(*, target_iterations: int = 10, science_input: dict[str, object] | None = None):
    return capture_science_intent(
        science_input or {"system": "hooke", "electrons": 2},
        target_iterations=target_iterations,
        source_config={"model": {"_target_": "tests.helpers.NeverConstructed"}},
        overrides=("trainer.max_steps=10",),
    )


def test_path_evidence_with_manifest_missing_required_field_refuses_typed(tmp_path: Path) -> None:
    """Expected RED at head c8570c21.

    Exposes: a complete, v2-shaped checkpoint directory whose manifest is
    missing ``completed_updates`` must still produce a *typed* refusal
    (INCOMPLETE or UNSUPPORTED), not an uncaught exception.
    Expected red signature: uncaught ``KeyError('completed_updates')`` raised
    from ``CheckpointManifest.from_mapping`` inside
    ``CheckpointEvidence.from_path`` -> ``CheckpointRef.from_directory``,
    because ``CheckpointEvidence.from_path`` (recovery.py:242) excludes
    ``KeyError`` from its except tuple.
    """

    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, step=5, payload=TrainResume().to_manifest())
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    del manifest["completed_updates"]
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    decision = select_recovery(
        intent,
        phase=RecoveryPhase.RECOVERY,
        evidence=CheckpointEvidence.from_path(
            checkpoint_dir,
            science_input=intent.science_input,
            target_iterations=intent.target_iterations,
        ),
    )

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code in (RefusalCode.INCOMPLETE, RefusalCode.UNSUPPORTED)
    assert decision.load is None


def test_ref_evidence_with_manifest_degraded_after_ref_build_refuses_typed(tmp_path: Path) -> None:
    """Expected RED at head c8570c21.

    Exposes: a valid ``CheckpointRef`` whose on-disk manifest is degraded
    (``completed_updates`` removed) *after* the ref was built must still
    produce a typed refusal when re-checked by ``select_recovery``.
    Expected red signature: uncaught ``KeyError('completed_updates')`` from
    ``ref.validate()`` -> ``CheckpointRef.from_directory`` ->
    ``CheckpointManifest.from_mapping``, because the except tuple at
    recovery.py:479 (around ``ref.validate()``) lacks ``KeyError``. The raw
    pre-gate (schema_version/kind check) passes because those two keys are
    left intact.
    """

    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, step=5, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)

    manifest = json.loads((checkpoint_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["kind"] == "tpen.checkpoint"
    del manifest["completed_updates"]
    (checkpoint_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    evidence = CheckpointEvidence(
        ref=ref,
        science_input_digest=intent.science_input_digest,
        target_iterations=intent.target_iterations,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code in (RefusalCode.INCOMPLETE, RefusalCode.UNSUPPORTED)


def test_decision_load_is_consumable_by_existing_load_path(tmp_path: Path) -> None:
    """Expected RED at head c8570c21.

    Exposes acceptance clause "overlay exactly equal to what the existing
    load path consumes": equality of the mapping contents is not enough if
    the existing ``_load_config`` consumer cannot accept the type that
    ``decision.load`` actually is.
    Expected red signature: ``TypeError: load config must be a mapping or
    OmegaConf container`` -- ``decision.load`` is a ``MappingProxyType``,
    and ``tpen.checkpoint.restore._load_config`` (restore.py:266-273) only
    accepts ``None``, an OmegaConf container, or a plain ``dict``.
    """

    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, step=5, completed_updates=3, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input=intent.science_input,
        target_iterations=intent.target_iterations,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)

    assert _load_config(decision.load) == {"mode": "train_resume", "path": str(checkpoint_dir)}


def test_genuine_v1_checkpoint_via_path_refuses_unsupported(tmp_path: Path) -> None:
    """Expected RED at head c8570c21.

    Exposes: a genuinely from-scratch v1 (``spenn.checkpoint``) legacy
    checkpoint, reachable only via a path (never a pre-built ``CheckpointRef``,
    since v1 cannot build one), must refuse with UNSUPPORTED -- the raw
    pre-gate at recovery.py:469-475 exists precisely for this population.
    Expected red signature: ``decision.refusal.code`` is actually
    ``RefusalCode.INCOMPLETE`` ("does not identify a committed complete
    checkpoint"), not ``UNSUPPORTED``, because ``CheckpointRef.from_directory``
    refuses v1 manifests (reference.py:111-115), so
    ``CheckpointEvidence.from_path`` catches that ``ValueError`` and sets
    ``ref=None``; ``select_recovery`` then refuses at the "evidence.ref is
    None" branch (recovery.py:386-392) before the raw-manifest UNSUPPORTED
    pre-gate in ``_checkpoint_refusal`` is ever reached.
    """

    intent = _intent()
    path = tmp_path / "step_000005"
    path.mkdir(parents=True)
    (path / "model.pt").write_bytes(b"payload")
    manifest = {
        "schema_version": 1,
        "kind": "spenn.checkpoint",
        "step": 5,
        "created_at_unix": 1.0,
        "files": {"model": "model.pt"},
        "hashes": {},
        "runtime": {"device": "cpu", "dtype": "float64"},
        "provenance": {"run_id": "e1-review-r1-v1", "git_sha": "test"},
    }
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "COMPLETE").write_text("complete\n", encoding="utf-8")

    decision = select_recovery(
        intent,
        phase=RecoveryPhase.RECOVERY,
        evidence=CheckpointEvidence.from_path(
            path,
            science_input=intent.science_input,
            target_iterations=intent.target_iterations,
        ),
    )

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.UNSUPPORTED


@pytest.mark.parametrize("schema_version", [[2], {}], ids=["list", "dict"])
def test_path_evidence_with_unhashable_schema_version_refuses_unsupported(
    tmp_path: Path, schema_version: object
) -> None:
    """Malformed raw schema versions must be typed unsupported refusals."""

    intent = _intent()
    path = _checkpoint(tmp_path, step=5, payload=TrainResume().to_manifest())
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    decision = select_recovery(
        intent,
        phase=RecoveryPhase.RECOVERY,
        evidence=CheckpointEvidence.from_path(
            path,
            science_input=intent.science_input,
            target_iterations=intent.target_iterations,
        ),
    )

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.UNSUPPORTED
    assert decision.load is None


@pytest.mark.parametrize("schema_version", [[2], {}], ids=["list", "dict"])
def test_ref_evidence_with_unhashable_schema_version_refuses_unsupported(
    tmp_path: Path, schema_version: object
) -> None:
    """A degraded ref manifest must classify malformed schema versions safely."""

    intent = _intent()
    path = _checkpoint(tmp_path, step=5, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    evidence = CheckpointEvidence(
        ref=ref,
        science_input_digest=intent.science_input_digest,
        target_iterations=intent.target_iterations,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.UNSUPPORTED
    assert decision.load is None


def test_raw_path_recovery_of_foreign_lineage_checkpoint_must_not_emit_overlay(tmp_path: Path) -> None:
    """Expected RED at head c8570c21.

    Exposes: a recovery selection over raw path/ref evidence must not emit a
    load overlay for a checkpoint with no actual binding to the current
    intent's science input.
    Expected red signature: assertion failure -- ``decision.outcome`` is
    ``RecoveryOutcome.RECOVERY`` with a non-None ``decision.load``, because
    ``_coerce_evidence`` (recovery.py:448-452) stamps path/ref evidence with
    the *current* intent's own digest and target before any binding check
    runs, making the binding gate (recovery.py:372-378) structurally
    vacuous for these evidence forms.
    """

    intent_a = _intent(science_input={"system": "hooke"})
    checkpoint_dir = _checkpoint(tmp_path, step=5, completed_updates=3, payload=TrainResume().to_manifest())

    intent_b = capture_science_intent({"system": "completely-different"}, target_iterations=10)

    decision = select_recovery(
        intent_b,
        phase=RecoveryPhase.RECOVERY,
        evidence=checkpoint_dir,
    )

    assert decision.load is None
    assert intent_a.science_input_digest != intent_b.science_input_digest


def test_evidence_target_horizon_mismatch_refuses_incompatible(tmp_path: Path) -> None:
    """Expected GREEN at head c8570c21.

    Pins: evidence bound to a different fixed target horizon than the
    current intent refuses INCOMPATIBLE.
    """

    intent = _intent(target_iterations=10)
    checkpoint_dir = _checkpoint(tmp_path, step=5, completed_updates=3, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input=intent.science_input,
        target_iterations=intent.target_iterations + 10,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPATIBLE


def test_checkpoint_counters_beyond_horizon_refuse_incompatible(tmp_path: Path) -> None:
    """Expected GREEN at head c8570c21.

    Pins: a checkpoint whose own recorded counters exceed the intent's fixed
    target horizon refuses INCOMPATIBLE, even when the binding digest and
    the evidence's declared target_iterations both match the intent.
    """

    intent = _intent(target_iterations=10)
    checkpoint_dir = _checkpoint(tmp_path, step=15, completed_updates=12, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input=intent.science_input,
        target_iterations=intent.target_iterations,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPATIBLE


def test_uncommitted_evidence_refuses_incomplete(tmp_path: Path) -> None:
    """Expected GREEN at head c8570c21.

    Pins: evidence explicitly marked not committed refuses INCOMPLETE before
    any filesystem or binding check runs.
    """

    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, step=5, completed_updates=3, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence = CheckpointEvidence(
        ref=ref,
        science_input_digest=intent.science_input_digest,
        target_iterations=10,
        committed=False,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPLETE


def test_initial_phase_with_valid_evidence_selects_initial_checkpoint(tmp_path: Path) -> None:
    """Expected GREEN at head c8570c21.

    Pins: the ``initial`` phase with correctly bound, valid evidence selects
    ``INITIAL_CHECKPOINT`` and carries the ref's own counters and the exact
    train-resume load overlay.
    """

    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, step=5, completed_updates=3, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input=intent.science_input,
        target_iterations=intent.target_iterations,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.INITIAL, evidence=evidence)

    assert decision.outcome is RecoveryOutcome.INITIAL_CHECKPOINT
    assert decision.load == {"mode": "train_resume", "path": str(checkpoint_dir)}
    assert decision.next_iteration == ref.next_iteration
    assert decision.completed_updates == ref.completed_updates


def test_missing_required_component_file_refuses_incomplete(tmp_path: Path) -> None:
    """Expected GREEN at head c8570c21.

    Pins: a full-state checkpoint whose manifest and model bytes still match
    a previously built ``CheckpointRef`` (so ``ref.validate()`` passes) but
    whose required ``rng.pt`` component file has since been unlinked refuses
    INCOMPLETE with a missing-files message.
    """

    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, step=5, completed_updates=3, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    (checkpoint_dir / "rng.pt").unlink()
    evidence = CheckpointEvidence.from_ref(
        ref,
        science_input=intent.science_input,
        target_iterations=intent.target_iterations,
    )

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence)

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPLETE
    assert "rng" in decision.refusal.message


def test_durable_mapping_evidence_binding_is_real(tmp_path: Path) -> None:
    """Expected GREEN at head c8570c21.

    Pins: a durable serialized evidence mapping (``CheckpointEvidence.to_dict``
    via ``from_ref``) carries its own binding digest, and that binding is
    actually enforced on ``select_recovery`` -- unlike the raw path/ref forms
    exercised by the red foreign-lineage test above.
    """

    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, step=5, completed_updates=3, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence_dict = CheckpointEvidence.from_ref(
        ref,
        science_input={"system": "other"},
        target_iterations=10,
    ).to_dict()

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence_dict)

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPATIBLE


def test_tampered_serialized_evidence_refuses_without_exception(tmp_path: Path) -> None:
    """Expected GREEN at head c8570c21.

    Pins: a serialized evidence mapping whose embedded ref content_id has
    been corrupted must be refused as a typed decision, not raise, because
    ``_coerce_evidence`` catches ``CheckpointEvidence.from_mapping`` failures
    and falls back to an unbound (``ref=None``) evidence record.
    """

    intent = _intent()
    checkpoint_dir = _checkpoint(tmp_path, step=5, completed_updates=3, payload=TrainResume().to_manifest())
    ref = CheckpointRef.from_directory(checkpoint_dir)
    evidence_dict = CheckpointEvidence.from_ref(
        ref,
        science_input=intent.science_input,
        target_iterations=intent.target_iterations,
    ).to_dict()
    original = evidence_dict["ref"]["content_id"]
    flipped_char = "0" if original[0] != "0" else "1"
    evidence_dict["ref"]["content_id"] = flipped_char + original[1:]

    decision = select_recovery(intent, phase=RecoveryPhase.RECOVERY, evidence=evidence_dict)

    assert decision.outcome is RecoveryOutcome.REFUSED
    assert decision.refusal is not None
    assert decision.refusal.code is RefusalCode.INCOMPLETE
