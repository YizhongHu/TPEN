"""Round 5 E1 raw-evidence refusal control, ported from fddb5ee0."""

from pathlib import Path


def test_r5_unbound_raw_evidence_forms_never_reach_coerce(monkeypatch):
    import tpen.execution.recovery as recovery
    from tpen.execution import RecoveryPhase, RefusalCode, capture_science_intent, select_recovery

    original = recovery._coerce_evidence
    seen = []

    def spy(evidence, *args, **kwargs):
        seen.append(type(evidence))
        return original(evidence, *args, **kwargs)

    monkeypatch.setattr(recovery, "_coerce_evidence", spy)
    intent = capture_science_intent({"x": 1}, target_iterations=2)
    from tpen.checkpoint.reference import CheckpointRef

    ref = CheckpointRef(Path("some/step_000001"), 2, "tpen.checkpoint", 1, 1, "a" * 64, "b" * 64, {})
    for phase in (RecoveryPhase.INITIAL, RecoveryPhase.RECOVERY):
        for value in ("some/step_000001", Path("some/step_000001"), ref):
            decision = select_recovery(intent, phase=phase, evidence=value)
            assert decision.refusal.code is RefusalCode.INCOMPATIBLE, f"observed refusal={decision.refusal!r}"
    assert seen == [], f"_coerce_evidence observed calls: {seen!r}"
