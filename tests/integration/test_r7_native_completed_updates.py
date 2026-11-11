"""Round-7 review P7-2: native recovery parity for ``completed_updates``.

Scenario
--------
``test_native_e1_to_e4_recovery_parity_divergence_and_receipt`` (E5 line 321,
CR line 464) asserts only that the successor's ``completed_updates`` is at most
the target; it never compares it with the uninterrupted reference worker. The
CR ``trainer_cursor`` channel observes ``next_iteration`` only. A restore that
keeps ``next_iteration`` but loses ``completed_updates`` would therefore pass.
This test reruns that native flow, captures both worker results, and compares
the two counters.

Owning layer and SHA
--------------------
E5 ``0918afe83829b3c3cd203646931501663255411d``; also run at CR
``c368d7aeb8c6e566407f2a07df344a850218973b``.

Expected polarity
-----------------
GREEN on clean E5 and CR. RED under mutant M-P7-2, which makes
``tpen.checkpoint.restore._load_trainer`` load ``completed_updates=0``. If the
wrapped base test itself fails, its original exception propagates unchanged.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.integration
def test_r7_successor_completed_updates_matches_reference(tmp_path, monkeypatch) -> None:
    import tests.integration.test_native_checkpoint_recovery as base

    captured: dict = {}
    original = base._result_from_log

    def capture(root, label):
        result = original(root, label)
        captured[label] = result
        return result

    monkeypatch.setattr(base, "_result_from_log", capture)
    base.test_native_e1_to_e4_recovery_parity_divergence_and_receipt(tmp_path)

    recovery = captured["recovery"]["invocation"]
    reference = captured["reference"]["invocation"]
    observed = {
        "recovery_completed_updates": recovery["completed_updates"],
        "reference_completed_updates": reference["completed_updates"],
        "recovery_next_iteration": recovery["next_iteration"],
        "reference_next_iteration": reference["next_iteration"],
    }
    (tmp_path / "r7-completed-updates.json").write_text(
        json.dumps(observed, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(observed, sort_keys=True))
    assert recovery["completed_updates"] == reference["completed_updates"], json.dumps(
        observed, sort_keys=True
    )
