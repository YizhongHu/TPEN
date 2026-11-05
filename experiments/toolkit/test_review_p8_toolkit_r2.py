"""Delta-focused parity and evidence-type controls for the P8 reader repair."""
import copy
import json
from pathlib import Path
import pytest
from experiments.toolkit.run_results import read, ManifestRefusal
from test_review_p8_toolkit_r1 import executor, materialize, write


@pytest.mark.parametrize("mutation", ["manifest-uppercase", "start-nonhex", "outcome-short", "result-uppercase", "outcome-group", "start-group"])
def test_r2_t4_digest_guards_refuse_before_inspection(tmp_path, mutation):
    payload, _ = materialize(tmp_path)
    start = payload["attempts"][0]["start_receipt"]
    outcome = payload["attempts"][0]["outcome_receipt"]
    if mutation == "manifest-uppercase": payload["group_digest"] = "A" * 64
    elif mutation == "start-nonhex": start["group_digest"] = "z" * 64
    elif mutation == "outcome-short": outcome["group_digest"] = "abc"
    elif mutation == "result-uppercase": outcome["result_digest"] = "A" * 64
    elif mutation == "outcome-group": outcome["group_digest"] = "b" * 64
    else: start["group_digest"] = "b" * 64
    path = write(tmp_path, payload)
    rejected = executor("import sys; from tpen_executor.results import RunManifest\ntry: RunManifest.decode(open(sys.argv[1],'rb').read())\nexcept (TypeError,ValueError): print('REFUSED')\nelse: raise AssertionError('accepted')", path)
    assert rejected.returncode == 0 and rejected.stdout.strip() == "REFUSED", rejected.stderr
    with pytest.raises(ManifestRefusal):
        read(path, lambda _: pytest.fail("must reject malformed wire before core inspection"))


@pytest.mark.parametrize("status", ["completed", "failed", "timed_out", "launch_failed"])
def test_r2_t5_closed_status_set_preserves_valid_unselected_records(tmp_path, status):
    payload, _ = materialize(tmp_path)
    payload["segments"] = []
    payload["attempts"][0]["outcome_receipt"]["status"] = status
    path = write(tmp_path, payload)
    decoded = executor("import sys; from tpen_executor.results import RunManifest; print(len(RunManifest.decode(open(sys.argv[1],'rb').read()).attempts))", path)
    assert decoded.returncode == 0 and decoded.stdout.strip() == "1", decoded.stderr
    result = read(path, lambda _: pytest.fail("nonselected attempts must not be inspected"))
    assert result.chains == ()
    assert len(result.preserved_attempts) == 1
    assert result.preserved_attempts[0].outcome == status


@pytest.mark.parametrize("field,value", [("schema_version", True), ("schema_version", "2"), ("next_iteration", True), ("completed_updates", -1)])
def test_r2_t6_core_evidence_types_cannot_compare_equal_to_valid_coordinates(tmp_path, field, value):
    payload, _ = materialize(tmp_path)
    path = write(tmp_path, payload)
    checkpoint = dict(checkpoint_dir=payload["segments"][0]["checkpoint_dir"], schema_version=2, next_iteration=1, completed_updates=1)
    checkpoint[field] = value
    with pytest.raises(ManifestRefusal):
        read(path, lambda _: dict(verdict="completed", checkpoint=checkpoint))
