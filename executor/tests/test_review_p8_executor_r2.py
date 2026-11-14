"""Round-two probes for changed chain traversal and outcome/schema guards."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
from tpen_executor.runners import CoreRunEvidence, EvidenceUnavailable
from tpen_executor.storage import FenceRejected
from test_review_p8_executor_r1 import setup_store, finish


@pytest.mark.parametrize("bad", ["two-roots", "fork", "disconnected", "cycle", "backwards"])
def test_r2_t1_malformed_prior_chain_refused_before_publication(tmp_path, bad):
    store, group, allocation = setup_store(tmp_path)
    records = []
    for step in (1, 2, 3):
        fence, evidence = finish(store, group, allocation, step)
        segment = store.accept_segment(fence, evidence, records[-1][1].checkpoint_dir if records else None)
        records.append((fence, segment))
    assert len(store.manifest(group).segments) == 3
    pending, evidence = finish(store, group, allocation, 4)
    if bad == "two-roots": index, changed = 1, replace(records[1][1], predecessor_checkpoint=None)
    elif bad == "fork": index, changed = 2, replace(records[2][1], predecessor_checkpoint=records[0][1].checkpoint_dir)
    elif bad == "disconnected": index, changed = 2, replace(records[2][1], predecessor_checkpoint="missing-checkpoint")
    elif bad == "cycle": index, changed = 2, replace(records[2][1], checkpoint_dir=records[0][1].checkpoint_dir)
    else: index, changed = 2, replace(records[2][1], next_iteration=1)
    (Path(records[index][0].output_dir) / "segment.json").write_bytes(changed.encode())
    with pytest.raises(FenceRejected):
        store.accept_segment(pending, evidence, records[-1][1].checkpoint_dir)
    assert not (Path(pending.output_dir) / "segment.json").exists()


@pytest.mark.parametrize("field,value", [("task_id", "foreign"), ("attempt_id", "foreign"), ("group_digest", "b" * 64), ("output_dir", "/foreign")])
def test_r2_t2_each_outcome_binding_is_checked_independently(tmp_path, field, value):
    store, group, allocation = setup_store(tmp_path)
    fence, evidence = finish(store, group, allocation, 1)
    path = Path(fence.output_dir) / "outcome.json"
    payload = json.loads(path.read_text())
    payload[field] = value
    path.write_text(json.dumps(payload))
    with pytest.raises(FenceRejected, match="outcome receipt crosses"):
        store.accept_segment(fence, evidence)
    assert not (Path(fence.output_dir) / "segment.json").exists()


@pytest.mark.parametrize("value", [True, 2.0, "2", "v2", None])
def test_r2_t3_schema_version_is_nonbool_integer(tmp_path, value):
    store, group, allocation = setup_store(tmp_path)
    _, evidence = finish(store, group, allocation, 1)
    payload = dict(schema=evidence.schema, run_dir=evidence.run_dir, verdict=evidence.verdict, run_status=evidence.run_status, checkpoint=dict(evidence.checkpoint), refusal=None, resolved_config_identity=dict(evidence.resolved_config_identity))
    assert payload["checkpoint"]["schema_version"] == 2
    assert type(payload["checkpoint"]["schema_version"]) is int
    payload["checkpoint"]["schema_version"] = value
    with pytest.raises(EvidenceUnavailable, match="schema_version must be an integer"):
        CoreRunEvidence.from_mapping(payload)
