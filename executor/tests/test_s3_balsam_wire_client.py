"""S3 fake-bridge tests for the closed Balsam boundary."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.contracts import AllocationId, BackendJobId, BatchJobId, GroupId
from tpen_executor.lifecycle.balsam import BalsamGroupParked, BalsamGroupRefusal, BalsamGroupRegistrationIntent, register_group
from tpen_executor.lifecycle.balsam_client import BalsamBridgeAmbiguous, BalsamProcessClient, select_balsam_client_path
from tpen_executor.lifecycle.balsam_wire import BalsamWireError, BalsamWireRequest, BalsamWireResult, decode_request, decode_result, encode_request, encode_result
from tpen_executor.storage import DirectoryStore


def _sha(seed: str) -> str:
    return digest({"seed": seed})


def _client_file(tmp_path: Path) -> Path:
    path = tmp_path / "client.yml"
    path.write_text("api_root: https://example.invalid\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def _intent() -> BalsamGroupRegistrationIntent:
    return BalsamGroupRegistrationIntent(GroupId("s3-group"), _sha("group"), _sha("deployment"))


class FakeBridge:
    def __init__(self, response, *, error: BaseException | None = None):
        self.response = response
        self.error = error
        self.requests: list[BalsamWireRequest] = []

    def call(self, payload: bytes, *, timeout: float) -> bytes:
        assert timeout > 0
        request = decode_request(payload)
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return encode_result(self.response(request) if callable(self.response) else self.response)


def test_closed_requests_cover_all_operations_and_round_trip() -> None:
    group = GroupId("group")
    job = BackendJobId("job")
    requests = (
        BalsamWireRequest.register(group, _sha("group"), _sha("deployment")),
        BalsamWireRequest.observe(group, job),
        BalsamWireRequest.submit(group, job, _sha("batch")),
        BalsamWireRequest.rearm(group, job, _sha("recovery")),
        BalsamWireRequest.readback(group, job),
    )
    for request in requests:
        assert decode_request(encode_request(request)) == request


def test_closed_results_interoperate_and_reject_foreign_handles() -> None:
    request = BalsamWireRequest.submit(GroupId("group"), BackendJobId("job"), _sha("batch"))
    result = BalsamWireResult.ack(request, backend_job_id=request.backend_job_id, batch_job_id=BatchJobId("batch"))
    assert decode_result(encode_result(result)) == result
    foreign = result.to_dict()
    foreign["backend_job_id"] = AllocationId("allocation").to_dict()
    with pytest.raises(ValueError):
        decode_result(foreign)


@pytest.mark.parametrize("field", ["operation", "status", "group_id", "backend_job_id", "extra"])
def test_malformed_wire_is_rejected(field: str) -> None:
    request = BalsamWireRequest.observe(GroupId("group"), BackendJobId("job"))
    payload = request.to_dict()
    if field == "operation":
        payload[field] = "delete"
    elif field == "status":
        payload["status"] = "made-up"
    elif field == "group_id":
        payload[field] = AllocationId("wrong").to_dict()
    elif field == "backend_job_id":
        payload[field] = AllocationId("wrong").to_dict()
    else:
        payload[field] = True
    with pytest.raises((BalsamWireError, TypeError, ValueError)):
        decode_request(payload)


def test_result_refusal_and_ambiguous_effect_are_distinct() -> None:
    request = BalsamWireRequest.rearm(GroupId("group"), BackendJobId("job"), _sha("recovery"))
    refusal = BalsamWireResult.refused(request, "grant expired")
    ambiguous = BalsamWireResult.ambiguous(request, "connection ended after submit")
    assert refusal.status == "refused" and ambiguous.status == "ambiguous"
    with pytest.raises(BalsamWireError):
        BalsamWireResult("observe", "ambiguous", GroupId("group"), reason="bad")


def test_client_selects_path_before_any_balsam_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _client_file(tmp_path)
    original = builtins.__import__

    def no_sdk(name, *args, **kwargs):
        if name == "balsam" or name.startswith("balsam."):
            raise AssertionError("executor selected client path after importing the SDK")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sdk)
    assert select_balsam_client_path(environ={"BALSAM_CLIENT_PATH": str(path)}) == path


def test_client_requires_explicit_path_and_rejects_unsafe_file(tmp_path: Path) -> None:
    with pytest.raises(BalsamGroupRefusal):
        select_balsam_client_path(environ={})
    path = _client_file(tmp_path)
    path.chmod(0o666)
    with pytest.raises(BalsamGroupRefusal):
        select_balsam_client_path(path)


def test_real_process_client_register_lost_ack_is_parked(tmp_path: Path) -> None:
    bridge = FakeBridge(None, error=BalsamBridgeAmbiguous("lost acknowledgement"))
    client = BalsamProcessClient("/bin/sh", client_path=_client_file(tmp_path), bridge=bridge)
    store = DirectoryStore(tmp_path / "store")
    with pytest.raises(BalsamGroupParked):
        register_group(_intent(), store, client)
    with pytest.raises(BalsamGroupParked):
        register_group(_intent(), DirectoryStore(store.root), client)
    assert len(bridge.requests) == 1


def test_process_client_refusal_is_not_ambiguous_and_is_not_retried(tmp_path: Path) -> None:
    intent = _intent()
    bridge = FakeBridge(lambda request: BalsamWireResult.refused(request, "deployment rejected"))
    client = BalsamProcessClient("/bin/sh", client_path=_client_file(tmp_path), bridge=bridge)
    store = DirectoryStore(tmp_path / "store")
    for _ in range(2):
        with pytest.raises(BalsamGroupRefusal):
            register_group(intent, store, client)
    assert len(bridge.requests) == 1


def test_mutation_journal_prevents_submit_retry_and_rejects_wrong_result(tmp_path: Path) -> None:
    job = BackendJobId("job")
    group = GroupId("group")
    bridge = FakeBridge(lambda request: BalsamWireResult.ack(request, backend_job_id=job, batch_job_id=BatchJobId("batch")))
    client = BalsamProcessClient("/bin/sh", client_path=_client_file(tmp_path), state_dir=tmp_path / "state", bridge=bridge)
    assert client.submit_batch(group, job, _sha("batch")) == BatchJobId("batch")
    assert client.submit_batch(group, job, _sha("batch")) == BatchJobId("batch")
    assert len(bridge.requests) == 1


def test_journaled_mutation_parks_and_never_auto_retries(tmp_path: Path) -> None:
    job = BackendJobId("job")
    group = GroupId("group")
    request = BalsamWireRequest.rearm(group, job, _sha("recovery"))
    intent_dir = tmp_path / "state" / "balsam-wire" / digest(request.to_dict())
    intent_dir.mkdir(parents=True)
    (intent_dir / "intent.json").write_bytes(canonical_bytes(request.to_dict()))
    bridge = FakeBridge(lambda request: BalsamWireResult.ack(request, backend_job_id=job, state="queued"))
    client = BalsamProcessClient("/bin/sh", client_path=_client_file(tmp_path), state_dir=tmp_path / "state", bridge=bridge)
    with pytest.raises(BalsamGroupParked):
        client.rearm_group(group, job, _sha("recovery"))
    assert bridge.requests == []


def test_client_does_not_accept_an_unbound_observation_handle(tmp_path: Path) -> None:
    bridge = FakeBridge(None)
    client = BalsamProcessClient("/bin/sh", client_path=_client_file(tmp_path), bridge=bridge)
    with pytest.raises(BalsamGroupRefusal):
        client.observe_job(BackendJobId("job"))
    assert bridge.requests == []


def test_client_rejects_crossed_response_identity(tmp_path: Path) -> None:
    group = GroupId("group")
    job = BackendJobId("job")
    bridge = FakeBridge(lambda request: BalsamWireResult("readback", "ack", request.group_id, request.backend_job_id, state="queued"))
    client = BalsamProcessClient("/bin/sh", client_path=_client_file(tmp_path), bridge=bridge)
    with pytest.raises(Exception, match="crosses request identity"):
        client.observe(group, job)
