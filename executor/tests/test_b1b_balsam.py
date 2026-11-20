"""Fake-only B1b adapter controls; no allocation controller or SDK required."""
from __future__ import annotations

import ast
import builtins
from dataclasses import replace
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpen_executor._codec import canonical_bytes, digest
from tpen_executor.contracts import AllocationId, BackendJobId, GroupId
from tpen_executor.lifecycle import balsam
from tpen_executor.lifecycle.balsam import (
    BalsamGroupParked, BalsamGroupRefusal, BalsamGroupRegistrationIntent,
    BalsamGroupRegistrationReceipt, GroupExitContract, GroupFailure,
    group_exit_code, make_group_application, register_group,
)
from tpen_executor.recovery import GroupReconciliationV2, MemberReconciliationV2
from tpen_executor.storage import DirectoryStore


CONTRACT = GroupExitContract((-11, 1, 255, 12345))


def _intent():
    return BalsamGroupRegistrationIntent(GroupId("group"), "a" * 64, "b" * 64)


def _result(*actions):
    return GroupReconciliationV2(GroupId("group"), "a" * 64, tuple(
        MemberReconciliationV2(f"task-{index}", action, "fake", None)
        for index, action in enumerate(actions)
    ))


class Client:
    def __init__(self, store, response=None):
        self.store = store
        self.response = BackendJobId("job") if response is None else response
        self.calls = []
        self.observations = []

    def register_group(self, intent):
        # The durable intent, not a later receipt, must precede any backend call.
        paths = list(Path(self.store.root).rglob("intent.json"))
        assert len(paths) == 1
        assert paths[0].read_bytes() == intent.encode()
        self.calls.append(intent)
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response

    def observe_job(self, job_id):
        self.observations.append(job_id)
        return "COMPLETED"


def test_registration_reuses_durable_job_and_never_observes_completion(tmp_path):
    store = DirectoryStore(tmp_path / "store")
    client = Client(store)
    receipt = register_group(_intent(), store, client)
    assert receipt == BalsamGroupRegistrationReceipt(_intent(), BackendJobId("job"))
    assert BalsamGroupRegistrationReceipt.decode(receipt.encode()) == receipt
    assert BalsamGroupRegistrationIntent.decode(_intent().encode()) == _intent()
    reopened = DirectoryStore(store.root)
    assert register_group(_intent(), reopened, client) == receipt
    assert len(client.calls) == 1
    assert client.observations == []


@pytest.mark.parametrize("field", ["group_digest", "backend_options_digest"])
def test_changed_intent_is_typed_refusal(tmp_path, field):
    store = DirectoryStore(tmp_path / "store")
    client = Client(store)
    register_group(_intent(), store, client)
    with pytest.raises(BalsamGroupRefusal, match="different intent"):
        register_group(replace(_intent(), **{field: "c" * 64}), store, client)
    assert len(client.calls) == 1


@pytest.mark.parametrize("response", [RuntimeError("lost ack"), "untyped-job", AllocationId("allocation")])
def test_lost_ack_never_registers_twice(tmp_path, response):
    store = DirectoryStore(tmp_path / "store")
    client = Client(store, response)
    with pytest.raises(BalsamGroupParked):
        register_group(_intent(), store, client)
    for _ in range(2):
        with pytest.raises(BalsamGroupParked):
            register_group(_intent(), DirectoryStore(store.root), client)
    assert len(client.calls) == 1
    assert not list(Path(store.root).rglob("receipt.json"))


def test_typed_client_refusal_is_durable_and_never_retried(tmp_path):
    store = DirectoryStore(tmp_path / "store")
    client = Client(store, BalsamGroupRefusal("deployment rejected"))
    for _ in range(2):
        with pytest.raises(BalsamGroupRefusal):
            register_group(_intent(), store, client)
    assert len(client.calls) == 1
    assert len(list(Path(store.root).rglob("refusal.json"))) == 1


def test_intent_without_ack_parks_before_backend_call(tmp_path):
    store = DirectoryStore(tmp_path / "store")
    directory = Path(store.root, "balsam-groups", digest(_intent().group_id.to_dict()))
    directory.mkdir(parents=True)
    (directory / "intent.json").write_bytes(_intent().encode())
    client = Client(store)
    with pytest.raises(BalsamGroupParked):
        register_group(_intent(), store, client)
    assert client.calls == []


def test_ack_publication_failure_cannot_resubmit(tmp_path, monkeypatch):
    store = DirectoryStore(tmp_path / "store")
    client = Client(store)
    original = balsam._write_new

    def fail_ack(path, payload):
        if path.endswith("receipt.json"):
            raise OSError("write failed")
        return original(path, payload)

    monkeypatch.setattr(balsam, "_write_new", fail_ack)
    with pytest.raises(BalsamGroupParked):
        register_group(_intent(), store, client)
    monkeypatch.setattr(balsam, "_write_new", original)
    with pytest.raises(BalsamGroupParked):
        register_group(_intent(), store, client)
    assert len(client.calls) == 1


def test_competing_registration_cannot_call_backend(tmp_path):
    store = DirectoryStore(tmp_path / "store")
    second = Client(store)

    class First(Client):
        def register_group(self, intent):
            with pytest.raises(BalsamGroupParked):
                register_group(intent, DirectoryStore(store.root), second)
            return super().register_group(intent)

    first = First(store)
    receipt = register_group(_intent(), store, first)
    assert len(first.calls) == 1 and second.calls == []
    assert register_group(_intent(), store, second) == receipt
    assert second.calls == []


def test_crossed_receipt_refuses_reuse(tmp_path):
    store = DirectoryStore(tmp_path / "store")
    client = Client(store)
    register_group(_intent(), store, client)
    path = next(Path(store.root).rglob("receipt.json"))
    path.write_bytes(BalsamGroupRegistrationReceipt(replace(_intent(), group_digest="c" * 64), BackendJobId("job")).encode())
    with pytest.raises(BalsamGroupRefusal, match="crosses intent"):
        register_group(_intent(), store, client)
    assert len(client.calls) == 1


def test_registration_identity_types_and_strict_schema():
    with pytest.raises(TypeError):
        BalsamGroupRegistrationIntent(BackendJobId("job"), "a" * 64, "b" * 64)
    with pytest.raises(TypeError):
        BalsamGroupRegistrationReceipt(_intent(), AllocationId("allocation"))
    data = _intent().to_dict()
    data["unknown"] = True
    with pytest.raises(ValueError):
        BalsamGroupRegistrationIntent.decode(data)
    data = BalsamGroupRegistrationReceipt(_intent(), BackendJobId("job")).to_dict()
    data["backend_job_id"] = AllocationId("allocation").to_dict()
    with pytest.raises(ValueError):
        BalsamGroupRegistrationReceipt.decode(data)


@pytest.mark.parametrize("actions,expected", [
    (("preserve_completed",), 0),
    (("preserve_completed", "preserve_completed"), 0),
    (("preserve_terminal",), 2),
    (("preserve_completed", "preserve_terminal"), 2),
    (("park",), 2),
    (("start_fresh",), 2),
    (("admit_successor",), 2),
    (("preserve_completed", "admit_successor", "start_fresh", "preserve_terminal", "park"), 2),
])
def test_exit_table(actions, expected):
    assert group_exit_code(_result(*actions), CONTRACT) == expected


@pytest.mark.parametrize("failure", list(GroupFailure))
def test_failure_exit_table(failure):
    assert group_exit_code(failure, CONTRACT) == 2


@pytest.mark.parametrize("retry_codes", [(2,), (-11, 1, 2, 255, 12345), (0,)])
def test_exit_contract_rejects_retry_overlap(retry_codes):
    with pytest.raises(BalsamGroupRefusal):
        GroupExitContract(retry_codes)


def test_exit_contract_roundtrip_and_types():
    assert GroupExitContract.decode(canonical_bytes(CONTRACT.to_dict())) == CONTRACT
    with pytest.raises(TypeError):
        GroupExitContract((True,))
    with pytest.raises(BalsamGroupRefusal):
        GroupExitContract((), no_retry_code=1)
    with pytest.raises(ValueError):
        GroupExitContract.decode({**CONTRACT.to_dict(), "schema": "unknown"})


def test_backend_completed_is_not_group_success(tmp_path):
    client = Client(DirectoryStore(tmp_path / "store"))
    assert client.observe_job(BackendJobId("job")) == "COMPLETED"
    assert group_exit_code(_result("park"), CONTRACT) == 2
    with pytest.raises(BalsamGroupRefusal):
        group_exit_code(client.observe_job(BackendJobId("job")), CONTRACT)


def _app(controller, recorder):
    app_type = make_group_application(SimpleNamespace(ApplicationDefinition=object),
        controller=controller, contract=CONTRACT, record_failure=recorder)
    app = app_type()
    app.job = SimpleNamespace(id="job", state="COMPLETED")
    return app


def test_application_delegates_once_and_preserves_environment(monkeypatch):
    monkeypatch.setenv("PATH", "/inherited/path")
    monkeypatch.setenv("PYTHONPATH", "/inherited/python")
    calls = []
    failures = []
    app = _app(lambda job: calls.append(job) or _result("preserve_completed"),
               lambda job, error: failures.append((job, error)))
    assert app.shell_preamble() == ""
    assert app.environment_variables == {}
    assert app.run() == 0
    assert calls == [app.job] and failures == []
    assert balsam.os.environ["PATH"] == "/inherited/path"
    assert balsam.os.environ["PYTHONPATH"] == "/inherited/python"


@pytest.mark.parametrize("outcome", [*list(GroupFailure),
    _result("park"), _result("preserve_terminal"), _result("start_fresh")])
def test_application_non_success_exits_process(outcome):
    app = _app(lambda job: outcome, lambda job, error: pytest.fail("unexpected failure"))
    with pytest.raises(SystemExit) as caught:
        app.run()
    assert caught.value.code == 2


@pytest.mark.parametrize("error", [RuntimeError("failure"), SystemExit(1), KeyboardInterrupt()])
def test_application_uncaught_records_failure_and_never_exits_one(error):
    calls = []

    def fail(job):
        raise error

    app = _app(fail, lambda job, exc: calls.append((job, exc)))
    with pytest.raises(SystemExit) as caught:
        app.run()
    assert caught.value.code == 2
    assert calls == [(app.job, error)]
    assert caught.value.__cause__ is error


def test_pre_start_failure_and_recorder_failure_still_exit_two():
    errors = []

    def recorder(job, error):
        assert job is None
        errors.append(error)
        raise OSError("receipt disk unavailable")

    app = _app(lambda job: pytest.fail("no controller call"), recorder)
    del app.job
    with pytest.raises(SystemExit) as caught:
        app.run()
    assert caught.value.code == 2
    assert len(errors) == 1 and isinstance(errors[0], AttributeError)
    assert any("receipt disk unavailable" in note for note in errors[0].__notes__)


def test_application_rejects_untyped_controller_result():
    errors = []
    app = _app(lambda job: 0, lambda job, error: errors.append(error))
    with pytest.raises(SystemExit) as caught:
        app.run()
    assert caught.value.code == 2
    assert len(errors) == 1 and isinstance(errors[0], BalsamGroupRefusal)


def test_absent_sdk_import_and_factory_refusal(monkeypatch):
    original = builtins.__import__

    def without_balsam(name, *args, **kwargs):
        if name == "balsam" or name.startswith("balsam."):
            raise ModuleNotFoundError("no balsam in fake environment")
        return original(name, *args, **kwargs)

    # Execute a fresh module namespace without reloading shared class identities.
    monkeypatch.setattr(builtins, "__import__", without_balsam)
    spec = importlib.util.spec_from_file_location("tpen_executor.lifecycle._b1b_absent", balsam.__file__)
    module = importlib.util.module_from_spec(spec)
    import sys
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    with pytest.raises(module.BalsamGroupRefusal, match="unavailable"):
        module.make_group_application(controller=lambda job: None,
            contract=module.GroupExitContract((-11, 1, 255, 12345)), record_failure=lambda job, error: None)


# Exact single-edit mutants for independent cluster verification; kills are
# predictions until the verifier records call-phase assertions for each arm.
MUTANTS = (
    ("blind-resubmit", "lifecycle/balsam.py", '        if not owns_intent:\n', '        if False:\n', "test_lost_ack_never_registers_twice"),
    ("discard-intent", "lifecycle/balsam.py", '        if stored != intent:\n', '        if False:\n', "test_changed_intent_is_typed_refusal"),
    ("discard-receipt-binding", "lifecycle/balsam.py", '            if receipt.intent != intent:\n', '            if False:\n', "test_crossed_receipt_refuses_reuse"),
    ("late-intent", "lifecycle/balsam.py", '                _write_new(str(intent_path), intent.encode())', '                client.register_group(intent)\n                _write_new(str(intent_path), intent.encode())', "test_registration_reuses_durable_job_and_never_observes_completion"),
    ("job-allocation-confusion", "lifecycle/balsam.py", 'BackendJobId.from_mapping(data["backend_job_id"])', 'BackendJobId(data["backend_job_id"]["value"])', "test_registration_identity_types_and_strict_schema"),
    ("terminal-success", "lifecycle/balsam.py", '        if all(member.action == "preserve_completed" for member in outcome.members):', '        if outcome.aggregate == "settled":', "test_exit_table"),
    ("parked-success", "lifecycle/balsam.py", '    if isinstance(outcome, GroupFailure):\n        return contract.no_retry_code', '    if isinstance(outcome, GroupFailure):\n        return 0', "test_failure_exit_table"),
    ("retry-overlap", "lifecycle/balsam.py", '        if self.no_retry_code in self.backend_retry_codes or 0 in self.backend_retry_codes:', '        if False:', "test_exit_contract_rejects_retry_overlap"),
    ("backend-success", "lifecycle/balsam.py", '    raise BalsamGroupRefusal("controller returned no typed group outcome")', '    return 0', "test_backend_completed_is_not_group_success"),
    ("return-integer", "lifecycle/balsam.py", '                raise SystemExit(code)', '                return code', "test_application_non_success_exits_process"),
    ("uncaught-one", "lifecycle/balsam.py", '                raise SystemExit(group_exit_code(GroupFailure.UNCAUGHT, contract)) from error', '                raise SystemExit(1) from error', "test_application_uncaught_records_failure_and_never_exits_one"),
    ("drop-failure-receipt", "lifecycle/balsam.py", '                    record_failure(job, error)', '                    pass', "test_application_uncaught_records_failure_and_never_exits_one"),
    ("replace-environment", "lifecycle/balsam.py", '        environment_variables = {}', '        environment_variables = {"PATH": "/replacement"}', "test_application_delegates_once_and_preserves_environment"),
    ("module-sdk-import", "lifecycle/balsam.py", 'from enum import Enum', 'from enum import Enum\nimport balsam', "test_balsam_allowlist_is_exact"),
)


def test_mutant_anchor_uniqueness():
    source = Path(balsam.__file__).read_text()
    names = set()
    for name, filename, before, after, killer in MUTANTS:
        assert filename == "lifecycle/balsam.py"
        assert name not in names
        names.add(name)
        assert source.count(before) == 1, name
        assert before != after
        ast.parse(source.replace(before, after, 1))
        assert any(f"def {killer}(" in path.read_text() for path in Path(__file__).parent.glob("test_*.py")), name
