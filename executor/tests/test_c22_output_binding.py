"""C2.2 common configured-program output/start binding controls."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import sys

import pytest

from tpen_executor._codec import digest, thaw
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.configured_program import CONFIGURED_PROGRAM_ALLOWLIST, AttemptDelta, ConfiguredLoad, ConfiguredProgram
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef
from tpen_executor.dispatch import LocalDispatcher
from tpen_executor.dispatch import contract as dispatch_contract
from tpen_executor.dispatch import parsl as parsl_dispatch
from tpen_executor.dispatch.transport import LeafRequest, LeafResult, run_leaf, validate_leaf_result
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import AllocationSlot, PLACEMENT_OUTPUT_ENV, PlacementRefusal
from tpen_executor.placement import physical_slots_from_allocation
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun
from tpen_executor.programs import WorkerProgram
from tpen_executor.workers import ProcessOutcome, SubprocessWorker

from test_s2b2_placement_v2 import _allocation as _slurm_allocation
from test_s3_transport_parsl import BINDING, _Parsl, _group, _lifecycle, _worker


ROOT = Path(__file__).parents[2]
STAGE_Q = ROOT / "experiments" / "atomistic" / "he-importance" / "run_stage_q.py"
MANIFEST = ROOT / "experiments" / "atomistic" / "he-importance" / "o1_scan_manifest_v1.json"
GROUP = "g" * 64
ROW = "0" * 64
DOC_EXAMPLE = (
    '{"attempt_core_dir":"/store/groups/g/t/a/core","attempt_id":"a",'
    '"group_digest":"g","load":{"mode":"fresh","path":null},'
    '"schema":"tpen-executor/attempt-delta-v1","task_id":"t"}'
)
DOC_EXAMPLE_SHA256 = "95f504768be45378d0d6a9396a88fc85b6524bbfda6ced9b9723b2701f775159"


def _program(tmp_path: Path, *, load: dict[str, object] | None = None) -> ConfiguredProgram:
    return ConfiguredProgram(
        "stage-q", str(STAGE_Q), str(MANIFEST), ROW, str(tmp_path / "campaign"),
        load={"mode": "fresh", "path": None} if load is None else load,
    )


def _configured(tmp_path: Path, *, load: dict[str, object] | None = None, task_id: str = "task") -> ConfiguredRun:
    return ConfiguredRun(task_id, "", (), _program(tmp_path, load=load))


def _request(tmp_path: Path, *, load: dict[str, object] | None = None) -> LeafRequest:
    root = tmp_path / "store"
    output = root / "groups" / GROUP / "task" / "attempt"
    allocation = _slurm_allocation(scheduler={"SLURM_NODELIST": socket.gethostname()}, device=())
    fence = SimpleNamespace(
        allocation_id=allocation.allocation_id.value, group_digest=GROUP, task_id="task", attempt_id=SimpleNamespace(value="attempt"),
        root=str(root), output_dir=str(output),
    )
    slot = physical_slots_from_allocation(allocation)[0]
    return LeafRequest.for_attempt(
        fence, allocation, _configured(tmp_path, load=load),
        PreparedPythonEnvironment(sys.executable, env=slot.env), 60,
    )


def _physical_lifecycle(tmp_path: Path, name: str = "physical-store") -> AttachedAllocation:
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    allocation = _slurm_allocation(scheduler={"SLURM_NODELIST": socket.gethostname()}, device=())
    return AttachedAllocation(allocation, DirectoryStore(root))


def _allocation_with_cpu(cpu: int) -> ConfiguredAllocation:
    slot = AllocationSlot("logical-slot", "logical-label", (), (cpu,), (), ()).to_dict()
    return ConfiguredAllocation(
        AllocationId("allocation-a"), BackendJobId("job-a"),
        {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot]},
         "scheduler": {"SLURM_NODELIST": socket.gethostname()}},
    )


def _worker_with_environment(tmp_path: Path, values: dict[str, str]) -> WorkerProgram:
    worker = _worker(tmp_path)
    raw = thaw(worker.spec.config_ref)
    raw["environment"]["env"].update(values)
    return WorkerProgram(ProgramSpecRef(raw, digest(raw)))


def _physical_leaf_request(
    tmp_path: Path, configured: ConfiguredRun, *, allocation: ConfiguredAllocation | None = None,
) -> LeafRequest:
    root = tmp_path / "leaf-store"
    output = root / "groups" / GROUP / configured.id / "attempt"
    allocation = allocation or _slurm_allocation(scheduler={"SLURM_NODELIST": socket.gethostname()}, device=())
    slot = physical_slots_from_allocation(allocation)[0]
    output.mkdir(parents=True)
    fence = SimpleNamespace(
        allocation_id=allocation.allocation_id.value,
        group_digest=GROUP,
        task_id=configured.id,
        attempt_id=SimpleNamespace(value="attempt"),
        root=str(root),
        output_dir=str(output),
    )
    base = _worker(tmp_path).materialize(None)["environment"]
    environment = PreparedPythonEnvironment(base.python, base.setup, {**base.env, **slot.env})
    return LeafRequest.for_attempt(fence, allocation, configured, environment, 60)


def _marker_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[ConfiguredRun, Path]:
    """Build an allowlisted marker that records the delivered shim environment."""

    marker = tmp_path / "marker.json"
    script = tmp_path / "marker.py"
    script.write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "core = Path(os.environ['TPEN_PLACEMENT_OUTPUT_DIR']) / 'core'\n"
        "core.mkdir(parents=True, exist_ok=True)\n"
        f"Path({str(marker)!r}).write_text(json.dumps({{'argv': sys.argv, 'env': {{key: os.environ.get(key) for key in ('TPEN_PLACEMENT_OUTPUT', 'TPEN_PLACEMENT_OUTPUT_DIR', 'TPEN_PLACEMENT_TASK_ID', 'TPEN_PLACEMENT_ATTEMPT_ID', 'TPEN_PLACEMENT_GROUP_DIGEST', 'TPEN_PLACEMENT_SLOT_ID', 'TPEN_PLACEMENT_PAYLOAD_DIGEST')}}}}))\n"
        "(core / 'evidence.json').write_text(json.dumps({"
        "'schema': 'tpen.run-evidence/v2', 'run_dir': str(core), 'verdict': 'completed', "
        "'run_status': 'completed', 'checkpoint': {'checkpoint_dir': str(core / 'checkpoint'), "
        "'schema_version': 1, 'next_iteration': 1, 'completed_updates': 1}, 'refusal': None, "
        "'resolved_config_identity': {'schema': 'tpen-config-identity-v1', 'sha256': 'a' * 64}}))\n",
        encoding="utf-8",
    )
    monkeypatch.setitem(CONFIGURED_PROGRAM_ALLOWLIST, "marker", {
        "trusted_root": str(tmp_path),
        "relative_path": "marker.py",
        "arguments": ("manifest", "row", "output_root"),
        "output_contract": "attempt-core",
    })
    program = ConfiguredProgram(
        "marker", str(script), str(MANIFEST), ROW, str(tmp_path / "campaign"),
        load={"mode": "fresh", "path": None},
    )
    return ConfiguredRun("member", "", (), program), marker


def test_attempt_delta_encoder_matches_normative_doc_example() -> None:
    delta = AttemptDelta(
        "/store/groups/g/t/a/core", ConfiguredLoad("fresh", None), "g", "t", "a",
    )
    assert delta.canonical_bytes.decode("utf-8") == DOC_EXAMPLE
    assert hashlib.sha256(delta.canonical_bytes).hexdigest() == DOC_EXAMPLE_SHA256
    assert delta.digest == DOC_EXAMPLE_SHA256


def test_load_mode_is_explicit_and_train_resume_is_not_own_leaf(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fresh|train_resume"):
        ConfiguredLoad("implicit", "/checkpoint")
    with pytest.raises(ValueError, match="keys"):
        ConfiguredLoad.from_mapping({"path": None})
    own_leaf = tmp_path / "store" / "groups" / GROUP / "task" / "attempt" / "checkpoint"
    with pytest.raises(ValueError, match="own attempt leaf"):
        AttemptDelta(
            str(own_leaf.parent / "core"), ConfiguredLoad("train_resume", str(own_leaf)),
            GROUP, "task", "attempt",
        )


def test_serialized_leaf_carries_delta_and_rejects_crossed_output(tmp_path: Path) -> None:
    request = _request(tmp_path)
    assert request.program is not None and request.attempt_delta is not None
    args = request.science_args()
    assert "--attempt-delta" in args
    assert not any(argument.startswith("run.") for argument in args)
    output_root_index = args.index("--output-root")
    assert args[output_root_index + 1] == request.program.output_root
    decoded = LeafRequest.decode(request.encode())
    assert decoded == request

    foreign = dataclasses.replace(
        request.attempt_delta,
        attempt_core_dir=str(tmp_path / "foreign" / "groups" / GROUP / "task" / "attempt" / "core"),
    )
    payload = request.to_dict()
    payload["attempt_delta"] = foreign.to_dict()
    payload["payload_digest"] = digest({"program_digest": request.program.program_digest, "attempt_delta_digest": foreign.digest})
    payload["environment"]["env"]["TPEN_PLACEMENT_PAYLOAD_DIGEST"] = payload["payload_digest"]
    with pytest.raises(ValueError, match="attempt delta does not bind"):
        LeafRequest.from_mapping(payload)


def test_common_configured_binding_has_no_legacy_run_overrides(tmp_path: Path) -> None:
    args = _configured(tmp_path).command_for_attempt(
        f"/store/groups/{GROUP}/task/attempt", root="/store", group_digest=GROUP, task_id="task", attempt_id="attempt",
    ).args
    assert "--attempt-delta" in args
    assert not any(argument.startswith("run.") for argument in args)


def test_no_slot_configured_local_refuses_before_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configured, marker = _marker_run(tmp_path, monkeypatch)
    calls: list[tuple[object, tuple[str, ...]]] = []
    monkeypatch.setattr(SubprocessWorker, "run", lambda self, environment, args, **kwargs: calls.append((environment, tuple(args))))
    manifest = LocalDispatcher(_worker(tmp_path)).run(
        _group("c22-no-slot-local", "member"), {"member": configured}, _lifecycle(tmp_path), timeout=60,
    )
    assert calls == []
    assert not marker.exists()
    assert manifest.attempts[0].outcome_receipt.status == "launch_failed"


def test_no_slot_configured_parsl_refuses_before_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_parsl = _Parsl().install(monkeypatch)
    configured, marker = _marker_run(tmp_path, monkeypatch)
    group = _group("c22-no-slot-parsl", "member")
    lifecycle = _lifecycle(tmp_path)
    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), binding=BINDING)
    manifest = dispatcher.run(group, {"member": configured}, lifecycle, timeout=60)
    records = [record for record in fake_parsl.submissions if record["function"] == "_run_parsl_child_captured"]
    assert not records
    assert not marker.exists()
    assert manifest.attempts[0].outcome_receipt.status == "launch_failed"


def test_no_slot_configured_serialized_for_attempt_refuses_before_leaf(tmp_path: Path) -> None:
    configured = _configured(tmp_path, task_id="member")
    root = tmp_path / "store"
    output = root / "groups" / GROUP / "member" / "attempt"
    allocation = ConfiguredAllocation(AllocationId("allocation"), BackendJobId("job"), {})
    fence = SimpleNamespace(
        allocation_id="allocation", group_digest=GROUP, task_id="member",
        attempt_id=SimpleNamespace(value="attempt"), root=str(root), output_dir=str(output),
    )
    with pytest.raises(PlacementRefusal, match="physical slot") as raised:
        LeafRequest.for_attempt(
            fence, allocation, configured, PreparedPythonEnvironment(sys.executable, env={}), 60,
        )
    assert raised.value.code == "configured-physical-slot-required"


def test_carried_slot_id_without_cpus_refuses_before_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    configured, marker = _marker_run(tmp_path, monkeypatch)
    calls: list[tuple[object, tuple[str, ...]]] = []
    seen_environment: dict[str, str] = {}
    original_placement = dispatch_contract._with_child_placement

    def observe_placement(environment, *args, **kwargs):
        seen_environment.update(environment.env)
        return original_placement(environment, *args, **kwargs)

    monkeypatch.setattr(dispatch_contract, "_with_child_placement", observe_placement)
    monkeypatch.setattr(SubprocessWorker, "run", lambda self, environment, args, **kwargs: calls.append((environment, tuple(args))))
    manifest = LocalDispatcher(_worker_with_environment(tmp_path, {"TPEN_SLOT_ID": "carried"})).run(
        _group("c22-carried-id-no-cpus", "member"), {"member": configured}, _lifecycle(tmp_path),
        timeout=60,
    )
    assert seen_environment["TPEN_SLOT_ID"] == "carried"
    assert "TPEN_SLOT_CPUS" not in seen_environment
    assert calls == []
    assert not marker.exists()
    assert manifest.attempts[0].outcome_receipt.status == "launch_failed"


def test_legacy_no_slot_request_remains_unchanged(tmp_path: Path) -> None:
    legacy = ConfiguredRun("task", "/prepared/config.yaml", ("payload.mode=attached",))
    output = f"/store/groups/{GROUP}/task/attempt"
    args = legacy.command_for_attempt(output).args
    assert args[-3:] == (f"run.root={output}", "run.layout=flat", "run.run_id=core")


@pytest.mark.parametrize("carrier", ["local", "parsl"])
@pytest.mark.parametrize("crossed", ["group", "task", "attempt"])
def test_actual_dispatchers_refuse_crossed_identity_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, carrier: str, crossed: str,
) -> None:
    """Actual dispatchers reject each crossed identity before process/submission."""

    configured = _configured(tmp_path, task_id="member")
    original = ConfiguredRun.command_for_attempt
    seam_calls: list[tuple[str, dict[str, str]]] = []
    refusal_reasons: list[str] = []

    def crossed_command(self, output_dir, **kwargs):
        if self.program is not None and self.id == "member":
            seam_calls.append((output_dir, dict(kwargs)))
            actual_path = Path(output_dir)
            actual_identity = {
                "group": actual_path.parent.parent.name,
                "task": actual_path.parent.name,
                "attempt": actual_path.name,
            }
            assert {
                "group": kwargs["group_digest"], "task": kwargs["task_id"], "attempt": kwargs["attempt_id"],
            } == actual_identity
            path_parts = list(actual_path.parts)
            path_parts[-{"group": 3, "task": 2, "attempt": 1}[crossed]] = f"foreign-{actual_identity[crossed]}"
            output_dir = str(Path(*path_parts))
        try:
            return original(self, output_dir, **kwargs)
        except ValueError as error:
            refusal_reasons.append(str(error))
            raise

    monkeypatch.setattr(ConfiguredRun, "command_for_attempt", crossed_command)
    group = _group(f"c22-crossed-{carrier}-{crossed}", "member")
    lifecycle = _physical_lifecycle(tmp_path)
    if carrier == "local":
        calls: list[tuple[object, tuple[str, ...]]] = []

        def spy(self, environment, args, **kwargs):
            calls.append((environment, tuple(args)))
            return ProcessOutcome(tuple(args), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)

        monkeypatch.setattr(SubprocessWorker, "run", spy)
        LocalDispatcher(_worker(tmp_path)).run(group, {"member": configured}, lifecycle, timeout=60)
        assert calls == []
    else:
        fake_parsl = _Parsl().install(monkeypatch)
        parsl_dispatch.ParslDispatcher(_worker(tmp_path)).run(
            group, {"member": configured}, lifecycle, timeout=60,
        )
        assert not [record for record in fake_parsl.submissions if record["function"] == "_run_parsl_child_captured"]
    assert len(seam_calls) == 1
    assert refusal_reasons == ["configured attempt output does not bind the placement fence"]


@pytest.mark.parametrize("carrier", ["local", "parsl"])
def test_actual_carriers_capture_matching_full_argv_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, carrier: str,
) -> None:
    """Matching actual carriers deliver the common configured argv and fence env."""

    configured, _marker = _marker_run(tmp_path, monkeypatch)
    configured = dataclasses.replace(configured, id="member")
    group = _group(f"c22-matching-{carrier}", "member")
    lifecycle = _physical_lifecycle(tmp_path)
    if carrier == "local":
        calls: list[tuple[PreparedPythonEnvironment, tuple[str, ...]]] = []

        def spy(self, environment, args, **kwargs):
            calls.append((environment, tuple(args)))
            return ProcessOutcome(tuple(args), 0, kwargs["stdout_path"], kwargs["stderr_path"], False)

        monkeypatch.setattr(SubprocessWorker, "run", spy)
        manifest = LocalDispatcher(_worker(tmp_path)).run(group, {"member": configured}, lifecycle, timeout=60)
        _environment, args = calls[0]
        attempt = manifest.attempts[0]
        output = Path(lifecycle.store.root) / "groups" / group.identity / "member" / attempt.attempt_id.value
        assert args == configured.command_for_attempt(
            str(output), root=str(lifecycle.store.root), group_digest=group.identity,
            task_id="member", attempt_id=attempt.attempt_id.value,
        ).args
        assert _environment.env[PLACEMENT_OUTPUT_ENV] == str(output / "child-placement.json")
    else:
        fake_parsl = _Parsl().install(monkeypatch)
        manifest = parsl_dispatch.ParslDispatcher(_worker(tmp_path)).run(
            group, {"member": configured}, lifecycle, timeout=60,
        )
        [record] = [record for record in fake_parsl.submissions if record["function"] == "_run_parsl_child_captured"]
        _python, _setup, environment, args = record["args"]
        attempt = manifest.attempts[0]
        output = Path(lifecycle.store.root) / "groups" / group.identity / "member" / attempt.attempt_id.value
        expected = configured.command_for_attempt(
            str(output), root=str(lifecycle.store.root), group_digest=group.identity,
            task_id="member", attempt_id=attempt.attempt_id.value,
        ).args
        assert tuple(args) == expected
        assert environment[PLACEMENT_OUTPUT_ENV] == str(output / "child-placement.json")


def test_physical_configured_serialized_leaf_completes_and_validates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured, marker = _marker_run(tmp_path, monkeypatch)
    configured = dataclasses.replace(configured, id="member")
    request = _physical_leaf_request(tmp_path, configured)
    result = LeafResult.decode(run_leaf(request.encode()))
    assert validate_leaf_result(request, result) is result
    expected = configured.attempt_binding_digest(
        request.output_dir, root=request.root, group_digest=GROUP, task_id="member", attempt_id="attempt",
    )
    assert request.environment.env["TPEN_PLACEMENT_PAYLOAD_DIGEST"] == expected
    declared_cpus = [int(value) for value in request.environment.env["TPEN_PLACEMENT_DECLARED_CPUS"].split(",")]
    assert declared_cpus == [min(os.sched_getaffinity(0))]
    assert result.science["returncode"] == 0
    assert marker.exists()
    observation = json.loads(Path(request.environment.env[PLACEMENT_OUTPUT_ENV]).read_text(encoding="utf-8"))
    assert observation["affinity"] == declared_cpus
    observed = json.loads(marker.read_text(encoding="utf-8"))
    assert observed["env"]["TPEN_PLACEMENT_PAYLOAD_DIGEST"] == expected
    resumed = _configured(tmp_path, task_id="member", load={"mode": "train_resume", "path": "/predecessor/core"})
    assert resumed.attempt_binding_digest(
        request.output_dir, root=request.root, group_digest=GROUP, task_id="member", attempt_id="attempt",
    ) != expected


def test_serialized_configured_unavailable_cpu_refuses_before_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured, marker = _marker_run(tmp_path, monkeypatch)
    configured = dataclasses.replace(configured, id="member")
    unavailable = max(os.sched_getaffinity(0)) + 10000
    request = _physical_leaf_request(tmp_path, configured, allocation=_allocation_with_cpu(unavailable))
    result = LeafResult.decode(run_leaf(request.encode()))
    assert result.science is None
    assert result.launch_error is not None
    assert "placement-cpu-outside-worker-affinity" in result.launch_error["message"]
    assert not marker.exists()


@pytest.mark.parametrize("carrier", ["local", "parsl"])
def test_physical_configured_carrier_completes_with_leaf_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, carrier: str,
) -> None:
    configured, marker = _marker_run(tmp_path, monkeypatch)
    configured = dataclasses.replace(configured, id="member")
    group = _group(f"c22-physical-{carrier}", "member")
    lifecycle = _physical_lifecycle(tmp_path)
    if carrier == "local":
        manifest = LocalDispatcher(_worker(tmp_path)).run(group, {"member": configured}, lifecycle, timeout=60)
    else:
        _Parsl().install(monkeypatch)
        manifest = parsl_dispatch.ParslDispatcher(_worker(tmp_path)).run(
            group, {"member": configured}, lifecycle, timeout=60,
        )
    attempt = manifest.attempts[0]
    assert marker.exists()
    assert attempt.outcome_receipt.status == "completed"
    receipt = attempt.placement_receipt
    base_receipt = getattr(receipt, "base", receipt)
    assert base_receipt.comparison["host"] == "matched"
    expected = configured.attempt_binding_digest(
        str(Path(lifecycle.store.root) / "groups" / group.identity / "member" / attempt.attempt_id.value),
        root=str(lifecycle.store.root), group_digest=group.identity, task_id="member",
        attempt_id=attempt.attempt_id.value,
    )
    observed = json.loads(marker.read_text(encoding="utf-8"))
    assert observed["env"]["TPEN_PLACEMENT_PAYLOAD_DIGEST"] == expected
    resumed = _configured(tmp_path, task_id="member", load={"mode": "train_resume", "path": "/predecessor/core"})
    assert resumed.attempt_binding_digest(
        str(Path(lifecycle.store.root) / "groups" / group.identity / "member" / attempt.attempt_id.value),
        root=str(lifecycle.store.root), group_digest=group.identity, task_id="member",
        attempt_id=attempt.attempt_id.value,
    ) != expected


def test_serialized_variants_call_shared_command_method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both serialized configured and legacy argv construction call the shared seam."""

    request = _request(tmp_path)
    legacy_config = "/prepared/config.yaml"
    legacy_overrides = ("payload.mode=attached",)
    legacy = dataclasses.replace(
        request,
        slot_id=None,
        program=None,
        attempt_delta=None,
        config=legacy_config,
        overrides=legacy_overrides,
        environment=PreparedPythonEnvironment(sys.executable, env={}),
        payload_digest=digest({"config": legacy_config, "overrides": list(legacy_overrides)}),
    )
    calls: list[bool] = []
    original = ConfiguredRun.command_for_attempt

    def spy(self, *args, **kwargs):
        calls.append(self.program is not None)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ConfiguredRun, "command_for_attempt", spy)
    configured_args = request.science_args()
    legacy_args = legacy.science_args()
    assert configured_args == _configured(tmp_path).command_for_attempt(
        request.output_dir, root=request.root, group_digest=GROUP, task_id="task", attempt_id="attempt",
    ).args
    assert legacy_args == ConfiguredRun("task", legacy_config, legacy_overrides).command_for_attempt(request.output_dir).args
    assert calls == [True, False, True, False]


def test_attempt_binding_digest_changes_with_placement_delta(tmp_path: Path) -> None:
    configured = _configured(tmp_path)
    resumed = _configured(tmp_path, load={"mode": "train_resume", "path": "/predecessor/core"})
    assert configured.program is not None and resumed.program is not None
    assert configured.program.program_digest != resumed.program.program_digest
    first = configured.attempt_binding_digest(f"/store/groups/{GROUP}/task/a", root="/store", group_digest=GROUP, task_id="task", attempt_id="a")
    second = configured.attempt_binding_digest(f"/store/groups/{GROUP}/task/b", root="/store", group_digest=GROUP, task_id="task", attempt_id="b")
    assert first != second


MUTANTS = (
    {
        "id": "configured-local-duplicate-run-injection",
        "target": "executor/src/tpen_executor/tasks.py",
        "old": "        return ConfiguredProgramCommand(self.program, delta)\n",
        "new": "        command = ConfiguredProgramCommand(self.program, delta)\n        return type(\"MutatedConfiguredCommand\", (), {\"args\": command.args + (f\"run.root={output_dir}\", \"run.layout=flat\", \"run.run_id=core\")})()\n",
        "killer": "test_common_configured_binding_has_no_legacy_run_overrides",
    },
    {
        "id": "parsl-duplicate-run-injection",
        "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": "                args = configured.command_for_attempt(\n                    output_dir,\n                    root=fence.root,\n                    group_digest=group.identity,\n                    task_id=member.task_id,\n                    attempt_id=fence.attempt_id.value,\n                ).args\n",
        "new": "                args = configured.command_for_attempt(\n                    output_dir,\n                    root=fence.root,\n                    group_digest=group.identity,\n                    task_id=member.task_id,\n                    attempt_id=fence.attempt_id.value,\n                ).args + ((f\"run.root={output_dir}\", \"run.layout=flat\", \"run.run_id=core\") if configured.program is not None else ())\n",
        "killer": "test_actual_carriers_capture_matching_full_argv_and_environment",
    },
    {
        "id": "transport-duplicate-run-injection",
        "target": "executor/src/tpen_executor/dispatch/transport.py",
        "old": "        return configured.command_for_attempt(\n            output_dir,\n            root=self.root if self.program is not None else None,\n            group_digest=self.group_digest if self.program is not None else None,\n            task_id=self.task_id if self.program is not None else None,\n            attempt_id=self.attempt_id if self.program is not None else None,\n        ).args\n",
        "new": "        return configured.command_for_attempt(\n            output_dir,\n            root=self.root if self.program is not None else None,\n            group_digest=self.group_digest if self.program is not None else None,\n            task_id=self.task_id if self.program is not None else None,\n            attempt_id=self.attempt_id if self.program is not None else None,\n        ).args + (f\"run.root={output_dir}\", \"run.layout=flat\", \"run.run_id=core\")\n",
        "killer": "test_serialized_leaf_carries_delta_and_rejects_crossed_output",
    },
    {
        "id": "configured-overlay-legacy-payload-digest",
        "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": "        \"TPEN_PLACEMENT_PAYLOAD_DIGEST\": payload_digest,\n",
        "new": "        \"TPEN_PLACEMENT_PAYLOAD_DIGEST\": digest({\"config\": configured.config, \"overrides\": list(configured.overrides)}),\n",
        "killer": "test_physical_configured_carrier_completes_with_leaf_result",
    },
    {
        "id": "serialized-slot-gate-bypass",
        "target": "executor/src/tpen_executor/dispatch/transport.py",
        "old": "            slot = _require_configured_physical_slot(allocation)\n",
        "new": "            slot = type(\"MutatedSlot\", (), {\"slot_id\": \"mutant-slot\", \"node\": \"mutant-host\", \"cpus\": (0,), \"devices\": (), \"env\": {\"TPEN_SLOT_ID\": \"mutant-slot\", \"TPEN_SLOT_NODE\": \"mutant-host\", \"TPEN_SLOT_CPUS\": \"0\"}})()\n",
        "killer": "test_no_slot_configured_serialized_for_attempt_refuses_before_leaf",
    },
    {
        "id": "configured-without-physical-slot-admitted",
        "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": "    if getattr(configured, \"program\", None) is not None:\n",
        "new": "    if False:\n",
        "killer": "test_no_slot_configured_local_refuses_before_child",
    },
    {
        "id": "carried-id-only-admitted",
        "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": "    if getattr(configured, \"program\", None) is not None:\n",
        "new": "    if configured.program is not None and not slot_id:\n",
        "killer": "test_carried_slot_id_without_cpus_refuses_before_child",
    },
    {
        "id": "crossed-group-check",
        "target": "executor/src/tpen_executor/tasks.py",
        "old": "        if os.path.basename(task_parent) != group_digest:\n            raise ValueError(\"configured attempt output does not bind the placement fence\")\n",
        "new": "        if False:\n            raise ValueError(\"configured attempt output does not bind the placement fence\")\n",
        "killer": "test_actual_dispatchers_refuse_crossed_identity_before_child[group-local]",
    },
    {
        "id": "crossed-task-check",
        "target": "executor/src/tpen_executor/tasks.py",
        "old": "        if os.path.basename(attempt_parent) != task_id:\n            raise ValueError(\"configured attempt output does not bind the placement fence\")\n",
        "new": "        if False:\n            raise ValueError(\"configured attempt output does not bind the placement fence\")\n",
        "killer": "test_actual_dispatchers_refuse_crossed_identity_before_child[task-local]",
    },
    {
        "id": "crossed-attempt-check",
        "target": "executor/src/tpen_executor/tasks.py",
        "old": "        if os.path.basename(output_dir) != attempt_id:\n            raise ValueError(\"configured attempt output does not bind the placement fence\")\n",
        "new": "        if False:\n            raise ValueError(\"configured attempt output does not bind the placement fence\")\n",
        "killer": "test_actual_dispatchers_refuse_crossed_identity_before_child[attempt-local]",
    },
    {
        "id": "cross-output-dir",
        "target": "executor/src/tpen_executor/dispatch/transport.py",
        "old": "            if self.attempt_delta != expected_delta:\n                raise ValueError(\"attempt delta does not bind the request fence\")\n",
        "new": "            if False:\n                raise ValueError(\"attempt delta does not bind the request fence\")\n",
        "killer": "test_serialized_leaf_carries_delta_and_rejects_crossed_output",
    },
    {
        "id": "misdirect-hi-output-root",
        "target": "executor/src/tpen_executor/runners.py",
        "old": "            *self.program.argv(),\n",
        "new": "            \"--manifest\", self.program.manifest, \"--row\", self.program.row, \"--output-root\", self.attempt_delta.attempt_core_dir if self.attempt_delta else self.program.output_root,\n",
        "killer": "test_serialized_leaf_carries_delta_and_rejects_crossed_output",
    },
    {
        "id": "own-leaf-resume-path-check",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "            if os.path.commonpath((attempt_leaf, self.load.path)) == attempt_leaf:\n                raise ValueError(\"train_resume load path must not lie in its own attempt leaf\")\n",
        "new": "            if False:\n                raise ValueError(\"train_resume load path must not lie in its own attempt leaf\")\n",
        "killer": "test_load_mode_is_explicit_and_train_resume_is_not_own_leaf",
    },
    {
        "id": "missing-load-mode-defaults-to-fresh",
        "target": "executor/src/tpen_executor/configured_program.py",
        "old": "        if not isinstance(data, Mapping) or set(data) != {\"mode\", \"path\"}:\n            raise ValueError(\"ConfiguredLoad keys differ\")\n",
        "new": "        if not isinstance(data, Mapping):\n            raise ValueError(\"ConfiguredLoad keys differ\")\n        if \"mode\" not in data:\n            data = {**data, \"mode\": \"fresh\", \"path\": None}\n        if set(data) != {\"mode\", \"path\"}:\n            raise ValueError(\"ConfiguredLoad keys differ\")\n",
        "killer": "test_load_mode_is_explicit_and_train_resume_is_not_own_leaf",
    },
    {
        "id": "delta-digest-omission",
        "target": "executor/src/tpen_executor/tasks.py",
        "old": "        return digest({\"program_digest\": self.program.program_digest, \"attempt_delta_digest\": delta.digest})\n",
        "new": "        return self.program.program_digest\n",
        "killer": "test_attempt_binding_digest_changes_with_placement_delta",
    },
)


def test_mutant_sources_compile_and_mapping_rows_are_unique() -> None:
    seen: set[str] = set()
    for mutant in MUTANTS:
        assert set(mutant) == {"id", "target", "old", "new", "killer"}
        assert mutant["id"] not in seen
        seen.add(mutant["id"])
        source_path = ROOT / mutant["target"]
        source = source_path.read_text(encoding="utf-8")
        assert source.count(mutant["old"]) == 1
        compile(source.replace(mutant["old"], mutant["new"], 1), str(source_path), "exec")
        assert mutant["killer"]
