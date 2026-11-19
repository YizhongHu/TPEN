"""S3 D2b: routed task labels and strict child-host admission."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys

import pytest

from tpen_executor.dispatch import parsl as parsl_dispatch
from tpen_executor.dispatch.contract import _child_precore_refusal, _with_child_placement
from tpen_executor.dispatch.transport import LeafRequest, LeafResult, LeafTeardownReceipt
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import ChildPlacementObservation, WorkerBinding, read_child_placement
from tpen_executor.runners import ROUTED_STRICT_HOST_ENV, RUN_SHIM
from tpen_executor.storage import DirectoryStore
from tpen_executor.workers import SubprocessWorker

from test_s3_transport_parsl import (
    _attach_routed_leaf_answers,
    _attach_routed_probe_answers,
    _group,
    _lifecycle,
    _placed_lifecycle,
    _manifest_attempts,
    _routed_allocation,
    _routed_transport_payload,
    _runs,
    _transport,
    _worker,
    fake_parsl,
)


PARSL_MODULE = "executor/src/tpen_executor/dispatch/parsl.py"
RUNNERS_MODULE = "executor/src/tpen_executor/runners.py"
CONTRACT_MODULE = "executor/src/tpen_executor/dispatch/contract.py"


D2B_MUTANT_TABLE = (
    {
        "id": "M1", "target": PARSL_MODULE,
        "old": "        executors = (\n            [self._routed_label_for_node(binding.node)]\n            if type(self.transport) is RoutedAttachedTransportSpec\n            else list(self._active_executor_labels)\n        )\n",
        "new": "        executors = list(self._active_executor_labels)\n",
        "killer": "test_routed_leaf_submission_uses_the_binding_node_label",
        "suite": "executor", "assertion": "each routed leaf submission names only its binding node label",
    },
    {
        "id": "M2", "target": PARSL_MODULE,
        "old": "        labels = [pool.label for pool in transport.pools if pool.node == node]\n",
        "new": "        labels = [transport.pools[0].label]\n",
        "killer": "test_routed_leaf_submission_uses_the_binding_node_label",
        "suite": "executor", "assertion": "label selection follows the physical node rather than pool order",
    },
    {
        "id": "M3", "target": PARSL_MODULE,
        "old": "            for binding in self._resolve_bindings(lifecycle):\n                self._routed_label_for_node(binding.node)\n",
        "new": "            for binding in self._resolve_bindings(lifecycle):\n                pass\n",
        "killer": "test_missing_routed_binding_is_refused_before_begin",
        "suite": "executor", "assertion": "an absent node-to-label binding is refused before any lifecycle fence",
    },
    {
        "id": "M4", "target": RUNNERS_MODULE,
        "old": "    \"        placement_abort = strict_host\\n\"\n",
        "new": "    \"        placement_abort = False\\n\"\n",
        "killer": "test_routed_child_host_mismatch_aborts_before_core",
        "suite": "executor", "assertion": "strict routed host mismatch writes its sidecar and aborts before core import",
    },
    {
        "id": "M5", "target": CONTRACT_MODULE,
        "old": "    if refusal.get(\"code\") == \"routed-child-node-mismatch\":\n        if environment.env.get(ROUTED_STRICT_HOST_ENV) != \"1\":\n            return None\n",
        "new": "    if False:\n        return None\n",
        "killer": "test_routed_child_host_mismatch_aborts_before_core",
        "suite": "executor", "assertion": "only the explicit routed flag turns the exact sidecar case into PlacementRefusal",
    },
    {
        "id": "M6", "target": PARSL_MODULE,
        "old": "                completion_lifecycle = (\n                    _RemoteCompletionLifecycle(lifecycle, record[\"handle\"])\n                    if is_remote else lifecycle\n                )\n",
        "new": "                completion_lifecycle = lifecycle\n",
        "killer": "test_routed_success_uses_finish_remote",
        "suite": "executor", "assertion": "routed normal completion persists through finish_remote and its teardown receipt",
    },
    {
        "id": "M7", "target": PARSL_MODULE,
        "old": "                    strict_host=type(self.transport) is RoutedAttachedTransportSpec,\n",
        "new": "                    strict_host=False,\n",
        "killer": "test_routed_child_mismatch_is_launch_failed_without_segment",
        "suite": "executor", "assertion": "routed leaf requests enable strict host admission before science",
    },
    {
        "id": "M8", "target": CONTRACT_MODULE,
        "old": "    else:\n        # Neither prepared nor slot values may enable this controller policy.\n        values.pop(ROUTED_STRICT_HOST_ENV, None)\n",
        "new": "    else:\n        pass  # mutant: preserve carried strict flag\n",
        "killer": "test_v1_child_host_mismatch_remains_advisory",
        "suite": "executor", "assertion": "prepared strict flag is cleared for advisory placement",
    },
    {
        "id": "M9", "target": "executor/src/tpen_executor/workers.py",
        "old": "        child_env.pop(ROUTED_STRICT_HOST_ENV, None)\n",
        "new": "        pass  # mutant: inherit owner strict flag\n",
        "killer": "test_v1_child_host_mismatch_remains_advisory",
        "suite": "executor", "assertion": "ambient owner strict flag cannot change advisory child policy",
    },
    {
        "id": "M10", "target": CONTRACT_MODULE,
        "old": "                environment = PreparedPythonEnvironment(environment.python, environment.setup, values)\n",
        "new": "                pass  # mutant: retain unsanitized early-return descriptor\n",
        "killer": "test_advisory_no_slot_clears_carried_strict_host",
        "suite": "executor", "assertion": "legacy no-slot return also clears the prepared strict flag",
    },
)


def test_d2b_mutant_table_targets_unique_executable_anchors() -> None:
    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    assert [row["id"] for row in D2B_MUTANT_TABLE] == ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"], (
        f"anchor ids={tuple(row['id'] for row in D2B_MUTANT_TABLE)!r}; "
        "outcome/receipt details unavailable before execution"
    )
    for row in D2B_MUTANT_TABLE:
        source = (root / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, (
            f"row={row!r}; anchor_count={source.count(row['old'])!r}; "
            "outcome/receipt details unavailable before execution"
        )
        assert row["old"] != row["new"] and row["suite"] == "executor", (
            f"row={row!r}; outcome/receipt details unavailable before execution"
        )
        ast.parse(source.replace(row["old"], row["new"]))
        assert f"def {row['killer']}(" in module, (
            f"row={row!r}; outcome/receipt details unavailable before execution"
        )


def test_routed_leaf_submission_uses_the_binding_node_label(tmp_path: Path, fake_parsl) -> None:
    """A reversed slot order proves routing uses node identity, not index."""

    allocation = _routed_allocation(("node-b", "node-a"))
    group = _group("s3d2b-routing", "first", "second")
    store_root = tmp_path / "store"
    store_root.mkdir()
    lifecycle = AttachedAllocation(allocation, DirectoryStore(store_root))
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path),
        transport=parsl_dispatch.RoutedAttachedTransportSpec.from_mapping(_routed_transport_payload()),
    )
    _attach_routed_probe_answers(dispatcher, fake_parsl, {"pool-a": "node-a", "pool-b": "node-b"})
    _attach_routed_leaf_answers(fake_parsl)

    manifest = dispatcher.run(group, _runs(tmp_path, "first", "second"), lifecycle, timeout=60)

    details = f"attempts={_manifest_attempts(manifest)!r}; segments={manifest.segments!r}"
    assert manifest.attempts, details
    assert [record["executors"] for record in fake_parsl.leaf_submissions()] == [["pool-b"], ["pool-a"]], details
    for record in fake_parsl.leaf_submissions():
        request = LeafRequest.decode(record["args"][0])
        assert request.environment.env.get(ROUTED_STRICT_HOST_ENV) == "1", (
            f"{details}; task_id={request.task_id!r}; environment={request.environment.env!r}"
        )


def test_attached_leaf_submission_does_not_enable_strict_host(tmp_path: Path, fake_parsl) -> None:
    group = _group("s3d2b-attached-advisory", "member")
    lifecycle = _placed_lifecycle(tmp_path, min(os.sched_getaffinity(0)))
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path), transport=_transport(),
    )

    manifest = dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)

    details = f"attempts={_manifest_attempts(manifest)!r}; segments={manifest.segments!r}"
    assert [attempt.outcome_receipt.status for attempt in manifest.attempts] == ["completed"], details
    assert len(fake_parsl.leaf_submissions()) == 1, details
    request = LeafRequest.decode(fake_parsl.leaf_submissions()[0]["args"][0])
    for key in ("TPEN_PLACEMENT_TASK_ID", "TPEN_PLACEMENT_ATTEMPT_ID",
                "TPEN_PLACEMENT_SLOT_ID", "TPEN_PLACEMENT_DECLARED_HOST"):
        assert request.environment.env.get(key), f"{details}; missing placement identity: {key}"
    assert ROUTED_STRICT_HOST_ENV not in request.environment.env, (
        f"{details}; task_id={request.task_id!r}; environment={request.environment.env!r}"
    )


def test_advisory_no_slot_clears_carried_strict_host(tmp_path: Path) -> None:
    """Legacy placement returns must serialize no controller-owned flag."""

    group = _group("s3d2b-no-slot", "member")
    lifecycle = _lifecycle(tmp_path)
    fence = lifecycle.begin(group, "member", "2026-11-18T00:00:00+00:00")
    environment = _with_child_placement(
        PreparedPythonEnvironment(sys.executable, None, {ROUTED_STRICT_HOST_ENV: "1"}),
        lifecycle.allocation, fence, group, group.members[0], _runs(tmp_path, "member")["member"],
        strict_host=False,
    )
    assert "TPEN_PLACEMENT_TASK_ID" not in environment.env
    assert ROUTED_STRICT_HOST_ENV not in environment.env, environment.env


def test_missing_routed_binding_is_refused_before_begin(tmp_path: Path, fake_parsl, monkeypatch: pytest.MonkeyPatch) -> None:
    allocation = _routed_allocation()
    group = _group("s3d2b-missing-label", "member")
    store_root = tmp_path / "store"
    store_root.mkdir()
    lifecycle = AttachedAllocation(allocation, DirectoryStore(store_root))
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path),
        transport=parsl_dispatch.RoutedAttachedTransportSpec.from_mapping(_routed_transport_payload()),
    )
    missing = WorkerBinding({"TPEN_SLOT_ID": "bad-slot", "TPEN_SLOT_NODE": "node-missing", "TPEN_SLOT_CPUS": "0"})
    monkeypatch.setattr(
        parsl_dispatch.ParslDispatcher,
        "_resolve_bindings_for_allocation",
        lambda _self, _allocation: (missing,),
    )

    with pytest.raises(parsl_dispatch.UnsupportedTransportError, match="exactly one executor label"):
        dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)

    details = f"submissions={fake_parsl.submissions!r}; store_groups={(store_root / 'groups').exists()!r}"
    assert not (store_root / "groups").exists(), details
    assert fake_parsl.submissions == [], details


def _placement_environment(tmp_path: Path, *, strict: bool, carried_flag: bool = False) -> PreparedPythonEnvironment:
    package = tmp_path / "core" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        "from pathlib import Path\n"
        "def main(argv):\n"
        "    Path(argv[0]).write_text('core-ran', encoding='utf-8')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    group = _group("s3d2b-placement", "member")
    allocation = _routed_allocation(("node-that-cannot-be-this-host",))
    store_root = tmp_path / "store"
    store_root.mkdir()
    lifecycle = AttachedAllocation(allocation, DirectoryStore(store_root))
    fence = lifecycle.begin(group, "member", "2026-11-18T00:00:00+00:00", transport_kind="remote")
    configured = _runs(tmp_path, "member")["member"]
    return _with_child_placement(
        PreparedPythonEnvironment(
            sys.executable,
            None,
            {"PYTHONPATH": str(tmp_path / "core"), **({ROUTED_STRICT_HOST_ENV: "1"} if carried_flag else {})},
        ),
        allocation,
        fence,
        group,
        group.members[0],
        configured,
        check_controller_affinity=False,
        slot_resolver=parsl_dispatch.routed_slots_from_allocation,
        strict_host=strict,
    )


def test_routed_child_host_mismatch_aborts_before_core(tmp_path: Path) -> None:
    environment = _placement_environment(tmp_path, strict=True)
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, str(tmp_path / "core-ran.txt"), f"run.root={tmp_path}"),
        cwd=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    details = f"returncode={outcome.returncode!r}; stdout={outcome.stdout_path!r}; stderr={outcome.stderr_path!r}"
    assert outcome.returncode == 2, details
    assert not (tmp_path / "core-ran.txt").exists(), details
    observation = read_child_placement(environment.env["TPEN_PLACEMENT_OUTPUT"])
    details = f"returncode={outcome.returncode!r}; observation={observation.to_dict()!r}"
    assert observation.refusal is not None, details
    assert observation.refusal["code"] == "routed-child-node-mismatch", details
    refusal = _child_precore_refusal(environment, outcome)
    assert refusal is not None and refusal.code == "routed-child-node-mismatch", (
        f"returncode={outcome.returncode!r}; observation={observation.to_dict()!r}; refusal={refusal!r}"
    )


def test_routed_refusal_sidecar_and_teardown_are_json_serializable() -> None:
    """Public refusal and teardown records must remain JSON-shaped."""

    group_digest = "a" * 64
    refusal = {"code": "routed-child-node-mismatch", "reason": "synthetic wrong child host"}
    observation = ChildPlacementObservation(
        "task", "attempt", group_digest, "allocation", "backend", "slot", "b" * 64,
        "/tmp/output", "wrong-child-node", 4243, 55, (), {"OMP_NUM_THREADS": None}, None,
        "2026-11-18T00:00:00Z", refusal,
    )
    teardown = LeafTeardownReceipt(
        group_digest, "task", "attempt", "allocation", "backend", True, "not-run",
    )

    encoded = json.dumps({"placement": observation.to_dict(), "teardown": teardown.to_dict()})
    decoded = json.loads(encoded)
    assert decoded["placement"]["refusal"] == refusal
    assert decoded["teardown"]["child_group_reaped"] is True


def _attach_routed_mismatch_leaf_answers(fake_parsl, errors: list[str] | None = None) -> None:
    """Mirror the shim's strict/advisory mismatch using the submitted flag."""

    _attach_routed_leaf_answers(fake_parsl)
    completed = fake_parsl.hooks["run_leaf"]

    def refused(_index, _function, args, _kwargs):
        try:
            return build_refusal(args)
        except BaseException as error:
            if errors is not None:
                errors.append(f"{type(error).__name__}: {error}")
            raise

    def build_refusal(args):
        request = LeafRequest.decode(args[0])
        output_dir, _core_dir, stdout_path, stderr_path, _evidence_stdout, _evidence_stderr = request.paths()
        values = request.environment.env
        declared_cpus = tuple(int(item) for item in values["TPEN_SLOT_CPUS"].split(","))
        strict_host = values.get(ROUTED_STRICT_HOST_ENV) == "1"
        placement_abort = strict_host
        refusal = {
            "code": "routed-child-node-mismatch" if strict_host else "child-placement-mismatch",
            "reason": "synthetic wrong child host",
        }
        # An advisory mismatch still runs core. Reuse the successful leaf's
        # evidence, then replace its placement sidecar with the mismatch.
        advisory_result = None if placement_abort else completed(0, None, args, {})
        sidecar = ChildPlacementObservation(
            request.task_id,
            request.attempt_id,
            request.group_digest,
            request.allocation_id,
            request.backend_job_id,
            values["TPEN_SLOT_ID"],
            request.payload_digest,
            output_dir,
            "wrong-child-node",
            4243,
            55,
            declared_cpus,
            {"OMP_NUM_THREADS": None},
            values.get("CUDA_VISIBLE_DEVICES"),
            "2026-11-18T00:00:00Z",
            refusal,
        )
        Path(values["TPEN_PLACEMENT_OUTPUT"]).write_text(
            json.dumps(sidecar.to_dict()) + "\n", encoding="utf-8",
        )
        if not placement_abort:
            return advisory_result
        return LeafResult(
            request.digest(),
            request.group_digest,
            request.task_id,
            request.attempt_id,
            request.allocation_id,
            request.backend_job_id,
            request.slot_id,
            request.payload_digest,
            values["TPEN_SLOT_NODE"],
            4242,
            sys.executable,
            ("science",),
            {
                "argv": [request.environment.python, *request.science_args()],
                "returncode": 2,
                "timed_out": False,
                "pid": 4243,
                "proc_start_ticks": 55,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
            },
            None,
            refusal,
            "not-run",
            None,
            None,
            LeafTeardownReceipt.for_request(request, inspection="not-run", child_group_reaped=True).to_dict(),
        ).encode()

    fake_parsl.hooks["run_leaf"] = refused


def test_routed_child_mismatch_is_launch_failed_without_segment(
    tmp_path: Path, fake_parsl, monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocation = _routed_allocation()
    group = _group("s3d2b-strict-refusal", "member")
    store_root = tmp_path / "store"
    store_root.mkdir()
    lifecycle = AttachedAllocation(allocation, DirectoryStore(store_root))
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path),
        transport=parsl_dispatch.RoutedAttachedTransportSpec.from_mapping(_routed_transport_payload()),
    )
    resolve_errors: list[str] = []
    real_resolve = parsl_dispatch._RemoteLeaf.resolve

    def tracing_resolve(handle, payload):
        try:
            return real_resolve(handle, payload)
        except BaseException as error:
            resolve_errors.append(f"{type(error).__name__}: {error}")
            raise

    # Preserve the production exception path while making a swallowed decode
    # or identity error visible in this test's assertion details.
    monkeypatch.setattr(parsl_dispatch._RemoteLeaf, "resolve", tracing_resolve)
    _attach_routed_probe_answers(dispatcher, fake_parsl, {"pool-a": "node-a", "pool-b": "node-b"})
    leaf_errors: list[str] = []
    _attach_routed_mismatch_leaf_answers(fake_parsl, leaf_errors)

    manifest = dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)

    details = (
        f"attempts={_manifest_attempts(manifest)!r}; segments={manifest.segments!r}; "
        f"remote_teardowns={list(store_root.rglob('remote-teardown.json'))!r}; "
        f"leaf_errors={leaf_errors!r}; resolve_errors={resolve_errors!r}"
    )
    assert [attempt.outcome_receipt.status for attempt in manifest.attempts] == ["launch_failed"], details
    assert manifest.segments == (), details
    assert list(store_root.rglob("remote-teardown.json")), details


@pytest.mark.parametrize("flag_source", ["none", "prepared", "owner"])
def test_v1_child_host_mismatch_remains_advisory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag_source: str,
) -> None:
    # Isolate prepared contamination from ambient owner contamination.
    monkeypatch.delenv(ROUTED_STRICT_HOST_ENV, raising=False)
    environment = _placement_environment(tmp_path, strict=False, carried_flag=flag_source == "prepared")
    if flag_source == "owner":
        monkeypatch.setenv(ROUTED_STRICT_HOST_ENV, "1")
    assert ROUTED_STRICT_HOST_ENV not in environment.env
    # The shim passes the first argument to the fake core, which proves that
    # advisory v1 mismatch handling still reaches core.
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, str(tmp_path / "core-ran.txt"), f"run.root={tmp_path}"),
        cwd=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    details = f"returncode={outcome.returncode!r}; stdout={outcome.stdout_path!r}; stderr={outcome.stderr_path!r}"
    assert outcome.returncode == 0 and (tmp_path / "core-ran.txt").read_text(encoding="utf-8") == "core-ran", details
    observation = read_child_placement(environment.env["TPEN_PLACEMENT_OUTPUT"])
    details = f"returncode={outcome.returncode!r}; observation={observation.to_dict()!r}"
    assert observation.refusal is not None and observation.refusal["code"] == "child-placement-mismatch", details
    assert _child_precore_refusal(environment, outcome) is None, (
        f"returncode={outcome.returncode!r}; observation={observation.to_dict()!r}; "
        f"refusal={_child_precore_refusal(environment, outcome)!r}"
    )


def test_routed_success_uses_finish_remote(tmp_path: Path, fake_parsl, monkeypatch: pytest.MonkeyPatch) -> None:
    allocation = _routed_allocation()
    group = _group("s3d2b-remote-finish", "member")
    store_root = tmp_path / "store"
    store_root.mkdir()
    lifecycle = AttachedAllocation(allocation, DirectoryStore(store_root))
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path),
        transport=parsl_dispatch.RoutedAttachedTransportSpec.from_mapping(_routed_transport_payload()),
    )
    _attach_routed_probe_answers(dispatcher, fake_parsl, {"pool-a": "node-a", "pool-b": "node-b"})
    _attach_routed_leaf_answers(fake_parsl)
    calls = []
    real_finish_remote = AttachedAllocation.finish_remote

    def spying_finish_remote(self, *args, **kwargs):
        calls.append(args[3])
        return real_finish_remote(self, *args, **kwargs)

    monkeypatch.setattr(AttachedAllocation, "finish_remote", spying_finish_remote)
    dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)

    details = f"finish_remote_statuses={calls!r}; remote_teardowns={list(store_root.rglob('remote-teardown.json'))!r}"
    assert calls == ["completed"], details
    assert list(store_root.rglob("remote-teardown.json")), details
