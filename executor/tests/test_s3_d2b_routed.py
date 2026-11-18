"""S3 D2b: routed task labels and strict child-host admission."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

from tpen_executor.dispatch import parsl as parsl_dispatch
from tpen_executor.dispatch.contract import _child_precore_refusal
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
    _manifest_attempts,
    _routed_allocation,
    _routed_transport_payload,
    _runs,
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
)


def test_d2b_mutant_table_targets_unique_executable_anchors() -> None:
    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    assert [row["id"] for row in D2B_MUTANT_TABLE] == ["M1", "M2", "M3", "M4", "M5", "M6"]
    for row in D2B_MUTANT_TABLE:
        source = (root / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, row
        assert row["old"] != row["new"] and row["suite"] == "executor"
        ast.parse(source.replace(row["old"], row["new"]))
        assert f"def {row['killer']}(" in module, row


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

    assert manifest.attempts, f"manifest attempts={_manifest_attempts(manifest)!r}"
    assert [record["executors"] for record in fake_parsl.leaf_submissions()] == [["pool-b"], ["pool-a"]]


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

    assert not (store_root / "groups").exists()
    assert fake_parsl.submissions == []


def _placement_environment(tmp_path: Path, *, strict: bool) -> PreparedPythonEnvironment:
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
    values = {
        "PYTHONPATH": str(tmp_path / "core"),
        "TPEN_PLACEMENT_OUTPUT": str(tmp_path / "child-placement.json"),
        "TPEN_PLACEMENT_OUTPUT_DIR": str(tmp_path),
        "TPEN_PLACEMENT_TASK_ID": "member",
        "TPEN_PLACEMENT_ATTEMPT_ID": "attempt",
        "TPEN_PLACEMENT_GROUP_DIGEST": "group",
        "TPEN_PLACEMENT_ALLOCATION_ID": "allocation",
        "TPEN_PLACEMENT_BACKEND_JOB_ID": "job",
        "TPEN_PLACEMENT_SLOT_ID": "slot",
        "TPEN_PLACEMENT_PAYLOAD_DIGEST": "payload",
        "TPEN_PLACEMENT_DECLARED_HOST": "node-that-cannot-be-this-host",
        "CORE_MARKER": str(tmp_path / "core-ran.txt"),
    }
    if strict:
        values[ROUTED_STRICT_HOST_ENV] = "1"
    return PreparedPythonEnvironment(sys.executable, None, values)


def test_routed_child_host_mismatch_aborts_before_core(tmp_path: Path) -> None:
    environment = _placement_environment(tmp_path, strict=True)
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, f"run.root={tmp_path}"),
        cwd=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert outcome.returncode == 2
    assert not (tmp_path / "core-ran.txt").exists()
    observation = read_child_placement(tmp_path / "child-placement.json")
    assert observation.refusal is not None
    assert observation.refusal["code"] == "routed-child-node-mismatch"
    refusal = _child_precore_refusal(environment, outcome)
    assert refusal is not None and refusal.code == "routed-child-node-mismatch"


def _attach_routed_mismatch_leaf_answers(fake_parsl) -> None:
    """Return a valid remote refusal result with an identity-bound sidecar."""

    def refused(_index, _function, args, _kwargs):
        request = LeafRequest.decode(args[0])
        output_dir, core_dir, stdout_path, stderr_path, _evidence_stdout, _evidence_stderr = request.paths()
        values = request.environment.env
        declared_cpus = tuple(int(item) for item in values["TPEN_SLOT_CPUS"].split(","))
        refusal = {"code": "routed-child-node-mismatch", "reason": "synthetic wrong child host"}
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


def test_routed_child_mismatch_is_launch_failed_without_segment(tmp_path: Path, fake_parsl) -> None:
    allocation = _routed_allocation()
    group = _group("s3d2b-strict-refusal", "member")
    store_root = tmp_path / "store"
    store_root.mkdir()
    lifecycle = AttachedAllocation(allocation, DirectoryStore(store_root))
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path),
        transport=parsl_dispatch.RoutedAttachedTransportSpec.from_mapping(_routed_transport_payload()),
    )
    _attach_routed_probe_answers(dispatcher, fake_parsl, {"pool-a": "node-a", "pool-b": "node-b"})
    _attach_routed_mismatch_leaf_answers(fake_parsl)

    manifest = dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)

    assert [attempt.outcome_receipt.status for attempt in manifest.attempts] == ["launch_failed"]
    assert manifest.segments == ()


def test_v1_child_host_mismatch_remains_advisory(tmp_path: Path) -> None:
    environment = _placement_environment(tmp_path, strict=False)
    # The shim passes the first argument to the fake core, which proves that
    # advisory v1 mismatch handling still reaches core.
    outcome = SubprocessWorker().run(
        environment,
        ("-c", RUN_SHIM, str(tmp_path / "core-ran.txt"), f"run.root={tmp_path}"),
        cwd=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert outcome.returncode == 0 and (tmp_path / "core-ran.txt").read_text(encoding="utf-8") == "core-ran"
    observation = read_child_placement(tmp_path / "child-placement.json")
    assert observation.refusal is not None and observation.refusal["code"] == "child-placement-mismatch"
    assert _child_precore_refusal(environment, outcome) is None


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

    assert calls == ["completed"]
    assert list(store_root.rglob("remote-teardown.json"))
