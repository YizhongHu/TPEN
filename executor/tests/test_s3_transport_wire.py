"""S3 Layer A: remote-leaf wire and worker-owned execution.

The wire is :mod:`tpen_executor.dispatch.transport`.  These tests prove the
codec round trip, refusal of live objects and crossed identities, and, with a
real serialized leaf in a separate local Python process and a fake-core
``tpen`` package, that science and then inspection run on the leaf side,
that the output is fresh, that the science argv is exactly ``RUN_SHIM``, and
that only the controller publishes store records.

They also prove the r2 review fixes: leaf-side slot admission against the
store-derived durable quarantine, request binding of task and attempt
independently of placement, controller refusal of foreign evidence and of
skipped inspection, exact nested types, and canonical-only decoding.

``S3A_MUTANT_TABLE`` lists the contract mutants R1-R3, R5 and R6 (R4 is
retired); each names its killer.
"""

from __future__ import annotations

import ast
from concurrent.futures import Future
import dataclasses
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import threading

import pytest

from tpen_executor import workers
from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch import LocalDispatcher
from tpen_executor.dispatch.contract import _with_child_context_keys, _with_child_placement
from tpen_executor.dispatch.parsl import _ChildProcessHandle
from tpen_executor.dispatch.transport import (
    LEAF_REQUEST_SCHEMA,
    LEAF_RESULT_SCHEMA,
    LeafRequest,
    LeafResult,
    LeafTeardownReceipt,
    LeafResultRejected,
    run_leaf,
    validate_leaf_result,
)
from tpen_executor.environments import PreparedPythonEnvironment, SourceScript
from tpen_executor.gpu_sampler import GPU_SAMPLER_OUTPUT_ENV, QUARANTINE_SCHEMA, SLOT_QUARANTINE_ENV, slot_quarantine_dir
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import PLACEMENT_REFUSAL_ENV, AllocationSlot, read_child_placement
from tpen_executor.programs import WorkerProgram
from tpen_executor.runners import CHILD_OBSERVATION_FILENAME, RUN_SHIM
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun


EXECUTOR_ROOT = Path(__file__).parents[1]
TRANSPORT = "executor/src/tpen_executor/dispatch/transport.py"
WORKER_MARK = "S3A_WORKER_MARK"
# The leaf host is a fresh Python process that sees only the request bytes on
# stdin and writes only the result bytes to stdout.
LEAF_HOST = (
    "import sys\n"
    "from tpen_executor.dispatch.transport import run_leaf\n"
    "sys.stdout.buffer.write(run_leaf(sys.stdin.buffer.read()))\n"
)


S3A_MUTANT_TABLE = (
    {
        "id": "R1", "target": TRANSPORT,
        "old": "        for name in _REQUEST_PLAIN_FIELDS:\n            _require_plain(getattr(self, name), name)\n",
        "new": "        for name in _REQUEST_PLAIN_FIELDS:\n            pass\n",
        "killer": "test_request_rejects_live_objects_in_every_field",
        "suite": "executor", "assertion": "a lock, future, process handle or handle-carrying builtin subclass never rides in the request",
    },
    {
        "id": "R2", "target": TRANSPORT,
        "old": "        \"attempt\": result.attempt_id == request.attempt_id,\n",
        "new": "",
        "killer": "test_result_validation_rejects_each_crossed_identity",
        "suite": "executor", "assertion": "a result naming another attempt is rejected even when its request digest matches",
    },
    {
        "id": "R3", "target": TRANSPORT,
        "old": "    if science is not None and child_refusal is None:\n        phases.append(\"inspection\")\n",
        "new": "    if False:\n        phases.append(\"inspection\")\n",
        "killer": "test_real_serialized_leaf_runs_science_then_inspection_on_the_worker",
        "suite": "executor", "assertion": "public inspection runs in the leaf after science, not in the controller after the leaf returns",
    },
    {
        "id": "R5", "target": TRANSPORT,
        "old": "        return (*TPENCommand(self.config, self.overrides).args, f\"run.root={output_dir}\"",
        "new": "        return (\"-m\", \"tpen.run\", \"--config\", self.config, *self.overrides, f\"run.root={output_dir}\"",
        "killer": "test_real_serialized_leaf_runs_science_then_inspection_on_the_worker",
        "suite": "executor", "assertion": "the leaf launches science through RUN_SHIM, never python -m tpen.run",
    },
    {
        "id": "R6", "target": TRANSPORT,
        "old": "        _admit_slot(environment, request)\n",
        "new": "        pass\n",
        "killer": "test_leaf_admits_its_slot_against_the_store_quarantine",
        "suite": "executor", "assertion": "a fresh leaf refuses a durably quarantined slot before any spawn, whatever quarantine overlay it carries",
    },
)
# R4 (the leaf calls DirectoryStore.finish_attempt) is retired in fix round r2
# to keep the table at five rows: it is the weakest control because the
# static authority test already refuses the storage import it needs, and the
# real-leaf test still asserts that no outcome exists before the controller
# publishes.


def test_s3a_mutant_table_targets_unique_executable_anchors() -> None:
    """Every mutant names a unique executable anchor, parses, and has a killer."""

    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    assert [row["id"] for row in S3A_MUTANT_TABLE] == ["R1", "R2", "R3", "R5", "R6"]
    for row in S3A_MUTANT_TABLE:
        source = (root / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, row
        assert row["old"] != row["new"] and row["suite"] == "executor" and row["assertion"]
        # Anchors are code, not docstrings or comments, and the mutant parses.
        assert not row["old"].strip().startswith(("#", '"""', "'"))
        ast.parse(source.replace(row["old"], row["new"]))
        assert f"def {row['killer'].split('[')[0]}(" in module, row


# --------------------------------------------------------------- fixtures


def _group(*task_ids: str) -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    members = tuple(TaskSpec(task_id, task_id, program, digest(task_id), 1, "steps") for task_id in task_ids)
    return ReadyGroup("s3a-group", members, digest({"plan": "s3a"}))


def _allocation(facts: dict | None = None) -> ConfiguredAllocation:
    return ConfiguredAllocation(AllocationId("s3a-allocation"), BackendJobId("job-s3a"), facts or {})


def _runs(tmp_path: Path, *task_ids: str, modes: dict[str, str] | None = None) -> dict[str, ConfiguredRun]:
    modes = {} if modes is None else modes
    result = {}
    for task_id in task_ids:
        config = tmp_path / f"{task_id}.cfg"
        config.write_text(modes.get(task_id, "ok"), encoding="utf-8")
        result[task_id] = ConfiguredRun(task_id, str(config), (f"payload.task={task_id}",))
    return result


def _make_fake_core(tmp_path: Path) -> Path:
    """Write a fake ``tpen`` with ``run.main`` and ``inspection``.

    Like real core, ``tpen/run.py`` has no ``__main__`` guard, so
    ``python -m tpen.run`` exits 0 without running anything.  Both modules
    append ``[phase, parent pid, worker mark]`` to ``S3A_EVENT_LOG``.
    """

    package = tmp_path / "fake-core" / "tpen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "run.py").write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            import time
            from pathlib import Path

            def _event(phase):
                with open(os.environ["S3A_EVENT_LOG"], "a", encoding="utf-8") as log:
                    log.write(json.dumps([phase, os.getppid(), os.environ.get({WORKER_MARK!r})]) + "\\n")

            def main(argv):
                _event("science")
                config = Path(argv[argv.index("--config") + 1])
                mode = config.read_text(encoding="utf-8")
                overrides = dict(item.split("=", 1) for item in argv if "=" in item)
                run_dir = Path(overrides["run.root"]) / overrides["run.run_id"]
                if mode == "sleep":
                    time.sleep(4)
                if mode == "fail":
                    return 7
                run_dir.mkdir(parents=True)
                (run_dir / "argv.json").write_text(json.dumps({{"argv": argv, "cwd": os.getcwd()}}), encoding="utf-8")
                (run_dir / "evidence.json").write_text(json.dumps({{
                    "schema": "tpen.run-evidence/v2",
                    "run_dir": str(run_dir),
                    "verdict": "completed",
                    "run_status": "completed",
                    "checkpoint": {{"checkpoint_dir": str(run_dir / "checkpoint"), "schema_version": 1, "next_iteration": 1, "completed_updates": 1}},
                    "refusal": None,
                    "resolved_config_identity": {{"schema": "tpen-config-identity-v1", "sha256": "a" * 64}},
                }}), encoding="utf-8")
                return 0
            """
        ),
        encoding="utf-8",
    )
    (package / "inspection.py").write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            import sys
            from pathlib import Path

            with open(os.environ["S3A_EVENT_LOG"], "a", encoding="utf-8") as log:
                log.write(json.dumps(["inspection", os.getppid(), os.environ.get({WORKER_MARK!r})]) + "\\n")
            print(json.dumps(json.loads(Path(sys.argv[1], "evidence.json").read_text(encoding="utf-8"))))
            """
        ),
        encoding="utf-8",
    )
    return package.parent


def _environment(tmp_path: Path) -> PreparedPythonEnvironment:
    package = _make_fake_core(tmp_path)
    return PreparedPythonEnvironment(
        sys.executable,
        env={"PYTHONPATH": str(package), "S3A_EVENT_LOG": str(tmp_path / "events.jsonl")},
    )


def _events(tmp_path: Path) -> list[list]:
    path = tmp_path / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _begin(tmp_path: Path, task_id: str = "member", *, facts: dict | None = None, mode: str = "ok"):
    """Begin one attempt on the controller and return its parts."""

    group = _group(task_id)
    allocation = _allocation(facts)
    lifecycle = AttachedAllocation(allocation, DirectoryStore(tmp_path / "store"))
    configured = _runs(tmp_path, task_id, modes={task_id: mode})[task_id]
    fence = lifecycle.begin(group, task_id, "2026-11-11T00:00:00+00:00")
    return group, allocation, lifecycle, configured, fence


def _request(tmp_path: Path, *, timeout: float | None = 60, **kwargs) -> tuple[LeafRequest, tuple]:
    group, allocation, lifecycle, configured, fence = _begin(tmp_path, **kwargs)
    request = LeafRequest.for_attempt(fence, allocation, configured, _environment(tmp_path), timeout)
    return request, (group, allocation, lifecycle, configured, fence)


def _serialized_leaf(request: LeafRequest) -> tuple[int, bytes]:
    """Run :func:`run_leaf` in a fresh worker-marked Python process."""

    search = [str(EXECUTOR_ROOT / "src"), *filter(None, [os.environ.get("PYTHONPATH")])]
    environment = {**os.environ, WORKER_MARK: "s3a-worker-0", "PYTHONPATH": os.pathsep.join(search)}
    process = subprocess.Popen(
        [sys.executable, "-c", LEAF_HOST],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    stdout, stderr = process.communicate(request.encode(), timeout=120)
    assert process.returncode == 0, stderr.decode("utf-8", "replace")
    return process.pid, stdout


def _publish(lifecycle: AttachedAllocation, fence, result: LeafResult) -> None:
    """Controller-side publication with the local completion's acceptance rule."""

    lifecycle.finish(fence, result.status(), "2026-11-11T00:00:01+00:00", result.result_digest())
    evidence = result.core_evidence()
    if (
        result.status() == "completed"
        and evidence is not None
        and evidence.verdict in {"completed", "incomplete"}
        and evidence.checkpoint is not None
    ):
        lifecycle.store.accept_segment(fence, evidence, None)


def _synthetic_result(request: LeafRequest, **changes) -> LeafResult:
    """Return a consistent completed result for ``request``, then ``changes``."""

    _output, core_dir, stdout_path, stderr_path, _evidence_stdout, _evidence_stderr = request.paths()
    result = LeafResult(
        request.digest(), request.group_digest, request.task_id, request.attempt_id, request.allocation_id,
        request.backend_job_id, request.slot_id, request.payload_digest, "leaf-host", 4242, sys.executable,
        ("science", "inspection"),
        {
            "argv": [request.environment.python, *request.science_args()], "returncode": 0, "timed_out": False,
            "pid": 4243, "proc_start_ticks": None, "stdout_path": stdout_path, "stderr_path": stderr_path,
        },
        None, None, "evidence",
        {
            "schema": "tpen.run-evidence/v2", "run_dir": core_dir, "verdict": "completed", "run_status": "completed",
            "checkpoint": {"checkpoint_dir": core_dir + "/checkpoint", "schema_version": 1, "next_iteration": 1, "completed_updates": 1},
            "refusal": None, "resolved_config_identity": {"schema": "tpen-config-identity-v1", "sha256": "a" * 64},
        },
        None, LeafTeardownReceipt.for_request(request, inspection="complete").to_dict(),
    )
    return dataclasses.replace(result, **changes) if changes else result


# ------------------------------------------------------------------ codec


def test_request_round_trips_through_canonical_bytes(tmp_path: Path) -> None:
    """The request survives encode/decode exactly and only bytes are wire data."""

    request, (_group, _allocation, _lifecycle, configured, fence) = _request(tmp_path)
    encoded = request.encode()
    decoded = LeafRequest.decode(encoded)
    assert decoded == request and decoded.encode() == encoded
    assert json.loads(encoded)["schema"] == LEAF_REQUEST_SCHEMA
    assert decoded.attempt_id == fence.attempt_id.value and decoded.output_dir == fence.output_dir
    assert decoded.payload_digest == digest({"config": configured.config, "overrides": list(configured.overrides)})
    assert decoded.digest() == request.digest()
    # In-process objects, decoded mappings and text are not wire bytes.
    for not_bytes in (request, request.to_dict(), encoded.decode("utf-8"), bytearray(encoded)):
        with pytest.raises(TypeError, match="canonical JSON bytes"):
            LeafRequest.decode(not_bytes)  # type: ignore[arg-type]
    # A live handle cannot be smuggled in as an extra wire member.
    for extra in ("process_handle", "pid", "future"):
        with pytest.raises(ValueError, match="keys differ"):
            LeafRequest.from_mapping({**request.to_dict(), extra: 1})
    with pytest.raises(ValueError, match="duplicate JSON key"):
        LeafRequest.decode(encoded[:-1] + b',"timeout":60}')


class _HandleCarryingText(str):
    """A ``str`` subclass that carries a live controller handle."""


class _HandleCarryingEnvironment(PreparedPythonEnvironment):
    """A prepared-environment subclass that can carry a live attribute."""


def _with_handle_text(value: str) -> str:
    text = _HandleCarryingText(value)
    text.handle = _ChildProcessHandle()  # type: ignore[attr-defined]
    return text


@pytest.mark.parametrize(
    "field_name, live",
    [
        ("timeout", lambda request: threading.Lock()),
        ("timeout", lambda request: Future()),
        ("slot_id", lambda request: Future()),
        ("overrides", lambda request: (*request.overrides, _ChildProcessHandle())),
        ("task_id", lambda request: _with_handle_text(request.task_id)),
        ("output_dir", lambda request: _with_handle_text(request.output_dir)),
    ],
    ids=["timeout-lock", "timeout-future", "slot-future", "overrides-handle", "task-text-subclass", "output-text-subclass"],
)
def test_request_rejects_live_objects_in_every_field(tmp_path: Path, field_name: str, live) -> None:
    """R1 killer: no lock, future, handle or handle-carrying subclass is plain."""

    request, _parts = _request(tmp_path)
    with pytest.raises(TypeError, match="is not plain wire data"):
        dataclasses.replace(request, **{field_name: live(request)})


def test_request_rejects_a_live_environment_and_non_numeric_timeout(tmp_path: Path) -> None:
    """The environment must be the exact descriptor; a bool is not a timeout."""

    request, _parts = _request(tmp_path)
    environment = _HandleCarryingEnvironment(request.environment.python, request.environment.setup, dict(request.environment.env))
    object.__setattr__(environment, "handle", _ChildProcessHandle())
    with pytest.raises(TypeError, match="environment is not plain wire data"):
        dataclasses.replace(request, environment=environment)
    with pytest.raises(TypeError, match="environment is not plain wire data"):
        dataclasses.replace(request, environment=request.environment.to_dict())
    with pytest.raises(TypeError, match="timeout must be"):
        dataclasses.replace(request, timeout=True)
    for bad in (0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="timeout must be finite"):
            dataclasses.replace(request, timeout=bad)


class _HandleCarryingScript(SourceScript):
    """A setup-script subclass that can carry a live attribute."""


def test_request_and_result_reject_nested_live_subclasses(tmp_path: Path) -> None:
    """Nested setup descriptors and fact leaves/keys must be exact types."""

    request, _parts = _request(tmp_path)
    script = str(tmp_path / "setup.sh")
    # Control: an exact setup descriptor is wire data and round-trips.
    exact = PreparedPythonEnvironment(request.environment.python, SourceScript(script), dict(request.environment.env))
    with_setup = dataclasses.replace(request, environment=exact)
    assert LeafRequest.decode(with_setup.encode()) == with_setup
    live = _HandleCarryingScript(script)
    object.__setattr__(live, "handle", threading.Lock())
    smuggled = PreparedPythonEnvironment(request.environment.python, live, dict(request.environment.env))
    with pytest.raises(TypeError, match="environment.setup is not plain wire data"):
        dataclasses.replace(request, environment=smuggled)

    result = _synthetic_result(request)
    assert LeafResult.decode(result.encode()) == result
    stdout_path = _with_handle_text(result.science["stdout_path"])
    with pytest.raises(TypeError, match=r"science\.stdout_path is not plain wire data"):
        dataclasses.replace(result, science={**dict(result.science), "stdout_path": stdout_path})
    with pytest.raises(TypeError, match="launch_error key is not plain wire data"):
        dataclasses.replace(
            result, phases=(), science=None, inspection="not-run", evidence=None,
            launch_error={_with_handle_text("type"): "OSError", "message": "m"},
        )
    evidence = {_with_handle_text("verdict") if key == "verdict" else key: item for key, item in result.evidence.items()}
    with pytest.raises(TypeError, match="evidence key is not plain wire data"):
        dataclasses.replace(result, evidence=evidence)


def test_decoders_accept_only_canonical_bytes(tmp_path: Path) -> None:
    """Bytes that decode to a valid record but re-encode differently are refused."""

    request, _parts = _request(tmp_path)
    encoded = request.encode()
    assert LeafRequest.decode(encoded).encode() == encoded and b'"timeout":60' in encoded
    wire = request.to_dict()
    for variant in (
        json.dumps(wire, sort_keys=True, indent=1).encode("utf-8"),
        json.dumps(dict(reversed(list(wire.items()))), separators=(",", ":")).encode("utf-8"),
        encoded.replace(b'"timeout":60', b'"timeout":6e1'),
        encoded + b"\n",
    ):
        assert variant != encoded and json.loads(variant) == json.loads(encoded)
        with pytest.raises(ValueError, match="leaf request bytes are not canonical"):
            LeafRequest.decode(variant)

    result = _synthetic_result(request)
    encoded = result.encode()
    assert LeafResult.decode(encoded).encode() == encoded
    for variant in (
        json.dumps(result.to_dict(), sort_keys=True, indent=1).encode("utf-8"),
        json.dumps(dict(reversed(list(result.to_dict().items()))), separators=(",", ":")).encode("utf-8"),
        b" " + encoded,
    ):
        assert variant != encoded
        with pytest.raises(ValueError, match="leaf result bytes are not canonical"):
            LeafResult.decode(variant)


def _placed_environment(request: LeafRequest, **overrides: str) -> PreparedPythonEnvironment:
    """Return the request environment with a controller-style placement identity."""

    values = {
        **request.environment.env,
        "TPEN_PLACEMENT_TASK_ID": request.task_id,
        "TPEN_PLACEMENT_ATTEMPT_ID": request.attempt_id,
        "TPEN_PLACEMENT_GROUP_DIGEST": request.group_digest,
        "TPEN_PLACEMENT_ALLOCATION_ID": request.allocation_id,
        "TPEN_PLACEMENT_BACKEND_JOB_ID": request.backend_job_id,
        "TPEN_PLACEMENT_PAYLOAD_DIGEST": request.payload_digest,
        "TPEN_PLACEMENT_OUTPUT_DIR": request.output_dir,
        **overrides,
    }
    return PreparedPythonEnvironment(request.environment.python, request.environment.setup, values)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: {"payload_digest": "b" * 64}, "payload_digest does not match"),
        (lambda r: {"overrides": (*r.overrides, "payload.extra=1")}, "payload_digest does not match"),
        (lambda r: {"environment": _placed_environment(r), "attempt_id": "0" * 32}, "TPEN_PLACEMENT_ATTEMPT_ID"),
        (lambda r: {"environment": _placed_environment(r), "task_id": "other"}, "TPEN_PLACEMENT_TASK_ID"),
        (lambda r: {"environment": _placed_environment(r), "allocation_id": "other-allocation"}, "TPEN_PLACEMENT_ALLOCATION_ID"),
        (lambda r: {"environment": _placed_environment(r, TPEN_PLACEMENT_OUTPUT_DIR=r.root + "/elsewhere")}, "TPEN_PLACEMENT_OUTPUT_DIR"),
        (lambda r: {"slot_id": "slot-a"}, "slot_id does not match"),
        (lambda r: {"environment": _placed_environment(r, **{PLACEMENT_REFUSAL_ENV: "{}"})}, "placement-refused"),
        (lambda r: {"output_dir": r.root}, "escapes"),
        (lambda r: {"output_dir": os.path.dirname(r.root) + "/foreign"}, "escapes"),
        (lambda r: {"output_dir": r.output_dir + "/../x"}, "normalized"),
    ],
    ids=[
        "payload-digest", "payload-overrides", "placement-attempt", "placement-task", "placement-allocation",
        "placement-output", "slot", "placement-refused", "output-is-root", "output-outside-root", "output-unnormalized",
    ],
)
def test_request_rejects_crossed_identity(tmp_path: Path, change, message: str) -> None:
    """The payload digest and any placement identity must name this attempt."""

    request, _parts = _request(tmp_path)
    assert dataclasses.replace(request, environment=_placed_environment(request)).task_id == request.task_id
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(request, **change(request))


def test_request_decode_rejects_a_crossed_wire_identity(tmp_path: Path) -> None:
    """A tampered wire attempt is refused by decode, not trusted."""

    request, (_group, allocation, _lifecycle, configured, fence) = _request(tmp_path)
    placed = dataclasses.replace(request, environment=_placed_environment(request))
    wire = placed.to_dict()
    wire["attempt_id"] = "0" * 32
    with pytest.raises(ValueError, match="crosses the request"):
        LeafRequest.decode(json.dumps(wire).encode("utf-8"))
    other = ConfiguredAllocation(AllocationId("other-allocation"), allocation.backend_job_id, {})
    with pytest.raises(ValueError, match="fence allocation"):
        LeafRequest.for_attempt(fence, other, configured, request.environment)


def test_request_binds_task_and_attempt_without_placement(tmp_path: Path) -> None:
    """Producer task and fenced output bind without any optional placement identity."""

    request, (_group, allocation, _lifecycle, configured, fence) = _request(tmp_path)
    assert not any(key.startswith("TPEN_PLACEMENT_") for key in request.environment.env)
    assert request.output_dir == os.path.join(fence.root, "groups", fence.group_digest, fence.task_id, fence.attempt_id.value)
    # Another task's configured run never rides on this task's fence.
    foreign = ConfiguredRun("other", str(tmp_path / "other.cfg"), ("payload.task=other",))
    with pytest.raises(ValueError, match="configured run does not belong"):
        LeafRequest.for_attempt(fence, allocation, foreign, request.environment)
    renamed = ConfiguredRun("other", configured.config, configured.overrides)
    with pytest.raises(ValueError, match="configured run does not belong"):
        LeafRequest.for_attempt(fence, allocation, renamed, request.environment)
    # Each identity is bound to the fenced output on its own.
    for change in ({"attempt_id": "0" * 32}, {"task_id": "other"}, {"group_digest": "d" * 64}):
        with pytest.raises(ValueError, match="output_dir does not bind"):
            dataclasses.replace(request, **change)
    for change in ({"task_id": ".."}, {"attempt_id": "a/b"}):
        with pytest.raises(ValueError, match="single store path component"):
            dataclasses.replace(request, **change)
    # The same rule holds on the wire.
    wire = request.to_dict()
    wire["attempt_id"] = "0" * 32
    with pytest.raises(ValueError, match="output_dir does not bind"):
        LeafRequest.decode(json.dumps(wire, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def test_result_round_trips_and_rejects_non_primitive_or_inconsistent_facts(tmp_path: Path) -> None:
    """The result is plain, canonical, and internally consistent."""

    request, _parts = _request(tmp_path)
    result = _synthetic_result(request)
    encoded = result.encode()
    assert LeafResult.decode(encoded) == result and LeafResult.decode(encoded).encode() == encoded
    assert json.loads(encoded)["schema"] == LEAF_RESULT_SCHEMA
    assert result.status() == "completed" and result.core_evidence() is not None
    with pytest.raises(TypeError, match="canonical JSON bytes"):
        LeafResult.decode(result)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="science pid must be an integer"):
        dataclasses.replace(result, science={**dict(result.science), "pid": threading.Lock()})
    with pytest.raises(TypeError, match="not JSON-shaped"):
        dataclasses.replace(result, launch_error={"type": threading.Lock(), "message": "m"})
    with pytest.raises(TypeError, match="is not plain wire data"):
        dataclasses.replace(result, leaf_host=_with_handle_text("leaf-host"))
    with pytest.raises(TypeError, match="leaf_pid"):
        dataclasses.replace(result, leaf_pid=True)
    with pytest.raises(ValueError, match="evidence is present exactly"):
        dataclasses.replace(result, inspection="unavailable", inspection_error={"type": "E", "message": "m"})
    with pytest.raises(ValueError, match="inspection state disagrees"):
        dataclasses.replace(result, phases=("science",))
    with pytest.raises(ValueError, match="exactly one of science and launch_error"):
        dataclasses.replace(result, launch_error={"type": "OSError", "message": "m"})
    with pytest.raises(ValueError, match="valid public core evidence"):
        dataclasses.replace(result, evidence={**dict(result.evidence), "verdict": "accepted"})
    with pytest.raises(ValueError, match="teardown"):
        dataclasses.replace(result, teardown={**dict(result.teardown), "pid": 1})


@pytest.mark.parametrize(
    "check, change",
    [
        ("request", lambda r, q: {"request_digest": "c" * 64}),
        ("group", lambda r, q: {"group_digest": "d" * 64}),
        ("task", lambda r, q: {"task_id": "other"}),
        ("attempt", lambda r, q: {"attempt_id": "0" * 32}),
        ("allocation", lambda r, q: {"allocation_id": "other-allocation"}),
        ("backend", lambda r, q: {"backend_job_id": "other-job"}),
        ("slot", lambda r, q: {"slot_id": "slot-a"}),
        ("payload", lambda r, q: {"payload_digest": "e" * 64}),
        ("argv", lambda r, q: {"science": {**dict(r.science), "argv": [q.environment.python, "-m", "tpen.run"]}}),
        ("logs", lambda r, q: {"science": {**dict(r.science), "stdout_path": q.root + "/stdout.log"}}),
        ("teardown", lambda r, q: {"teardown": {**dict(r.teardown), "child_group_reaped": False}}),
        ("evidence", lambda r, q: {"evidence": {**dict(r.evidence), "run_dir": q.root + "/foreign/core"}}),
        ("evidence", lambda r, q: {"evidence": {
            **dict(r.evidence), "checkpoint": {**dict(r.evidence["checkpoint"]), "checkpoint_dir": q.root + "/foreign/checkpoint"},
        }}),
        ("inspection", lambda r, q: {"phases": ("science",), "inspection": "not-run", "evidence": None}),
    ],
    ids=[
        "request", "group", "task", "attempt", "allocation", "backend", "slot", "payload", "argv", "logs", "teardown",
        "evidence-run-dir", "evidence-checkpoint", "science-without-inspection",
    ],
)
def test_result_validation_rejects_each_crossed_identity(tmp_path: Path, check: str, change) -> None:
    """R2 killer: each echoed field and fact is compared, not only the request digest."""

    request, (group, _allocation, lifecycle, _configured, _fence) = _request(tmp_path)
    result = _synthetic_result(request)
    assert validate_leaf_result(request, result) is result
    crossed = dataclasses.replace(result, **change(result, request))
    # Only the named field differs; the request digest still matches.
    assert check == "request" or crossed.request_digest == request.digest()
    # The forgery is refused as decoded wire bytes, before any publication.
    for forged in (crossed, LeafResult.decode(crossed.encode())):
        with pytest.raises(LeafResultRejected, match=rf"crosses its request: {check}$"):
            validate_leaf_result(request, forged)
    assert lifecycle.store.manifest(group).attempts[0].outcome_receipt is None


def test_result_validation_accepts_a_refused_science_without_inspection(tmp_path: Path) -> None:
    """A pre-core refusal is the only science outcome published uninspected."""

    request, _parts = _request(tmp_path)
    refused = dataclasses.replace(
        _synthetic_result(request),
        phases=("science",), inspection="not-run", evidence=None,
        child_refusal={"code": "child-placement-refused", "reason": "declared CPU set is invalid"},
    )
    assert validate_leaf_result(request, LeafResult.decode(refused.encode())).status() == "launch_failed"


# ---------------------------------------------------------- real leaf


def test_real_serialized_leaf_runs_science_then_inspection_on_the_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R3/R5 killer: worker-side science then inspection, RUN_SHIM, controller-only publication."""

    monkeypatch.delenv(WORKER_MARK, raising=False)
    request, (group, _allocation, lifecycle, configured, fence) = _request(tmp_path)
    # Fresh output: the controller's begin wrote only its start receipt.
    assert sorted(os.listdir(fence.output_dir)) == ["start.json"]

    leaf_pid, encoded = _serialized_leaf(request)
    result = validate_leaf_result(request, LeafResult.decode(encoded))

    # Science then inspection ran as children of the leaf process, in the
    # worker-marked environment the controller does not have.
    assert leaf_pid != os.getpid() and result.leaf_pid == leaf_pid
    assert result.leaf_host == socket.gethostname()
    assert _events(tmp_path) == [["science", leaf_pid, "s3a-worker-0"], ["inspection", leaf_pid, "s3a-worker-0"]]
    assert result.phases == ("science", "inspection") and result.inspection == "evidence"

    # The exact RUN_SHIM argv, never python -m tpen.run.
    output_dir, core_dir, stdout_path, stderr_path, _evidence_stdout, _evidence_stderr = request.paths()
    expected_argv = (
        sys.executable, "-c", RUN_SHIM, "--config", configured.config, *configured.overrides,
        f"run.root={output_dir}", "run.layout=flat", "run.run_id=core",
    )
    assert tuple(result.science["argv"]) == expected_argv
    assert (result.science["stdout_path"], result.science["stderr_path"]) == (stdout_path, stderr_path)
    observed = json.loads((Path(core_dir) / "argv.json").read_text(encoding="utf-8"))
    assert observed["argv"] == list(expected_argv[3:])
    assert Path(observed["cwd"]).resolve() == Path(output_dir).resolve()
    # RUN_SHIM itself ran in the science child and wrote a fresh observation.
    observation = json.loads((Path(output_dir) / CHILD_OBSERVATION_FILENAME).read_text(encoding="utf-8"))
    assert observation["pid"] == result.science["pid"] and observation["hostname"] == socket.gethostname()
    assert result.evidence["run_dir"] == core_dir and result.status() == "completed"

    # The leaf published nothing: no outcome, segment or placement receipt.
    assert not {"outcome.json", "segment.json", "placement.json"} & set(os.listdir(output_dir))
    before = lifecycle.store.manifest(group)
    assert before.attempts[0].outcome_receipt is None and before.segments == ()

    # The controller publishes from the plain facts alone.
    _publish(lifecycle, fence, result)
    after = lifecycle.store.manifest(group)
    assert after.attempts[0].outcome_receipt.status == "completed"  # type: ignore[union-attr]
    assert after.attempts[0].outcome_receipt.result_digest == result.result_digest()  # type: ignore[union-attr]
    assert [segment.task_id for segment in after.segments] == [fence.task_id]


def test_leaf_refuses_replay_and_an_invisible_attempt_store(tmp_path: Path) -> None:
    """A replayed request never reruns science; a missing fence never launches."""

    request, (group, _allocation, lifecycle, _configured, fence) = _request(tmp_path)
    first = validate_leaf_result(request, LeafResult.decode(run_leaf(request.encode())))
    assert first.status() == "completed" and len(_events(tmp_path)) == 2
    replay = validate_leaf_result(request, LeafResult.decode(run_leaf(request.encode())))
    assert replay.status() == "launch_failed" and replay.phases == () and replay.inspection == "not-run"
    assert replay.launch_error["type"] == "LeafRefused" and "not fresh" in replay.launch_error["message"]
    assert len(_events(tmp_path)) == 2
    assert lifecycle.store.manifest(group).attempts[0].outcome_receipt is None

    missing = dataclasses.replace(request, attempt_id="f" * 32, output_dir=os.path.join(os.path.dirname(fence.output_dir), "f" * 32))
    invisible = validate_leaf_result(missing, LeafResult.decode(run_leaf(missing.encode())))
    assert invisible.status() == "launch_failed" and "not visible" in invisible.launch_error["message"]
    assert len(_events(tmp_path)) == 2
    with pytest.raises(TypeError, match="canonical JSON bytes"):
        run_leaf(request)  # type: ignore[arg-type]


# One physical slot of the test allocation, as the controller placement
# overlay names it; with no sidecar path RUN_SHIM enforces nothing, so the
# test is portable.
SLOT_PLACEMENT = {"TPEN_SLOT_ID": "slot-q", "TPEN_PLACEMENT_SLOT_ID": "slot-q", "TPEN_PLACEMENT_DECLARED_HOST": "s3a-host"}


def test_leaf_admits_its_slot_against_the_store_quarantine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R6 killer: a fresh leaf refuses a durably quarantined slot before any spawn."""

    request, (group, _allocation, lifecycle, _configured, fence) = _request(tmp_path)
    # An unsampled placed descriptor carries no quarantine overlay.
    placed = dataclasses.replace(request, slot_id="slot-q", environment=_placed_environment(request, **SLOT_PLACEMENT))
    quarantine_dir = slot_quarantine_dir(fence.root, placed.environment.env)
    assert quarantine_dir is not None and quarantine_dir.startswith(fence.root + os.sep)
    correct = dataclasses.replace(
        placed, environment=_placed_environment(request, **SLOT_PLACEMENT, **{SLOT_QUARANTINE_ENV: quarantine_dir}),
    )
    # An earlier sampled launch on this physical slot, under another
    # controller process, never confirmed its cleanup.
    os.makedirs(quarantine_dir)
    marker = Path(quarantine_dir) / "earlier-controller.json"
    marker.write_text(json.dumps({"schema": QUARANTINE_SCHEMA}), encoding="utf-8")
    spawned: list = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        spawned.append(args[0])
        return real_popen(*args, **kwargs)

    monkeypatch.setattr(workers.subprocess, "Popen", recording_popen)

    # Missing and correct overlays are both refused by leaf admission.
    for held in (placed, correct):
        result = validate_leaf_result(held, LeafResult.decode(run_leaf(held.encode())))
        assert result.status() == "launch_failed" and result.phases == () and result.inspection == "not-run"
        assert result.launch_error["type"] == "SlotQuarantined"
        assert "durable quarantine marker" in result.launch_error["message"]
    # No child was spawned and the fenced output is still fresh.
    assert spawned == [] and _events(tmp_path) == []
    assert sorted(os.listdir(fence.output_dir)) == ["start.json"]

    # A wrong overlay, or a sampled launch without one, is never a request.
    for overlay, message in (
        ({SLOT_QUARANTINE_ENV: os.path.join(fence.root, "slot-quarantine", "0" * 64)}, "not the store-derived"),
        ({SLOT_QUARANTINE_ENV: str(tmp_path / "elsewhere")}, "not the store-derived"),
        ({GPU_SAMPLER_OUTPUT_ENV: os.path.join(fence.output_dir, "gpu-sampler.json")}, "must carry its store-derived"),
    ):
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(placed, environment=_placed_environment(request, **SLOT_PLACEMENT, **overlay))

    # Clean reuse: once the marker is released the same slot is admitted.
    marker.unlink()
    result = validate_leaf_result(placed, LeafResult.decode(run_leaf(placed.encode())))
    assert result.status() == "completed" and result.phases == ("science", "inspection")
    assert len(spawned) == 2 and [event[0] for event in _events(tmp_path)] == ["science", "inspection"]
    assert lifecycle.store.manifest(group).attempts[0].outcome_receipt is None


def _worker_spec(environment: PreparedPythonEnvironment) -> WorkerProgram:
    raw = {
        "_target_": "tpen_executor.programs.WorkerProgram",
        "_recursive_": False,
        "environment": {
            "_target_": "tpen_executor.environments.PreparedPythonEnvironment",
            "_recursive_": False,
            "python": environment.python,
            "env": dict(environment.env),
        },
        "process": {"_target_": "tpen_executor.workers.SubprocessWorker", "_recursive_": False},
        "runner": {"_target_": "tpen_executor.runners.TPENCommand", "_recursive_": False, "config": "/s3a/config"},
        "evidence": {"_target_": "tpen_executor.runners.TPENEvidence", "_recursive_": False},
    }
    return WorkerProgram(ProgramSpecRef(raw, digest(raw)))


def test_leaf_outcomes_match_local_dispatch(tmp_path: Path) -> None:
    """Completed, failed and timed-out members publish as local dispatch does."""

    environment = _environment(tmp_path)
    group = _group("ok", "bad", "late")
    runs = _runs(tmp_path, "ok", "bad", "late", modes={"bad": "fail", "late": "sleep"})
    local = LocalDispatcher(_worker_spec(environment)).run(
        group, runs, AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "local")), timeout=1,
    )

    lifecycle = AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "leaf"))
    for member in group.members:
        fence = lifecycle.begin(group, member.task_id, "2026-11-11T00:00:00+00:00")
        request = LeafRequest.for_attempt(fence, lifecycle.allocation, runs[member.task_id], environment, 1)
        _publish(lifecycle, fence, validate_leaf_result(request, LeafResult.decode(run_leaf(request.encode()))))
    leaf = lifecycle.store.manifest(group)

    def summary(manifest):
        return {
            attempt.task_id: (attempt.outcome_receipt.status, attempt.outcome_receipt.result_digest is None)  # type: ignore[union-attr]
            for attempt in manifest.attempts
        }

    assert summary(leaf) == summary(local) == {"ok": ("completed", False), "bad": ("failed", True), "late": ("timed_out", True)}
    assert {segment.task_id for segment in leaf.segments} == {segment.task_id for segment in local.segments} == {"ok"}


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux sched affinity and /proc")
def test_placed_leaf_carries_the_controller_placement_identity(tmp_path: Path) -> None:
    """An S2b-1 placement overlay crosses the wire and binds the child sidecar."""

    cpu = min(os.sched_getaffinity(0))
    slot = AllocationSlot("slot-a", socket.gethostname(), ("0",), (cpu,), (), ())
    facts = {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot.to_dict()]}}
    group, allocation, lifecycle, configured, fence = _begin(tmp_path, facts=facts)
    member = group.members[0]
    environment = _with_child_context_keys(_environment(tmp_path), allocation)
    environment = _with_child_placement(environment, allocation, fence, group, member, configured)
    request = LeafRequest.for_attempt(fence, allocation, configured, environment, 60)
    assert request.slot_id == "slot-a"

    leaf_pid, encoded = _serialized_leaf(request)
    result = validate_leaf_result(request, LeafResult.decode(encoded))
    assert result.status() == "completed" and result.child_refusal is None and result.leaf_pid == leaf_pid
    sidecar = read_child_placement(environment.env["TPEN_PLACEMENT_OUTPUT"])
    assert (sidecar.task_id, sidecar.attempt_id, sidecar.slot_id, sidecar.payload_digest) == (
        request.task_id, request.attempt_id, "slot-a", request.payload_digest,
    )
    assert sidecar.pid == result.science["pid"] and tuple(sidecar.affinity) == (cpu,)
    assert lifecycle.store.manifest(group).attempts[0].outcome_receipt is None


# ------------------------------------------------------------ boundaries


def test_leaf_module_has_no_publication_or_scheduler_authority() -> None:
    """The transport module never imports the store/lifecycle or calls their writers."""

    path = EXECUTOR_ROOT / "src" / "tpen_executor" / "dispatch" / "transport.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_modules = {"storage", "lifecycle", "lifecycle.attached", "recovery", "allocations", "composition", "cli"}
    forbidden_calls = {
        "begin", "finish", "begin_attempt", "finish_attempt", "accept_segment", "publish_placement",
        "manifest", "observe", "submit", "cancel", "retry",
    }
    imports = [
        node.module for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 2 and node.module is not None
    ]
    assert not set(imports) & forbidden_modules, imports
    calls = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & forbidden_calls, calls & forbidden_calls
    leaf = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_leaf")
    assert [arg.arg for arg in leaf.args.args] == ["payload"]
    assert not (leaf.args.vararg or leaf.args.kwarg or leaf.args.kwonlyargs)


def test_local_and_threadpool_dispatch_do_not_route_through_the_wire() -> None:
    """Layer A adds the wire only; local and shared dispatch do not import it.

    Layer B imports the wire in ``parsl.py`` for the explicitly selected
    attached route only; ``test_threadpool_and_local_routes_never_build_a_leaf``
    in ``test_s3_transport_parsl.py`` proves behaviourally that the default
    ThreadPool route never reaches it.
    """

    dispatch = EXECUTOR_ROOT / "src" / "tpen_executor" / "dispatch"
    for name in ("__init__.py", "local.py", "contract.py"):
        assert "transport" not in (dispatch / name).read_text(encoding="utf-8"), name
