"""S3 Layer B: explicit attached HTEX/LocalProvider/MpiExec selection and submit.

The attached route lives in :mod:`tpen_executor.dispatch.parsl`.  These tests
prove, with Parsl doubles, that only the typed attached form is admitted and
everything else is refused before any attempt exists; that each run builds a
fresh HTEX config and DFK, preflights the shared store on every worker, and
submits only the Layer A leaf's canonical request bytes; that the slot binding
survives controller ambient values at the remote boundary; that Parsl never
retries a submission; that leaf facts are validated before the controller
publishes them; and that ThreadPool and local dispatch never build a leaf.

Fix round r2 adds: shared-store probes in separate processes that overlap
their lock acquisition still both reach the barrier (review F1), and a remote
slot's CPUs are checked against the worker's affinity, not the controller's
(review F2).

A real local HighThroughputExecutor with a LocalProvider and an MpiExecLauncher
(``test_real_attached_htex_runs_the_leaf_on_its_workers``) runs when the Parsl
extra is installed on Linux.  It uses a test-owned ``mpiexec`` stand-in that
starts the worker pool locally, because a test-partition job has no MPI
launcher configured and no facility value may enter the test.

``S3B_MUTANT_TABLE`` lists the contract mutants M1-M4 and M6; each names its
killer.
"""

from __future__ import annotations

import ast
from concurrent.futures import Future
import dataclasses
import importlib.util
import json
import os
from pathlib import Path
import pickle
import shlex
import stat
import sys
import textwrap
import threading
import types

import socket
import subprocess
import time

import pytest

from tpen_executor._codec import digest
from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor import cli, composition
from tpen_executor.contracts import AllocationId, BackendJobId, ProgramSpecRef, ReadyGroup, TaskSpec
from tpen_executor.dispatch import LocalDispatcher
from tpen_executor.dispatch import parsl as parsl_dispatch
from tpen_executor.dispatch.transport import LeafRequest, LeafResult, LeafTeardownReceipt, run_leaf
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.lifecycle import AttachedAllocation
from tpen_executor.placement import AllocationSlot, ChildPlacementObservation
from tpen_executor.programs import WorkerProgram
from tpen_executor.storage import DirectoryStore
from tpen_executor.tasks import ConfiguredRun, ConfiguredRunTable


EXECUTOR_ROOT = Path(__file__).parents[1]
PARSL_MODULE = "executor/src/tpen_executor/dispatch/parsl.py"
WORKERS_MODULE = "executor/src/tpen_executor/workers.py"
HAS_PARSL = importlib.util.find_spec("parsl") is not None
WORKER_MARK = "S3B_WORKER_MARK"
LABEL = "s3b-htex"


S3B_MUTANT_TABLE = (
    {
        "id": "M1", "target": PARSL_MODULE,
        "old": "                    config = self._attached_config(self.transport, run_dir)\n                    labels = self._validate_attached_config(config, self.transport)\n",
        "new": "                    config = self._config_type(executors=[self._thread_pool_type(label=self.transport.label, max_threads=len(bindings))], run_dir=run_dir, usage_tracking=False)\n                    labels = (self.transport.label,)\n",
        "killer": "test_attached_run_submits_the_leaf_through_a_fresh_htex_dfk",
        "suite": "executor", "assertion": "a selected attached transport is never silently routed through an in-process ThreadPool",
    },
    {
        "id": "M2", "target": PARSL_MODULE,
        "old": "        if form != _ATTACHED_FORM:\n            raise ValueError(",
        "new": "        if False:\n            raise ValueError(",
        "killer": "test_spec_admits_only_the_attached_form",
        "suite": "executor", "assertion": "any other Parsl executor, provider or launcher is refused, never mapped onto the attached form",
    },
    {
        "id": "M3", "target": WORKERS_MODULE,
        "old": "        child_env = os.environ.copy()\n        child_env.update(environment.env)\n",
        "new": "        child_env = dict(environment.env)\n        child_env.update(os.environ)\n",
        "killer": "test_slot_binding_survives_controller_ambient_at_the_remote_boundary",
        "suite": "executor", "assertion": "controller/worker ambient CUDA and CPU values never override the slot binding in the leaf child",
    },
    {
        "id": "M4", "target": PARSL_MODULE,
        "old": "            dfk = self._dfk_type(config=config)\n",
        "new": "            dfk = ParslDispatcher.__dict__.get(\"_s3b_shared_dfk\") or self._dfk_type(config=config)\n            setattr(ParslDispatcher, \"_s3b_shared_dfk\", dfk)\n",
        "killer": "test_each_attached_run_uses_a_fresh_dfk",
        "suite": "executor", "assertion": "no DFK is reused across runs or allocations",
    },
    {
        "id": "M6", "target": PARSL_MODULE,
        "old": "    _lockf_probe(private_lock)\n",
        "new": "    _lockf_probe(token)\n",
        "killer": "test_overlapping_store_probes_in_separate_processes_both_reach_the_barrier",
        "suite": "executor", "assertion": "probes never contend for one shared lock, so ordinary overlap never refuses a healthy store",
    },
)
# M5 (retries=2 in the attached config) is retired in fix round r2 to keep the
# table at five rows.  It was the weakest control: its killer failed only
# through the duplicate retries check in _validate_attached_config, not by
# observing a retry.  test_a_failed_remote_submission_is_not_retried still
# asserts one leaf call and one durable remote-quiescence-unconfirmed
# attempt (S3 Layer C review P1-2: worker loss is never launch_failed).


def test_s3b_mutant_table_targets_unique_executable_anchors() -> None:
    """Every mutant names a unique executable anchor, parses, and has a killer."""

    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    assert [row["id"] for row in S3B_MUTANT_TABLE] == ["M1", "M2", "M3", "M4", "M6"]
    for row in S3B_MUTANT_TABLE:
        source = (root / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, row
        assert row["old"] != row["new"] and row["suite"] == "executor" and row["assertion"]
        # Anchors are code, not docstrings or comments, and the mutant parses.
        assert not row["old"].strip().startswith(("#", '"""', "'"))
        ast.parse(source.replace(row["old"], row["new"]))
        assert f"def {row['killer'].split('[')[0]}(" in module, row


# ----------------------------------------------------------- parsl doubles


# Same leading script name and placeholder style as Parsl's DEFAULT_LAUNCH_CMD.
_DEFAULT_LAUNCH_CMD = (
    "process_worker_pool.py {debug} {max_workers_per_node} -a {addresses} -p {prefetch_capacity} "
    "--block_id={{block_id}} --cpu-affinity {cpu_affinity} --available-accelerators {accelerators}"
)


class _Config:
    def __init__(self, executors=None, run_dir="runinfo", retries=0, retry_handler=None, strategy="simple", usage_tracking=0, **options):
        self.executors = list(executors or [])
        self.run_dir, self.retries, self.retry_handler = run_dir, retries, retry_handler
        self.strategy, self.usage_tracking, self.options = strategy, usage_tracking, options


class _ThreadPool:
    def __init__(self, label="threads", max_threads=2, **options):
        self.label, self.max_threads, self.options = label, max_threads, options


class _HTEX:
    def __init__(self, label="HighThroughputExecutor", provider=None, **options):
        self.label, self.provider, self.options = label, provider, options


class _LocalProvider:
    def __init__(self, launcher=None, **options):
        self.launcher, self.options = launcher, options


class _MpiExec:
    def __init__(self, debug=True, bind_cmd="--bind-to", overrides=""):
        self.bind_cmd, self.overrides = bind_cmd, overrides


class _Parsl:
    """Parsl doubles that record DFKs and submissions.

    Every app call runs in its own thread, as a worker would.  An HTEX app is
    pickled and unpickled first, as Parsl serializes it for a remote worker;
    a ThreadPool app is not, because Parsl runs it in-process.  The DFK honors
    ``config.retries`` by re-running a failed app, as Parsl would.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []
        self.dfks: list = []
        self.submissions: list[dict] = []
        self.hooks: dict[str, object] = {}
        self.calls: dict[str, int] = {}
        self.routed_probe_hosts: dict[str, str] | None = None
        self.routed_probe_labels: list[str] = []
        recorder = self

        class DataFlowKernel:
            def __init__(self, config):
                self.config, self.cleaned = config, 0
                recorder.dfks.append(self)
                recorder.events.append(("dfk", id(self)))

            def cleanup(self):
                self.cleaned += 1
                recorder.events.append(("cleanup", id(self)))

        def python_app(data_flow_kernel=None, cache=False, executors="all"):
            def decorate(function):
                def app(*args, **kwargs):
                    dfk = data_flow_kernel
                    assert dfk.cleaned == 0, "app submitted to a cleaned DFK"
                    remote = isinstance(dfk.config.executors[0], _HTEX)
                    if remote:
                        # The serialization boundary of a remote worker.
                        function_, args, kwargs = pickle.loads(pickle.dumps((function, args, kwargs)))
                    else:
                        function_ = function
                    name = function_.__name__
                    recorder.submissions.append({"function": name, "executors": list(executors), "args": args, "dfk": dfk, "remote": remote})
                    future: Future = Future()

                    def work() -> None:
                        error: BaseException | None = None
                        for _attempt in range(1 + dfk.config.retries):
                            index = recorder.calls[name] = recorder.calls.get(name, 0) + 1
                            hook = recorder.hooks.get(name)
                            try:
                                if name == "_shared_store_probe" and recorder.routed_probe_hosts is not None:
                                    label = tuple(executors)[0]
                                    recorder.routed_probe_labels.append(label)
                                    result = {
                                        "host": recorder.routed_probe_hosts[label],
                                        "pid": 1,
                                        "root": os.path.realpath(args[0]),
                                    }
                                else:
                                    result = hook(index, function_, args, kwargs) if hook else function_(*args, **kwargs)
                            except Exception as caught:
                                error = caught
                                continue
                            future.set_result(result)
                            return
                        future.set_exception(error)

                    threading.Thread(target=work, daemon=True).start()
                    return future

                return app

            return decorate

        self.modules = {
            "parsl": types.ModuleType("parsl"),
            "parsl.config": types.SimpleNamespace(Config=_Config),
            "parsl.dataflow": types.ModuleType("parsl.dataflow"),
            "parsl.dataflow.dflow": types.SimpleNamespace(DataFlowKernel=DataFlowKernel),
            "parsl.app": types.ModuleType("parsl.app"),
            "parsl.app.app": types.SimpleNamespace(python_app=python_app),
            "parsl.executors": types.SimpleNamespace(HighThroughputExecutor=_HTEX),
            "parsl.executors.high_throughput": types.ModuleType("parsl.executors.high_throughput"),
            "parsl.executors.high_throughput.executor": types.SimpleNamespace(DEFAULT_LAUNCH_CMD=_DEFAULT_LAUNCH_CMD),
            "parsl.executors.threads": types.SimpleNamespace(ThreadPoolExecutor=_ThreadPool),
            "parsl.launchers": types.SimpleNamespace(MpiExecLauncher=_MpiExec),
            "parsl.providers": types.SimpleNamespace(LocalProvider=_LocalProvider),
        }

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "_Parsl":
        for name, module in self.modules.items():
            monkeypatch.setitem(sys.modules, name, module)
        return self

    def leaf_submissions(self) -> list[dict]:
        return [record for record in self.submissions if record["function"] == "run_leaf"]


@pytest.fixture
def fake_parsl(monkeypatch: pytest.MonkeyPatch) -> _Parsl:
    return _Parsl().install(monkeypatch)


# --------------------------------------------------------------- fixtures


def _group(name: str, *task_ids: str) -> ReadyGroup:
    program = ProgramSpecRef({}, digest({}))
    members = tuple(TaskSpec(task_id, task_id, program, digest(task_id), 1, "steps") for task_id in task_ids)
    return ReadyGroup(name, members, digest({"plan": name}))


def _allocation() -> ConfiguredAllocation:
    return ConfiguredAllocation(AllocationId("s3b-allocation"), BackendJobId("job-s3b"), {})


def _runs(tmp_path: Path, *task_ids: str, modes: dict[str, str] | None = None) -> dict[str, ConfiguredRun]:
    modes = {} if modes is None else modes
    result = {}
    for task_id in task_ids:
        config = tmp_path / f"{task_id}.cfg"
        config.write_text(modes.get(task_id, "ok"), encoding="utf-8")
        result[task_id] = ConfiguredRun(task_id, str(config), (f"payload.task={task_id}",))
    return result


def _make_fake_core(tmp_path: Path) -> Path:
    """Write a fake ``tpen`` whose science records its process and binding.

    ``run.main`` writes ``observed.json`` (parent pid, worker mark and the
    bound CUDA/CPU values) and public evidence into the flat core directory.
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

            def main(argv):
                config = Path(argv[argv.index("--config") + 1])
                mode = config.read_text(encoding="utf-8")
                overrides = dict(item.split("=", 1) for item in argv if "=" in item)
                run_dir = Path(overrides["run.root"]) / overrides["run.run_id"]
                if mode == "sleep":
                    time.sleep(4)
                if mode == "fail":
                    return 7
                run_dir.mkdir(parents=True)
                (run_dir / "observed.json").write_text(json.dumps({{
                    "ppid": os.getppid(),
                    "pid": os.getpid(),
                    "mark": os.environ.get({WORKER_MARK!r}),
                    "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "omp": os.environ.get("OMP_NUM_THREADS"),
                }}), encoding="utf-8")
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
            """
            import json
            import sys
            from pathlib import Path

            print(json.dumps(json.loads(Path(sys.argv[1], "evidence.json").read_text(encoding="utf-8"))))
            """
        ),
        encoding="utf-8",
    )
    return package.parent


def _worker(tmp_path: Path, *, process: str = "tpen_executor.workers.SubprocessWorker") -> WorkerProgram:
    raw = {
        "_target_": "tpen_executor.programs.WorkerProgram",
        "_recursive_": False,
        "environment": {
            "_target_": "tpen_executor.environments.PreparedPythonEnvironment",
            "_recursive_": False,
            "python": sys.executable,
            "env": {"PYTHONPATH": str(_make_fake_core(tmp_path))},
        },
        "process": {"_target_": process, "_recursive_": False},
        "runner": {"_target_": "tpen_executor.runners.TPENCommand", "_recursive_": False, "config": "/s3b/config"},
        "evidence": {"_target_": "tpen_executor.runners.TPENEvidence", "_recursive_": False},
    }
    return WorkerProgram(ProgramSpecRef(raw, digest(raw)))


BINDING = {"environment": {"CUDA_VISIBLE_DEVICES": "slot-gpu-3", "OMP_NUM_THREADS": "3"}}


def _transport(**changes) -> dict:
    spec = {"schema": parsl_dispatch.ATTACHED_TRANSPORT_SCHEMA, "label": LABEL, "max_workers_per_node": 2, "preflight_timeout": 30}
    spec.update(changes)
    return spec


def _routed_transport_payload() -> dict:
    return {
        "schema": parsl_dispatch.ROUTED_ATTACHED_TRANSPORT_SCHEMA,
        "executor": "htex",
        "provider": "local",
        "launcher": "mpiexec",
        "pools": [
            {
                "node": "node-a",
                "label": "pool-a",
                "workers": 1,
                "worker_init": "export D2A_POOL=A",
                "address": None,
                "mpiexec_bind_cmd": "--bind-to",
                "mpiexec_overrides": "--a",
            },
            {
                "node": "node-b",
                "label": "pool-b",
                "workers": 1,
                "worker_init": "export D2A_POOL=B",
                "address": "127.0.0.1",
                "mpiexec_bind_cmd": "--bind-to",
                "mpiexec_overrides": "--b",
            },
        ],
        "preflight_timeout": 5,
        "teardown_timeout": 2,
    }


def _routed_allocation(nodes: tuple[str, ...] = ("node-a", "node-b")) -> ConfiguredAllocation:
    available_cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(len(nodes)))
    if len(available_cpus) < len(nodes):
        available_cpus = list(range(len(nodes)))
    slots = [
        AllocationSlot(f"routed-slot-{index}", node, (), (available_cpus[index],), (), ()).to_dict()
        for index, node in enumerate(nodes)
    ]
    return ConfiguredAllocation(
        AllocationId("s3b-routed-allocation"),
        BackendJobId("s3b-routed-job"),
        {
            "scheduler": {"pbs_nodefile_text": "\n".join(dict.fromkeys(nodes)) + "\n"},
            "slots": {"schema": "tpen-executor/allocation-slots-v2", "slots": slots},
        },
    )


def _attach_routed_probe_answers(dispatcher: object, fake_parsl: _Parsl, hosts: dict[str, str]) -> list[str]:
    """Make only the production probe app synthetic; leaves stay fake-Parsl remote apps."""

    del dispatcher
    fake_parsl.routed_probe_hosts = hosts
    fake_parsl.routed_probe_labels.clear()
    return fake_parsl.routed_probe_labels


def _attach_routed_leaf_answers(fake_parsl: _Parsl) -> None:
    """Return valid leaf facts without pretending synthetic nodes are local hosts."""

    def completed(_index, _function, args, _kwargs):
        request = LeafRequest.decode(args[0])
        output_dir, core_dir, stdout_path, stderr_path, _evidence_stdout, _evidence_stderr = request.paths()
        values = request.environment.env
        declared_cpus = tuple(int(item) for item in values["TPEN_SLOT_CPUS"].split(","))
        sidecar = ChildPlacementObservation(
            request.task_id,
            request.attempt_id,
            request.group_digest,
            request.allocation_id,
            request.backend_job_id,
            values["TPEN_SLOT_ID"],
            request.payload_digest,
            output_dir,
            values["TPEN_SLOT_NODE"],
            4243,
            55,
            declared_cpus,
            {"OMP_NUM_THREADS": None},
            values.get("CUDA_VISIBLE_DEVICES"),
            "2026-11-18T00:00:00Z",
            None,
        )
        Path(values["TPEN_PLACEMENT_OUTPUT"]).write_text(json.dumps(sidecar.to_dict()) + "\n", encoding="utf-8")
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
            ("science", "inspection"),
            {
                "argv": [request.environment.python, *request.science_args()],
                "returncode": 0,
                "timed_out": False,
                "pid": 4243,
                "proc_start_ticks": 55,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
            },
            None,
            None,
            "evidence",
            {
                "schema": "tpen.run-evidence/v2",
                "run_dir": core_dir,
                "verdict": "completed",
                "run_status": "completed",
                "checkpoint": {
                    "checkpoint_dir": core_dir + "/checkpoint",
                    "schema_version": 1,
                    "next_iteration": 1,
                    "completed_updates": 1,
                },
                "refusal": None,
                "resolved_config_identity": {
                    "schema": "tpen-config-identity-v1",
                    "sha256": "a" * 64,
                },
            },
            None,
            LeafTeardownReceipt.for_request(request, inspection="complete", child_group_reaped=True).to_dict(),
        ).encode()

    fake_parsl.hooks["run_leaf"] = completed


def _manifest_attempts(manifest) -> dict[str, dict]:
    """Expose each member's durable outcome and placement receipt in failures."""

    return {attempt.task_id: attempt.to_dict() for attempt in manifest.attempts}


def _cli_inputs(tmp_path: Path, allocation: ConfiguredAllocation, *, runner: str = "tpen_executor.runners.TPENCommand") -> dict[str, Path | ReadyGroup]:
    """Write one CLI-shaped group, allocation, worker, and run table."""

    group = _group("s3b-cli", "member")
    core = _make_fake_core(tmp_path)
    config = tmp_path / "member.cfg"
    config.write_text("ok", encoding="utf-8")
    worker = tmp_path / "worker.yaml"
    runner_lines = [
        "  runner:",
        f"    _target_: {runner}",
        "    _recursive_: false",
    ]
    if runner.endswith("TPENCommand"):
        runner_lines.append(f"    config: {json.dumps(str(config))}")
    worker.write_text(
        "\n".join(
            [
                "worker:",
                "  _target_: tpen_executor.programs.WorkerProgram",
                "  _recursive_: false",
                "  environment:",
                "    _target_: tpen_executor.environments.PreparedPythonEnvironment",
                "    _recursive_: false",
                f"    python: {json.dumps(sys.executable)}",
                "    env:",
                f"      PYTHONPATH: {json.dumps(str(core))}",
                "  process:",
                "    _target_: tpen_executor.workers.SubprocessWorker",
                "    _recursive_: false",
                *runner_lines,
                "  evidence:",
                "    _target_: tpen_executor.runners.TPENEvidence",
                "    _recursive_: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths: dict[str, Path | ReadyGroup] = {
        "input": tmp_path / "executor_input.json",
        "allocation": tmp_path / "allocation.json",
        "runs": tmp_path / "runs.json",
        "worker": worker,
        "store": tmp_path / "store",
        "group": group,
    }
    paths["input"].write_bytes(group.encode())  # type: ignore[union-attr]
    paths["allocation"].write_bytes(allocation.encode())  # type: ignore[union-attr]
    paths["runs"].write_bytes(ConfiguredRunTable(group.identity, (ConfiguredRun("member", str(config)),)).encode())  # type: ignore[union-attr]
    return paths


def _lifecycle(tmp_path: Path, name: str = "store") -> AttachedAllocation:
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    return AttachedAllocation(_allocation(), DirectoryStore(root))


def _placed_lifecycle(tmp_path: Path, cpu: int) -> AttachedAllocation:
    """Return an attached lifecycle whose one slot declares ``cpu`` on this host."""

    slot = AllocationSlot("slot-a", socket.gethostname(), (), (cpu,), (), ())
    facts = {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot.to_dict()]}}
    root = tmp_path / "placed-store"
    root.mkdir()
    return AttachedAllocation(
        ConfiguredAllocation(AllocationId("s3b-allocation"), BackendJobId("job-s3b"), facts), DirectoryStore(root),
    )


def _summary(manifest) -> dict:
    return {
        attempt.task_id: (attempt.outcome_receipt.status, attempt.outcome_receipt.result_digest is None)  # type: ignore[union-attr]
        for attempt in manifest.attempts
    }


def _observed(lifecycle: AttachedAllocation, group: ReadyGroup, task_id: str) -> dict:
    attempt = next(attempt for attempt in lifecycle.store.manifest(group).attempts if attempt.task_id == task_id)
    output_dir = Path(lifecycle.store.root) / "groups" / group.identity / task_id / attempt.attempt_id.value
    return json.loads((output_dir / "core" / "observed.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------- admission


@pytest.mark.parametrize(
    "change",
    [
        {"executor": "threads"},
        {"executor": "workqueue"},
        {"provider": "slurm"},
        {"provider": "pbspro"},
        {"launcher": "srun"},
        {"launcher": "simple"},
    ],
    ids=["executor-threads", "executor-workqueue", "provider-slurm", "provider-pbspro", "launcher-srun", "launcher-simple"],
)
def test_spec_admits_only_the_attached_form(change: dict) -> None:
    """M2 killer: any other executor, provider or launcher is refused."""

    admitted = parsl_dispatch.AttachedTransportSpec.from_mapping(_transport())
    assert (admitted.executor, admitted.provider, admitted.launcher) == ("htex", "local", "mpiexec")
    assert parsl_dispatch.AttachedTransportSpec.from_mapping(admitted.to_dict()) == admitted
    assert admitted.workers == 2
    with pytest.raises(ValueError, match="attached htex/local/mpiexec form"):
        parsl_dispatch.AttachedTransportSpec.from_mapping(_transport(**change))
    with pytest.raises(ValueError, match="attached htex/local/mpiexec form"):
        parsl_dispatch.AttachedTransportSpec(**change)


@pytest.mark.parametrize(
    "change, error",
    [
        ({"label": ""}, ValueError),
        ({"label": "_parsl_internal"}, ValueError),
        ({"label": "a b"}, ValueError),
        ({"nodes_per_block": 0}, ValueError),
        ({"max_workers_per_node": True}, ValueError),
        ({"preflight_timeout": float("inf")}, ValueError),
        ({"address": ""}, TypeError),
        ({"worker_init": None}, TypeError),
        ({"scheduler_account": "x"}, ValueError),
        ({"schema": "tpen-executor/attached-transport-v0"}, ValueError),
    ],
    ids=["empty-label", "internal-label", "space-label", "zero-nodes", "bool-workers", "inf-timeout", "empty-address", "none-init", "unknown-key", "schema"],
)
def test_spec_rejects_invalid_values(change: dict, error: type) -> None:
    with pytest.raises(error):
        parsl_dispatch.AttachedTransportSpec.from_mapping(_transport(**change))


def test_spec_has_no_facility_defaults() -> None:
    """Executor defaults carry no facility value: profiles inject them."""

    spec = parsl_dispatch.AttachedTransportSpec()
    assert (spec.worker_init, spec.address, spec.mpiexec_overrides) == ("", None, "")
    assert (spec.nodes_per_block, spec.max_workers_per_node) == (1, 1)


@pytest.mark.parametrize(
    "case",
    ["executor-config", "unknown-provider", "not-attached", "non-standard-worker", "more-slots-than-workers", "missing-store-root"],
)
def test_transport_refusal_leaves_no_attempt(tmp_path: Path, fake_parsl: _Parsl, case: str) -> None:
    """Every admission refusal happens before a DFK or lifecycle write exists."""

    group = _group("s3b-refuse", "member")
    runs = _runs(tmp_path, "member")
    lifecycle = _lifecycle(tmp_path)
    worker = _worker(tmp_path)
    kwargs: dict = {"binding": BINDING, "transport": _transport()}
    if case == "executor-config":
        kwargs["executor_config"] = _Config(executors=[_HTEX(label=LABEL)])
        with pytest.raises(parsl_dispatch.UnsupportedTransportError, match="mutually exclusive"):
            parsl_dispatch.ParslDispatcher(worker, **kwargs)
    elif case == "unknown-provider":
        kwargs["transport"] = _transport(provider="slurm")
        with pytest.raises(ValueError, match="attached htex/local/mpiexec form"):
            parsl_dispatch.ParslDispatcher(worker, **kwargs)
    else:
        if case == "non-standard-worker":
            worker = _worker(tmp_path / "custom", process="tpen_executor.programs.DummyProcess")
        if case == "more-slots-than-workers":
            kwargs["transport"] = _transport(max_workers_per_node=1)
            slots = [AllocationSlot(f"slot-{index}", "node-a", (str(index),), (index,), (), ()).to_dict() for index in range(2)]
            facts = {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": slots}}
            lifecycle = AttachedAllocation(
                ConfiguredAllocation(AllocationId("s3b-allocation"), BackendJobId("job-s3b"), facts),
                lifecycle.store,
            )
            kwargs.pop("binding")
        if case == "missing-store-root":
            lifecycle = AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "absent-store"))
        dispatcher = parsl_dispatch.ParslDispatcher(worker, **kwargs)
        target = lifecycle
        if case == "not-attached":
            target = types.SimpleNamespace(begin=lifecycle.begin, finish=lifecycle.finish, store=lifecycle.store, allocation=lifecycle.allocation)
        reasons = {
            "not-attached": "requires an AttachedAllocation",
            "non-standard-worker": "process must be exactly SubprocessWorker",
            "more-slots-than-workers": "fewer workers than admitted slots",
            "missing-store-root": "must already exist",
        }
        with pytest.raises(parsl_dispatch.UnsupportedTransportError, match=reasons[case]):
            dispatcher.run(group, runs, target, timeout=60)
    assert fake_parsl.dfks == [] and fake_parsl.submissions == []
    assert not (Path(lifecycle.store.root) / "groups").exists()


@pytest.mark.parametrize("case", ["capacity", "non-standard-runner"])
def test_cli_transport_shape_refusal_precedes_store_reservation(
    tmp_path: Path, fake_parsl: _Parsl, capsys: pytest.CaptureFixture[str], case: str,
) -> None:
    """F1: capacity and exact worker-shape refusals are public admission errors."""

    if case == "capacity":
        slots = [
            AllocationSlot("slot-a", "node-a", ("device-a",), (0,), (), ()).to_dict(),
            AllocationSlot("slot-b", "node-a", ("device-b",), (1,), (), ()).to_dict(),
        ]
        allocation = ConfiguredAllocation(
            AllocationId("s3b-cli-allocation"), BackendJobId("s3b-cli-job"),
            {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": slots}},
        )
        runner = "tpen_executor.runners.TPENCommand"
    else:
        allocation = ConfiguredAllocation(
            AllocationId("s3b-cli-allocation"), BackendJobId("s3b-cli-job"), {"binding": BINDING},
        )
        runner = "tpen_executor.programs.DummyRunner"
    paths = _cli_inputs(tmp_path, allocation, runner=runner)
    transport = tmp_path / "transport.json"
    payload = _transport(max_workers_per_node=1)
    transport.write_text(json.dumps(payload), encoding="utf-8")

    code = cli.main(
        [
            "dispatch", "--input", str(paths["input"]), "--store", str(paths["store"]),
            "--allocation", str(paths["allocation"]), "--worker-config", str(paths["worker"]),
            "--runs", str(paths["runs"]), "--dispatcher", "parsl", "--transport", str(transport),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ""
    error = json.loads(captured.err)
    assert error["schema"] == "tpen-executor/cli-error-v1"
    assert error["code"] == "dispatcher-admission"
    assert not Path(paths["store"]).exists()
    assert fake_parsl.dfks == [] and fake_parsl.submissions == []


@pytest.mark.parametrize(
    "allocation_kind",
    [
        "legacy-binding",
        pytest.param("cpu-slot", marks=pytest.mark.skipif(not sys.platform.startswith("linux"), reason="CPU slot affinity is Linux-only")),
    ],
)
def test_cli_admitted_plan_dispatches_allocation_binding_with_real_dispatcher(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch, allocation_kind: str,
) -> None:
    """F1 test 3: composition and dispatch share one real admitted plan."""

    if allocation_kind == "legacy-binding":
        allocation = ConfiguredAllocation(
            AllocationId("s3b-cli-allocation"), BackendJobId("s3b-cli-job"), {"binding": BINDING},
        )
    else:
        cpu = min(os.sched_getaffinity(0))
        slot = AllocationSlot("slot-cpu", socket.gethostname(), (), (cpu,), (), ())
        allocation = ConfiguredAllocation(
            AllocationId("s3b-cli-allocation"), BackendJobId("s3b-cli-job"),
            {"slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot.to_dict()]}},
        )
    paths = _cli_inputs(tmp_path, allocation)
    transport = tmp_path / "transport.json"
    transport.write_text(json.dumps(_transport(max_workers_per_node=1)), encoding="utf-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "controller-ambient-0")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    plan = composition.admit(
        paths["input"], paths["store"], paths["allocation"], paths["worker"], paths["runs"],
        dispatcher_name="parsl", transport_path=transport,
    )
    manifest = composition.dispatch(plan)

    assert _summary(manifest) == {"member": ("completed", False)}
    assert plan.dispatcher.transport is plan.transport
    [record] = fake_parsl.leaf_submissions()
    request = LeafRequest.decode(record["args"][0])
    if allocation_kind == "legacy-binding":
        assert request.environment.env["CUDA_VISIBLE_DEVICES"] == BINDING["environment"]["CUDA_VISIBLE_DEVICES"]
        assert request.environment.env["OMP_NUM_THREADS"] == BINDING["environment"]["OMP_NUM_THREADS"]
    else:
        assert request.environment.env["TPEN_SLOT_ID"] == "slot-cpu"
        assert request.environment.env["TPEN_SLOT_NODE"] == socket.gethostname()
    observed = _observed(plan.lifecycle, paths["group"], "member")  # type: ignore[arg-type]
    if allocation_kind == "legacy-binding":
        assert (observed["cuda"], observed["omp"]) == ("slot-gpu-3", "3")
    [dfk] = fake_parsl.dfks
    assert [type(executor) for executor in dfk.config.executors] == [_HTEX]
    assert dfk.config.executors[0].label == LABEL
    assert [item["function"] for item in fake_parsl.submissions] == ["_shared_store_probe", "run_leaf"]


def test_routed_composition_dispatches_v2_transport_file(tmp_path: Path, fake_parsl: _Parsl) -> None:
    """D2a public path: decode, admit, run and publish a two-pool v2 manifest."""

    allocation = _routed_allocation()
    paths = _cli_inputs(tmp_path, allocation)
    transport = tmp_path / "routed-transport.json"
    transport.write_text(json.dumps(_routed_transport_payload()), encoding="utf-8")

    plan = composition.admit(
        paths["input"], paths["store"], paths["allocation"], paths["worker"], paths["runs"],
        dispatcher_name="parsl", transport_path=transport,
    )
    probe_labels = _attach_routed_probe_answers(plan.dispatcher, fake_parsl, {"pool-a": "node-a", "pool-b": "node-b"})
    _attach_routed_leaf_answers(fake_parsl)
    manifest = composition.dispatch(plan)

    assert _summary(manifest) == {"member": ("completed", False)}, f"manifest attempts={_manifest_attempts(manifest)!r}"
    assert json.loads((Path(paths["store"]) / "run-manifest.json").read_text(encoding="utf-8"))["schema"] == "tpen-executor/run-manifest-v2"
    assert probe_labels == ["pool-a", "pool-b"]
    assert [record["executors"] for record in fake_parsl.submissions[:2]] == [["pool-a"], ["pool-b"]]
    [dfk] = fake_parsl.dfks
    assert [executor.label for executor in dfk.config.executors] == ["pool-a", "pool-b"]
    assert [executor.provider.options["nodes_per_block"] for executor in dfk.config.executors] == [1, 1]
    assert [executor.provider.options["worker_init"] for executor in dfk.config.executors] == [
        "export D2A_POOL=A", "export D2A_POOL=B",
    ]


def test_routed_run_rejects_wrong_probe_hostname_before_begin(tmp_path: Path, fake_parsl: _Parsl) -> None:
    """D2a public run path: a wrong routed probe has no lifecycle or science side effect."""

    group = _group("s3b-routed-wrong-host", "member")
    store_root = tmp_path / "store"
    store_root.mkdir()
    lifecycle = AttachedAllocation(_routed_allocation(), DirectoryStore(store_root))
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path), transport=parsl_dispatch.RoutedAttachedTransportSpec.from_mapping(_routed_transport_payload()),
    )
    _attach_routed_probe_answers(dispatcher, fake_parsl, {"pool-a": "node-a", "pool-b": "wrong-node"})

    with pytest.raises(parsl_dispatch.RoutedTransportPreflightError):
        dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)

    manifest = lifecycle.store.manifest(group)
    assert not (Path(lifecycle.store.root) / "groups").exists(), f"manifest attempts={_manifest_attempts(manifest)!r}"
    assert manifest.attempts == (), f"manifest attempts={_manifest_attempts(manifest)!r}"
    assert fake_parsl.leaf_submissions() == [], f"manifest attempts={_manifest_attempts(manifest)!r}"
    assert not list(Path(tmp_path).rglob("observed.json")), f"manifest attempts={_manifest_attempts(manifest)!r}"


def test_routed_run_uses_routed_validator(tmp_path: Path, fake_parsl: _Parsl) -> None:
    """The public run path must accept its two-executor v2 config with the v2 validator."""

    group = _group("s3b-routed-validator", "member")
    store_root = tmp_path / "store"
    store_root.mkdir()
    lifecycle = AttachedAllocation(_routed_allocation(), DirectoryStore(store_root))
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path), transport=parsl_dispatch.RoutedAttachedTransportSpec.from_mapping(_routed_transport_payload()),
    )
    _attach_routed_probe_answers(dispatcher, fake_parsl, {"pool-a": "node-a", "pool-b": "node-b"})
    _attach_routed_leaf_answers(fake_parsl)

    manifest = dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)

    assert _summary(manifest) == {"member": ("completed", False)}, f"manifest attempts={_manifest_attempts(manifest)!r}"
    assert len(fake_parsl.dfks[0].config.executors) == 2


@pytest.mark.parametrize("case", ["node-set", "capacity"], ids=["node-set-mismatch", "capacity-mismatch"])
def test_routed_cli_rejects_v2_node_set_or_capacity_mismatch(
    tmp_path: Path, fake_parsl: _Parsl, capsys: pytest.CaptureFixture[str], case: str,
) -> None:
    """D2a public CLI admission refuses v2 topology before reserving the store."""

    nodes = ("node-a", "node-c") if case == "node-set" else ("node-a", "node-a", "node-b")
    paths = _cli_inputs(tmp_path, _routed_allocation(nodes))
    transport = tmp_path / "routed-transport.json"
    transport.write_text(json.dumps(_routed_transport_payload()), encoding="utf-8")

    code = cli.main(
        [
            "dispatch", "--input", str(paths["input"]), "--store", str(paths["store"]),
            "--allocation", str(paths["allocation"]), "--worker-config", str(paths["worker"]),
            "--runs", str(paths["runs"]), "--dispatcher", "parsl", "--transport", str(transport),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ""
    error = json.loads(captured.err)
    assert error["schema"] == "tpen-executor/cli-error-v1"
    assert error["code"] == "dispatcher-admission"
    assert not Path(paths["store"]).exists()
    assert fake_parsl.dfks == [] and fake_parsl.submissions == []


# ------------------------------------------------------------- submission


def test_attached_run_submits_the_leaf_through_a_fresh_htex_dfk(tmp_path: Path, fake_parsl: _Parsl) -> None:
    """M1 killer: one HTEX/LocalProvider/MpiExec DFK, preflight, then leaf bytes only."""

    group = _group("s3b-submit", "first", "second")
    lifecycle = _lifecycle(tmp_path)
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path),
        binding=BINDING,
        transport=_transport(worker_init="export S3B_PROFILE=operator", address="10.0.0.1", mpiexec_overrides="--ppn 1"),
    )
    manifest = dispatcher.run(group, _runs(tmp_path, "first", "second"), lifecycle, timeout=60)

    assert _summary(manifest) == {"first": ("completed", False), "second": ("completed", False)}
    assert {segment.task_id for segment in manifest.segments} == {"first", "second"}

    # Exactly one DFK, built from exactly the attached form, cleaned once.
    [dfk] = fake_parsl.dfks
    assert dfk.cleaned == 1 and fake_parsl.events[-1] == ("cleanup", id(dfk))
    config = dfk.config
    [executor] = config.executors
    assert type(executor) is _HTEX and executor.label == LABEL
    assert type(executor.provider) is _LocalProvider and type(executor.provider.launcher) is _MpiExec
    assert executor.provider.options == {
        "nodes_per_block": 1, "init_blocks": 1, "min_blocks": 1, "max_blocks": 1, "worker_init": "export S3B_PROFILE=operator",
    }
    assert executor.provider.launcher.overrides == "--ppn 1"
    assert executor.options["address"] == "10.0.0.1" and executor.options["max_workers_per_node"] == 2
    assert executor.options["cpu_affinity"] == "none" and executor.options["available_accelerators"] == ()
    assert (config.retries, config.retry_handler, config.strategy) == (0, None, "none")
    assert config.run_dir == str(Path(lifecycle.store.root) / "parsl")

    # The shared-store preflight ran once per worker before any leaf.
    names = [record["function"] for record in fake_parsl.submissions]
    assert names == ["_shared_store_probe", "_shared_store_probe", "run_leaf", "run_leaf"]
    assert all(record["executors"] == [LABEL] and record["remote"] and record["dfk"] is dfk for record in fake_parsl.submissions)
    # Each leaf receives only canonical request bytes for its own attempt.
    for record, task_id in zip(fake_parsl.leaf_submissions(), ("first", "second")):
        [payload] = record["args"]
        assert type(payload) is bytes
        request = LeafRequest.decode(payload)
        assert request.task_id == task_id and request.allocation_id == "s3b-allocation"


def test_attached_htex_starts_its_interchange_and_workers_with_this_interpreter(tmp_path: Path, fake_parsl: _Parsl) -> None:
    """Cannon baseline 59775545: neither HTEX process is resolved on PATH.

    The interchange argv and the worker-pool template both run the current
    interpreter with ``-m``; the template keeps every Parsl placeholder.
    """

    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), binding=BINDING, transport=_transport())
    manifest = dispatcher.run(_group("s3b-interpreter", "member"), _runs(tmp_path, "member"), _lifecycle(tmp_path), timeout=60)
    assert _summary(manifest) == {"member": ("completed", False)}

    [executor] = fake_parsl.dfks[0].config.executors
    assert executor.options["interchange_launch_cmd"] == [sys.executable, "-m", "parsl.executors.high_throughput.interchange"]
    assert executor.options["launch_cmd"] == (
        f"{shlex.quote(sys.executable)} -m parsl.executors.high_throughput.process_worker_pool "
        + _DEFAULT_LAUNCH_CMD.removeprefix("process_worker_pool.py ")
    )
    assert "interchange.py" not in executor.options["interchange_launch_cmd"]
    assert "process_worker_pool.py" not in executor.options["launch_cmd"]
    # Any other Parsl template is refused rather than half-rewritten.
    with pytest.raises(parsl_dispatch.UnsupportedTransportError, match="reviewed process_worker_pool.py form"):
        parsl_dispatch._interpreter_launch_commands("worker_pool {debug}")


def test_slot_binding_survives_controller_ambient_at_the_remote_boundary(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M3 killer: ambient CUDA/CPU values never replace the slot binding."""

    # The doubled worker inherits the controller environment, exactly as a
    # LocalProvider block does.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "controller-ambient-0")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    # Only the bound keys conflict; the child still finds the fake core.
    monkeypatch.delenv("PYTHONPATH", raising=False)
    group = _group("s3b-binding", "member")
    lifecycle = _lifecycle(tmp_path)
    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), binding=BINDING, transport=_transport())
    manifest = dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)

    assert _summary(manifest) == {"member": ("completed", False)}
    [record] = fake_parsl.leaf_submissions()
    request = LeafRequest.decode(record["args"][0])
    assert {key: request.environment.env[key] for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")} == BINDING["environment"]
    observed = _observed(lifecycle, group, "member")
    assert (observed["cuda"], observed["omp"]) == ("slot-gpu-3", "3")


def test_each_attached_run_uses_a_fresh_dfk(tmp_path: Path, fake_parsl: _Parsl) -> None:
    """M4 killer: a DFK and its executor never outlive one run."""

    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), binding=BINDING, transport=_transport())
    for index in range(2):
        group = _group(f"s3b-fresh-{index}", "member")
        manifest = dispatcher.run(group, _runs(tmp_path, "member"), _lifecycle(tmp_path, f"store-{index}"), timeout=60)
        assert _summary(manifest) == {"member": ("completed", False)}

    first, second = fake_parsl.dfks
    assert first is not second and first.config is not second.config
    assert first.config.executors[0] is not second.config.executors[0]
    assert first.config.executors[0].provider is not second.config.executors[0].provider
    # The first DFK is cleaned exactly once before the second one exists.
    assert fake_parsl.events == [("dfk", id(first)), ("cleanup", id(first)), ("dfk", id(second)), ("cleanup", id(second))]
    assert [record["dfk"] for record in fake_parsl.leaf_submissions()] == [first, second]


def test_a_failed_remote_submission_is_not_retried(tmp_path: Path, fake_parsl: _Parsl) -> None:
    """A lost leaf is one durable remote-quiescence-unconfirmed outcome, never
    retried and never published completed/accepted (S3 Layer C review P1-2:
    worker loss is never an ordinary launch failure, because there is no
    leaf ack proving no live work remains).
    """

    def lost(_index, _function, _args, _kwargs):
        raise RuntimeError("worker lost")

    fake_parsl.hooks["run_leaf"] = lost
    group = _group("s3b-noretry", "member")
    lifecycle = _lifecycle(tmp_path)
    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), binding=BINDING, transport=_transport())
    manifest = dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)

    assert fake_parsl.calls["run_leaf"] == 1
    assert _summary(manifest) == {"member": ("remote-quiescence-unconfirmed", True)}
    assert len(manifest.attempts) == 1 and manifest.segments == ()


def test_attached_outcomes_match_local_dispatch(tmp_path: Path, fake_parsl: _Parsl) -> None:
    """Completed, failed and timed-out members publish exactly as local dispatch does."""

    group = _group("s3b-parity", "ok", "bad", "late")
    runs = _runs(tmp_path, "ok", "bad", "late", modes={"bad": "fail", "late": "sleep"})
    worker = _worker(tmp_path)
    local = LocalDispatcher(worker).run(group, runs, _lifecycle(tmp_path, "local"), timeout=1)
    attached = parsl_dispatch.ParslDispatcher(worker, binding=BINDING, transport=_transport()).run(
        group, runs, _lifecycle(tmp_path, "attached"), timeout=1,
    )
    assert _summary(attached) == _summary(local) == {"ok": ("completed", False), "bad": ("failed", True), "late": ("timed_out", True)}
    assert {segment.task_id for segment in attached.segments} == {segment.task_id for segment in local.segments} == {"ok"}


@pytest.mark.parametrize("case", ["not-bytes", "not-canonical", "crossed-attempt", "leaf-refusal"])
def test_leaf_results_are_validated_before_publication(tmp_path: Path, fake_parsl: _Parsl, case: str) -> None:
    """A validated leaf refusal is a launch failure; anything that cannot
    even be trusted to answer its request (non-bytes, non-canonical, or a
    well-formed but crossed identity) settles remote-quiescence-unconfirmed
    instead (S3 Layer C review P1-2).
    """

    def tamper(_index, function, args, kwargs):
        if case == "not-bytes":
            return {"status": "completed"}
        if case == "not-canonical":
            return b"{}"
        request = LeafRequest.decode(args[0])
        if case == "leaf-refusal":
            # A replayed request: the fenced core directory already exists.
            Path(request.paths()[1]).mkdir()
            return function(*args, **kwargs)
        result = LeafResult.decode(function(*args, **kwargs))
        return dataclasses.replace(result, attempt_id="another-attempt").encode()

    fake_parsl.hooks["run_leaf"] = tamper
    group = _group(f"s3b-validate-{case}", "member")
    lifecycle = _lifecycle(tmp_path)
    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), binding=BINDING, transport=_transport())
    manifest = dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)
    expected_status = "launch_failed" if case == "leaf-refusal" else "remote-quiescence-unconfirmed"
    assert _summary(manifest) == {"member": (expected_status, True)}
    assert manifest.segments == ()


# ---------------------------------------------------------- shared store


def _probe_setup(tmp_path: Path, nonce: str = "nonce") -> tuple[str, str]:
    root = tmp_path / "shared"
    probe_dir = root / "parsl" / f"store-preflight-{nonce}"
    probe_dir.mkdir(parents=True)
    (probe_dir / "token").write_text(nonce, encoding="utf-8")
    return str(root), os.path.realpath(root)


def test_shared_store_probe_checks_identity_lockf_and_every_worker(tmp_path: Path) -> None:
    root, real_root = _probe_setup(tmp_path)
    facts = parsl_dispatch._shared_store_probe(root, real_root, "nonce", 1, 5)
    assert facts["root"] == real_root and facts["pid"] == os.getpid()
    with pytest.raises(RuntimeError, match="different canonical path"):
        parsl_dispatch._shared_store_probe(root, real_root + "-elsewhere", "nonce", 1, 5)
    other_root, other_real = _probe_setup(tmp_path / "other", "other")
    (Path(other_root) / "parsl" / "store-preflight-other" / "token").write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="token differs"):
        parsl_dispatch._shared_store_probe(other_root, other_real, "other", 1, 5)
    with pytest.raises(FileNotFoundError):
        parsl_dispatch._shared_store_probe(root, real_root, "unknown-nonce", 1, 5)
    # A worker that never arrives keeps the barrier closed until the bound.
    with pytest.raises(TimeoutError, match="every worker"):
        parsl_dispatch._shared_store_probe(root, real_root, "nonce", 3, 0.3)


@pytest.mark.parametrize("case", ["foreign-root", "worker-error", "missing-worker", "controller-lockf", "worker-lockf"])
def test_shared_store_preflight_fails_closed_before_any_attempt(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    """A store not shared and lockable on every worker refuses the whole run."""

    def probe(index, function, args, kwargs):
        if case == "foreign-root":
            return {"host": "worker", "pid": 1, "root": "/elsewhere"}
        if case == "worker-error" and index == 2:
            raise OSError("store not mounted on this worker")
        if case == "missing-worker" and index == 2:
            raise TimeoutError("never registered")
        return function(*args, **kwargs)

    fake_parsl.hooks["_shared_store_probe"] = probe
    if case in ("controller-lockf", "worker-lockf"):
        real_lockf_probe = parsl_dispatch._lockf_probe

        def no_lock(path):
            # The controller locks the shared token; each worker probe locks
            # only its private lock-capability file.
            if case == "controller-lockf" or os.path.basename(path).startswith("lock-"):
                raise OSError("lockf unsupported")
            real_lockf_probe(path)

        monkeypatch.setattr(parsl_dispatch, "_lockf_probe", no_lock)
    group = _group(f"s3b-preflight-{case}", "member")
    lifecycle = _lifecycle(tmp_path)
    dispatcher = parsl_dispatch.ParslDispatcher(_worker(tmp_path), binding=BINDING, transport=_transport(preflight_timeout=0.5))
    with pytest.raises(parsl_dispatch.SharedStorePreflightError):
        dispatcher.run(group, _runs(tmp_path, "member"), lifecycle, timeout=60)
    assert not (Path(lifecycle.store.root) / "groups").exists()
    assert fake_parsl.leaf_submissions() == []
    [dfk] = fake_parsl.dfks
    assert dfk.cleaned == 1


# Each probe process runs the production probe.  The "holder" pauses right
# after any successful exclusive lockf, before releasing it, until the test
# creates <control>/go: a peer probe then acquires while the holder holds.
_PROBE_PROCESS = """
import fcntl, json, os, sys, time
from tpen_executor.dispatch import parsl as dispatch

control, role, root, real_root, nonce, expected, timeout = sys.argv[1:8]
real_lockf = fcntl.lockf

def lockf(fd, command, *args):
    result = real_lockf(fd, command, *args)
    if role == "holder" and command & fcntl.LOCK_EX:
        open(os.path.join(control, "held"), "w").close()
        deadline = time.monotonic() + 60
        while not os.path.exists(os.path.join(control, "go")):
            if time.monotonic() > deadline:
                raise TimeoutError("the test never released the held lock")
            time.sleep(0.01)
    return result

fcntl.lockf = lockf
print(json.dumps(dispatch._shared_store_probe(root, real_root, nonce, int(expected), float(timeout))))
"""


def _start_probe(tmp_path: Path, role: str, root: str, real_root: str, expected: int, timeout: float) -> subprocess.Popen:
    control = tmp_path / "control"
    control.mkdir(exist_ok=True)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(EXECUTOR_ROOT / "src"), os.environ.get("PYTHONPATH")]))}
    return subprocess.Popen(
        [sys.executable, "-c", _PROBE_PROCESS, str(control), role, root, real_root, "nonce", str(expected), str(timeout)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )


def _wait_for(condition, bound: float = 30) -> bool:
    deadline = time.monotonic() + bound
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_overlapping_store_probes_in_separate_processes_both_reach_the_barrier(tmp_path: Path) -> None:
    """M6 killer (review F1): overlapping lock acquisition never refuses a healthy store.

    POSIX record locks belong to processes, so only separate processes can
    contend.  The holder probe pauses while it holds its lock; the peer probe
    then takes its own lock and arrives at the barrier during the hold.
    """

    root, real_root = _probe_setup(tmp_path)
    probe_dir = Path(root) / "parsl" / "store-preflight-nonce"
    holder = _start_probe(tmp_path, "holder", root, real_root, 2, 30)
    peer = None
    try:
        assert _wait_for(lambda: (tmp_path / "control" / "held").exists() or holder.poll() is not None)
        assert holder.poll() is None, holder.communicate()
        peer = _start_probe(tmp_path, "peer", root, real_root, 2, 30)
        # The peer arrives while the holder still holds its lock: the two
        # acquisitions overlapped.  On contention the peer exits instead.
        arrived = _wait_for(lambda: len(list(probe_dir.glob("arrival-*"))) == 1 or peer.poll() is not None)
        assert arrived and peer.poll() is None, peer.communicate()
        assert holder.poll() is None
        (tmp_path / "control" / "go").touch()
        results = [process.communicate(timeout=60) for process in (holder, peer)]
    finally:
        for process in (holder, peer):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
    assert (holder.returncode, peer.returncode) == (0, 0), results
    facts = [json.loads(stdout) for stdout, _stderr in results]
    assert [fact["root"] for fact in facts] == [real_root, real_root]
    assert facts[0]["pid"] != facts[1]["pid"]
    assert len(list(probe_dir.glob("arrival-*"))) == 2
    # Each probe proved lockf on its own file, never on the shared token.
    assert len(list(probe_dir.glob("lock-*"))) == 2


def test_a_missing_worker_process_keeps_the_barrier_closed(tmp_path: Path) -> None:
    """Missing-worker control: one probe process alone never passes a two-worker barrier."""

    root, real_root = _probe_setup(tmp_path)
    probe = _start_probe(tmp_path, "peer", root, real_root, 2, 0.5)
    _stdout, stderr = probe.communicate(timeout=60)
    assert probe.returncode != 0 and "not every worker reached" in stderr
    assert len(list((Path(root) / "parsl" / "store-preflight-nonce").glob("arrival-*"))) == 1


# ------------------------------------------------------- remote CPU affinity


def _split_affinity(monkeypatch: pytest.MonkeyPatch, controller_mask, worker_mask) -> list[str]:
    """Give the controller thread and the doubled worker threads different masks.

    Each mask is a callable of the real mask; it may raise.  Returns the list
    of callers, in order, that queried the affinity.
    """

    real = os.sched_getaffinity
    controller = threading.get_ident()
    callers: list[str] = []

    def affinity(pid):
        if threading.get_ident() == controller:
            callers.append("controller")
            return controller_mask(set(real(pid)))
        callers.append("worker")
        return worker_mask(set(real(pid)))

    monkeypatch.setattr(os, "sched_getaffinity", affinity)
    return callers


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux sched affinity and /proc")
def test_a_remote_slot_is_admitted_by_the_worker_affinity_not_the_controller(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review F2: disjoint controller and worker masks admit a valid worker slot."""

    cpu = min(os.sched_getaffinity(0))
    # The controller may not use the slot CPU; the worker may.
    callers = _split_affinity(monkeypatch, lambda real: {max(real) + 1}, lambda real: real)
    group = _group("s3b-affinity-disjoint", "member")
    lifecycle = _placed_lifecycle(tmp_path, cpu)
    manifest = parsl_dispatch.ParslDispatcher(_worker(tmp_path), transport=_transport()).run(
        group, _runs(tmp_path, "member"), lifecycle, timeout=60,
    )

    assert _summary(manifest) == {"member": ("completed", False)}
    [attempt] = manifest.attempts
    assert attempt.placement_receipt.refusal is None
    # The leaf checked its worker mask; the controller mask was never consulted.
    assert "worker" in callers and "controller" not in callers


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs Linux sched affinity and /proc")
@pytest.mark.parametrize(
    "case, code",
    [("outside", "placement-cpu-outside-worker-affinity"), ("unavailable", "placement-worker-affinity-unavailable")],
)
def test_an_invalid_worker_mask_is_refused_before_science(
    tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch, case: str, code: str,
) -> None:
    """Review F2: a slot CPU the worker may not use is refused on the worker before any child."""

    cpu = min(os.sched_getaffinity(0))

    def worker_mask(_real):
        if case == "unavailable":
            raise OSError("affinity is not supported on this worker")
        return {cpu + 1}

    # The controller may use the slot CPU, so only the worker check refuses.
    _split_affinity(monkeypatch, lambda real: real, worker_mask)
    group = _group(f"s3b-affinity-{case}", "member")
    lifecycle = _placed_lifecycle(tmp_path, cpu)
    manifest = parsl_dispatch.ParslDispatcher(_worker(tmp_path), transport=_transport()).run(
        group, _runs(tmp_path, "member"), lifecycle, timeout=60,
    )

    assert _summary(manifest) == {"member": ("launch_failed", True)} and manifest.segments == ()
    [attempt] = manifest.attempts
    assert attempt.placement_receipt.refusal["code"] == code
    # No science child ran: no core directory, science log or sidecar exists.
    output_dir = Path(lifecycle.store.root) / "groups" / group.identity / "member" / attempt.attempt_id.value
    assert not any((output_dir / name).exists() for name in ("core", "stdout.log", "stderr.log", "child-placement.json"))
    assert len(fake_parsl.leaf_submissions()) == 1


# ------------------------------------------------------- unchanged routes


def test_threadpool_and_local_routes_never_build_a_leaf(tmp_path: Path, fake_parsl: _Parsl, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default dispatch is unchanged: no leaf request, no leaf, no HTEX."""

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the default route reached the remote leaf")

    monkeypatch.setattr(parsl_dispatch, "run_leaf", forbidden)
    monkeypatch.setattr(parsl_dispatch.LeafRequest, "for_attempt", forbidden)
    worker = _worker(tmp_path)
    group = _group("s3b-default", "member")
    threads = parsl_dispatch.ParslDispatcher(worker, binding=BINDING).run(group, _runs(tmp_path, "member"), _lifecycle(tmp_path, "threads"), timeout=60)
    local = LocalDispatcher(worker).run(group, _runs(tmp_path, "member"), _lifecycle(tmp_path, "local"), timeout=60)
    assert _summary(threads) == _summary(local) == {"member": ("completed", False)}
    [dfk] = fake_parsl.dfks
    assert [type(executor) for executor in dfk.config.executors] == [_ThreadPool]
    assert [record["function"] for record in fake_parsl.submissions] == ["_run_parsl_child_captured"]
    assert not any(record["remote"] for record in fake_parsl.submissions)


# ------------------------------------------------------------- real HTEX


_MPIEXEC_STANDIN = """#!/bin/sh
# Test-owned stand-in for mpiexec: start -n copies of the trailing command on
# this host and record the invocation.  It carries no facility value.
echo "$@" >> "$S3B_MPIEXEC_LOG"
count=1
while [ $# -gt 0 ]; do
  case "$1" in
    -n) count="$2"; shift 2 ;;
    --hostfile) shift 2 ;;
    /usr/bin/sh|/bin/sh) break ;;
    *) shift ;;
  esac
done
index=0
while [ "$index" -lt "$count" ]; do
  "$@" &
  index=$((index + 1))
done
wait
"""


@pytest.mark.skipif(not HAS_PARSL, reason="requires the approved executor parsl extra")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real attached HTEX is exercised on the Linux test partition")
def test_real_attached_htex_runs_the_leaf_on_its_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real serialized HTEX/LocalProvider/MpiExec app runs the leaf on its workers.

    It proves the selected executor label, a fresh DFK per run, the worker
    preflight barrier over both workers, worker-side science under the
    immutable slot binding despite conflicting controller ambient values,
    and controller publication.
    """

    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "mpiexec").write_text(_MPIEXEC_STANDIN, encoding="utf-8")
    (shim / "mpiexec").chmod(stat.S_IRWXU)
    mpiexec_log = tmp_path / "mpiexec.log"
    venv_bin = os.path.dirname(sys.executable)
    # The interchange and worker pool run as modules of this interpreter, so
    # this environment's bin directory is deliberately kept OFF PATH (Cannon
    # baseline 59775545 failed on a missing interchange.py there).  Only the
    # mpiexec stand-in is added.  Controller ambient CUDA/CPU values conflict
    # with the binding and reach the workers by inheritance.
    path = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry and os.path.realpath(entry) != os.path.realpath(venv_bin)]
    monkeypatch.setenv("PATH", os.pathsep.join([str(shim), *path]))
    monkeypatch.setenv("S3B_MPIEXEC_LOG", str(mpiexec_log))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "controller-ambient-0")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    monkeypatch.delenv(WORKER_MARK, raising=False)
    worker_init = "\n".join([
        f"export PATH={shim}:$PATH",
        f"export PYTHONPATH={EXECUTOR_ROOT / 'src'}${{PYTHONPATH:+:$PYTHONPATH}}",
        f"export {WORKER_MARK}=htex-worker",
    ])
    dispatcher = parsl_dispatch.ParslDispatcher(
        _worker(tmp_path),
        binding=BINDING,
        transport=_transport(worker_init=worker_init, address="127.0.0.1", max_workers_per_node=2, preflight_timeout=120),
    )
    lifecycle = _lifecycle(tmp_path)
    first = _group("s3b-real-first", "alpha", "beta")
    second = _group("s3b-real-second", "gamma")
    first_manifest = dispatcher.run(first, _runs(tmp_path, "alpha", "beta"), lifecycle, timeout=120)
    second_manifest = dispatcher.run(second, _runs(tmp_path, "gamma"), lifecycle, timeout=120)

    assert _summary(first_manifest) == {"alpha": ("completed", False), "beta": ("completed", False)}
    assert _summary(second_manifest) == {"gamma": ("completed", False)}
    assert {segment.task_id for segment in (*first_manifest.segments, *second_manifest.segments)} == {"alpha", "beta", "gamma"}
    for group, task_id in ((first, "alpha"), (first, "beta"), (second, "gamma")):
        observed = _observed(lifecycle, group, task_id)
        # Science ran under a worker-side leaf, never the controller, with
        # exactly the slot binding.
        assert observed["mark"] == "htex-worker" and observed["ppid"] != os.getpid()
        assert (observed["cuda"], observed["omp"]) == ("slot-gpu-3", "3")

    # One fresh DFK per run, each with the selected executor label, and a
    # worker block launched through MpiExecLauncher once per run.
    parsl_root = Path(lifecycle.store.root) / "parsl"
    run_dirs = sorted(path for path in parsl_root.iterdir() if path.is_dir() and path.name.isdigit())
    assert len(run_dirs) == 2 and all((run_dir / LABEL).is_dir() for run_dir in run_dirs)
    launches = mpiexec_log.read_text(encoding="utf-8").splitlines()
    assert len(launches) == 2 and all("-n 1" in launch for launch in launches)
    # Both workers held a preflight probe at once in each run.
    preflights = sorted(parsl_root.glob("store-preflight-*"))
    assert len(preflights) == 2
    assert all(len(list(path.glob("arrival-*"))) == 2 for path in preflights)


@pytest.mark.skipif(not HAS_PARSL, reason="requires the approved executor parsl extra")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real attached HTEX is exercised on the Linux test partition")
def test_real_attached_htex_cli_entry_runs_allocation_bound_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """F1 test 4: the public CLI reaches the real attached HTEX route."""

    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "mpiexec").write_text(_MPIEXEC_STANDIN, encoding="utf-8")
    (shim / "mpiexec").chmod(stat.S_IRWXU)
    mpiexec_log = tmp_path / "mpiexec.log"
    venv_bin = os.path.dirname(sys.executable)
    path = [entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry and os.path.realpath(entry) != os.path.realpath(venv_bin)]
    monkeypatch.setenv("PATH", os.pathsep.join([str(shim), *path]))
    monkeypatch.setenv("S3B_MPIEXEC_LOG", str(mpiexec_log))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "controller-ambient-0")
    monkeypatch.setenv("OMP_NUM_THREADS", "99")
    monkeypatch.delenv(WORKER_MARK, raising=False)
    worker_init = "\n".join([
        f"export PATH={shim}:$PATH",
        f"export PYTHONPATH={EXECUTOR_ROOT / 'src'}${{PYTHONPATH:+:$PYTHONPATH}}",
        f"export {WORKER_MARK}=htex-worker",
    ])
    allocation = ConfiguredAllocation(
        AllocationId("s3b-cli-allocation"), BackendJobId("s3b-cli-job"), {"binding": BINDING},
    )
    paths = _cli_inputs(tmp_path, allocation)
    transport = tmp_path / "transport.json"
    transport.write_text(
        json.dumps(_transport(worker_init=worker_init, address="127.0.0.1", max_workers_per_node=2, preflight_timeout=120)),
        encoding="utf-8",
    )
    code = cli.main(
        [
            "dispatch", "--input", str(paths["input"]), "--store", str(paths["store"]),
            "--allocation", str(paths["allocation"]), "--worker-config", str(paths["worker"]),
            "--runs", str(paths["runs"]), "--dispatcher", "parsl", "--transport", str(transport),
            "--timeout", "120",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0 and captured.err == ""
    stdout_manifest = json.loads(captured.out)
    durable_manifest = json.loads((Path(paths["store"]) / "run-manifest.json").read_text(encoding="utf-8"))
    assert stdout_manifest == durable_manifest
    assert stdout_manifest["attempts"][0]["outcome_receipt"]["status"] == "completed"

    lifecycle = AttachedAllocation(allocation, DirectoryStore(Path(paths["store"])))
    observed = _observed(lifecycle, paths["group"], "member")  # type: ignore[arg-type]
    assert observed["mark"] == "htex-worker" and observed["ppid"] != os.getpid()
    assert (observed["cuda"], observed["omp"]) == ("slot-gpu-3", "3")
    parsl_root = Path(paths["store"]) / "parsl"
    run_dirs = [path for path in parsl_root.iterdir() if path.is_dir() and path.name.isdigit()]
    assert len(run_dirs) == 1 and (run_dirs[0] / LABEL).is_dir()
    assert len(mpiexec_log.read_text(encoding="utf-8").splitlines()) == 1
    preflights = list(parsl_root.glob("store-preflight-*"))
    assert len(preflights) == 1 and len(list(preflights[0].glob("arrival-*"))) == 2
