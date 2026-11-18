"""S3 D2a: strict routed transport pools, slot admission and host preflight.

All tests use lightweight Parsl doubles.  D2b owns task-to-label routing and
child placement mismatch policy; this file only proves the D2a admission and
pool/preflight topology.
"""

from __future__ import annotations

import ast
from concurrent.futures import Future
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId
from tpen_executor.dispatch import parsl as parsl_dispatch
from tpen_executor.dispatch.contract import _with_child_placement
from tpen_executor.environments import PreparedPythonEnvironment
from tpen_executor.placement import AllocationSlot, PlacementRefusal, routed_slots_from_allocation
from tpen_executor.storage import DirectoryStore
from tpen_executor.lifecycle import AttachedAllocation


PARSL_MODULE = "executor/src/tpen_executor/dispatch/parsl.py"
PLACEMENT_MODULE = "executor/src/tpen_executor/placement.py"


D2A_MUTANT_TABLE = (
    {
        "id": "H1", "target": PARSL_MODULE,
        "old": '            if facts["root"] != real_root or facts["host"] != pool.node or type(facts["pid"]) is not int or facts["pid"] <= 0:\n',
        "new": '            if facts["root"] != real_root or type(facts["pid"]) is not int or facts["pid"] <= 0:\n',
        "killer": "test_routed_run_rejects_wrong_probe_hostname_before_begin",
        "suite": "executor", "assertion": "every probe host equals its pool node before lifecycle.begin",
    },
    {
        "id": "H2", "target": PARSL_MODULE,
        "old": "            if pool_nodes != set(counts):\n",
        "new": "            if False:\n",
        "killer": "test_routed_cli_rejects_v2_node_set_or_capacity_mismatch",
        "suite": "executor", "assertion": "pool and captured allocation node sets are equal",
    },
    {
        "id": "H3", "target": PARSL_MODULE,
        "old": "                nodes_per_block=1,\n",
        "new": "                nodes_per_block=len(transport.pools),\n",
        "killer": "test_routed_composition_dispatches_v2_transport_file",
        "suite": "executor", "assertion": "each v2 pool is a single-node provider",
    },
    {
        "id": "H4", "target": PARSL_MODULE,
        "old": "                executors=[pool.label],\n",
        "new": "                executors=list(self._active_executor_labels),\n",
        "killer": "test_routed_composition_dispatches_v2_transport_file",
        "suite": "executor", "assertion": "a pool's probes use only that pool label",
    },
    {
        "id": "H5", "target": PARSL_MODULE,
        "old": "                    labels = self._validate_routed_config(config, self.transport)\n",
        "new": "                    labels = self._validate_attached_config(config, self.transport)\n",
        "killer": "test_routed_run_uses_routed_validator",
        "suite": "executor", "assertion": "v2 dispatch selects its multi-pool validator",
    },
    {
        "id": "H6", "target": PARSL_MODULE,
        "old": "                if counts[pool.node] != pool.workers:\n",
        "new": "                if False:\n",
        "killer": "test_routed_cli_rejects_v2_node_set_or_capacity_mismatch",
        "suite": "executor", "assertion": "each pool worker count equals its node's slot count",
    },
    {
        "id": "H7", "target": "executor/src/tpen_executor/dispatch/contract.py",
        "old": "    resolve_slots = slot_resolver or physical_slots_from_allocation\n",
        "new": "    resolve_slots = physical_slots_from_allocation\n",
        "killer": "test_launch_placement_accepts_routed_v2_declared_host_and_preserves_v1",
        "suite": "executor", "assertion": "v2 resolver not used under routed transport",
    },
    {
        "id": "F1", "target": PARSL_MODULE,
        "old": "            if not isinstance(available_accelerators, (list, tuple)) or available_accelerators:\n",
        "new": "            if available_accelerators != ():\n",
        "killer": "test_review_r1_real_htex_routed_config_passes_its_validator",
        "suite": "executor", "assertion": "an empty list from real HTEX is equivalent to an empty tuple",
    },
    {
        "id": "F2", "target": PLACEMENT_MODULE,
        "old": "    cpu_entitlements = [(slot.node, cpu) for slot in decoded for cpu in slot.cpus]\n",
        "new": "    cpu_entitlements = [cpu for slot in decoded for cpu in slot.cpus]\n",
        "killer": "test_review_r1_cpu_ownership_is_node_local",
        "suite": "executor", "assertion": "CPU ownership is keyed by physical node and CPU",
    },
    {
        "id": "F3", "target": "executor/src/tpen_executor/allocations.py",
        "old": "                        captured_scheduler_host_map(scheduler)\n                        if routed_slots\n                        else scheduler_host_map(scheduler, environ={})\n",
        "new": "                        scheduler_host_map(scheduler, environ={})\n",
        "killer": "test_review_r1_pbs_capture_preserves_exact_routed_names",
        "suite": "executor", "assertion": "v2 captures exact PBS host spellings before freezing the allocation",
    },
    {
        "id": "F4", "target": PARSL_MODULE,
        "old": "                facts = future.result(timeout=max(deadline - time.monotonic(), 0))\n",
        "new": "                facts = future.result(timeout=max(deadline - time.monotonic(), 0) + 5)\n",
        "killer": "test_review_r1_routed_preflight_checks_completion_deadline",
        "suite": "executor", "assertion": "late successful preflight facts refuse before lifecycle.begin",
    },
)


class _Config:
    def __init__(self, executors=None, run_dir="runinfo", retries=0, retry_handler=None, strategy="simple", usage_tracking=0, **options):
        self.executors = list(executors or [])
        self.run_dir = run_dir
        self.retries = retries
        self.retry_handler = retry_handler
        self.strategy = strategy
        self.usage_tracking = usage_tracking
        self.options = options


class _ThreadPool:
    def __init__(self, label="threads", max_threads=2, **options):
        self.label = label
        self.max_threads = max_threads
        self.options = options


class _HTEX:
    def __init__(self, label="htex", provider=None, **options):
        self.label = label
        self.provider = provider
        self.options = options


class _LocalProvider:
    def __init__(self, launcher=None, **options):
        self.launcher = launcher
        self.options = options


class _MpiExec:
    def __init__(self, debug=True, bind_cmd="--bind-to", overrides=""):
        self.bind_cmd = bind_cmd
        self.overrides = overrides


class _DFK:
    def __init__(self, config):
        self.config = config


def _install_fake_parsl(
    monkeypatch: pytest.MonkeyPatch,
    *,
    probe_hosts: dict[str, str] | None = None,
    probe_labels: list[str] | None = None,
) -> None:
    probe_hosts = {} if probe_hosts is None else probe_hosts

    def python_app(*, executors="all", **_kwargs):
        labels = tuple(executors) if not isinstance(executors, str) else (executors,)

        def decorate(function):
            if function.__name__ != "_shared_store_probe" or not probe_hosts:
                return function

            def submit(*args, **_submit_kwargs):
                label = labels[0]
                if probe_labels is not None:
                    probe_labels.append(label)
                future: Future = Future()
                future.set_result(
                    {"host": probe_hosts[label], "pid": 1, "root": str(Path(args[0]).resolve())}
                )
                return future

            return submit

        return decorate

    modules = {
        "parsl.config": SimpleNamespace(Config=_Config),
        "parsl.dataflow.dflow": SimpleNamespace(DataFlowKernel=_DFK),
        "parsl.app.app": SimpleNamespace(python_app=python_app),
        "parsl.executors": SimpleNamespace(HighThroughputExecutor=_HTEX),
        "parsl.executors.high_throughput.executor": SimpleNamespace(
            DEFAULT_LAUNCH_CMD="process_worker_pool.py {debug} {max_workers_per_node} -a {addresses} -p {prefetch_capacity}"
        ),
        "parsl.executors.threads": SimpleNamespace(ThreadPoolExecutor=_ThreadPool),
        "parsl.launchers": SimpleNamespace(MpiExecLauncher=_MpiExec),
        "parsl.providers": SimpleNamespace(LocalProvider=_LocalProvider),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _spec() -> parsl_dispatch.RoutedAttachedTransportSpec:
    return parsl_dispatch.RoutedAttachedTransportSpec.from_mapping(
        {
            "schema": parsl_dispatch.ROUTED_ATTACHED_TRANSPORT_SCHEMA,
            "executor": "htex", "provider": "local", "launcher": "mpiexec",
            "pools": [
                {"node": "node-a", "label": "pool-a", "workers": 2, "worker_init": "A", "address": None, "mpiexec_bind_cmd": "--bind-to", "mpiexec_overrides": "--a"},
                {"node": "node-b", "label": "pool-b", "workers": 1, "worker_init": "B", "address": "127.0.0.1", "mpiexec_bind_cmd": "--bind-to", "mpiexec_overrides": "--b"},
            ],
            "preflight_timeout": 5,
            "teardown_timeout": 2,
        }
    )


def _allocation(*, nodes=("node-a", "node-a", "node-b"), scheduler="node-a\nnode-b\n") -> ConfiguredAllocation:
    slots = [AllocationSlot(f"slot-{index}", node, (), (index,), (), ()).to_dict() for index, node in enumerate(nodes)]
    return ConfiguredAllocation(
        AllocationId("d2a-allocation"), BackendJobId("d2a-job"),
        {"scheduler": {"pbs_nodefile_text": scheduler}, "slots": {"schema": "tpen-executor/allocation-slots-v2", "slots": slots}},
    )


def test_routed_transport_decoder_is_strict_and_discriminated() -> None:
    spec = _spec()
    assert type(spec) is parsl_dispatch.RoutedAttachedTransportSpec
    assert parsl_dispatch.decode_attached_transport(spec.to_dict()) == spec
    with pytest.raises(ValueError):
        parsl_dispatch.RoutedAttachedTransportSpec.from_mapping({**spec.to_dict(), "unexpected": True})
    with pytest.raises(ValueError):
        parsl_dispatch.RoutedAttachedTransportSpec.from_mapping({**spec.to_dict(), "pools": [{**spec.to_dict()["pools"][0], "workers": 0}]})
    with pytest.raises(ValueError):
        parsl_dispatch.decode_attached_transport({"schema": "tpen-executor/attached-transport-v0"})


def test_routed_slots_allow_multiple_slots_per_node_and_use_exact_captured_nodes() -> None:
    slots = routed_slots_from_allocation(_allocation())
    assert [slot.node for slot in slots] == ["node-a", "node-a", "node-b"]
    with pytest.raises(PlacementRefusal, match="differ"):
        routed_slots_from_allocation(_allocation(scheduler="node-a.example\nnode-b\n"))


def test_launch_placement_accepts_routed_v2_declared_host_and_preserves_v1(tmp_path: Path) -> None:
    """The routed launch path resolves v2 hosts while the default stays v1."""

    cpu = min(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    slot = AllocationSlot("slot-a", "node-a", (), (cpu,), (), ())
    routed = ConfiguredAllocation(
        AllocationId("placement-routed"),
        BackendJobId("placement-job"),
        {
            "scheduler": {"pbs_nodefile_text": "node-a\n"},
            "slots": {"schema": "tpen-executor/allocation-slots-v2", "slots": [slot.to_dict()]},
        },
    )
    fence = SimpleNamespace(attempt_id=SimpleNamespace(value="attempt-a"), output_dir=str(tmp_path / "output"))
    group = SimpleNamespace(identity="group-digest")
    member = SimpleNamespace(task_id="task-a")
    configured = SimpleNamespace(config={}, overrides=())
    prepared = PreparedPythonEnvironment(sys.executable, None, dict(slot.env))

    routed_environment = _with_child_placement(
        prepared,
        routed,
        fence,
        group,
        member,
        configured,
        check_controller_affinity=False,
        slot_resolver=routed_slots_from_allocation,
    )
    assert routed_environment.env["TPEN_SLOT_NODE"] == "node-a"
    assert routed_environment.env["TPEN_PLACEMENT_DECLARED_HOST"] == "node-a"
    assert "TPEN_PLACEMENT_REFUSAL" not in routed_environment.env

    with pytest.raises(PlacementRefusal, match="allocation slots must use schema 'tpen-executor/allocation-slots-v1'"):
        _with_child_placement(
            prepared,
            routed,
            fence,
            group,
            member,
            configured,
            check_controller_affinity=False,
        )

    legacy = ConfiguredAllocation(
        AllocationId("placement-v1"),
        BackendJobId("placement-job-v1"),
        {
            "scheduler": {"pbs_nodefile_text": "physical-a\n"},
            "slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot.with_node("logical-a").to_dict()]},
        },
    )
    legacy_environment = _with_child_placement(
        PreparedPythonEnvironment(sys.executable, None, {}),
        legacy,
        fence,
        group,
        member,
        configured,
        check_controller_affinity=False,
    )
    assert legacy_environment.env["TPEN_SLOT_NODE"] == "physical-a"
    assert legacy_environment.env["TPEN_PLACEMENT_DECLARED_HOST"] == "physical-a"


def test_routed_shape_requires_exact_node_set_and_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_parsl(monkeypatch)
    monkeypatch.setattr(parsl_dispatch, "_require_standard_worker", lambda _: None)
    worker = SimpleNamespace(materialize=lambda _env=None: {})
    dispatcher = parsl_dispatch.ParslDispatcher(worker, transport=_spec())
    dispatcher.admit_transport_shape(_allocation())
    with pytest.raises(parsl_dispatch.UnsupportedTransportError, match="slot count"):
        dispatcher.admit_transport_shape(_allocation(nodes=("node-a", "node-b", "node-b")))
    with pytest.raises(PlacementRefusal, match="differ"):
        dispatcher.admit_transport_shape(_allocation(scheduler="node-a\nnode-c\n"))


def test_routed_config_builds_one_single_node_pool_per_node(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_parsl(monkeypatch)
    dispatcher = parsl_dispatch.ParslDispatcher(SimpleNamespace(materialize=lambda _env=None: {}), transport=_spec())
    config = dispatcher._attached_routed_config(_spec(), str(tmp_path))
    assert [executor.label for executor in config.executors] == ["pool-a", "pool-b"]
    assert [executor.provider.options["nodes_per_block"] for executor in config.executors] == [1, 1]
    assert [executor.options["max_workers_per_node"] for executor in config.executors] == [2, 1]
    assert dispatcher._validate_routed_config(config, _spec()) == ("pool-a", "pool-b")


def test_routed_preflight_submits_exact_workers_to_each_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _install_fake_parsl(
        monkeypatch,
        probe_hosts={"pool-a": "node-a", "pool-b": "node-b"},
        probe_labels=calls,
    )
    spec = _spec()
    allocation = _allocation()
    lifecycle = AttachedAllocation(allocation, DirectoryStore(tmp_path / "store"))
    dispatcher = parsl_dispatch.ParslDispatcher(SimpleNamespace(materialize=lambda _env=None: {}), transport=spec)
    dispatcher._preflight_routed_store(lifecycle, spec)
    assert calls == ["pool-a", "pool-a", "pool-b"]


def test_routed_preflight_rejects_wrong_hostname_before_begin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_parsl(
        monkeypatch,
        probe_hosts={"pool-a": "node-a", "pool-b": "wrong-node"},
    )
    spec = _spec()
    lifecycle = AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "store"))
    dispatcher = parsl_dispatch.ParslDispatcher(SimpleNamespace(materialize=lambda _env=None: {}), transport=spec)

    with pytest.raises(parsl_dispatch.RoutedTransportPreflightError):
        dispatcher._preflight_routed_store(lifecycle, spec)
    assert not (Path(lifecycle.store.root) / "groups").exists()


def test_routed_config_is_not_validated_as_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_parsl(monkeypatch)
    spec = _spec()
    dispatcher = parsl_dispatch.ParslDispatcher(SimpleNamespace(materialize=lambda _env=None: {}), transport=spec)
    config = dispatcher._attached_routed_config(spec, "runinfo")
    with pytest.raises(parsl_dispatch.UnsupportedTransportError, match="exactly one executor"):
        dispatcher._validate_attached_config(config, spec)  # type: ignore[arg-type]


def test_d2a_mutant_table_targets_unique_executable_anchors() -> None:
    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    public_module = (root / "executor/tests/test_s3_transport_parsl.py").read_text(encoding="utf-8")
    assert [row["id"] for row in D2A_MUTANT_TABLE] == [
        "H1", "H2", "H3", "H4", "H5", "H6", "H7", "F1", "F2", "F3", "F4",
    ]
    for row in D2A_MUTANT_TABLE:
        source = (root / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, row
        assert row["old"] != row["new"] and row["suite"] == "executor"
        ast.parse(source.replace(row["old"], row["new"]))
        assert f"def {row['killer']}(" in module + public_module


@pytest.mark.parametrize("same_node", [False, True], ids=["different-nodes", "same-node"])
def test_review_r1_cpu_ownership_is_node_local(same_node: bool) -> None:
    """CPU zero on another machine is not the same CPU entitlement."""

    from tpen_executor.placement import BindingError

    nodes = ("node-a", "node-a" if same_node else "node-b")
    slots = [AllocationSlot(f"slot-{index}", node, (), (0,), (), ()) for index, node in enumerate(nodes)]
    allocation = ConfiguredAllocation(
        AllocationId("review-cpus"), BackendJobId("review-job"),
        {
            "scheduler": {"pbs_nodefile_text": "\n".join(dict.fromkeys(nodes)) + "\n"},
            "slots": {"schema": "tpen-executor/allocation-slots-v2", "slots": [slot.to_dict() for slot in slots]},
        },
    )
    if same_node:
        with pytest.raises(BindingError, match="CPU entitlements"):
            routed_slots_from_allocation(allocation)
    else:
        assert routed_slots_from_allocation(allocation) == tuple(slots)


def test_review_r1_real_htex_routed_config_passes_its_validator(tmp_path: Path) -> None:
    """Consume real HTEX constructor output without starting a DFK or workers."""

    pytest.importorskip("parsl")
    spec = _spec()
    dispatcher = parsl_dispatch.ParslDispatcher(SimpleNamespace(materialize=lambda _env=None: {}), transport=spec)
    config = dispatcher._attached_routed_config(spec, str(tmp_path / "parsl"))
    assert dispatcher._validate_routed_config(config, spec) == ("pool-a", "pool-b")


@pytest.mark.parametrize("via_path", [False, True], ids=["captured-text", "nodefile-path"])
def test_review_r1_pbs_capture_preserves_exact_routed_names(tmp_path: Path, via_path: bool) -> None:
    """Real allocation capture must preserve the spelling used by v2 pools."""

    raw_hosts = "Node-A.example\nnode-b.example\n"
    nodefile = tmp_path / "nodes"
    nodefile.write_text(raw_hosts, encoding="utf-8")
    scheduler = {"pbs_nodefile_path": str(nodefile)} if via_path else {"pbs_nodefile_text": raw_hosts}
    slots = [
        AllocationSlot(f"slot-{index}", node, (), (index,), (), ())
        for index, node in enumerate(("Node-A.example", "node-b.example"))
    ]
    allocation = ConfiguredAllocation(
        AllocationId("review-pbs"), BackendJobId("review-job"),
        {"scheduler": scheduler, "slots": {"schema": "tpen-executor/allocation-slots-v2", "slots": [slot.to_dict() for slot in slots]}},
    )
    # Consume the actual frozen producer output after the input file changes.
    nodefile.write_text("replacement-node\n", encoding="utf-8")
    round_tripped = ConfiguredAllocation.from_mapping(allocation.to_dict())
    assert [slot.node for slot in routed_slots_from_allocation(round_tripped)] == ["Node-A.example", "node-b.example"]


def test_review_r1_pbs_v1_path_keeps_legacy_normalization(tmp_path: Path) -> None:
    """An exact-name v2 repair must leave v1's short-host policy intact."""

    from tpen_executor.placement import physical_slots_from_allocation

    nodefile = tmp_path / "nodes"
    nodefile.write_text("Node-A.example\n", encoding="utf-8")
    slot = AllocationSlot("slot-a", "logical-a", (), (0,), (), ())
    allocation = ConfiguredAllocation(
        AllocationId("review-pbs-v1"), BackendJobId("review-job"),
        {
            "scheduler": {"pbs_nodefile_path": str(nodefile)},
            "slots": {"schema": "tpen-executor/allocation-slots-v1", "slots": [slot.to_dict()]},
        },
    )
    nodefile.write_text("replacement-node\n", encoding="utf-8")
    assert [slot.node for slot in physical_slots_from_allocation(allocation)] == ["node-a"]


@pytest.mark.parametrize("reply_time", [0.0, 6.0], ids=["on-time", "late"])
def test_review_r1_routed_preflight_checks_completion_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reply_time: float,
) -> None:
    """Successful facts arriving after the shared bound must still refuse."""

    _install_fake_parsl(monkeypatch)
    now = [0.0]
    monkeypatch.setattr(parsl_dispatch, "time", SimpleNamespace(monotonic=lambda: now[0]))
    spec = _spec()  # preflight_timeout=5, two workers on A and one on B
    hosts = {pool.label: pool.node for pool in spec.pools}

    def python_app(*, executors, **_kwargs):
        label, = executors

        def decorate(_function):
            def submit(root, *_args):
                class ProbeFuture:
                    def result(self, timeout):
                        # Model a reply which arrives during the currently
                        # allowed five-second grace, without sleeping.
                        delay = max(reply_time - now[0], 0)
                        if delay > timeout:
                            now[0] += timeout
                            raise TimeoutError("probe is still pending")
                        now[0] += delay
                        return {"host": hosts[label], "pid": 1, "root": os.path.realpath(root)}

                return ProbeFuture()

            return submit

        return decorate

    monkeypatch.setattr(sys.modules["parsl.app.app"], "python_app", python_app)
    dispatcher = parsl_dispatch.ParslDispatcher(SimpleNamespace(materialize=lambda _env=None: {}), transport=spec)
    lifecycle = AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "store"))
    if reply_time > spec.preflight_timeout:
        with pytest.raises(parsl_dispatch.RoutedTransportPreflightError):
            dispatcher._preflight_routed_store(lifecycle, spec)
    else:
        dispatcher._preflight_routed_store(lifecycle, spec)
    assert not (Path(lifecycle.store.root) / "groups").exists()
