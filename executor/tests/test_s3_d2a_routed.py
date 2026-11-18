"""S3 D2a: strict routed transport pools, slot admission and host preflight.

All tests use lightweight Parsl doubles.  D2b owns task-to-label routing and
child placement mismatch policy; this file only proves the D2a admission and
pool/preflight topology.
"""

from __future__ import annotations

import ast
from concurrent.futures import Future
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tpen_executor.allocations import ConfiguredAllocation
from tpen_executor.contracts import AllocationId, BackendJobId
from tpen_executor.dispatch import parsl as parsl_dispatch
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


def _install_fake_parsl(monkeypatch: pytest.MonkeyPatch) -> None:
    modules = {
        "parsl.config": SimpleNamespace(Config=_Config),
        "parsl.dataflow.dflow": SimpleNamespace(DataFlowKernel=_DFK),
        "parsl.app.app": SimpleNamespace(python_app=lambda **_: lambda function: function),
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
    _install_fake_parsl(monkeypatch)
    spec = _spec()
    allocation = _allocation()
    lifecycle = AttachedAllocation(allocation, DirectoryStore(tmp_path / "store"))
    dispatcher = parsl_dispatch.ParslDispatcher(SimpleNamespace(materialize=lambda _env=None: {}), transport=spec)
    calls: list[str] = []

    def app_factory(*, executors, **_kwargs):
        calls.extend(executors)

        def decorate(function):
            def submit(*args, **_submit_kwargs):
                future: Future = Future()
                pool = next(pool for pool in spec.pools if pool.label == executors[0])
                future.set_result({"host": pool.node, "pid": 1, "root": str(Path(args[1]).resolve())})
                return future
            return submit
        return decorate

    # The fake's root fact is derived from the probe argument, matching the
    # controller's realpath without launching a worker.
    dispatcher._python_app_factory = app_factory
    dispatcher._active_dfk = object()
    dispatcher._preflight_routed_store(lifecycle, spec)
    assert calls == ["pool-a", "pool-a", "pool-b"]


def test_routed_preflight_rejects_wrong_hostname_before_begin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_parsl(monkeypatch)
    spec = _spec()
    lifecycle = AttachedAllocation(_allocation(), DirectoryStore(tmp_path / "store"))
    dispatcher = parsl_dispatch.ParslDispatcher(SimpleNamespace(materialize=lambda _env=None: {}), transport=spec)

    def app_factory(*, executors, **_kwargs):
        def decorate(_function):
            def submit(*_args, **_submit_kwargs):
                future: Future = Future()
                pool = next(pool for pool in spec.pools if pool.label == executors[0])
                future.set_result({"host": "wrong-node" if pool.label == "pool-b" else pool.node, "pid": 1, "root": str(Path(lifecycle.store.root).resolve())})
                return future
            return submit
        return decorate

    dispatcher._python_app_factory = app_factory
    dispatcher._active_dfk = object()
    with pytest.raises(parsl_dispatch.RoutedTransportPreflightError):
        dispatcher._preflight_routed_store(lifecycle, spec)
    assert not (Path(lifecycle.store.root) / "groups").exists()


def test_routed_config_is_not_validated_as_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_parsl(monkeypatch)
    spec = _spec()
    dispatcher = parsl_dispatch.ParslDispatcher(SimpleNamespace(materialize=lambda _env=None: {}), transport=spec)
    config = dispatcher._attached_routed_config(spec, "runinfo")
    with pytest.raises(AttributeError, match="label"):
        dispatcher._validate_attached_config(config, spec)  # type: ignore[arg-type]


def test_d2a_mutant_table_targets_unique_executable_anchors() -> None:
    root = Path(__file__).parents[2]
    module = Path(__file__).read_text(encoding="utf-8")
    public_module = (root / "executor/tests/test_s3_transport_parsl.py").read_text(encoding="utf-8")
    assert [row["id"] for row in D2A_MUTANT_TABLE] == ["H1", "H2", "H3", "H4", "H5", "H6"]
    for row in D2A_MUTANT_TABLE:
        source = (root / row["target"]).read_text(encoding="utf-8")
        assert source.count(row["old"]) == 1, row
        assert row["old"] != row["new"] and row["suite"] == "executor"
        ast.parse(source.replace(row["old"], row["new"]))
        assert f"def {row['killer']}(" in module + public_module
