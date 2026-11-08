"""Unit tests for the optional Parsl allocation-attach executor."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from experiments.toolkit.dispatch import AllocationContext, DispatchExecutor, DispatchSpec
from experiments.toolkit.parsl_attach import (
    ParslAttachExecutor,
    _parsl_app_runner,
    _run_dispatch_payload,
    validate_accelerator_tiling,
    validate_pbs_nodefile,
)
from experiments.toolkit.specs import CompletionSpec


def _dispatch(tmp_path: Path, *, completion: CompletionSpec | None = None) -> DispatchSpec:
    return DispatchSpec(
        logical_task_id="logical-1",
        admission_id="admission-1",
        attempt_id="attempt-1",
        stage="train",
        run_id="run-1",
        argv=(sys.executable, "-c", "print('ok')"),
        result_dir=str(tmp_path / "result"),
        runtime="test",
        cwd=str(tmp_path),
        environment={"EXTRA": "one"},
        completion=completion or CompletionSpec(policy="none"),
    )


def _context(tmp_path: Path, **overrides: Any) -> AllocationContext:
    values: dict[str, Any] = {
        "allocation_id": "allocation-1",
        "visibility_variable": "CUDA_VISIBLE_DEVICES",
        "visibility_values": ("0",),
        "run_root": str(tmp_path / "launch"),
    }
    values.update(overrides)
    return AllocationContext(**values)


def test_protocol_records_verbatim_argv_and_worker_layout(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        output = Path(kwargs["output_directory"])
        output.mkdir(parents=True)
        status = output / "attempt_status.json"
        status.write_text(json.dumps({"status": "success"}) + "\n")
        (output / "stdout.log").write_text("")
        (output / "stderr.log").write_text("")
        return {"returncode": 0, "attempt_status_path": str(status)}

    spec = _dispatch(tmp_path)
    executor = ParslAttachExecutor(app_runner=fake_runner)
    assert isinstance(executor, DispatchExecutor)
    records = executor.dispatch((spec,), context=_context(tmp_path))

    assert records[0].submitted_command == spec.argv
    assert spec.argv == (sys.executable, "-c", "print('ok')")
    assert records[0].backend == "parsl_attach"
    assert records[0].launcher_job_id == "allocation-1"
    assert calls[0]["argv"] == spec.argv
    assert calls[0]["environment"]["EXTRA"] == "one"
    assert "CUDA_VISIBLE_DEVICES" not in calls[0]["environment"]
    assert "visibility_value" not in calls[0]
    assert Path(calls[0]["output_directory"]) == tmp_path / "launch" / "dispatch" / spec.attempt_id


def test_completion_rechecked_after_success(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    spec = _dispatch(tmp_path, completion=CompletionSpec(policy="file_exists", output_paths=(str(missing),)))

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        output = Path(kwargs["output_directory"])
        output.mkdir(parents=True)
        status = output / "attempt_status.json"
        status.write_text(json.dumps({"status": "success"}) + "\n")
        return {"returncode": 0, "attempt_status_path": str(status)}

    with pytest.raises(RuntimeError, match="completion predicate"):
        ParslAttachExecutor(app_runner=fake_runner).dispatch((spec,), context=_context(tmp_path))
    status = json.loads((tmp_path / "launch" / "dispatch" / spec.attempt_id / "attempt_status.json").read_text())
    assert status["status"] == "failed"
    assert "completion_error" in status


def test_deadline_guard_refuses_new_dispatches(tmp_path: Path) -> None:
    spec = _dispatch(tmp_path)
    context = _context(tmp_path, deadline=time.time() + 1, deadline_guard_min=1)
    calls = 0

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"returncode": 0}

    error: RuntimeError | None = None
    try:
        ParslAttachExecutor(app_runner=fake_runner).dispatch((spec,), context=context)
    except RuntimeError as exc:
        error = exc
    assert calls == 0
    assert not (tmp_path / "launch" / "dispatch" / spec.attempt_id).exists()
    assert str(error) == "allocation deadline guard reached; refusing new Parsl dispatches"


def test_module_imports_without_parsl_site_package() -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import experiments.toolkit.parsl_attach"],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_pbs_nodefile_missing_is_attributable(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"PBS_NODEFILE missing.*actual host count 0.*expected 2"):
        validate_pbs_nodefile(tmp_path / "missing", requested_node_count=2)


def test_pbs_nodefile_empty_is_attributable(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("\n  \n")
    with pytest.raises(RuntimeError, match=r"PBS_NODEFILE empty.*actual host count 0.*expected 2"):
        validate_pbs_nodefile(nodefile, requested_node_count=2)


def test_pbs_nodefile_unreadable_is_attributable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node01\n")
    def deny_read(*args: Any, **kwargs: Any) -> str:
        raise PermissionError("permission denied")
    monkeypatch.setattr(Path, "read_text", deny_read)
    with pytest.raises(RuntimeError, match=r"PBS_NODEFILE unreadable.*actual host count unavailable.*expected 1"):
        validate_pbs_nodefile(nodefile, requested_node_count=1)


def test_pbs_nodefile_duplicates_report_unique_count(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node01\nnode01\nnode02\n")
    with pytest.raises(RuntimeError, match=r"host count mismatch.*actual host count 2.*expected 3"):
        validate_pbs_nodefile(nodefile, requested_node_count=3)


def test_pbs_nodefile_one_when_two_reports_counts(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node01\n")
    with pytest.raises(RuntimeError, match=r"host count mismatch.*actual host count 1.*expected 2"):
        validate_pbs_nodefile(nodefile, requested_node_count=2)


def test_pbs_nodefile_ten_hosts_passes(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("".join(f"node-{index:02d}\n" for index in range(10)))
    assert validate_pbs_nodefile(nodefile, requested_node_count=10) == tuple(
        f"node-{index:02d}" for index in range(10)
    )


def test_pbs_nodefile_ten_when_two_reports_counts(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("".join(f"node-{index:02d}\n" for index in range(10)))
    with pytest.raises(RuntimeError, match=r"host count mismatch.*actual host count 10.*expected 2"):
        validate_pbs_nodefile(nodefile, requested_node_count=2)


def test_pbs_nodefile_ignores_blank_lines_and_canonicalizes_whitespace(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("  NODE-01  \n\n node-02\t\n NODE-01\n")
    assert validate_pbs_nodefile(nodefile, requested_node_count=2) == ("node-01", "node-02")


def test_pbs_nodefile_unknown_hostname_format_is_attributable(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node 01\nnode-02\n")
    with pytest.raises(RuntimeError, match=r"unknown hostname format on line 1.*expected 2"):
        validate_pbs_nodefile(nodefile, requested_node_count=2)


def test_single_node_four_gpus_validates_one_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_parsl(monkeypatch)
    nodefile = tmp_path / "PBS_NODEFILE"
    nodefile.write_text("node-01\n")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    context = _context(tmp_path, visibility_values=("0", "1", "2", "3"), nodes_per_block=1)

    _parsl_app_runner(context, tmp_path / "launch")

    assert captured["provider"]["nodes_per_block"] == 1


def test_four_nodes_four_gpus_per_node_validate_four_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _install_fake_parsl(monkeypatch)
    nodefile = tmp_path / "PBS_NODEFILE"
    nodefile.write_text("node-01\nnode-02\nnode-03\nnode-04\n")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    context = _context(tmp_path, visibility_values=("0", "1", "2", "3"), nodes_per_block=4)

    _parsl_app_runner(context, tmp_path / "launch")

    assert captured["provider"]["nodes_per_block"] == 4


def test_multi_node_attach_rejects_host_count_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nodefile = tmp_path / "PBS_NODEFILE"
    nodefile.write_text("node-01\n")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    context = _context(tmp_path, visibility_values=("0", "1", "2", "3"), nodes_per_block=4)

    with pytest.raises(RuntimeError, match=r"actual host count 1.*expected 4"):
        _parsl_app_runner(context, tmp_path / "launch")


def test_multi_node_attach_rejects_extra_hosts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nodefile = tmp_path / "PBS_NODEFILE"
    nodefile.write_text("node-01\nnode-02\nnode-03\nnode-04\n")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    context = _context(tmp_path, visibility_values=("0", "1", "2", "3"), nodes_per_block=1)

    with pytest.raises(RuntimeError, match=r"actual host count 4.*expected 1"):
        _parsl_app_runner(context, tmp_path / "launch")


def test_legacy_single_node_does_not_validate_gpu_count_as_hosts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_parsl(monkeypatch)
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    context = _context(tmp_path, visibility_values=("0", "1", "2", "3"))

    _parsl_app_runner(context, tmp_path / "launch")


def test_multi_node_attach_rejects_missing_nodefile_before_parsl_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    with pytest.raises(RuntimeError, match=r"PBS_NODEFILE missing.*expected 2"):
        _parsl_app_runner(
            _context(tmp_path, visibility_values=("0", "1", "2", "3"), nodes_per_block=2),
            tmp_path / "launch",
        )


def test_stdlib_worker_writes_output_layout(tmp_path: Path) -> None:
    output = tmp_path / "dispatch" / "attempt-1"
    payload = _run_dispatch_payload(
        (
            sys.executable,
            "-c",
            "import os; print(os.environ['EXTRA']); print(os.environ['CUDA_VISIBLE_DEVICES'])",
        ),
        str(tmp_path),
        {"EXTRA": "one"},
        str(output),
        "attempt-1",
        "CUDA_VISIBLE_DEVICES",
        "0",
    )
    assert payload["returncode"] == 0
    assert (output / "stdout.log").read_text() == "one\n0\n"
    assert (output / "stderr.log").is_file()
    assert json.loads((output / "attempt_status.json").read_text())["visibility_value"] == "0"


def test_inherit_visibility_keeps_scheduler_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "scheduler-mig")
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        output = Path(kwargs["output_directory"])
        output.mkdir(parents=True)
        status = output / "attempt_status.json"
        status.write_text(json.dumps({"status": "success"}) + "\n")
        return {"returncode": 0, "attempt_status_path": str(status)}

    spec = _dispatch(tmp_path)
    context = _context(tmp_path, visibility_values=())
    assert context.validate() is context
    ParslAttachExecutor(app_runner=fake_runner).dispatch((spec,), context=context)
    assert calls[0]["environment"]["CUDA_VISIBLE_DEVICES"] == "scheduler-mig"
    assert "visibility_variable" not in calls[0]
    assert "visibility_value" not in calls[0]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "scheduler-mig"

    output = tmp_path / "inherit-worker"
    payload = _run_dispatch_payload(
        (sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"),
        str(tmp_path),
        calls[0]["environment"],
        str(output),
        calls[0]["attempt_id"],
    )
    assert payload["returncode"] == 0
    assert (output / "stdout.log").read_text() == "scheduler-mig\n"
    status = json.loads((output / "attempt_status.json").read_text())
    assert payload["inherited_visibility_value"] is None
    assert payload["placement"]["cuda_visible_devices"] == "scheduler-mig"
    assert status["inherited_visibility_value"] is None
    assert status["placement"]["cuda_visible_devices"] == "scheduler-mig"

    parsl = pytest.importorskip("parsl")
    captured: dict[str, Any] = {}

    def fake_load(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(parsl, "load", fake_load)
    _parsl_app_runner(context, tmp_path / "inherit-launch")
    assert captured["config"].executors[0].max_workers_per_node == 1


def test_tiled_visibility_protects_worker_binding_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker's binding survives dispatch and a real child process."""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "dispatcher-mig")
    calls: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "worker-mig")
        payload = _run_dispatch_payload(
            (sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"),
            kwargs["cwd"],
            kwargs["environment"],
            kwargs["output_directory"],
            kwargs["attempt_id"],
            kwargs["visibility_variable"],
            kwargs.get("visibility_value"),
        )
        payloads.append(payload)
        return payload

    ParslAttachExecutor(app_runner=fake_runner).dispatch(
        (_dispatch(tmp_path),), context=_context(tmp_path)
    )

    assert "CUDA_VISIBLE_DEVICES" not in calls[0]["environment"]
    assert (tmp_path / "launch" / "dispatch" / "attempt-1" / "stdout.log").read_text() == (
        "worker-mig\n"
    )
    assert payloads[0]["inherited_visibility_value"] == "worker-mig"
    assert payloads[0]["placement"]["cuda_visible_devices"] == "worker-mig"


def test_payload_protects_worker_binding_from_serialized_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A serialized visibility override cannot replace the worker binding."""

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "worker-mig")
    output = tmp_path / "payload"
    payload = _run_dispatch_payload(
        (sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"),
        str(tmp_path),
        {"CUDA_VISIBLE_DEVICES": "dispatcher-mig"},
        str(output),
        "attempt-1",
        "CUDA_VISIBLE_DEVICES",
    )

    assert (output / "stdout.log").read_text() == "worker-mig\n"
    assert payload["inherited_visibility_value"] == "worker-mig"
    assert payload["placement"]["cuda_visible_devices"] == "worker-mig"


@pytest.mark.parametrize("source", ["context", "dispatch"])
def test_tiled_visibility_override_is_rejected_before_runner(
    source: str, tmp_path: Path
) -> None:
    """Reject ambiguous tiled visibility overrides before submitting work."""

    calls = 0

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"returncode": 0}

    spec = _dispatch(tmp_path)
    context = _context(
        tmp_path,
        environment={"CUDA_VISIBLE_DEVICES": "context-mig"} if source == "context" else {},
    )
    if source == "dispatch":
        spec = replace(spec, environment={**spec.environment, "CUDA_VISIBLE_DEVICES": "task-mig"})

    with pytest.raises(ValueError, match=rf"TILED visibility override from {source}") as raised:
        ParslAttachExecutor(app_runner=fake_runner).dispatch((spec,), context=context)

    assert "CUDA_VISIBLE_DEVICES" in str(raised.value)
    if source == "dispatch":
        assert spec.attempt_id in str(raised.value)
    assert calls == 0


def test_worker_binding_is_measured_not_selected_by_dispatch_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A worker assignment differing from submission order must be preserved."""

    observed: list[str] = []
    # The first submitted task happens to execute on worker GPU 3 and the
    # second on worker GPU 0.  Dispatch-index modulo four would report 0, 1.
    worker_bindings = iter(("3", "0"))

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        worker_binding = next(worker_bindings)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", worker_binding)
        output = Path(kwargs["output_directory"])
        payload = _run_dispatch_payload(
            kwargs["argv"],
            kwargs["cwd"],
            kwargs["environment"],
            str(output),
            kwargs["attempt_id"],
            kwargs["visibility_variable"],
            kwargs.get("visibility_value"),
        )
        observed.append(str(payload["inherited_visibility_value"]))
        return payload

    first = _dispatch(tmp_path)
    second = replace(first, attempt_id="attempt-2", logical_task_id="logical-2")
    context = _context(tmp_path, visibility_values=("0", "1", "2", "3"))
    ParslAttachExecutor(app_runner=fake_runner).dispatch((first, second), context=context)

    assert observed == ["3", "0"]
    assert observed != [context.visibility_values[i % 4] for i in range(2)]


def test_rendered_provider_options_keep_single_node_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_parsl(monkeypatch)
    _parsl_app_runner(_context(tmp_path), tmp_path / "launch")
    assert captured["provider"] == {"init_blocks": 1, "min_blocks": 1, "max_blocks": 1}


def test_parsl_payload_is_by_value_and_keeps_optional_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_parsl(monkeypatch)

    payload = _parsl_app_runner(_context(tmp_path), tmp_path / "launch")

    assert payload.__module__ == "__parsl_worker_payload__"
    assert payload.__defaults__ == (None, None)


def test_rendered_provider_options_use_alcf_multinode_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _install_fake_parsl(monkeypatch)
    nodefile = tmp_path / "PBS_NODEFILE"
    nodefile.write_text("node-01\nnode-02\nnode-03\n")
    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    context = _context(tmp_path, visibility_values=("0", "1", "2", "3"), nodes_per_block=3)
    _parsl_app_runner(context, tmp_path / "launch")
    assert captured["provider"]["nodes_per_block"] == 3
    assert captured["provider"]["launcher"].__dict__ == {
        "bind_cmd": "--cpu-bind",
        "overrides": "--depth=64 --ppn 1",
    }
    assert captured["provider"]["worker_init"] == "export TMPDIR=/tmp"
    assert captured["executor"]["max_workers_per_node"] == 4
    assert captured["executor"]["available_accelerators"] == ("0", "1", "2", "3")


def _install_fake_parsl(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Provide tiny Parsl constructor doubles for rendered-config tests."""

    captured: dict[str, Any] = {}

    class Config:
        def __init__(self, *, executors: list[Any], **kwargs: Any) -> None:
            self.executors = executors
            self.__dict__.update(kwargs)

    class HighThroughputExecutor:
        def __init__(self, **kwargs: Any) -> None:
            captured["executor"] = kwargs
            self.__dict__.update(kwargs)

    class LocalProvider:
        def __init__(self, **kwargs: Any) -> None:
            captured["provider"] = kwargs
            self.__dict__.update(kwargs)

    class MpiExecLauncher:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)
            captured["provider_launcher"] = kwargs

        def __eq__(self, other: object) -> bool:
            return isinstance(other, dict) and self.__dict__ == other

    parsl = types.ModuleType("parsl")
    parsl.load = lambda config: None
    config_module = types.ModuleType("parsl.config")
    config_module.Config = Config
    executors_module = types.ModuleType("parsl.executors")
    executors_module.HighThroughputExecutor = HighThroughputExecutor
    providers_module = types.ModuleType("parsl.providers")
    providers_module.LocalProvider = LocalProvider
    launchers_module = types.ModuleType("parsl.launchers")
    launchers_module.MpiExecLauncher = MpiExecLauncher
    app_module = types.ModuleType("parsl.app")
    app_app_module = types.ModuleType("parsl.app.app")
    app_app_module.python_app = lambda function: function
    for name, module in {
        "parsl": parsl,
        "parsl.config": config_module,
        "parsl.executors": executors_module,
        "parsl.providers": providers_module,
        "parsl.launchers": launchers_module,
        "parsl.app": app_module,
        "parsl.app.app": app_app_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(parsl, "load", lambda config: captured.update(config=config))
    return captured


def test_parsl_config_has_no_retries_and_accelerator_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parsl = pytest.importorskip("parsl")
    captured: dict[str, Any] = {}

    def fake_load(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr(parsl, "load", fake_load)
    _parsl_app_runner(_context(tmp_path), tmp_path / "launch")
    config = captured["config"]
    executor = config.executors[0]
    assert config.retries == 0
    assert config.run_dir == str(tmp_path / "launch" / "parsl")
    assert config.usage_tracking is False
    assert executor.max_workers_per_node == 1
    assert executor.available_accelerators == ["0"]

    _parsl_app_runner(
        _context(tmp_path, visibility_values=("0", "1")),
        tmp_path / "two-worker-launch",
    )
    assert captured["config"].executors[0].max_workers_per_node == 2
    assert captured["config"].executors[0].available_accelerators == ["0", "1"]

    _parsl_app_runner(
        _context(tmp_path, visibility_values=()),
        tmp_path / "inherit-launch",
    )
    inherit_executor = captured["config"].executors[0]
    assert inherit_executor.max_workers_per_node == 1
    assert inherit_executor.available_accelerators == []


def test_real_parsl_reuses_one_dfk_across_sequential_dispatches(tmp_path: Path) -> None:
    pytest.importorskip("parsl")
    executor = ParslAttachExecutor()
    context = _context(tmp_path, visibility_values=())
    first = _dispatch(tmp_path)
    second = replace(first, attempt_id="attempt-2", logical_task_id="logical-2")
    try:
        assert len(executor.dispatch((first,), context=context)) == 1
        assert len(executor.dispatch((second,), context=context)) == 1
        with pytest.raises(RuntimeError, match="different AllocationContext"):
            executor.dispatch((second,), context=_context(tmp_path, allocation_id="allocation-2"))
    finally:
        executor.close()


def test_accelerator_tiling_accepts_every_width_that_partitions_the_node() -> None:
    """Every row width that tiles a node is admissible, not just one-GPU rows.

    The replaced check required exactly four bindings, which rejected `2` and `4`
    GPU rows outright and confined multi-node dispatch to one GPU per row.
    """

    for bindings in (
        ("0", "1", "2", "3"),
        ("0,1", "2,3"),
        ("0,1,2,3",),
        # No facility number is baked in, so a wider node needs no edit here.
        ("0,1,2,3", "4,5,6,7"),
    ):
        validate_accelerator_tiling(bindings)  # must not raise


@pytest.mark.parametrize(
    "bindings, fragment",
    [
        (("0,1", "1,2"), "exactly once"),
        (("0", "0"), "exactly once"),
        (("0", "2"), "exactly once"),
        (("1", "2"), "exactly once"),
        (("0,0",), "exactly once"),
        (("",), "empty"),
        ((), "at least one"),
        (("a",), "must be integers"),
    ],
)
def test_accelerator_tiling_rejects_bindings_that_do_not_partition(
    bindings: tuple[str, ...], fragment: str
) -> None:
    """A guard that accepts everything is not a fix.

    Overlapping bindings would double-book an accelerator; gappy ones do not
    describe a node. Both must still raise, or relaxing the old check would have
    bought permissiveness rather than correctness.
    """

    with pytest.raises(ValueError, match=fragment):
        validate_accelerator_tiling(bindings)
