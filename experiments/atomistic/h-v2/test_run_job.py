"""Real-executor regression test for TO 8d9a31b3.

``run_job.py`` builds ``store = run_root / "attached" / "store" / task_id``
(the pilot TRAIN path and the science/pipeline-check path) and, before this
fix, never created ``store``'s parent directory. The real
``tpen_executor`` CLI's ``_reserve_store`` does a bare, non-recursive
``mkdir()`` on the store root and refuses with ``CLIRefusal("store-create",
...)`` whenever that parent is absent -- which it always is on a genuinely
fresh run root (see note ``pilot-g1-receipt-20261110`` on item
``406194c5``). This test drives the real, installed executor CLI (not a
mock) against a run root whose ``attached/store`` ancestry does not exist,
mirroring ``test_attached.py``'s
``test_the_real_executor_cli_produces_a_manifest_this_reader_accepts``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def _module(name: str):
    key = f"h_v2_{name}"
    cached = sys.modules.get(key)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


attached = _module("attached")
run_job = _module("run_job")


def _real_cli_allocation_and_worker(tmp_path: Path, *, science_python: str, placeholder_config: str) -> tuple[Path, Path]:
    """Minimal real allocation/worker-config fixtures for ``--dispatcher local``.

    Duplicated from ``test_attached.py``'s own identically-named helper
    (same ``WorkerProgram``/``PreparedPythonEnvironment``/``SubprocessWorker``
    + real ``TPENCommand``/``TPENEvidence`` runner shape) rather than
    cross-imported, so this file stays collectable on its own.
    """

    allocation_path = tmp_path / "allocation.json"
    allocation_path.write_text(
        json.dumps(
            {
                "schema": "tpen-executor/configured-allocation-v1",
                "allocation_id": "h-v2-real-cli-alloc",
                "backend_job_id": "h-v2-real-cli-job",
                "facts": {},
            }
        ),
        encoding="utf-8",
    )
    worker_path = tmp_path / "worker.yaml"
    worker_path.write_text(
        "worker:\n"
        "  _target_: tpen_executor.programs.WorkerProgram\n"
        "  _recursive_: false\n"
        "  environment:\n"
        "    _target_: tpen_executor.environments.PreparedPythonEnvironment\n"
        "    _recursive_: false\n"
        f"    python: {json.dumps(science_python)}\n"
        "    env: {}\n"
        "  process:\n"
        "    _target_: tpen_executor.workers.SubprocessWorker\n"
        "    _recursive_: false\n"
        "  runner:\n"
        "    _target_: tpen_executor.runners.TPENCommand\n"
        "    _recursive_: false\n"
        f"    config: {json.dumps(placeholder_config)}\n"
        "  evidence:\n"
        "    _target_: tpen_executor.runners.TPENEvidence\n"
        "    _recursive_: false\n",
        encoding="utf-8",
    )
    return allocation_path, worker_path


def _require_real_cli_env() -> tuple[str, str]:
    science_python = os.environ.get("H_V2_SCIENCE_PYTHON")
    executor_python = os.environ.get("H_V2_EXECUTOR_PYTHON")
    if not science_python:
        pytest.skip("named skip: H_V2_SCIENCE_PYTHON is unset")
    if not executor_python:
        pytest.skip("named skip: H_V2_EXECUTOR_PYTHON is unset")
    return science_python, executor_python


def _dispatch_against(store: Path, *, science_python: str, executor_python: str, tmp_path: Path):
    """Build a real, tiny H-v2 TRAIN task and dispatch it at *store*.

    Mirrors ``run_job.run``'s own layout (``handoff_dir``/``results_root``/
    ``store`` siblings under ``attached/``) and ``test_attached.py``'s real-CLI
    test, but lets the caller control exactly where ``store`` sits so the
    fresh-run-root case (no pre-created ``attached/store`` ancestry) can be
    exercised against the real executor.
    """

    plan, overrides, config = attached.build_h_v2_train_plan(
        output_dir=tmp_path / "handoff",
        results_root=tmp_path / "results",
        lineage="store-parent-fix",
        n_walkers=4,
        max_steps=2,
    )
    prepared = attached.prepare_h_v2_run(plan, overrides, config, output_dir=tmp_path / "handoff")
    allocation_path, worker_path = _real_cli_allocation_and_worker(
        tmp_path, science_python=science_python, placeholder_config=prepared.effective_config
    )
    return attached.run_executor_cli(
        executor_python,
        prepared.handoff,
        allocation=allocation_path,
        worker_config=worker_path,
        store=store,
        dispatcher="local",
        timeout=180.0,
    )


def test_fresh_run_root_store_parent_created_before_real_dispatch(tmp_path: Path) -> None:
    """TO 8d9a31b3 acceptance: a fresh run root, no pre-created dirs, succeeds.

    ``store``'s ancestry (``<run_root>/attached/store``) does not exist
    anywhere on disk before this test runs ``run_job.py``'s own fix
    (``store.parent.mkdir(parents=True, exist_ok=True)``) and dispatches.
    Before the fix, the real executor CLI refused with
    ``CLIRefusal("store-create", ...)`` (exit 2) exactly as the pilot g1 run
    observed on Cannon job 59560298.
    """

    science_python, executor_python = _require_real_cli_env()
    run_root = tmp_path / "fresh_run_root"  # does not exist yet -- a genuinely fresh run root
    assert not run_root.exists()
    store = run_root / "attached" / "store" / "store-parent-fix-task"
    assert not store.parent.exists()

    # The exact fix under test (run_job._reserve_store_parent): called at both
    # dispatch sites in run_job.py before every executor dispatch that passes
    # `store`, and never creates `store` itself.
    run_job._reserve_store_parent(store)
    assert not store.exists()

    dispatch = _dispatch_against(
        store, science_python=science_python, executor_python=executor_python, tmp_path=tmp_path
    )
    assert dispatch.returncode == 0, (dispatch.returncode, dispatch.stdout, dispatch.stderr)
    assert dispatch.manifest is not None
    assert dispatch.manifest["schema"] == "tpen-executor/run-manifest-v2"
    assert store.is_dir()


def test_store_exists_refusal_still_holds(tmp_path: Path) -> None:
    """The store-exists refusal is untouched: a pre-existing store still refuses.

    This guards against a fix that over-corrects by also pre-creating
    ``store`` itself (the executor's atomic non-recursive ``store.mkdir()``
    reservation must still see the fresh root as absent).
    """

    science_python, executor_python = _require_real_cli_env()
    run_root = tmp_path / "fresh_run_root"
    store = run_root / "attached" / "store" / "store-parent-fix-task"
    store.mkdir(parents=True, exist_ok=True)  # pre-exists -- deliberately violates the fresh-store invariant

    dispatch = _dispatch_against(
        store, science_python=science_python, executor_python=executor_python, tmp_path=tmp_path
    )
    assert dispatch.returncode == 2, (dispatch.returncode, dispatch.stdout, dispatch.stderr)
    assert dispatch.manifest is None
    assert "store-exists" in dispatch.stderr
