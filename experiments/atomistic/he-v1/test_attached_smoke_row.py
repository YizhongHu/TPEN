"""Contract tests for the minimal He attached-route row selector."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.toolkit.attached_activation import ActivationRefusal

_MODULE_PATH = Path(__file__).with_name("attached_smoke_row.py")
_SPEC = importlib.util.spec_from_file_location("he_v1_attached_smoke_row", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


def _arguments(tmp_path: Path) -> dict[str, object]:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"plan_id": "smoke", "rows": []}), encoding="utf-8")
    return {
        "manifest_path": manifest,
        "output_dir": tmp_path / "output",
        "repo_root": tmp_path / "repo",
        "results_root": tmp_path / "results",
        "source_config": tmp_path / "train.yaml",
        "row_id": "seed-000",
    }


def _main_argv(arguments: dict[str, object]) -> list[str]:
    """Build the complete explicit launch packet for ``main``."""

    flags = {
        "manifest": arguments["manifest_path"],
        "output-dir": arguments["output_dir"],
        "repo-root": arguments["repo_root"],
        "results-root": arguments["results_root"],
        "source-config": arguments["source_config"],
        "science-python": Path("/science/python"),
        "executor-python": Path("/executor/python"),
        "allocation": Path("/allocation.json"),
        "worker-config": Path("/worker.yaml"),
        "store": Path("/store"),
        "receipt": Path("/receipt.json"),
        "row-id": arguments["row_id"],
    }
    return [item for key, value in flags.items() for item in (f"--{key}", str(value))]


def test_t1_forwards_explicit_settings_and_single_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """T1 kills the mutant that drops or changes a prepare_he_train argument."""

    arguments = _arguments(tmp_path)
    calls: list[tuple[object, dict[str, object]]] = []

    def record(manifest: object, **kwargs: object) -> object:
        calls.append((manifest, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(smoke, "prepare_he_train", record)
    smoke.prepare_smoke_row(**arguments)

    assert len(calls) == 1
    manifest, kwargs = calls[0]
    assert manifest == {"plan_id": "smoke", "rows": []}
    assert kwargs == {
        "output_dir": arguments["output_dir"],
        "repo_root": arguments["repo_root"],
        "results_root": arguments["results_root"],
        "source_config": arguments["source_config"],
        "enable_attached": True,
        "enable_reduced_callbacks": True,
        "checkpoint_every_n": 25,
        "row_ids": ("seed-000",),
    }


def test_t2_prepare_refusal_propagates_and_main_does_not_convert_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T2 kills the mutant that catches ActivationRefusal or returns success."""

    arguments = _arguments(tmp_path)
    refusal = ActivationRefusal("row-selection", "declared refusal")

    def refuse(*args: object, **kwargs: object) -> object:
        raise refusal

    monkeypatch.setattr(smoke, "prepare_he_train", refuse)
    with pytest.raises(ActivationRefusal) as raised:
        smoke.main(_main_argv(arguments))
    assert raised.value is refusal


def test_t3_relative_paths_are_refused_before_prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """T3 kills the mutant that silently resolves a relative launch path."""

    arguments = _arguments(tmp_path)
    arguments["manifest_path"] = "manifest.json"
    called = False

    def record(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return SimpleNamespace()

    monkeypatch.setattr(smoke, "prepare_he_train", record)
    with pytest.raises(ValueError, match="manifest_path must be absolute"):
        smoke.prepare_smoke_row(**arguments)
    assert not called


def test_t4_module_has_no_exception_handler_or_retry_loop() -> None:
    """T4 kills mutants that add exception swallowing or retry control flow."""

    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    assert not any(isinstance(node, (ast.Try, ast.TryStar)) for node in ast.walk(tree))
    assert not any(isinstance(node, (ast.For, ast.AsyncFor, ast.While)) for node in ast.walk(tree))
