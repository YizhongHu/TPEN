"""Golden and metamorphic tests for the public resolved-config identity."""

from __future__ import annotations

import copy

import pytest

from tpen.checkpoint.hashing import (
    RESOLVED_CONFIG_IDENTITY_SCHEMA,
    canonical_resolved_config_bytes,
    resolved_config_identity,
)


def _config() -> dict[str, object]:
    return {
        "run": {
            "root": "/tmp/a",
            "run_id": "x",
            "dir": "/tmp/a/x",
            "layout": "layout-a",
            "timezone": "America/New_York",
            "timestamp": "2026-11-09T00:00:00Z",
            "start_time_unix": 1.0,
        },
        "callbacks": [
            {"output_path": "/tmp/a/x/cb.json", "output_dir": "/tmp/a/x/cb", "name": "save"},
            {"name": "other", "output_path": "/tmp/a/x/other.json"},
        ],
        "loggers": [{"path": "/tmp/a/x/log.txt", "name": "main"}],
        "evaluator": {"tasks": [{"output_dir": "/tmp/a/x/trajectory", "metric": "energy"}]},
        "model": {"layers": 4},
        "science": {"charge": 2},
    }


def test_golden_identity_vector() -> None:
    assert canonical_resolved_config_bytes(_config()) == (
        b'{"callbacks":[{"name":"save"},{"name":"other"}],'
        b'"evaluator":{"tasks":[{"metric":"energy"}]},'
        b'"loggers":[{"name":"main"}],"model":{"layers":4},"run":{},'
        b'"science":{"charge":2}}'
    )
    assert resolved_config_identity(_config()) == {
        "schema": RESOLVED_CONFIG_IDENTITY_SCHEMA,
        "sha256": "b4d320579cc190945649a3782bea79fd9c64358473ae7485152d3b96bbd6602b",
    }


def test_identity_is_invariant_to_all_declared_run_fields_and_paths() -> None:
    original = resolved_config_identity(_config())
    changed = _config()
    changed["run"] = {
        "root": "/different/root",
        "run_id": "another-run",
        "dir": "/different/root/another-run",
        "layout": "different-layout",
        "timezone": "UTC",
        "timestamp": "later",
        "start_time_unix": 99.0,
    }
    changed["callbacks"] = [
        {"output_path": "/elsewhere/callback.json", "output_dir": "/elsewhere/callback", "name": "save"},
        {"name": "other", "output_path": "/elsewhere/other.json"},
    ]
    changed["loggers"] = [{"path": "/elsewhere/log.txt", "name": "main"}]
    changed["evaluator"] = {"tasks": [{"output_dir": "/elsewhere/trajectory", "metric": "energy"}]}
    assert resolved_config_identity(changed) == original


def test_identity_changes_for_each_science_field() -> None:
    base = _config()
    for path in (("model", "layers"), ("science", "charge"), ("evaluator", "tasks", 0, "metric")):
        changed = copy.deepcopy(base)
        cursor: object = changed
        for key in path[:-1]:
            cursor = cursor[key]  # type: ignore[index]
        cursor[path[-1]] = "changed"  # type: ignore[index]
        assert resolved_config_identity(changed) != resolved_config_identity(base), path


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_identity_rejects_nonfinite_values(value: float) -> None:
    config = {"science": {"value": value}}
    with pytest.raises(TypeError, match="non-finite"):
        resolved_config_identity(config)


def test_identity_requires_mapping_and_string_keys() -> None:
    with pytest.raises(TypeError, match="requires a mapping"):
        resolved_config_identity(["not", "a", "mapping"])
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        resolved_config_identity({1: "not-json-config"})
