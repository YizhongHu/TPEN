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
        },
        "callbacks": [
            {"_target_": "tpen.callback.ConfigSnapshot", "output_path": "/tmp/a/x/cb.json", "name": "save"},
            {"_target_": "tpen.callback.Status", "output_path": "/tmp/a/x/other.json", "name": "other"},
        ],
        "loggers": [{"_target_": "tpen.logging.JSONL", "path": "/tmp/a/x/log.txt", "name": "main"}],
        "evaluator": {"_target_": "tpen.evaluation.Evaluator", "tasks": [{"output_dir": "/tmp/a/x/trajectory", "metric": "energy"}]},
        "model": {"layers": 4},
        "science": {"charge": 2},
    }


def test_golden_identity_vector() -> None:
    assert canonical_resolved_config_bytes(_config()) == (
        b'{"callbacks":[{"_target_":"tpen.callback.ConfigSnapshot","name":"save"},'
        b'{"_target_":"tpen.callback.Status","name":"other"}],'
        b'"evaluator":{"_target_":"tpen.evaluation.Evaluator","tasks":[{"metric":"energy"}]},'
        b'"experiment":{"run_name":"tpen_run"},'
        b'"loggers":[{"_target_":"tpen.logging.JSONL","name":"main"}],"model":{"layers":4},"run":{},'
        b'"science":{"charge":2}}'
    )
    assert resolved_config_identity(_config()) == {
        "schema": RESOLVED_CONFIG_IDENTITY_SCHEMA,
        "sha256": "b58ece8d8f791a7954277ae4879695fb1fa196b8badd56539b891e996d46fc6a",
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
    }
    changed["callbacks"] = [
        {"_target_": "tpen.callback.ConfigSnapshot", "output_path": "/elsewhere/callback.json", "name": "save"},
        {"_target_": "tpen.callback.Status", "name": "other", "output_path": "/elsewhere/other.json"},
    ]
    changed["loggers"] = [{"_target_": "tpen.logging.JSONL", "path": "/elsewhere/log.txt", "name": "main"}]
    changed["evaluator"] = {"_target_": "tpen.evaluation.Evaluator", "tasks": [{"output_dir": "/elsewhere/trajectory", "metric": "energy"}]}
    assert resolved_config_identity(changed) == original


def test_identity_keeps_unproduced_run_like_fields_science_relevant() -> None:
    base = _config()
    changed = copy.deepcopy(base)
    changed["run"]["timestamp"] = "later"  # type: ignore[index]
    changed["run"]["start_time_unix"] = 99.0  # type: ignore[index]
    assert resolved_config_identity(changed) != resolved_config_identity(base)


def test_identity_changes_for_each_science_field() -> None:
    base = _config()
    for path in (("model", "layers"), ("science", "charge"), ("evaluator", "tasks", 0, "metric")):
        changed = copy.deepcopy(base)
        cursor: object = changed
        for key in path[:-1]:
            cursor = cursor[key]  # type: ignore[index]
        cursor[path[-1]] = "changed"  # type: ignore[index]
        assert resolved_config_identity(changed) != resolved_config_identity(base), path


def test_identity_resolves_run_derived_evaluator_copies_before_scrubbing() -> None:
    intended = {
        "experiment": {"name": "eval"},
        "run": {"dir": None, "run_id": None},
        "runner": {"_target_": "tpen.runner.Evaluate", "evaluator": "${evaluator}"},
        "evaluator": {
            "_target_": "tpen.evaluation.Evaluator",
            "tasks": [{"name": "trajectory", "output_dir": "${run.dir}/trajectory", "metric": "energy"}],
        },
        "science": {"charge": 1},
    }
    actual = copy.deepcopy(intended)
    actual["run"] = {"dir": "/different/run", "run_id": "different"}
    assert resolved_config_identity(intended) == resolved_config_identity(actual)


def test_identity_applies_core_run_name_default_on_both_sides() -> None:
    intended = {"experiment": {"name": "eval"}, "science": {"charge": 1}}
    actual = {"experiment": {"name": "eval", "run_name": "eval"}, "run": {"run_id": "id", "dir": "/tmp/id"}, "science": {"charge": 1}}
    assert resolved_config_identity(intended) == resolved_config_identity(actual)


def test_identity_handles_tuple_operational_slots_like_lists() -> None:
    listed = _config()
    tupled = copy.deepcopy(listed)
    tupled["callbacks"] = tuple(tupled["callbacks"])  # type: ignore[assignment]
    tupled["loggers"] = tuple(tupled["loggers"])  # type: ignore[assignment]
    assert resolved_config_identity(tupled) == resolved_config_identity(listed)


def test_identity_rejects_run_field_interpolation_into_science() -> None:
    with pytest.raises(ValueError, match="would feed science"):
        resolved_config_identity({"science": {"seed": "${run.run_id}"}})


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
