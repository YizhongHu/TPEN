"""Contract tests for the stdlib-only attempt-delta reader."""

from __future__ import annotations

import importlib
import json

import pytest


delta = importlib.import_module("experiments.atomistic.he-importance.attempt_delta")


def _value() -> dict[str, object]:
    return {
        "attempt_core_dir": "/store/groups/g/t/a/core",
        "attempt_id": "a",
        "group_digest": "g",
        "load": {"mode": "fresh", "path": None},
        "schema": delta.SCHEMA,
        "task_id": "t",
    }


def _raw(value: dict[str, object]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_attempt_delta_accepts_canonical_fresh_value() -> None:
    parsed = delta.parse_attempt_delta(_raw(_value()))
    assert parsed.load_mode == "fresh"
    assert parsed.load_path is None


@pytest.mark.parametrize("mutator", [lambda value: value.update(extra=True), lambda value: value["load"].update(extra=True), lambda value: value.update(schema="wrong")])
def test_attempt_delta_rejects_closed_shape(mutator) -> None:
    value = _value()
    mutator(value)
    with pytest.raises(delta.AttemptDeltaError):
        delta.parse_attempt_delta(_raw(value))


def test_attempt_delta_rejects_noncanonical_bytes() -> None:
    with pytest.raises(delta.AttemptDeltaError, match="canonical"):
        delta.parse_attempt_delta(json.dumps(_value()))


def test_attempt_delta_accepts_resume_outside_current_leaf() -> None:
    value = _value()
    value["load"] = {"mode": "train_resume", "path": "/store/groups/g/t/previous/core/checkpoints/final"}
    assert delta.parse_attempt_delta(_raw(value)).load_mode == "train_resume"


@pytest.mark.parametrize("path", ["/store/groups/g/t/a/core/", "/store/groups/g/t/a//core", "/store/groups/g/t/a/./core", "//store/groups/g/t/a/core"])
def test_attempt_delta_rejects_non_normalized_absolute_paths(path: str) -> None:
    value = _value()
    value["attempt_core_dir"] = path
    with pytest.raises(delta.AttemptDeltaError, match="absolute and normalized"):
        delta.parse_attempt_delta(_raw(value))


def test_attempt_delta_rejects_all_paths_in_current_attempt_root() -> None:
    value = _value()
    value["load"] = {"mode": "train_resume", "path": "/store/groups/g/t/a/checkpoint"}
    with pytest.raises(delta.AttemptDeltaError, match="outside the current attempt leaf"):
        delta.parse_attempt_delta(_raw(value))
