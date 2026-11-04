"""Conformance tests for prepared environment descriptors."""

from __future__ import annotations

import pytest

from tpen_executor.environments import PreparedPythonEnvironment, SourceScript


def test_source_script_round_trip_is_frozen() -> None:
    """Absolute script paths round-trip and cannot be reassigned."""

    record = SourceScript("/prepared/setup.sh")
    assert SourceScript.decode(record.encode()) == record
    assert SourceScript.from_mapping(record.to_dict()) == record
    with pytest.raises(AttributeError):
        record.path = "/other/setup.sh"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        record.extra = "not allowed"  # type: ignore[attr-defined]


def test_prepared_environment_round_trip_freezes_setup_and_env() -> None:
    """Nested setup and environment mappings survive canonical encoding."""

    record = PreparedPythonEnvironment(
        "/prepared/bin/python",
        SourceScript("/prepared/setup.sh"),
        {"OMP_NUM_THREADS": "4", "MODE": "control"},
    )
    assert PreparedPythonEnvironment.decode(record.encode()) == record
    assert record.env["MODE"] == "control"
    with pytest.raises(AttributeError):
        record.python = "/other/bin/python"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        record.extra = "not allowed"  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        record.setup.path = "/other/setup.sh"  # type: ignore[union-attr,assignment]
    with pytest.raises((TypeError, AttributeError)):
        record.env["MODE"] = "science"  # type: ignore[index]


@pytest.mark.parametrize("constructor", [SourceScript, PreparedPythonEnvironment])
@pytest.mark.parametrize("path", ["", "relative/setup.sh", "python"])
def test_environment_paths_require_nonempty_absolute_strings(constructor, path: str) -> None:
    """Descriptors reject empty and relative paths without filesystem access."""

    with pytest.raises((TypeError, ValueError)):
        constructor(path)


def test_each_environment_path_field_rejects_relative_values() -> None:
    """Every environment path-bearing field independently rejects relatives."""

    with pytest.raises(ValueError, match="python must be an absolute path"):
        PreparedPythonEnvironment("python")
    with pytest.raises(ValueError, match="path must be an absolute path"):
        PreparedPythonEnvironment("/prepared/bin/python", SourceScript("setup.sh"))


def test_environment_decoders_reject_unknown_missing_and_wrong_types() -> None:
    """Wire records have closed keys and scalar types."""

    record = PreparedPythonEnvironment("/prepared/bin/python")
    cases = []
    missing = record.to_dict(); missing.pop("env"); cases.append(missing)
    extra = record.to_dict(); extra["unexpected"] = True; cases.append(extra)
    wrong = record.to_dict(); wrong["python"] = 7; cases.append(wrong)
    wrong_setup = record.to_dict(); wrong_setup["setup"] = "setup.sh"; cases.append(wrong_setup)
    wrong_env = record.to_dict(); wrong_env["env"] = {"N": 1}; cases.append(wrong_env)
    for case in cases:
        with pytest.raises((TypeError, ValueError)):
            PreparedPythonEnvironment.from_mapping(case)
