"""Census and ownership checks for executor frozen records."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass, make_dataclass
import importlib
import inspect
import pkgutil

import pytest

import tpen_executor
from tpen_executor._records import is_frozen_record


_P5A_RECORDS = {
    ("tpen_executor.delivery", "PreparedInputBundle"),
    ("tpen_executor.environments", "SourceScript"),
    ("tpen_executor.environments", "PreparedPythonEnvironment"),
    ("tpen_executor.programs", "MaterializedProgram"),
}


def _record_classes() -> dict[tuple[str, str], type[object]]:
    """Discover every frozen, slotted dataclass defined by the executor."""

    modules = [tpen_executor]
    modules.extend(
        importlib.import_module(module_info.name)
        for module_info in pkgutil.walk_packages(tpen_executor.__path__, f"{tpen_executor.__name__}.")
    )
    records: dict[tuple[str, str], type[object]] = {}
    for module in modules:
        for candidate in vars(module).values():
            if not inspect.isclass(candidate) or candidate.__module__ != module.__name__:
                continue
            params = getattr(candidate, "__dataclass_params__", None)
            if not is_dataclass(candidate) or params is None or not params.frozen:
                continue
            if not hasattr(candidate, "__slots__"):
                continue
            records[(candidate.__module__, candidate.__qualname__)] = candidate
    return records


def test_frozen_record_census_rejects_field_and_non_field_mutation() -> None:
    """Every discovered record is slotted and rejects all mutation paths."""

    records = _record_classes()
    assert records
    discovered_names = {(record_type.__module__, record_type.__name__) for record_type in records.values()}
    assert _P5A_RECORDS <= discovered_names

    for record_type in records.values():
        record = object.__new__(record_type)
        record_name = f"{record_type.__module__}.{record_type.__qualname__}"
        assert not hasattr(record, "__dict__"), record_name
        record_fields = fields(record_type)
        field_name = record_fields[0].name if record_fields else "__census_field__"
        with pytest.raises(FrozenInstanceError):
            setattr(record, field_name, object())
        with pytest.raises(FrozenInstanceError):
            setattr(record, "__census_extra__", object())
        with pytest.raises(FrozenInstanceError):
            delattr(record, field_name)
        with pytest.raises(FrozenInstanceError):
            delattr(record, "__census_extra__")


def test_frozen_record_census_uses_the_owner_decorator() -> None:
    """Every discovered record must carry the owner decorator marker."""

    records = _record_classes()
    bypassed = sorted(
        f"{record_type.__module__}.{record_type.__qualname__}"
        for record_type in records.values()
        if not is_frozen_record(record_type)
    )
    assert not bypassed, f"records bypassing tpen_executor._records.frozen_record: {bypassed}"


def test_frozen_record_census_retains_same_named_plain_class_and_rejects_owner_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cross-module short-name collision remains visible to the owner guard."""

    module = importlib.import_module("tpen_executor.allocations")
    plain_record = make_dataclass("AttemptId", [("value", str)], frozen=True, slots=True)
    plain_record.__module__ = module.__name__
    monkeypatch.setattr(module, "AttemptId", plain_record, raising=False)

    records = _record_classes()
    assert records[("tpen_executor.allocations", "AttemptId")] is plain_record
    assert records[("tpen_executor.contracts", "AttemptId")] is not plain_record
    with pytest.raises(AssertionError, match="AttemptId"):
        test_frozen_record_census_uses_the_owner_decorator()
