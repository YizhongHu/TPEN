"""Census and ownership checks for executor frozen records."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import importlib
import inspect
import pkgutil

import pytest

import tpen_executor
from tpen_executor._records import is_frozen_record


_P5A_RECORDS = {
    "PreparedInputBundle",
    "SourceScript",
    "PreparedPythonEnvironment",
    "MaterializedProgram",
}


def _record_classes() -> dict[str, type[object]]:
    """Discover every frozen, slotted dataclass defined by the executor."""

    modules = [tpen_executor]
    modules.extend(
        importlib.import_module(module_info.name)
        for module_info in pkgutil.walk_packages(tpen_executor.__path__, f"{tpen_executor.__name__}.")
    )
    records: dict[str, type[object]] = {}
    for module in modules:
        for candidate in vars(module).values():
            if not inspect.isclass(candidate) or candidate.__module__ != module.__name__:
                continue
            params = getattr(candidate, "__dataclass_params__", None)
            if not is_dataclass(candidate) or params is None or not params.frozen:
                continue
            if not hasattr(candidate, "__slots__"):
                continue
            records[candidate.__name__] = candidate
    return records


def test_frozen_record_census_rejects_field_and_non_field_mutation() -> None:
    """Every discovered record is slotted and rejects all mutation paths."""

    records = _record_classes()
    assert records
    assert _P5A_RECORDS <= records.keys()

    for name, record_type in records.items():
        record = object.__new__(record_type)
        assert not hasattr(record, "__dict__"), name
        record_fields = fields(record_type)
        field_name = record_fields[0].name if record_fields else "__census_field__"
        with pytest.raises(FrozenInstanceError):
            setattr(record, field_name, object())
        with pytest.raises(FrozenInstanceError):
            setattr(record, "__census_extra__", object())
        with pytest.raises(FrozenInstanceError):
            delattr(record, field_name)


def test_frozen_record_census_uses_the_owner_decorator() -> None:
    """Every discovered record must carry the owner decorator marker."""

    records = _record_classes()
    bypassed = sorted(name for name, record_type in records.items() if not is_frozen_record(record_type))
    assert not bypassed, f"records bypassing tpen_executor._records.frozen_record: {bypassed}"
