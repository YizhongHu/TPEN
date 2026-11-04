"""Conformance tests for prepared input bundle descriptors."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tpen_executor.delivery import PreparedInputBundle


def test_descriptor_modules_have_no_filesystem_write_api() -> None:
    """Descriptor construction remains read-only by static inspection."""

    source_root = Path(__file__).parents[1] / "src" / "tpen_executor"
    forbidden = {"open", "write_text", "write_bytes", "mkdir", "makedirs", "remove", "unlink", "rename", "replace"}
    violations: list[str] = []
    for name in ("environments.py", "delivery.py"):
        tree = ast.parse((source_root / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in forbidden:
                violations.append(f"{name}:{node.lineno}:{node.func.attr}")
    assert not violations, "descriptor write APIs: " + ", ".join(violations)


def test_bundle_round_trip_and_optional_digest() -> None:
    """A bundle descriptor round-trips with or without a manifest digest."""

    digest = "a" * 64
    for record in (PreparedInputBundle("/prepared/inputs"), PreparedInputBundle("/prepared/inputs", digest)):
        assert PreparedInputBundle.decode(record.encode()) == record
        assert PreparedInputBundle.from_mapping(record.to_dict()) == record


@pytest.mark.parametrize("root", ["", "inputs", "./inputs"])
def test_bundle_root_requires_nonempty_absolute_path(root: str) -> None:
    """Bundle construction validates syntax but does not inspect the root."""

    with pytest.raises((TypeError, ValueError)):
        PreparedInputBundle(root)


def test_bundle_decoder_rejects_unknown_missing_and_wrong_types() -> None:
    """The bundle wire record has strict keys and digest typing."""

    record = PreparedInputBundle("/prepared/inputs")
    cases = []
    missing = record.to_dict(); missing.pop("root"); cases.append(missing)
    extra = record.to_dict(); extra["unexpected"] = True; cases.append(extra)
    wrong_root = record.to_dict(); wrong_root["root"] = 7; cases.append(wrong_root)
    wrong_digest = record.to_dict(); wrong_digest["manifest_digest"] = "bad"; cases.append(wrong_digest)
    for case in cases:
        with pytest.raises((TypeError, ValueError)):
            PreparedInputBundle.from_mapping(case)
