"""Conformance tests for prepared input bundle descriptors."""

from __future__ import annotations

import pytest

from tpen_executor.delivery import PreparedInputBundle


def test_bundle_round_trip_and_optional_digest() -> None:
    """A bundle descriptor round-trips with or without a manifest digest."""

    digest = "a" * 64
    for record in (PreparedInputBundle("/prepared/inputs"), PreparedInputBundle("/prepared/inputs", digest)):
        assert PreparedInputBundle.decode(record.encode()) == record
        assert PreparedInputBundle.from_mapping(record.to_dict()) == record


def test_bundle_record_is_frozen_and_slotted() -> None:
    """Bundle descriptors reject field reassignment and new attributes."""

    record = PreparedInputBundle("/prepared/inputs")
    with pytest.raises(AttributeError):
        record.root = "/other/inputs"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        record.extra = "not allowed"  # type: ignore[attr-defined]


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
