"""Tests for virtual-support path metadata."""

from __future__ import annotations

import json
from math import factorial

import pytest

from tpen.data.paths import (
    NormalizedChannels,
    NormalizedOrders,
    OutputPathLayout,
    PathEntry,
    PathLayout,
    PathMetadata,
    SupportPath,
    VirtualPath,
    generate_virtual_paths,
    iter_path_blocks,
    validate_linear_output_plus_input_cover_support,
    validate_tp_input_cover_support,
    validate_virtual_path,
    enumerate_linear_support_paths,
)


def _coverage_count(s: int, m1: int, m2: int) -> int:
    if not max(m1, m2) <= s <= m1 + m2:
        return 0
    return factorial(s) * factorial(m1) * factorial(m2) // (
        factorial(s - m1) * factorial(m1 + m2 - s) * factorial(s - m2)
    )


def _path_count(metadata: PathMetadata, s: int, m: int, m1: int, m2: int) -> int:
    return len(metadata.get(s, m, m1, m2))


def test_path_count_canonical_matches_coverage_formula() -> None:
    metadata = PathMetadata.generate(max_order=3, max_virtual_order=3, output_embedding="canonical")

    for (s, m, m1, m2), block in iter_path_blocks(metadata.paths):
        assert len(block) == _coverage_count(s, m1, m2)

    assert _path_count(metadata, 1, 1, 1, 1) == 1
    assert _path_count(metadata, 2, 2, 1, 1) == 2
    assert _path_count(metadata, 3, 3, 2, 2) == 24


def test_path_count_full_includes_output_injections() -> None:
    metadata = PathMetadata.generate(max_order=3, max_virtual_order=3, output_embedding="full")

    for (s, m, m1, m2), block in iter_path_blocks(metadata.paths):
        expected = factorial(s) // factorial(s - m) * _coverage_count(s, m1, m2)
        assert len(block) == expected

    assert _path_count(metadata, 1, 1, 1, 1) == 1
    assert _path_count(metadata, 2, 2, 1, 1) == 4
    assert _path_count(metadata, 3, 3, 2, 2) == 144


def test_path_metadata_roundtrip_json(tmp_path) -> None:
    metadata = PathMetadata.generate(max_order=3, max_virtual_order=3, output_embedding="canonical")
    path = tmp_path / "paths_canonical.json"

    metadata.save(path)
    loaded = PathMetadata.load(path)

    assert loaded.to_json_data() == metadata.to_json_data()
    assert loaded.all_paths() == metadata.all_paths()


def test_path_metadata_rejects_legacy_dict_path_storage(tmp_path) -> None:
    path = tmp_path / "legacy_paths.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": PathMetadata.schema_version,
                "index_base": 0,
                "max_order": 1,
                "max_virtual_order": 1,
                "output_embedding": "canonical",
                "path_order_version": PathMetadata.path_order_version,
                "paths": {"1": {}},
            }
        )
    )

    with pytest.raises(TypeError, match="compact nested-list"):
        PathMetadata.load(path)


def test_path_order_stability() -> None:
    first = PathMetadata.generate(max_order=3, max_virtual_order=3, output_embedding="full")
    second = PathMetadata.generate(max_order=3, max_virtual_order=3, output_embedding="full")

    assert [path.as_tuple() for path in first.all_paths()] == [path.as_tuple() for path in second.all_paths()]
    assert [path.global_id for path in first.all_paths()] == list(range(len(first.all_paths())))


def test_paths_have_injective_maps_and_cover_virtual_support() -> None:
    metadata = PathMetadata.generate(max_order=3, max_virtual_order=3, output_embedding="full")

    for path in metadata.all_paths():
        validate_virtual_path(path, max_virtual_order=3)
        assert len(set(path.tau)) == len(path.tau)
        assert len(set(path.tau1)) == len(path.tau1)
        assert len(set(path.tau2)) == len(path.tau2)
        assert path.input_support == set(range(path.s))


def test_canonical_paths_have_fixed_tau() -> None:
    metadata = PathMetadata.generate(max_order=3, max_virtual_order=3, output_embedding="canonical")

    assert metadata.all_paths()
    assert all(path.tau == tuple(range(path.m)) for path in metadata.all_paths())


def test_full_paths_have_all_tau() -> None:
    metadata = PathMetadata.generate(max_order=3, max_virtual_order=3, output_embedding="full")
    block = metadata.get(3, 2, 2, 2)

    assert {path.tau for path in block} == {
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
    }


def test_virtual_path_validation_rejects_uncovered_support() -> None:
    path = VirtualPath(
        s=2,
        m=1,
        m1=1,
        m2=1,
        local_id=0,
        global_id=0,
        tau=(0,),
        tau1=(0,),
        tau2=(0,),
    )

    with pytest.raises(ValueError, match="cover"):
        validate_virtual_path(path, max_virtual_order=2)


def _layout() -> PathLayout:
    paths = (
        SupportPath(1, 1, (0,), (0,), "sum"),
        SupportPath(1, 1, (0,), (1,), "completion_mean"),
    )
    return PathLayout(
        outputs=(OutputPathLayout(1, tuple(PathEntry(1, 1, path) for path in reversed(paths))),),
        input_orders=NormalizedOrders((1,)),
        output_orders=NormalizedOrders((1,)),
        input_channels=NormalizedChannels(((1, 3),)),
        output_channels=NormalizedChannels(((1, 4),)),
    )


def test_path_layout_normalizes_order_count_and_fingerprint() -> None:
    first = _layout()
    second = _layout()

    assert first.counts == ((1, 2),)
    assert first.fingerprint == second.fingerprint
    assert first.outputs[0].entries[0].path.tau_in == (1,)
    assert len(first.fingerprint) == 64


def test_path_position_is_meaningful_to_layout_identity() -> None:
    first = _layout()
    reordered = PathLayout(
        outputs=(OutputPathLayout(1, tuple(reversed(first.outputs[0].entries))),),
        input_orders=first.input_orders,
        output_orders=first.output_orders,
        input_channels=first.input_channels,
        output_channels=first.output_channels,
    )

    assert first.fingerprint != reordered.fingerprint


def test_path_layout_rejects_duplicate_semantic_paths() -> None:
    path = SupportPath(1, 1, (0,), (0,))
    with pytest.raises(ValueError, match="duplicate"):
        OutputPathLayout(1, (PathEntry(1, 1, path), PathEntry(1, 1, path)))


def test_support_validators_remain_explicit() -> None:
    tp_path = VirtualPath(1, 1, 1, 1, 0, 0, (0,), (0,), (0,))
    validate_tp_input_cover_support(tp_path)

    linear_path = SupportPath(1, 1, (0,), (0,))
    validate_linear_output_plus_input_cover_support(linear_path)
    with pytest.raises(ValueError, match="canonical"):
        SupportPath(1, 1, (1,), (0,))


def test_linear_paths_use_canonical_support_labels() -> None:
    paths = enumerate_linear_support_paths(1, 1)
    assert len(paths) == 2
    assert {(path.tau_out, path.tau_in) for path in paths} == {((0,), (0,)), ((0,), (1,))}
    with pytest.raises(ValueError, match="canonical"):
        SupportPath(1, 1, (1,), (0,))
    with pytest.raises(ValueError, match="canonical"):
        SupportPath(1, 2, (0,), (2, 1))


@pytest.mark.parametrize("output_order,input_order,expected", [(1, 1, 2), (1, 2, 3), (2, 1, 3), (2, 2, 7)])
def test_complete_linear_path_counts(output_order: int, input_order: int, expected: int) -> None:
    assert len(enumerate_linear_support_paths(output_order, input_order)) == expected
