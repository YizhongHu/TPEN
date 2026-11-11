"""Contract tests for the rank-local VMC statistics reducer seam."""

from __future__ import annotations

import pytest
import torch

from tpen.training.statistics import (
    IdentityStatisticsReducer,
    StatisticsReducer,
    StatisticsSums,
    center_statistics,
)


class _FakeReducer(StatisticsReducer):
    """Minimal fake proving the seam owns count and sum operations."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def reduce_count(self, count: int) -> int:
        self.calls.append(("count", count))
        return count + 10

    def reduce_sum(self, value: torch.Tensor) -> torch.Tensor:
        self.calls.append(("sum", value))
        return value + 1.0


def test_fake_reducer_contract_exposes_count_and_aggregate_sums() -> None:
    local = StatisticsSums(
        count=3,
        sums=(torch.tensor([2.0, 4.0], dtype=torch.float64),),
    )
    reducer = _FakeReducer()

    reduced = reducer.reduce(local)

    assert reduced.count == 13
    assert torch.equal(reduced.sums[0], torch.tensor([3.0, 5.0], dtype=torch.float64))
    assert reducer.calls[0] == ("count", 3)
    assert reducer.calls[1][0] == "sum"
    assert torch.equal(reducer.calls[1][1], local.sums[0])


def test_identity_reduction_is_single_process_behavior_without_centering() -> None:
    raw_score_rows = torch.tensor(
        [[1.0, 3.0], [5.0, 7.0]],
        dtype=torch.float64,
    )
    local = StatisticsSums(count=2, sums=(raw_score_rows.sum(dim=0),))

    reduced = IdentityStatisticsReducer().reduce(local)

    assert reduced.count == local.count
    assert reduced.sums[0] is local.sums[0]
    assert torch.equal(reduced.sums[0], raw_score_rows.sum(dim=0))
    # A reducer receives aggregate sums only.  Raw score rows are unchanged
    # until the score consumer explicitly owns this centering operation.
    assert torch.equal(raw_score_rows, torch.tensor([[1.0, 3.0], [5.0, 7.0]], dtype=torch.float64))
    centered = center_statistics(
        raw_score_rows,
        count=reduced.count,
        total=reduced.sums[0],
    )
    assert torch.equal(centered, torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=torch.float64))


def test_statistics_sums_rejects_nonfloating_aggregates() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        StatisticsSums(count=1, sums=(torch.ones(1, dtype=torch.int64),))
