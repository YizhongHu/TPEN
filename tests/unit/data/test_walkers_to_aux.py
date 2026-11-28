"""Tests for moving auxiliary walker state."""

from __future__ import annotations

import torch

from tpen.data.batch import Walkers


def test_walkers_to_moves_nested_aux_tensors_without_mutating_source() -> None:
    aux = {
        "accepted": torch.tensor([True, False], dtype=torch.bool),
        "attempts": torch.tensor([1, 2], dtype=torch.int64),
        "log_accept_ratio": torch.tensor([0.25, -0.5], dtype=torch.float64),
        "nested": (torch.tensor([1.0], dtype=torch.float64), ["metadata"]),
        "tag": "sampler",
    }
    walkers = Walkers(positions=torch.zeros(2, 1, 3, dtype=torch.float64), aux=aux)

    moved = walkers.to(dtype=torch.float32)

    assert moved is not walkers
    assert moved.aux is not walkers.aux
    assert moved.aux["accepted"].dtype == torch.bool
    assert moved.aux["attempts"].dtype == torch.int64
    assert moved.aux["log_accept_ratio"].dtype == torch.float32
    assert moved.aux["nested"][0].dtype == torch.float32
    assert moved.aux["nested"][1] == ["metadata"]
    assert moved.aux["tag"] == "sampler"

    # The source containers and tensor dtypes remain unchanged.
    assert walkers.aux["log_accept_ratio"].dtype == torch.float64
    assert walkers.aux["nested"][0].dtype == torch.float64
    assert walkers.aux["nested"] is not moved.aux["nested"]
