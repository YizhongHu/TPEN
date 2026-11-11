"""Tests for immutable polymorphic wavefunction forward requests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from tpen.data.batch import (
    CoordinateForwardPacket,
    CoordinateLogGradient,
    ElectronBatch,
    MaterializedParameterLogScores,
    ParameterLayout,
    ParameterScoreForwardPacket,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.nn.forward import (
    CoordinateGradientRequest,
    CoordinateGradientProvider,
    MaterializedParameterScoreRequest,
    ParameterScoreRequest,
    ParameterScoreProvider,
    WavefunctionForwardRequest,
)


class _Provider:
    def __init__(
        self,
        coordinate_packet: CoordinateForwardPacket,
        score_packet: ParameterScoreForwardPacket,
    ) -> None:
        self.coordinate_packet = coordinate_packet
        self.score_packet = score_packet
        self.coordinate_request: CoordinateGradientRequest | None = None
        self.score_request: MaterializedParameterScoreRequest | None = None

    def evaluate_coordinate_gradient_request(
        self,
        *,
        request: CoordinateGradientRequest,
        batch: ElectronBatch,
    ) -> CoordinateForwardPacket:
        assert batch.positions.shape == (2, 3, 2)
        self.coordinate_request = request
        return self.coordinate_packet

    def evaluate_materialized_parameter_score_request(
        self,
        *,
        request: MaterializedParameterScoreRequest,
        batch: ElectronBatch,
    ) -> ParameterScoreForwardPacket:
        assert batch.positions.shape == (2, 3, 2)
        self.score_request = request
        return self.score_packet


def _packets() -> tuple[CoordinateForwardPacket, ParameterScoreForwardPacket]:
    output = WavefunctionOutput(logabs=torch.zeros(2), sign=torch.ones(2))
    coordinate_packet = CoordinateForwardPacket(
        output=output,
        coordinates=CoordinateLogGradient(values=torch.zeros(2, 3, 2)),
    )
    layout = ParameterLayout(
        slots=(ParameterSlot(ordinal=0, shape=(2,), numel=2, dtype=torch.float32),)
    )
    score_packet = ParameterScoreForwardPacket(
        output=output,
        parameter_scores=MaterializedParameterLogScores(
            layout=layout,
            blocks=(torch.zeros(2, 2),),
        ),
    )
    return coordinate_packet, score_packet


def test_requests_are_nominal_frozen_polymorphic_strategies() -> None:
    coordinate_packet, score_packet = _packets()
    provider = _Provider(coordinate_packet, score_packet)
    batch = ElectronBatch(positions=torch.zeros(2, 3, 2))
    coordinate_request = CoordinateGradientRequest()
    score_request = MaterializedParameterScoreRequest(chunk_size=4)

    assert isinstance(provider, CoordinateGradientProvider)
    assert isinstance(provider, ParameterScoreProvider)
    assert isinstance(coordinate_request, WavefunctionForwardRequest)
    assert isinstance(score_request, WavefunctionForwardRequest)
    assert isinstance(score_request, ParameterScoreRequest)
    assert coordinate_request.evaluate(provider, batch) is coordinate_packet
    assert score_request.evaluate(provider, batch) is score_packet
    assert provider.coordinate_request is coordinate_request
    assert provider.score_request is score_request

    with pytest.raises(FrozenInstanceError):
        score_request.chunk_size = 8


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_materialized_parameter_score_request_rejects_invalid_chunk_size(chunk_size: object) -> None:
    with pytest.raises(ValueError, match="positive or None"):
        MaterializedParameterScoreRequest(chunk_size=chunk_size)  # type: ignore[arg-type]
