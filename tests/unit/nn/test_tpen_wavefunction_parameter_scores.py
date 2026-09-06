"""Focused materialized parameter-score tests for :class:`TPENWaveFunction`."""

from __future__ import annotations

import io

import pytest
import torch

from tpen.data.batch import ElectronBatch, ParameterScoreForwardPacket, WavefunctionOutput
from tpen.data.paths import (
    LinearPathMetadata,
    NormalizedChannels,
    NormalizedOrders,
    PathMetadata,
    compose_path_layout,
)
from tpen.data.real import Feature
from tpen.nn import (
    CompositeMixing,
    Embedding,
    EquivariantMixing,
    InteractionMode,
    LinearEquivariantMixing,
    MaterializedParameterScoreRequest,
    PathAggregation,
    ResidualUpdater,
    TPENLayer,
    TPENWaveFunction,
)
from tpen.nn.readout import PfaffianReadout

from tests.helpers.score_reachability import disconnected_parameter_names


def _build_model(
    mode: InteractionMode,
    *,
    readout: torch.nn.Module | None = None,
) -> TPENWaveFunction:
    """Build one of the landed TP-only, linear-only, and hybrid presets."""

    input_orders = NormalizedOrders((1, 2))
    channels = NormalizedChannels(((1, 1), (2, 1)))
    linear_metadata = LinearPathMetadata.generate(max_order=2)
    tensor_metadata = PathMetadata.generate(max_order=2, max_virtual_order=2, output_embedding="canonical")
    layout = compose_path_layout(
        linear=linear_metadata if mode is not InteractionMode.TENSOR_PRODUCT else None,
        tensor_product=tensor_metadata if mode is not InteractionMode.LINEAR else None,
        input_orders=input_orders,
        output_orders=input_orders,
        input_channels=channels,
        output_channels=channels,
    )
    producers = []
    if mode is not InteractionMode.TENSOR_PRODUCT:
        producers.append(LinearEquivariantMixing(max_order=2, channels=1, metadata=linear_metadata))
    if mode is not InteractionMode.LINEAR:
        producers.append(EquivariantMixing(max_order=2, channels=1, paths=tensor_metadata, activation=None))
    mixing = CompositeMixing(layout=layout, producers=tuple(producers), activation=torch.nn.SiLU())
    aggregation = PathAggregation(max_order=2, channels=1, layout=layout, activation=torch.nn.SiLU())
    layer = TPENLayer(mixing=mixing, path_aggregation=aggregation, update=ResidualUpdater(), layout=layout)
    return TPENWaveFunction(
        embedding=Embedding(max_order=2, spatial_dim=3, out_channels=1, hidden_channels=4, num_hidden_layers=1),
        layers=(layer,),
        readout=PfaffianReadout(channels=1) if readout is None else readout,
        layout=layout,
    ).to(dtype=torch.float64)


def _batch() -> ElectronBatch:
    """Return multidimensional sample axes that the readout flattens."""

    generator = torch.Generator().manual_seed(73)
    return ElectronBatch(
        positions=torch.randn(2, 2, 3, 3, generator=generator, dtype=torch.float64),
        spins=torch.tensor([[[1.0, -1.0, 1.0]] * 2] * 2, dtype=torch.float64),
    )


class _EmptyShapeEmbedding(torch.nn.Module):
    """Pass typed input through while accepting TPEN's context argument."""

    def forward(self, value: ElectronBatch, *, context: object) -> Feature:
        del value, context
        return Feature()


class _EmptyShapeReadout(torch.nn.Module):
    """Produce an explicitly multidimensional empty primal output."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, value: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        del value
        logabs = self.weight.expand(batch.sample_shape)
        return WavefunctionOutput(logabs=logabs, sign=torch.ones_like(logabs))


def _empty_multidimensional_batch() -> ElectronBatch:
    """Return a valid empty batch with two distinct sample axes."""

    return ElectronBatch(
        positions=torch.empty(0, 2, 1, 1, dtype=torch.float64),
        spins=torch.ones(0, 2, 1, dtype=torch.float64),
    )


def _empty_multidimensional_model() -> TPENWaveFunction:
    """Build the smallest model whose readout preserves ``(0, 2)``."""

    return TPENWaveFunction(
        embedding=_EmptyShapeEmbedding(),
        readout=_EmptyShapeReadout(),
    )


@pytest.mark.parametrize("mode", tuple(InteractionMode))
def test_slow_and_chunked_scores_agree_for_all_landed_modes(mode: InteractionMode) -> None:
    model = _build_model(mode)
    batch = _batch()

    slow = model(batch, request=MaterializedParameterScoreRequest())
    chunked = model(batch, request=MaterializedParameterScoreRequest(chunk_size=2))

    assert isinstance(slow, ParameterScoreForwardPacket)
    assert isinstance(chunked, ParameterScoreForwardPacket)
    assert tuple(slow.output.logabs.shape) == (4,)
    assert slow.parameter_scores.sample_shape == (4,)
    assert slow.parameter_scores.layout.compare(chunked.parameter_scores.layout)[0]
    for parameter, slow_block, chunked_block in zip(
        model.parameter_binding.parameters,
        slow.parameter_scores.blocks,
        chunked.parameter_scores.blocks,
    ):
        assert slow_block.shape == (4, *tuple(parameter.shape))
        assert not slow_block.requires_grad
        assert not chunked_block.requires_grad
        torch.testing.assert_close(slow_block, chunked_block, rtol=1.0e-10, atol=1.0e-10)


def test_flattened_j_and_jt_products_match_ordinary_autograd() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)
    batch = _batch()
    parameters = model.parameter_binding.parameters
    sample_count = batch.batch_size
    weights = torch.linspace(0.25, 1.25, sample_count, dtype=torch.float64)

    weighted_output = model(batch)
    ordinary_jt = torch.autograd.grad(
        (weights * weighted_output.logabs.reshape(-1)).sum(),
        parameters,
    )

    ordinary_output = model(batch)
    ordinary_rows = []
    for sample_index, value in enumerate(ordinary_output.logabs.reshape(-1)):
        gradients = torch.autograd.grad(
            value,
            parameters,
            retain_graph=sample_index + 1 < sample_count,
        )
        ordinary_rows.append(torch.cat(tuple(gradient.reshape(-1) for gradient in gradients)))
    ordinary_j = torch.stack(ordinary_rows)

    packet = model(batch, request=MaterializedParameterScoreRequest(chunk_size=2))
    assert isinstance(packet, ParameterScoreForwardPacket)
    materialized_j = torch.cat(
        tuple(block.reshape(sample_count, -1) for block in packet.parameter_scores.blocks),
        dim=1,
    )
    ordinary_jt_flat = torch.cat(tuple(gradient.reshape(-1) for gradient in ordinary_jt))
    direction = torch.linspace(
        -0.4,
        0.6,
        materialized_j.shape[1],
        dtype=torch.float64,
    )

    torch.testing.assert_close(materialized_j, ordinary_j, rtol=1.0e-10, atol=1.0e-10)
    torch.testing.assert_close(materialized_j.transpose(0, 1) @ weights, ordinary_jt_flat)
    torch.testing.assert_close(materialized_j @ direction, ordinary_j @ direction)


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_empty_multidimensional_scores_keep_every_sample_axis(chunk_size: int | None) -> None:
    model = _empty_multidimensional_model()
    packet = model(
        _empty_multidimensional_batch(),
        request=MaterializedParameterScoreRequest(chunk_size=chunk_size),
    )

    assert isinstance(packet, ParameterScoreForwardPacket)
    assert tuple(packet.output.logabs.shape) == (0, 2)
    assert tuple(packet.parameter_scores.blocks[0].shape) == (0, 2)
    assert packet.parameter_scores.sample_shape == (0, 2)
    assert not packet.output.logabs.requires_grad


class _UnusedPfaffianReadout(PfaffianReadout):
    """Add a registered parameter that the inherited readout never consumes.

    Retained as a COMMITTED MUTANT, not as a fixture of historical interest.
    The score blocks now pass ``allow_unused=True`` and substitute exact zeros,
    which turns what used to be a loud runtime refusal into a silent zero, so
    the strictness moved to `test_no_parameter_is_score_disconnected_at_odd_n`
    below. A guard that nothing in the repository can fail is worth nothing;
    this class is what that guard catches. Being unread by ANY code path, it is
    disconnected at every electron count, which is precisely the property that
    distinguishes a wiring defect from the design-level even-n disconnection.
    """

    def __init__(self) -> None:
        super().__init__(channels=1)
        self.unused = torch.nn.Parameter(torch.tensor(1.0))


class _ZeroGradientPfaffianReadout(PfaffianReadout):
    """Consume an extra parameter along a path whose gradient is exactly zero.

    The counterpart to `_UnusedPfaffianReadout`, and the reason the seam reports
    substituted ordinals separately from the values it emits. Both classes yield
    an exactly-zero score column and NOTHING IN THE VALUES TELLS THEM APART.
    Here autograd reaches the parameter and computes zero; there it never
    reaches it at all. The first is correct output, the second is a defect.
    """

    def __init__(self) -> None:
        super().__init__(channels=1)
        self.zero_gradient = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        """Add the extra parameter scaled by an exact zero tensor."""

        output = super().forward(features, batch)
        # Multiplying by exact zeros keeps the parameter ON the graph, so
        # autograd returns a computed 0.0 for it rather than None.
        contribution = self.zero_gradient * torch.zeros_like(output.logabs)
        return WavefunctionOutput(
            logabs=output.logabs + contribution,
            sign=output.sign,
            phase=output.phase,
            aux=dict(output.aux),
        )


_ZEROED_LOG_PREFIX = "parameter scores substituted exact zeros for "


def _zeroed_log_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    """Return only the score-seam zero-substitution log lines."""

    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith(_ZEROED_LOG_PREFIX)
    ]


def _names_in(message: str) -> tuple[str, ...]:
    """Parse the parameter names out of one zero-substitution log line."""

    return tuple(message.split(": ", 1)[1].split(", "))


def _ordinal_of(model: TPENWaveFunction, name: str) -> int:
    """Return the layout ordinal of one named trainable parameter."""

    target = dict(model.named_parameters())[name]
    ordinals = [
        slot.ordinal
        for slot, parameter in zip(
            model.parameter_binding.layout.slots,
            model.parameter_binding.parameters,
            strict=True,
        )
        if parameter is target
    ]
    assert len(ordinals) == 1, f"expected exactly one binding slot for {name}, got {ordinals}"
    return ordinals[0]


@pytest.mark.parametrize("mode", list(InteractionMode))
def test_no_parameter_is_score_disconnected_at_odd_n(mode: InteractionMode) -> None:
    """THE RELOCATED FAIL-LOUD GUARD: at odd n every parameter must be reachable.

    The score blocks used to refuse any request containing a parameter autograd
    could not reach. That refusal was correct for a genuinely unread parameter
    and wrong for TPEN's order-1 output weights, which reach ``logabs`` ONLY
    through the odd-electron Pfaffian padding block
    (`tpen.nn.readout.pfaffian._odd_padding_block`) and so are legitimately
    disconnected at even n in a one-layer model. Since the blocks now substitute
    exact zeros, that runtime refusal is gone and this test is what replaces it.

    ODD n IS THE WHOLE POINT. At odd n the padding block is present, so the
    design-level disconnection does not occur and ANY unreachable parameter is a
    wiring defect. `_batch` is three electrons, asserted below rather than
    assumed, because the guard silently becomes vacuous at an even count.

    Parametrized over every landed interaction preset, so a producer wired into
    the layout but not into the graph fails here whichever family it belongs to.
    """

    batch = _batch()
    assert batch.n_electrons % 2 == 1, "this guard is vacuous at an even electron count"
    assert disconnected_parameter_names(_build_model(mode), batch) == frozenset()


def test_the_odd_n_guard_flags_a_genuinely_unconsumed_parameter() -> None:
    """The guard above has teeth: this model fails the property it asserts.

    The other arm of a committed mutation pair. Its subject is a DIFFERENT model
    (`_UnusedPfaffianReadout`) and its expectation is a literal name, so neither
    arm is derived from the other and each can fail while the other passes: a
    seam that reported nothing would fail this test alone, and a wiring
    regression at odd n would fail the test above alone.
    """

    model = _build_model(InteractionMode.TENSOR_PRODUCT, readout=_UnusedPfaffianReadout())
    assert disconnected_parameter_names(model, _batch()) == frozenset({"readout.unused"})


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_a_disconnected_parameter_scores_exact_zero_instead_of_failing(chunk_size: int | None) -> None:
    """The request now succeeds, and the unreachable parameter's block is 0.

    Exact zero is asserted by counting nonzeros, not by a tolerance: the score
    of a parameter with no path into ``logabs`` is analytically zero, so any
    nonzero entry at all is wrong rather than merely imprecise. A second,
    reachable parameter is required to be nonzero in the same packet, otherwise
    a seam that zeroed everything would pass.
    """

    model = _build_model(InteractionMode.TENSOR_PRODUCT, readout=_UnusedPfaffianReadout())
    packet = model(_batch(), request=MaterializedParameterScoreRequest(chunk_size=chunk_size))
    assert isinstance(packet, ParameterScoreForwardPacket)

    blocks = packet.parameter_scores.blocks
    unused_block = blocks[_ordinal_of(model, "readout.unused")]
    assert int(torch.count_nonzero(unused_block)) == 0
    # A scalar parameter contributes no trailing axes, so the block is sample-shaped.
    assert tuple(unused_block.shape) == tuple(packet.parameter_scores.sample_shape)
    assert any(int(torch.count_nonzero(block)) > 0 for block in blocks), "every block came back zero"


def test_the_two_score_routes_agree_on_a_substituted_zero() -> None:
    """Both routes substitute, and they substitute BITWISE the same thing.

    The slow route materializes one gradient per sample; the chunked route uses
    batched VJPs. They accumulate in different orders, so their ORDINARY blocks
    agree only to floating-point precision -- measured at 8.9e-16 on one block
    of this model (Cannon job 44898289), which is float64 behaving normally and
    not a defect. A SUBSTITUTED block is not a computed quantity at all, so it
    must agree exactly, and separating the two claims is what makes this test
    about the substitution rather than about summation order.
    """

    model = _build_model(InteractionMode.TENSOR_PRODUCT, readout=_UnusedPfaffianReadout())
    batch = _batch()
    substituted = disconnected_parameter_names(model, batch)
    assert substituted, "nothing was substituted, so this test would assert nothing"

    names_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    slow = model(batch, request=MaterializedParameterScoreRequest()).parameter_scores
    chunked = model(batch, request=MaterializedParameterScoreRequest(chunk_size=2)).parameter_scores
    for parameter, slow_block, chunked_block in zip(
        model.parameter_binding.parameters, slow.blocks, chunked.blocks, strict=True
    ):
        name = names_by_identity[id(parameter)]
        if name in substituted:
            assert torch.equal(slow_block, chunked_block), f"{name} substituted differently per route"
            assert int(torch.count_nonzero(slow_block)) == 0
        else:
            torch.testing.assert_close(slow_block, chunked_block, rtol=1.0e-12, atol=1.0e-12)


def test_the_zeroed_parameter_log_names_them_and_fires_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Setup logs the substituted parameters by NAME, exactly once per model.

    Names, not a count: a count cannot be checked against anything, while a name
    can be compared to an independent probe, which is the second assertion here.
    Once, because which parameters are structurally inactive is fixed by the
    architecture and the particle count, so a per-step line would bury a run log
    under an invariant fact. Two requests are issued to make "once" observable
    at all -- with a single request the assertion would hold for a seam that
    logged on every call.
    """

    model = _build_model(InteractionMode.TENSOR_PRODUCT, readout=_UnusedPfaffianReadout())
    batch = _batch()
    with caplog.at_level("INFO", logger="tpen"):
        model(batch, request=MaterializedParameterScoreRequest())
        model(batch, request=MaterializedParameterScoreRequest(chunk_size=2))

    messages = _zeroed_log_messages(caplog)
    assert len(messages) == 1, f"expected exactly one substitution log line, got {messages}"
    assert frozenset(_names_in(messages[0])) == disconnected_parameter_names(model, batch)


def test_a_genuine_zero_gradient_is_neither_substituted_nor_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A computed zero and a substituted zero are different facts.

    This is the assertion that keeps the log honest. `_ZeroGradientPfaffianReadout`
    emits a score column indistinguishable from a substituted one -- exactly
    zero, same shape, same dtype -- yet autograd DID reach the parameter, so it
    must not appear in the substitution log. If the seam ever reported zeroness
    instead of unreachability, the block assertion here would still pass and
    this test would be the only thing that failed.
    """

    model = _build_model(InteractionMode.TENSOR_PRODUCT, readout=_ZeroGradientPfaffianReadout())
    batch = _batch()
    with caplog.at_level("INFO", logger="tpen"):
        packet = model(batch, request=MaterializedParameterScoreRequest())

    block = packet.parameter_scores.blocks[_ordinal_of(model, "readout.zero_gradient")]
    assert int(torch.count_nonzero(block)) == 0, "the fixture must produce an exactly-zero column"
    assert disconnected_parameter_names(model, batch) == frozenset(), "must be graph-reachable"
    assert _zeroed_log_messages(caplog) == []


def test_parameter_reordering_is_rejected_before_building_or_updating() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)
    batch = _batch()
    before = tuple((parameter, parameter.detach().clone()) for parameter in model.parameter_binding.parameters)
    module_items = tuple(model._modules.items())
    model._modules.clear()
    for module_name, module in reversed(module_items):
        model._modules[module_name] = module

    with pytest.raises(ValueError, match="binding/layout mismatch or reordering"):
        model(batch, request=MaterializedParameterScoreRequest())

    assert all(parameter.grad is None for parameter in model.parameters())
    for parameter, prior in before:
        torch.testing.assert_close(parameter, prior, rtol=0.0, atol=0.0)


def test_score_packets_are_value_only_and_graph_bearing_packets_reject_serialization() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)
    packet = model(_batch(), request=MaterializedParameterScoreRequest(chunk_size=2))
    assert isinstance(packet, ParameterScoreForwardPacket)
    assert not packet.output.logabs.requires_grad
    assert all(not block.requires_grad for block in packet.parameter_scores.blocks)
    torch.save(packet, io.BytesIO())

    graph_output = WavefunctionOutput(
        logabs=torch.ones_like(packet.output.logabs, requires_grad=True),
        sign=packet.output.sign,
    )
    graph_packet = ParameterScoreForwardPacket(
        output=graph_output,
        parameter_scores=packet.parameter_scores,
    )
    with pytest.raises(RuntimeError, match="graph-bearing"):
        torch.save(graph_packet, io.BytesIO())


def test_parameter_binding_is_direct_and_refreshes_after_model_owned_cast() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)
    binding = model.parameter_binding

    assert all(left is right for left, right in zip(model.parameters(), binding.parameters))
    assert all(slot.dtype == torch.float64 for slot in binding.layout.slots)
    assert binding.layout.total_numel == sum(parameter.numel() for parameter in binding.parameters)

    model.to(dtype=torch.float32)
    rebound = model.parameter_binding
    assert all(left is right for left, right in zip(model.parameters(), rebound.parameters))
    assert all(slot.dtype == torch.float32 for slot in rebound.layout.slots)


def test_parameter_score_request_rejects_inference_mode() -> None:
    model = _build_model(InteractionMode.TENSOR_PRODUCT)

    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference mode"):
        model(_batch(), request=MaterializedParameterScoreRequest())
