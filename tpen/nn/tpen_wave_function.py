"""Composed TPEN wavefunction scaffold."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from collections import OrderedDict
from dataclasses import replace
from typing import NamedTuple

from tpen.data.batch import (
    CoordinateForwardPacket,
    CoordinateLogGradient,
    ElectronBatch,
    FactorizedLocalEnergyInput,
    MaterializedParameterLogScores,
    ParameterBinding,
    ParameterLayout,
    ParameterScoreForwardPacket,
    ParameterSlot,
    WavefunctionOutput,
)
from tpen.data.paths import PathLayout
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap
from tpen.nn.context import TPENForwardContext
from tpen.nn.cusp import ElectronNucleusCusp
from tpen.nn.forward import CoordinateGradientRequest, MaterializedParameterScoreRequest
from tpen.nn.tpen_layer import TPENLayer
from tpen.nn.tpen_stack import TPENStack

torch = require_torch(feature="TPEN wavefunction modules")
nn = require_torch_nn(feature="TPEN wavefunction modules")

_LOGGER = logging.getLogger("tpen")


class TPENWaveFunction(EquivariantMap):
    """Compose basis, embedding, a TPEN layer stack, readout, and post-readout factors.

    The full pipeline is::

        ElectronBatch
          -> ElectronBasis (optional)
          -> ElectronBasisFeatures
          -> embedding
          -> TPENStack (TPEN layers)
          -> readout
          -> + envelope (legacy, optional)
          -> + sum(factors) (generic, optional)

    ``envelope`` and each entry of ``factors`` are additive post-readout
    log-amplitude contributions: every one of them accepts an
    :class:`ElectronBatch` and returns a ``[batch]``-shaped tensor (the
    contract shared by both `tpen.nn.envelope.Envelope` and
    `tpen.nn.factor.LogAmplitudeFactor`). There is no mutual exclusion
    between them -- they compose in one pipeline, and either (or both) may be
    omitted for a bare readout output.

    The raw :class:`ElectronBatch` is still passed to the readout and every
    factor so they see true coordinates; the basis only re-represents the
    per-particle input to the embedding.

    Parameters
    ----------
    embedding : torch.nn.Module
        Module mapping the basis output (or, when ``basis`` is ``None``, an
        :class:`ElectronBatch`) to :class:`tpen.data.real.Feature`.
    layers : iterable of torch.nn.Module or TPENStack
        TPEN layers, or an already-constructed :class:`TPENStack`. Iterables
        are wrapped into a stack; the layers always live in ``self.stack``.
    readout : torch.nn.Module
        Module mapping final real features to :class:`WavefunctionOutput`.
    envelope : torch.nn.Module or None, optional
        Legacy additive log-amplitude envelope (e.g. `tpen.nn.AdditiveEnvelope`).
    factors : iterable of torch.nn.Module, optional
        Generic additive post-readout log-amplitude factors (e.g.
        `tpen.nn.ElectronNucleusCusp`, `tpen.nn.AdditiveCusp`).
    basis : torch.nn.Module or None, optional
        Optional :class:`tpen.nn.ElectronBasis` applied before the embedding.
        When ``None``, the embedding consumes the raw :class:`ElectronBatch`.
    layout : PathLayout or None, optional
        Model-owned immutable interaction layout. When present, its fingerprint
        is serialized with the model and checked before any restore mutation.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        *,
        embedding: nn.Module,
        layers: Iterable[nn.Module] | TPENStack = (),
        readout: nn.Module,
        envelope: nn.Module | None = None,
        factors: Iterable[nn.Module] = (),
        basis: nn.Module | None = None,
        layout: PathLayout | None = None,
        analytic_cusp_provider: ElectronNucleusCusp | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.basis = basis
        self.embedding = embedding
        self.stack = layers if isinstance(layers, TPENStack) else TPENStack(layers)
        self.layout = layout if layout is not None else _layout_from_stack(self.stack)
        if self.layout is not None:
            for layer in self.stack.layers:
                if isinstance(layer, TPENLayer) and layer.layout is not None:
                    if layer.layout.fingerprint != self.layout.fingerprint:
                        raise ValueError("TPENWaveFunction layers do not share the model layout")
        self.readout = readout
        self.envelope = envelope
        factors = tuple(factors)
        self.factors = nn.ModuleList(factors)
        if analytic_cusp_provider is not None:
            if not isinstance(analytic_cusp_provider, ElectronNucleusCusp):
                raise TypeError("analytic_cusp_provider must be an ElectronNucleusCusp")
            occurrences = sum(factor is analytic_cusp_provider for factor in factors)
            if occurrences != 1:
                raise ValueError(
                    "analytic_cusp_provider must be the unique participating ElectronNucleusCusp factor"
                )
        self.analytic_cusp_provider = analytic_cusp_provider
        # Bind direct parameter references only after every model component has
        # been registered.  The tuple order is PyTorch's deterministic module
        # traversal order, and the binding is refreshed by model-owned casts.
        self._parameter_binding = self._make_parameter_binding()
        # Latch for the once-per-instance score-inactive parameter log below.
        # A plain bool, so it is neither a parameter nor a buffer and never
        # reaches ``state_dict``.
        self._logged_zeroed_score_parameters = False

    _LAYOUT_STATE_KEY = "_tpen_layout_fingerprint"

    @property
    def parameter_binding(self) -> ParameterBinding:
        """Return the immutable direct binding captured after initialization."""

        return self._parameter_binding

    @property
    def parameter_layout(self) -> ParameterLayout:
        """Return the model-owned immutable trainable-parameter layout."""

        return self._parameter_binding.layout

    def _make_parameter_binding(self) -> ParameterBinding:
        """Capture direct trainable references in module traversal order."""

        parameters = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        slots = []
        for ordinal, parameter in enumerate(parameters):
            if isinstance(parameter, nn.parameter.UninitializedParameter):
                raise ValueError("TPENWaveFunction cannot bind an uninitialized parameter")
            slots.append(
                ParameterSlot(
                    ordinal=ordinal,
                    shape=tuple(parameter.shape),
                    numel=parameter.numel(),
                    dtype=parameter.dtype,
                )
            )
        return ParameterBinding(layout=ParameterLayout(slots=tuple(slots)), parameters=parameters)

    def _apply(self, fn, recurse=True):
        """Refresh direct references after a model-owned device or dtype cast."""

        result = super()._apply(fn, recurse=recurse)
        self._parameter_binding = self._make_parameter_binding()
        return result

    def _validate_parameter_binding(self) -> ParameterBinding:
        """Reject parameter replacement, freezing, or reordering before a forward."""

        binding = self._parameter_binding
        try:
            binding.validate()
        except (TypeError, ValueError) as exc:
            raise ValueError("parameter binding/layout is no longer valid") from exc
        current = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        if len(current) != len(binding.parameters) or not all(
            current_parameter is bound_parameter
            for current_parameter, bound_parameter in zip(current, binding.parameters)
        ):
            raise ValueError(
                "parameter binding/layout mismatch or reordering; refusing forward before update"
            )
        return binding

    def state_dict(self, *args, **kwargs):
        """Return model state with layout identity owned by the model.

        TP-only composite layers retain the historical ``mixing.weights``
        namespace. The compatibility rewrite is limited to that exact model
        shape; hybrid and linear models keep their producer-qualified keys.
        """

        state = OrderedDict(super().state_dict(*args, **kwargs))
        if self.layout is None:
            return state
        state = self._legacy_tp_state_keys(state, to_legacy=True)
        state[self._LAYOUT_STATE_KEY] = torch.tensor(
            list(self.layout.fingerprint.encode("ascii")), dtype=torch.uint8
        )
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Validate layout identity before PyTorch can mutate parameters.

        A missing identity is accepted only for a TP-only model, where the
        historical public state namespace is intentionally preserved. New
        hybrid and linear checkpoints must carry identity metadata; an old
        TP-only checkpoint cannot load into either of those changed shapes.
        """

        incoming = OrderedDict(state_dict)
        if self.layout is not None:
            encoded = incoming.get(self._LAYOUT_STATE_KEY)
            if encoded is None:
                if not self._is_tp_only_layout():
                    raise ValueError(
                        "checkpoint has no layout fingerprint; only legacy TP-only checkpoints are compatible"
                    )
            else:
                actual = bytes(encoded.detach().cpu().tolist()).decode("ascii")
                if actual != self.layout.fingerprint:
                    raise ValueError(
                        "checkpoint layout fingerprint does not match the model; refusing restore before mutation"
                    )
            incoming.pop(self._LAYOUT_STATE_KEY, None)
            incoming = self._legacy_tp_state_keys(incoming, to_legacy=False)
        return super().load_state_dict(incoming, *args, **kwargs)

    def _is_tp_only_layout(self) -> bool:
        """Return whether this model has exactly one tensor-product family."""

        return (
            self.layout is not None
            and len(self.layout.family_slices) == 1
            and self.layout.family_slices[0].family == "tensor_product"
        )

    def _legacy_tp_state_keys(self, state, *, to_legacy: bool):
        """Adapt only the nested TP-only composite state namespace."""

        if not self._is_tp_only_layout():
            return state
        rewritten = OrderedDict()
        for key, value in state.items():
            if to_legacy and ".mixing.producers.0.weights." in key:
                key = key.replace(".mixing.producers.0.weights.", ".mixing.weights.", 1)
            elif not to_legacy and ".mixing.weights." in key:
                key = key.replace(".mixing.weights.", ".mixing.producers.0.weights.", 1)
            rewritten[key] = value
        return rewritten

    def forward_impl(
        self,
        batch: ElectronBatch,
        request: CoordinateGradientRequest | MaterializedParameterScoreRequest | None = None,
    ) -> WavefunctionOutput | CoordinateForwardPacket | ParameterScoreForwardPacket:
        """Evaluate the value or an explicitly requested coordinate packet."""

        if request is None:
            return self._construct_output(batch, include_analytic_cusp=True)
        if not isinstance(request, (CoordinateGradientRequest, MaterializedParameterScoreRequest)):
            raise TypeError(f"unsupported wavefunction forward request: {type(request)!r}")
        return request.evaluate(self, batch)

    def evaluate_coordinate_gradient_request(
        self,
        *,
        request: CoordinateGradientRequest,
        batch: ElectronBatch,
    ) -> CoordinateForwardPacket:
        """Return one value output and its real-logabs coordinate gradient."""

        del request
        if torch.is_inference_mode_enabled():
            raise RuntimeError("CoordinateGradientRequest is not supported in inference mode")
        with torch.enable_grad():
            positions = batch.positions.detach().requires_grad_(True)
            gradient_batch = replace(batch, positions=positions)
            output = self._construct_output(gradient_batch, include_analytic_cusp=True)
            values = torch.autograd.grad(
                output.logabs,
                positions,
                grad_outputs=torch.ones_like(output.logabs),
                create_graph=False,
            )[0]
            values = values.reshape(*output.logabs.shape, batch.n_electrons, batch.spatial_dim)
            output = WavefunctionOutput(
                logabs=output.logabs.detach(),
                sign=output.sign.detach(),
                phase=None if output.phase is None else output.phase.detach(),
                aux=dict(output.aux),
            )
        return CoordinateForwardPacket(
            output=output,
            coordinates=CoordinateLogGradient(values=values),
        )

    def evaluate_materialized_parameter_score_request(
        self,
        *,
        request: MaterializedParameterScoreRequest,
        batch: ElectronBatch,
    ) -> ParameterScoreForwardPacket:
        """Return raw, uncentered per-sample real-logabs parameter scores.

        Score block ``i`` has shape ``(*output.logabs.shape, *parameter_i.shape)``.
        The leading shape is deliberately the model-owned primal output shape:
        TPEN's readout may flatten multidimensional input sample axes before
        producing ``logabs``, so score blocks follow that flattened shape just
        like the value/derivative packet contracts.
        """

        if torch.is_inference_mode_enabled():
            raise RuntimeError("MaterializedParameterScoreRequest is not supported in inference mode")
        binding = self._validate_parameter_binding()
        if not binding.parameters:
            raise ValueError("materialized parameter scores require at least one trainable parameter")
        with torch.enable_grad():
            output = self._construct_output(batch, include_analytic_cusp=True)
            if not output.logabs.requires_grad:
                raise RuntimeError("parameter score request requires a differentiable logabs output")
            if request.chunk_size is None:
                result = _slow_parameter_score_blocks(output.logabs, binding.parameters)
            else:
                result = _chunked_parameter_score_blocks(
                    output.logabs,
                    binding.parameters,
                    chunk_size=request.chunk_size,
                )
        self._log_zeroed_score_parameters(result.zeroed_ordinals)
        scores = MaterializedParameterLogScores(layout=binding.layout, blocks=result.blocks)
        return ParameterScoreForwardPacket(
            output=_detach_wavefunction_output(output),
            parameter_scores=scores,
        )


    def _log_zeroed_score_parameters(self, zeroed_ordinals: frozenset[int]) -> None:
        """Name the score-inactive parameters once per model instance.

        :class:`~tpen.data.batch.ParameterBinding` deliberately stores no names,
        module-member paths, or reconstruction metadata, so the score blocks can
        only report ORDINALS. Resolving those back to names belongs here, the
        one place that both owns the binding and can call
        ``named_parameters()``; identity is the join key, because a name lookup
        by value would be ambiguous between tied or equal parameters.

        Emitted once, on the first request that substitutes any zero: which
        parameters are structurally inactive is a property of the architecture
        and the particle count, so repeating it per optimizer step would flood a
        run log with an invariant fact.

        Parameters
        ----------
        zeroed_ordinals : frozenset of int
            Layout ordinals for which autograd reported no path into ``logabs``.
            An empty set is not logged and does not consume the latch.
        """

        if self._logged_zeroed_score_parameters or not zeroed_ordinals:
            return
        self._logged_zeroed_score_parameters = True
        bound = self._parameter_binding.parameters
        names_by_identity = {id(parameter): name for name, parameter in self.named_parameters()}
        names = tuple(
            names_by_identity.get(id(bound[ordinal]), f"<unnamed ordinal {ordinal}>")
            for ordinal in sorted(zeroed_ordinals)
        )
        _LOGGER.info(
            "parameter scores substituted exact zeros for %d structurally inactive parameter(s): %s",
            len(names),
            ", ".join(names),
        )

    def factorized_local_energy_input(self, batch: ElectronBatch) -> FactorizedLocalEnergyInput:
        """Return the regular output and analytic data for local-energy evaluation.

        The explicitly bound cusp is omitted while constructing the regular
        output, then queried once from that same live factor instance.
        """

        provider = self.analytic_cusp_provider
        if provider is None:
            raise ValueError("factorized local-energy input requires an analytic_cusp_provider at construction")
        regular = self._construct_output(batch, include_analytic_cusp=False)
        evaluation = provider.analytic_evaluation(batch)
        return FactorizedLocalEnergyInput(regular, evaluation)

    def _construct_output(self, batch: ElectronBatch, *, include_analytic_cusp: bool) -> WavefunctionOutput:
        output = self._readout_output(batch)
        logabs = output.logabs
        if self.envelope is not None:
            logabs = logabs + _log_factor(self.envelope, batch, logabs.shape, name="Envelope")
        for index, factor in enumerate(self.factors):
            if not include_analytic_cusp and factor is self.analytic_cusp_provider:
                continue
            logabs = logabs + _log_factor(factor, batch, logabs.shape, name=f"factors[{index}]")
        return WavefunctionOutput(logabs=logabs, sign=output.sign, phase=output.phase, aux=dict(output.aux))

    def _readout_output(self, batch: ElectronBatch) -> WavefunctionOutput:
        """Build readout output once for the public evaluation path."""

        basis_features = self.basis(batch) if self.basis is not None else None
        context = TPENForwardContext(batch=batch, basis_features=basis_features)
        embedded_input = basis_features if basis_features is not None else batch
        features = self.embedding(embedded_input, context=context)
        features = self.stack(features, context)
        return self.readout(features, batch)


def _log_factor(module: nn.Module, batch: ElectronBatch, shape: torch.Size, *, name: str) -> torch.Tensor:
    value = module(batch)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} output must be a torch.Tensor, got {type(value)!r}")
    if value.shape != shape:
        raise ValueError(f"{name} output must have shape {tuple(shape)}, got {tuple(value.shape)}")
    return value


def _detach_wavefunction_output(output: WavefunctionOutput) -> WavefunctionOutput:
    """Return the value-only output paired with a materialized score packet."""

    return WavefunctionOutput(
        logabs=output.logabs.detach(),
        sign=output.sign.detach(),
        phase=None if output.phase is None else output.phase.detach(),
        aux={
            key: value.detach() if isinstance(value, torch.Tensor) else value
            for key, value in output.aux.items()
        },
    )


class _ScoreBlockResult(NamedTuple):
    """Carry score blocks alongside the ordinals autograd could not reach.

    The two fields answer two DIFFERENT questions and must not be collapsed
    into one. ``blocks`` is what the score consumer needs; a zero column in it
    may be a computed zero or a substituted one, and nothing in the values
    distinguishes them. ``zeroed_ordinals`` records only the substituted case --
    autograd returning ``None``, meaning no path from the parameter into
    ``logabs`` exists at all. A parameter whose gradient genuinely evaluates to
    zero is absent from this set.

    Parameters
    ----------
    blocks : tuple of torch.Tensor
        One block per layout slot, shaped ``sample_shape + parameter.shape``.
    zeroed_ordinals : frozenset of int
        Layout ordinals whose blocks are substituted exact zeros.
    """

    blocks: tuple[torch.Tensor, ...]
    zeroed_ordinals: frozenset[int]


def _slow_parameter_score_blocks(
    logabs: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
) -> _ScoreBlockResult:
    """Materialize one ordinary autograd gradient per flattened sample.

    A parameter with no path into ``logabs`` receives exact zeros rather than
    failing the whole request: zero IS its score, so the numerics are unchanged
    by construction. See `_ScoreBlockResult` for why the substitution is
    reported separately from the values it produces.
    """

    sample_shape = tuple(logabs.shape)
    values = logabs.reshape(-1)
    if values.numel() == 0:
        # No autograd call is made, so nothing can have been substituted.
        return _ScoreBlockResult(
            blocks=tuple(
                logabs.new_empty(sample_shape + tuple(parameter.shape)) for parameter in parameters
            ),
            zeroed_ordinals=frozenset(),
        )
    gradients = [[] for _ in parameters]
    zeroed_ordinals: set[int] = set()
    for sample_index, value in enumerate(values):
        try:
            sample_gradients = torch.autograd.grad(
                value,
                parameters,
                retain_graph=sample_index + 1 < values.numel(),
                create_graph=False,
                allow_unused=True,
            )
        except RuntimeError as exc:
            # Retained deliberately. ``allow_unused=True`` no longer routes a
            # disconnected parameter here, but any other autograd RuntimeError
            # still must surface under this contract's wording.
            raise RuntimeError(
                "materialized parameter scores found an unused or disconnected parameter"
            ) from exc
        for ordinal, (parameter, parameter_gradients, sample_gradient) in enumerate(
            zip(parameters, gradients, sample_gradients, strict=True)
        ):
            if sample_gradient is None:
                # Union across samples: structural disconnection is the same on
                # every sample, so a per-sample disagreement would mean a
                # data-dependent graph, and reporting the ordinal is then the
                # conservative record. The substituted value stays exact either way.
                zeroed_ordinals.add(ordinal)
                sample_gradient = parameter.new_zeros(parameter.shape)
            parameter_gradients.append(sample_gradient)
    return _ScoreBlockResult(
        blocks=tuple(
            torch.stack(parameter_gradients).reshape(sample_shape + tuple(parameter.shape))
            for parameter_gradients, parameter in zip(gradients, parameters)
        ),
        zeroed_ordinals=frozenset(zeroed_ordinals),
    )


def _chunked_parameter_score_blocks(
    logabs: torch.Tensor,
    parameters: tuple[nn.Parameter, ...],
    *,
    chunk_size: int,
) -> _ScoreBlockResult:
    """Materialize score blocks using batched vector-Jacobian products.

    Applies the same exact-zero substitution as
    `_slow_parameter_score_blocks`, one chunk-shaped zero block at a time, so
    the two implementations agree on both the values and the reported ordinals.
    """

    sample_shape = tuple(logabs.shape)
    values = logabs.reshape(-1)
    if values.numel() == 0:
        # No autograd call is made, so nothing can have been substituted.
        return _ScoreBlockResult(
            blocks=tuple(
                logabs.new_empty(sample_shape + tuple(parameter.shape)) for parameter in parameters
            ),
            zeroed_ordinals=frozenset(),
        )
    gradients = [[] for _ in parameters]
    zeroed_ordinals: set[int] = set()
    for start in range(0, values.numel(), chunk_size):
        stop = min(start + chunk_size, values.numel())
        grad_outputs = values.new_zeros((stop - start, values.numel()))
        row_indices = torch.arange(stop - start, device=values.device)
        column_indices = torch.arange(start, stop, device=values.device)
        grad_outputs[row_indices, column_indices] = 1
        try:
            chunk_gradients = torch.autograd.grad(
                values,
                parameters,
                grad_outputs=grad_outputs,
                retain_graph=stop < values.numel(),
                create_graph=False,
                allow_unused=True,
                is_grads_batched=True,
            )
        except RuntimeError as exc:
            # Retained for the same reason as in the slow path above.
            raise RuntimeError(
                "materialized parameter scores found an unused or disconnected parameter"
            ) from exc
        for ordinal, (parameter, parameter_gradients, chunk_gradient) in enumerate(
            zip(parameters, gradients, chunk_gradients, strict=True)
        ):
            if chunk_gradient is None:
                zeroed_ordinals.add(ordinal)
                chunk_gradient = parameter.new_zeros((stop - start, *parameter.shape))
            parameter_gradients.append(chunk_gradient)
    return _ScoreBlockResult(
        blocks=tuple(
            torch.cat(parameter_gradients, dim=0).reshape(sample_shape + tuple(parameter.shape))
            for parameter_gradients, parameter in zip(gradients, parameters)
        ),
        zeroed_ordinals=frozenset(zeroed_ordinals),
    )


def _layout_from_stack(stack: TPENStack) -> PathLayout | None:
    """Infer one model layout from already-constructed typed TPEN layers."""

    layouts = tuple(
        layer.layout
        for layer in stack.layers
        if isinstance(layer, TPENLayer) and layer.layout is not None
    )
    if not layouts:
        return None
    first = layouts[0]
    if any(layout.fingerprint != first.fingerprint for layout in layouts[1:]):
        raise ValueError("TPENWaveFunction layers do not share one layout fingerprint")
    return first


__all__ = ["TPENWaveFunction"]
