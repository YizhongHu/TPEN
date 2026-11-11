"""Activation functions owned by TPEN layer stages.

This module owns the ``Gamma`` / ``Gamma_c`` pointwise functions that
:class:`tpen.nn.EquivariantMixing`, :class:`tpen.nn.PathAggregation`, and
:class:`tpen.nn.MLP` accept through their ``activation`` argument. Per decision
D2 activation is an arbitrary function on the index axes owned by those stages,
and there is no gated activation anywhere in TPEN. Per decision D3 the first
activation form is elementwise with ``C_out == C_in``. The shipped
:class:`ChannelPreservingMLPActivation` refines that contract while retaining
``C_out == C_in``: it mixes channels inside one eager MLP per tensor order.
Channel mixing therefore belongs to the activation instance when that opt-in
module is used, while the existing pointwise activations retain their D3
behavior.

The D-series decision log lives in Task Orchestrator notes on the TPEN project
root (``decision-log-D1-D9`` and ``decision-log-D10-D18``); the former
``TPEN-MIGRATION.md`` was folded into them and removed on 2026-08-14.

Modules here are plain :class:`torch.nn.Module` maps, *not*
:class:`tpen.equivariance.EquivariantMap` subclasses. ``GaussianActivation`` is
pointwise, while ``ChannelPreservingMLPActivation`` owns its typed tensor-axis
and order layout and applies a channel-preserving, channel-mixing map.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from tpen.dependencies import require_torch, require_torch_nn
from tpen.nn.initialization import TorchInitializer
from tpen.nn.mlp import MLP

torch = require_torch(feature="TPEN activations")
nn = require_torch_nn(feature="TPEN activations")


@dataclass(frozen=True)
class ChannelActivationAxes:
    """Immutable tensor-axis contract for a channel-preserving activation.

    Parameters
    ----------
    channel_axis : int, optional
        Position of the channel axis. TPEN blocks use ``1`` by default.
    tuple_axes_start : int, optional
        First tuple-index axis. Use ``2`` for ``[B, C, N^m]`` blocks and
        ``3`` for ``[B, C, P, N^m]`` blocks.

    Raises
    ------
    ValueError
        If either axis is negative or if the tuple axes do not follow the
        channel axis.
    """

    channel_axis: int = 1
    tuple_axes_start: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.channel_axis, int) or isinstance(self.channel_axis, bool):
            raise TypeError("channel_axis must be an integer")
        if not isinstance(self.tuple_axes_start, int) or isinstance(self.tuple_axes_start, bool):
            raise TypeError("tuple_axes_start must be an integer")
        if self.channel_axis < 0:
            raise ValueError(f"channel_axis must be nonnegative, got {self.channel_axis}")
        if self.tuple_axes_start <= self.channel_axis:
            raise ValueError(
                "tuple_axes_start must be greater than channel_axis, got "
                f"{self.tuple_axes_start} <= {self.channel_axis}"
            )


@dataclass(frozen=True)
class OrderMLPSpec:
    """Immutable configuration for one order-specific channel MLP.

    The MLP is always channel-preserving: ``channels`` is used for both its
    input and output widths. ``activation`` is copied by :class:`MLP` for each
    hidden layer, just as it is for a directly constructed MLP.

    Parameters
    ----------
    order : int
        Positive tensor order selected from an input rank.
    channels : int
        Positive channel width, used for both input and output.
    hidden_channels : int, optional
        Width of each hidden layer. Defaults to ``64``.
    num_hidden_layers : int, optional
        Number of hidden layers. Defaults to ``2``.
    activation : torch.nn.Module or None, optional
        Hidden-layer activation passed to :class:`MLP`.
    bias : bool, optional
        Whether the MLP's linear layers include biases.
    """

    order: int
    channels: int
    hidden_channels: int = 64
    num_hidden_layers: int = 2
    activation: nn.Module | None = None
    bias: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            ("order", self.order),
            ("channels", self.channels),
            ("hidden_channels", self.hidden_channels),
            ("num_hidden_layers", self.num_hidden_layers),
        )
        for name, value in integer_fields:
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if self.order <= 0:
            raise ValueError(f"order must be positive, got {self.order}")
        if self.channels <= 0:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {self.hidden_channels}")
        if self.num_hidden_layers < 0:
            raise ValueError(
                f"num_hidden_layers must be nonnegative, got {self.num_hidden_layers}"
            )
        if self.activation is not None and not isinstance(self.activation, nn.Module):
            raise TypeError("activation must be a torch.nn.Module or None")
        if not isinstance(self.bias, bool):
            raise TypeError("bias must be a bool")


@dataclass(frozen=True)
class OrderMLPLayout:
    """Immutable axis and order specification for an activation instance.

    Parameters
    ----------
    axes : ChannelActivationAxes
        Channel and tuple-axis positions shared by every configured order.
    specs : tuple of OrderMLPSpec
        Non-empty, strictly increasing order specifications. Their tuple
        order is the exact order used by the activation's :class:`ModuleList`.
    """

    axes: ChannelActivationAxes
    specs: tuple[OrderMLPSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axes, ChannelActivationAxes):
            raise TypeError("axes must be a ChannelActivationAxes")
        if isinstance(self.specs, Mapping) or isinstance(self.specs, (str, bytes)):
            raise TypeError("specs must be an ordered sequence of OrderMLPSpec")
        specs = tuple(self.specs)
        if not specs:
            raise ValueError("specs must contain at least one order")
        if not all(isinstance(spec, OrderMLPSpec) for spec in specs):
            raise TypeError("specs must contain only OrderMLPSpec values")
        orders = tuple(spec.order for spec in specs)
        if orders != tuple(sorted(set(orders))):
            raise ValueError("specs must have strictly increasing, unique orders")
        object.__setattr__(self, "specs", specs)


class ChannelPreservingMLPActivation(nn.Module):
    """Apply one eager, channel-preserving MLP to each configured tensor order.

    Unlike :class:`GaussianActivation`, this map is not pointwise: each output
    channel can depend on every input channel at the same batch/path/tuple
    position. It nevertheless preserves the channel width and every inert
    position, so it is a drop-in channel-preserving activation at TPEN's
    existing callable slot.

    The input order is selected as ``input.ndim - axes.tuple_axes_start``. The
    channel axis is moved to the final position for :class:`MLP`, then moved
    back. All MLPs are constructed during initialization in the same order as
    ``layout.specs`` and are registered in ``mlps`` before the first forward.
    As with ordinary PyTorch modules, parameters are not automatically cast to
    an input's dtype or device; call ``.to(dtype=..., device=...)`` on this
    activation before passing matching tensors.

    Parameters
    ----------
    layout : OrderMLPLayout
        Immutable axis and order-specific MLP configuration.
    initializer : TorchInitializer or None, optional
        Side-effect-free initializer. Each order receives the stable child
        stream ``order_<m>``. If ``None``, :class:`MLP` uses PyTorch defaults.

    Raises
    ------
    ValueError
        If an input rank, implied order, or channel width is not represented
        by `layout`.
    """

    def __init__(
        self,
        layout: OrderMLPLayout,
        *,
        initializer: TorchInitializer | None = None,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.initializer = initializer
        self.mlps = nn.ModuleList(
            [
                MLP(
                    in_channels=spec.channels,
                    out_channels=spec.channels,
                    hidden_channels=spec.hidden_channels,
                    num_hidden_layers=spec.num_hidden_layers,
                    activation=spec.activation,
                    bias=spec.bias,
                    initializer=(
                        None
                        if initializer is None
                        else initializer.spawn(f"order_{spec.order}")
                    ),
                )
                for spec in layout.specs
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the MLP selected by the input rank, preserving input shape."""

        axes = self.layout.axes
        if inputs.ndim <= axes.tuple_axes_start:
            raise ValueError(
                "input rank must exceed tuple_axes_start to imply a positive order, "
                f"got rank={inputs.ndim}, tuple_axes_start={axes.tuple_axes_start}"
            )
        order = inputs.ndim - axes.tuple_axes_start
        spec_index = next(
            (index for index, spec in enumerate(self.layout.specs) if spec.order == order),
            None,
        )
        if spec_index is None:
            configured = tuple(spec.order for spec in self.layout.specs)
            raise ValueError(f"input order {order} is not configured; configured orders={configured}")
        spec = self.layout.specs[spec_index]
        channels = int(inputs.shape[axes.channel_axis])
        if channels != spec.channels:
            raise ValueError(
                f"input channels {channels} do not match order {order} channels {spec.channels}"
            )
        moved = inputs.movedim(axes.channel_axis, -1)
        activated = self.mlps[spec_index](moved)
        return activated.movedim(-1, axes.channel_axis)

    def extra_repr(self) -> str:
        """Return the immutable layout summary in module representations."""

        orders = tuple(spec.order for spec in self.layout.specs)
        return (
            f"channel_axis={self.layout.axes.channel_axis}, "
            f"tuple_axes_start={self.layout.axes.tuple_axes_start}, orders={orders}"
        )


class GaussianActivation(nn.Module):
    """Elementwise Gaussian activation ``f(x) = exp(-x**2 / (2 sigma**2))``.

    A shape-preserving pointwise map: every entry is transformed independently,
    so ``C_out == C_in`` and no channel, path, or particle axis is mixed (D3).
    That is what makes it safe to pass as the ``activation`` of an equivariant
    stage — an elementwise map commutes with particle permutation.

    This is deliberately *not*
    :class:`tpen.nn.coordinate_envelopes.GaussianDecayGate`, which is
    ``exp(-x / (2 sigma**2))`` and is only ever fed non-negative squared radii;
    that form diverges for negative inputs and so cannot be used on signed
    feature blocks.

    ``sigma`` is a fixed constructor float held as a plain attribute, not a
    :class:`torch.nn.Parameter`: the module has no learnable state, and the
    width is architecture metadata rather than something training moves.

    Parameters
    ----------
    sigma : float, optional
        Positive, finite Gaussian width. Defaults to ``1.0``.

    Raises
    ------
    ValueError
        If `sigma` is not positive and finite.

    Notes
    -----
    ``f(0) == 1``, so this activation is one of the ``Gamma(0) != 0`` choices
    described in :class:`tpen.nn.EquivariantMixing`: applied to a full mixing
    block it writes the invariant constant ``1`` onto the non-distinct tuple
    entries that mixing never writes.

    The mathematical range is ``(0, 1]``, but ``exp`` underflows to exactly
    ``0.0`` once ``x**2 / (2 sigma**2)`` exceeds the floating-point exponent
    range (``|x| > ~38.6`` at ``sigma == 1`` in ``float64``). Values stay
    finite and in ``[0, 1]`` for every representable input whose square is
    finite.
    """

    def __init__(self, *, sigma: float = 1.0) -> None:
        super().__init__()
        sigma = float(sigma)
        if not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"sigma must be positive and finite, got {sigma}")
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the Gaussian bump elementwise, preserving shape, dtype, device.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            ``exp(-x**2 / (2 sigma**2))`` with the same shape, dtype, and
            device as `x`.
        """

        # 2 sigma^2 is the full denominator of the Gaussian exponent; folding
        # the factor 2 in here keeps a single division on the hot path.
        scale = 2.0 * self.sigma * self.sigma
        return torch.exp(-x.square() / scale)

    def extra_repr(self) -> str:
        """Return the width in ``repr`` so configured modules are readable."""

        return f"sigma={self.sigma}"


class _SigmoidStableGrad(torch.autograd.Function):
    """Sigmoid whose double backward avoids ``1 - sigmoid`` cancellation."""

    @staticmethod
    def forward(ctx, z: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(z)
        return torch.sigmoid(z)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (z,) = ctx.saved_tensors
        return (grad_output * torch.sigmoid(z) * torch.sigmoid(-z),)


class _SoftplusBetaFunction(torch.autograd.Function):
    """Forward formula with explicit stable first and second derivatives."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, beta: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.beta = beta
        return torch.logaddexp(torch.zeros_like(x), beta * x) / beta

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        return grad_output * _SigmoidStableGrad.apply(ctx.beta * x), None


class SoftplusBeta(nn.Module):
    """Stable elementwise softplus ``logaddexp(0, beta*x) / beta``.

    This is the smooth A6 activation specified by the helium-importance scan
    design, section 2.4.  Unlike :class:`torch.nn.Softplus`, it has no finite
    threshold at which the function is replaced by a linear branch, so its
    coordinate second derivatives remain nonzero throughout the finite input
    domain.

    Parameters
    ----------
    beta : float, optional
        Positive inverse width. Defaults to ``2.0``.

    Raises
    ------
    ValueError
        If ``beta`` is not positive and finite.

    Notes
    -----
    The A6 beta-2 instance is nonzero at zero, with ``f(0) = log(2) / 2``.
    First and second coordinate derivatives use explicit sigmoid products so
    the Laplacian path remains finite beyond the exponent overflow range.
    Because the derivative path is implemented by a custom autograd Function,
    forward-mode AD and ``torch.func``/``vmap`` transforms are unsupported.
    """

    def __init__(self, *, beta: float = 2.0) -> None:
        super().__init__()
        beta = float(beta)
        if not math.isfinite(beta) or beta <= 0.0:
            raise ValueError(f"beta must be positive and finite, got {beta}")
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the stable softplus formula elementwise."""

        return _SoftplusBetaFunction.apply(x, self.beta)

    def extra_repr(self) -> str:
        """Return the inverse width in the module representation."""

        return f"beta={self.beta}"


__all__ = [
    "ChannelActivationAxes",
    "ChannelPreservingMLPActivation",
    "GaussianActivation",
    "OrderMLPLayout",
    "OrderMLPSpec",
    "SoftplusBeta",
]
