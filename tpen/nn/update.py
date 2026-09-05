"""Real-feature update modules."""

from __future__ import annotations

from collections.abc import Mapping

from tpen.data.real import (
    Feature,
    Update,
    common_real_dtype,
    validate_matching_real_blocks,
    validate_real_update_geometry,
)
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap

torch = require_torch(feature="TPEN update modules")
nn = require_torch_nn(feature="TPEN update modules")


class Updater(EquivariantMap):
    """Base class for real-feature update rules.

    Subclasses map a persistent :class:`Feature` and a proposed
    :class:`Update` to the next persistent feature state.
    """


class ReplaceUpdater(Updater):
    """Replace persistent features with the update proposal: ``x_next = u``.

    The A8 "replacement" feature-update arm. It asks whether the initial
    embedding must survive directly, or whether each layer may discard it.

    Previously carried as experimental and unexported. Nothing about its
    behaviour changed when it was admitted -- only its status. The formula was
    already the one the study wants, which is why this is exposure rather than
    reimplementation.
    """

    def forward_impl(self, x: Feature, u: Update) -> Feature:
        """Return the update proposal as the next real feature state."""

        validate_matching_real_blocks(x, u)
        return Feature([tensor.clone() for tensor in u.blocks])


class ResidualUpdater(Updater):
    """Add a scaled real update proposal to persistent features.

    Mathematical reference: ``main.typ`` section "Updates" and the final
    ``Feature update`` line in "Model Workflow". The usual TPEN update is the
    residual rule ``x^{t+1}_I = x^t_I + a u^{t+1}_I``. Here ``step`` is the
    scalar ``a`` and ``u`` is the real-space update produced by path
    aggregation.
    """

    def __init__(self, step: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step = float(step)

    def forward_impl(self, x: Feature, u: Update) -> Feature:
        """Return ``x + step * u`` blockwise."""

        validate_matching_real_blocks(x, u)
        # Same residual formula for every body order m and tuple I; no tuple
        # positions are mixed here, so the equivariance established upstream is
        # preserved by construction.
        return Feature([left + self.step * right for left, right in zip(x.blocks, u.blocks)])


class NormGatedUpdater(Updater):
    """Experimental residual update gated by an equivariant update norm.

    This class is not part of the baseline TPEN API and is intentionally not
    exported from ``tpen.nn`` or this module's ``__all__``.

    IT IS ALSO NOT AN A8 ARM, AND MUST NOT BE SUBSTITUTED FOR ONE. The study
    admits exactly three feature updates: ``x + u``
    (:class:`ResidualUpdater`), ``u`` (:class:`ReplaceUpdater`), and
    ``x + g(R) u`` (:class:`ResidualUpdater` plus
    :class:`~tpen.nn.coordinate_envelopes.GaussianCoordinateEnvelope`). The design
    explicitly rules out an RMS or norm-gate substitute for the third.

    The confusion is easy to fall into and is why this paragraph exists: this
    class and the Gaussian arm both multiply the update by a gate in ``(0, 1)``,
    so a plot of either looks like "the update is being damped". They ask
    different questions. This gate reads the UPDATE's own magnitude, so it damps
    wherever the network is already confident. The Gaussian gate reads the
    ELECTRON COORDINATES, so it damps far from the nucleus regardless of what
    the network is doing. Only the second is a physical hypothesis.
    """

    def __init__(self, step: float = 1.0, eps: float = 1.0e-12, **kwargs) -> None:
        super().__init__(**kwargs)
        self.step = float(step)
        self.eps = float(eps)

    def forward_impl(self, x: Feature, u: Update) -> Feature:
        """Return a norm-gated residual update."""

        validate_matching_real_blocks(x, u)
        output = []
        for feature, update in zip(x.blocks, u.blocks):
            if update.shape[1] == 0:
                output.append(feature.clone())
                continue
            norm = update.square().mean(dim=1, keepdim=True).clamp_min(self.eps).sqrt()
            gate = torch.sigmoid(norm)
            output.append(feature + self.step * gate * update)
        return Feature(output)


class ChannelMappedUpdater(Updater):
    """Experimental channel-mapped real update proposal.

    This class is not part of the baseline TPEN API and is intentionally not
    exported from ``tpen.nn`` or this module's ``__all__``.

    The learned map is shared across all tuple positions within each body
    order. This preserves particle equivariance because only channel axes are
    mixed.

    Parameters
    ----------
    step : float, optional
        Scalar multiplier for the mapped update.
    max_order : int
        Maximum positive body order to initialize.
    channels : int or mapping
        Persistent feature channels per body order.
    update_channels : int, mapping, or None, optional
        Real update channels per body order. If ``None``, uses `channels`.
    initial_weight : float, optional
        Initial value for non-identity channel maps.
    identity_init : bool, optional
        If ``True``, same-size channel maps start as identity matrices.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        step: float = 1.0,
        *,
        max_order: int,
        channels: int | Mapping[int, int],
        update_channels: int | Mapping[int, int] | None = None,
        initial_weight: float = 0.0,
        identity_init: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.step = float(step)
        self.max_order = int(max_order)
        if self.max_order <= 0:
            raise ValueError(f"max_order must be positive, got {self.max_order}")
        self.channels_by_order = _normalize_positive_channels(channels, max_order=self.max_order, name="channels")
        self.update_channels_by_order = _normalize_positive_channels(
            channels if update_channels is None else update_channels,
            max_order=self.max_order,
            name="update_channels",
        )
        self.initial_weight = float(initial_weight)
        self.identity_init = bool(identity_init)
        self.channel_maps = nn.ParameterDict()
        self._initialize_channel_maps()

    def forward_impl(self, x: Feature, u: Update) -> Feature:
        """Return ``x + step * W_m u_m`` for every body order ``m``."""

        validate_real_update_geometry(x, u)
        common_real_dtype(x, u)
        output = []
        for order, (feature, update) in enumerate(zip(x.blocks, u.blocks)):
            if feature.shape[1] == 0:
                output.append(feature.clone())
                continue
            weight = self._weight_for_order(
                order,
                out_channels=int(feature.shape[1]),
                in_channels=int(update.shape[1]),
            )
            mapped = torch.einsum("oc,bc...->bo...", weight, update)
            output.append(feature + self.step * mapped)
        return Feature(output)

    def _weight_for_order(
        self,
        order: int,
        *,
        out_channels: int,
        in_channels: int,
    ) -> torch.Tensor:
        key = str(order)
        shape = (out_channels, in_channels)
        if key not in self.channel_maps:
            raise RuntimeError(f"Missing eager ChannelMappedUpdater map for order {order}")
        weight = self.channel_maps[key]
        if tuple(weight.shape) != shape:
            raise ValueError(f"Order-{order} channel map shape {tuple(weight.shape)} does not match {shape}")
        return weight

    def _initialize_channel_maps(self) -> None:
        for order in range(1, self.max_order + 1):
            shape = (self.channels_by_order[order], self.update_channels_by_order[order])
            initial = torch.full(shape, self.initial_weight)
            if self.identity_init and shape[0] == shape[1]:
                initial = torch.eye(shape[0])
            self.channel_maps[str(order)] = nn.Parameter(initial)


def _normalize_positive_channels(
    value: int | Mapping[int, int],
    *,
    max_order: int,
    name: str,
) -> dict[int, int]:
    if isinstance(value, Mapping):
        channels = {int(order): int(count) for order, count in value.items()}
        missing = [order for order in range(1, max_order + 1) if order not in channels]
        if missing:
            raise ValueError(f"{name} is missing orders {missing}")
    else:
        count = int(value)
        channels = {order: count for order in range(1, max_order + 1)}
    for order, count in channels.items():
        if order < 1 or order > max_order:
            raise ValueError(f"{name} contains order {order} outside [1, {max_order}]")
        if count <= 0:
            raise ValueError(f"{name}[{order}] must be positive, got {count}")
    return dict(sorted(channels.items()))


__all__ = ["ReplaceUpdater", "ResidualUpdater", "Updater"]
