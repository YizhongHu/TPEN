"""Trainable embedding from electron batches to real tuple features."""

from __future__ import annotations

from collections.abc import Mapping

from tpen.data.batch import ElectronBatch
from tpen.data.indices import no_repeated_particle_mask, tuple_particle_inputs
from tpen.data.real import Feature, zero_block
from tpen.nn.basis import ElectronBasisFeatures
from tpen.nn.context import TPENForwardContext
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap
from tpen.nn.initialization import TorchInitializer
from tpen.nn.mlp import MLP

torch = require_torch(feature="TPEN embedding modules")
nn = require_torch_nn(feature="TPEN embedding modules")


_PAIR_ORDER = 2


class Embedding(EquivariantMap):
    """Encode raw non-repeating particle-vector tuples with per-order MLPs.

    For each body order ``m``, the input to the order-specific MLP is the
    concatenation ``(v_{i_1}, ..., v_{i_m})`` over ordered non-repeating tuple
    indices. By default ``v_i`` contains the electron coordinate and, when
    present, its spin. Extra per-particle vectors can be appended from
    ``ElectronBatch.aux``. Repeated tuple positions are stored as zeros in the
    dense :class:`Feature` output. No handcrafted geometry channels such as
    distances, radii, or inverse distances are constructed here.

    Parameters
    ----------
    max_order : int, optional
        Highest body order to return.
    spatial_dim : int
        Coordinate dimension of each particle vector.
    out_channels : int or mapping, optional
        Output channels per order for generated MLPs.
    hidden_channels : int, optional
        Hidden width for generated MLPs.
    num_hidden_layers : int, optional
        Number of hidden layers for generated MLPs.
    activation : torch.nn.Module or None, optional
        Activation copied into generated MLPs. If ``None``, SiLU is used.
    bias : bool, optional
        Whether generated MLP linear layers include bias terms.
    mlps : mapping of int to torch.nn.Module or None, optional
        Explicit per-order modules. Missing orders are filled with generated
        :class:`MLP` instances.
    include_spins : bool, optional
        If ``True``, append ``ElectronBatch.spins`` to the per-particle vector.
        Forward requires spins to be present in the batch.
    aux_feature_channels : mapping of str to int, optional
        Per-particle auxiliary feature widths keyed by ``ElectronBatch.aux``.
        Values must have shape ``[*sample_shape, n_electrons, channels]`` with
        the configured channel count.
    in_features : int or None, optional
        Explicit per-particle input width. When set, the order MLPs are sized
        for this width instead of the derived coordinate/spin/aux width; use it
        when an :class:`tpen.nn.ElectronBasis` supplies ``one_body`` features
        whose width is ``basis.out_features``.
    initializer : TorchInitializer or None, optional
        Explicit side-effect-free initializer for generated order MLPs. Supplied
        custom ``mlps`` are already constructed and are not modified.
    embedding_normalization : torch.nn.Module or None, optional
        Optional normalization applied to the embedded :class:`Feature`.
    embedding_envelope : torch.nn.Module or None, optional
        Optional context-dependent real-state envelope applied after
        ``embedding_normalization``.
    **kwargs : object
        Runtime-check options forwarded to :class:`EquivariantMap`.
    """

    def __init__(
        self,
        max_order: int = 3,
        *,
        spatial_dim: int,
        out_channels: int | Mapping[int, int] = 16,
        hidden_channels: int = 64,
        num_hidden_layers: int = 2,
        activation: nn.Module | None = None,
        bias: bool = True,
        mlps: Mapping[int, nn.Module] | None = None,
        include_spins: bool = True,
        aux_feature_channels: Mapping[str, int] | None = None,
        in_features: int | None = None,
        pair_input_channels: int = 0,
        initializer: TorchInitializer | None = None,
        embedding_envelope: nn.Module | None = None,
        embedding_normalization: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if max_order < 1:
            raise ValueError(f"max_order must be positive, got {max_order}")
        if spatial_dim <= 0:
            raise ValueError(f"spatial_dim must be positive, got {spatial_dim}")
        self.max_order = int(max_order)
        self.spatial_dim = int(spatial_dim)
        self.out_channels = {int(order): int(channels) for order, channels in out_channels.items()} if isinstance(out_channels, Mapping) else int(out_channels)
        self.include_spins = bool(include_spins)
        aux_feature_channels = {} if aux_feature_channels is None else dict(aux_feature_channels)
        self.aux_feature_channels = _normalize_aux_feature_channels(aux_feature_channels)
        # When an ElectronBasis supplies pre-built per-particle features, the
        # input width is the basis ``out_features`` rather than the raw
        # coordinate/spin/aux width. ``in_features`` overrides the derived width
        # so the order MLPs are sized for the basis path.
        derived_channels = self.spatial_dim + (1 if self.include_spins else 0) + sum(
            self.aux_feature_channels.values()
        )
        if in_features is not None:
            if int(in_features) <= 0:
                raise ValueError(f"in_features must be positive, got {in_features}")
            self.particle_input_channels = int(in_features)
        else:
            self.particle_input_channels = derived_channels
        self.in_features = None if in_features is None else int(in_features)
        # Ordered-pair ingress width. 0 (the default) means this embedding
        # consumes no pair channel, which is every historical configuration.
        self.pair_input_channels = int(pair_input_channels)
        if self.pair_input_channels < 0:
            raise ValueError(f"pair_input_channels must be non-negative, got {pair_input_channels}")
        if self.pair_input_channels and self.max_order < _PAIR_ORDER:
            raise ValueError(
                f"pair_input_channels={self.pair_input_channels} requires max_order >= "
                f"{_PAIR_ORDER}; an ordered pair has no tuple to enter at order "
                f"{self.max_order}"
            )
        self.embedding_envelope = embedding_envelope
        self.embedding_normalization = embedding_normalization
        self.order_mlps = nn.ModuleDict()
        supplied = {} if mlps is None else {int(order): module for order, module in mlps.items()}
        for order in range(1, self.max_order + 1):
            order_out_channels = self._out_channels(order)
            module = supplied.get(order)
            if module is None:
                module = MLP(
                    in_channels=self._order_in_channels(order),
                    out_channels=order_out_channels,
                    hidden_channels=hidden_channels,
                    num_hidden_layers=num_hidden_layers,
                    activation=activation,
                    bias=bias,
                    initializer=initializer.spawn(f"order_{order}") if initializer is not None else None,
                )
            self.order_mlps[str(order)] = module
        unknown = sorted(order for order in supplied if order < 1 or order > self.max_order)
        if unknown:
            raise ValueError(f"mlps contains orders outside [1, {self.max_order}]: {unknown}")

    def forward_impl(
        self,
        inputs: ElectronBatch | ElectronBasisFeatures,
        context: TPENForwardContext | None = None,
    ) -> Feature:
        """Embed electron inputs as persistent real tuple features.

        Accepts either a raw :class:`ElectronBatch` (the per-particle vector is
        built from coordinates, spins, and aux features) or an
        :class:`ElectronBasisFeatures` whose ``one_body`` tensor is used directly
        as the per-particle vector. The tuple construction, per-order MLPs, and
        repeated-particle masking are identical for both paths.
        """

        if isinstance(inputs, ElectronBasisFeatures):
            particle_vectors = inputs.one_body.reshape(-1, inputs.n_electrons, inputs.n_features)
            n_electrons = inputs.n_electrons
        else:
            flat = inputs.flatten_samples()
            particle_vectors = _particle_vectors(
                flat,
                spatial_dim=self.spatial_dim,
                include_spins=self.include_spins,
                aux_feature_channels=self.aux_feature_channels,
            )
            n_electrons = flat.n_electrons
        if particle_vectors.shape[-1] != self.particle_input_channels:
            raise ValueError(
                f"Embedding expected per-particle width {self.particle_input_channels}, "
                f"got {particle_vectors.shape[-1]}"
            )
        batch_size = int(particle_vectors.shape[0])
        device = particle_vectors.device
        dtype = particle_vectors.dtype
        blocks = [zero_block(batch_size=batch_size, device=device, dtype=dtype)]
        for order in range(1, self.max_order + 1):
            tuple_inputs = tuple_particle_inputs(particle_vectors, order)
            tuple_inputs = self._pair_tuple_inputs(
                tuple_inputs,
                inputs,
                order=order,
                batch_size=batch_size,
                n_electrons=n_electrons,
            )
            block = self.order_mlps[str(order)](tuple_inputs).movedim(-1, 1)
            block = block * no_repeated_particle_mask(n_electrons, order, device=device).reshape(
                1,
                1,
                *((n_electrons,) * order),
            ).to(dtype=block.dtype)
            blocks.append(block)
        features = Feature(blocks)
        if self.embedding_normalization is not None:
            features = self.embedding_normalization(features)
        if self.embedding_envelope is not None:
            if context is None:
                raise ValueError("embedding_envelope requires a TPENForwardContext")
            features = self.embedding_envelope(features, context)
        return features

    def _order_in_channels(self, order: int) -> int:
        """Return the MLP input width for one tuple order.

        Pair channels are added ONLY at order 2, because that is the only order
        whose tuple IS an ordered pair. The authority defines pair features and
        nothing wider, so orders above 2 receive no invented triple analogue.
        """

        width = order * self.particle_input_channels
        if order == _PAIR_ORDER:
            width += self.pair_input_channels
        return width

    def _pair_tuple_inputs(
        self,
        tuple_inputs: "torch.Tensor",
        inputs: ElectronBatch | ElectronBasisFeatures,
        *,
        order: int,
        batch_size: int,
        n_electrons: int,
    ) -> "torch.Tensor":
        """Append the ordered-pair channel to order-2 tuple inputs.

        Raises rather than silently proceeding in BOTH mismatch directions. A
        configured pair width with no pair tensor would otherwise fail later as
        a confusing shape error; an unconsumed pair tensor would silently drop
        typed features that a caller went to the trouble of producing, which is
        the same class of defect as broadcasting them in undocumented -- just
        inverted.
        """

        pair = getattr(inputs, "pair", None)

        if order != _PAIR_ORDER:
            return tuple_inputs
        if not self.pair_input_channels:
            if pair is not None:
                raise ValueError(
                    "ElectronBasisFeatures carries a pair channel but this Embedding has "
                    "pair_input_channels=0, so the pair features would be silently "
                    "discarded. Set pair_input_channels to consume them"
                )
            return tuple_inputs
        if pair is None:
            raise ValueError(
                f"Embedding has pair_input_channels={self.pair_input_channels} but the "
                "inputs carry no pair channel; a basis producing ordered pairs is required"
            )

        flat = pair.reshape(batch_size, n_electrons, n_electrons, -1)
        if flat.shape[-1] != self.pair_input_channels:
            raise ValueError(
                f"Embedding expected pair width {self.pair_input_channels}, "
                f"got {flat.shape[-1]}"
            )
        return torch.cat([tuple_inputs, flat.to(dtype=tuple_inputs.dtype)], dim=-1)

    def _out_channels(self, order: int) -> int:
        if isinstance(self.out_channels, dict):
            try:
                channels = self.out_channels[order]
            except KeyError as exc:
                raise KeyError(f"Missing out_channels for order {order}") from exc
        else:
            channels = self.out_channels
        if channels <= 0:
            raise ValueError(f"Embedding out_channels must be positive, got {channels}")
        return int(channels)


def _particle_vectors(
    batch: ElectronBatch,
    *,
    spatial_dim: int,
    include_spins: bool,
    aux_feature_channels: Mapping[str, int],
) -> torch.Tensor:
    """Return dense per-particle vectors for an electron batch."""

    if batch.spatial_dim != spatial_dim:
        raise ValueError(f"ElectronBatch spatial_dim={batch.spatial_dim} disagrees with Embedding spatial_dim={spatial_dim}")
    features = [batch.positions]
    if include_spins and batch.spins is None:
        raise ValueError("Embedding include_spins=True requires ElectronBatch.spins")
    if include_spins:
        features.append(batch.spins.unsqueeze(-1).to(dtype=batch.positions.dtype))
    for key, channels in aux_feature_channels.items():
        if key not in batch.aux:
            raise KeyError(f"ElectronBatch.aux is missing particle feature key {key!r}")
        value = batch.aux[key]
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=batch.device)
        tensor = tensor.to(device=batch.device, dtype=batch.dtype)
        if tensor.ndim != 3 or tuple(tensor.shape[:2]) != (batch.batch_size, batch.n_electrons):
            raise ValueError(
                f"ElectronBatch.aux[{key!r}] must have shape [batch, n_electrons, channels], "
                f"got {tuple(tensor.shape)}"
            )
        if int(tensor.shape[-1]) != channels:
            raise ValueError(
                f"ElectronBatch.aux[{key!r}] has {tensor.shape[-1]} channels, expected {channels}"
            )
        features.append(tensor)
    return torch.cat(features, dim=-1)


def _normalize_aux_feature_channels(value: Mapping[str, int]) -> dict[str, int]:
    normalized = {}
    for raw_key, raw_channels in value.items():
        key = str(raw_key)
        channels = int(raw_channels)
        if channels < 0:
            raise ValueError(f"aux_feature_channels[{key!r}] must be nonnegative, got {channels}")
        normalized[key] = channels
    return normalized


__all__ = [
    "matched_start_embedding",
    "transplant_raw_input_columns",
    "Embedding",
]


def matched_start_embedding(*, raw: "Embedding", augmented: "Embedding") -> "Embedding":
    r"""Return ``augmented`` after constructing a matched raw-feature start.

    Parameters
    ----------
    raw : Embedding
        The same-seed control embedding consuming only the raw per-particle
        features.
    augmented : Embedding
        The wider embedding consuming raw features plus bounded-distance
        features. Its added columns remain trainable after they are zeroed.

    Returns
    -------
    Embedding
        The ``augmented`` instance, after its shared parameters have been
        copied from ``raw`` and its added input columns have been zeroed.

    Raises
    ------
    TypeError
        If either argument is not an :class:`Embedding`.
    """

    if not isinstance(raw, Embedding) or not isinstance(augmented, Embedding):
        raise TypeError("matched_start_embedding expects two constructed Embedding instances")
    transplant_raw_input_columns(raw, augmented)
    return augmented


def transplant_raw_input_columns(source: "Embedding", target: "Embedding") -> None:
    r"""Copy `source`'s weights into `target`, zeroing `target`'s added columns.

    Implements the matched-B-seed rule: "copy shared raw-input weights exactly
    and set added input columns to zero", so that a bounded-distance embedding
    starts as the SAME FUNCTION as its raw counterpart. With the added columns
    zeroed, the two cells differ only in what they can learn, not in where they
    start -- which is what removes initial-function change as the first
    explanation for any difference between them.

    Parameters
    ----------
    source : Embedding
        The narrower embedding, consuming raw features only.
    target : Embedding
        The wider embedding, consuming the same raw features plus appended
        columns. Modified in place.

    Raises
    ------
    ValueError
        If the two do not differ only by appended input columns.

    Notes
    -----
    **The added columns are NOT contiguous, and that is the whole difficulty.**
    Order 2's input is ``[v_i, v_j, pair]``, so appending one column per
    particle inserts a zero at the END OF EACH PARTICLE BLOCK, not at the end
    of the row:

    .. code-block:: text

        source order 2 : [ v_i(w0) | v_j(w0) ]
        target order 2 : [ v_i(w0) 0 | v_j(w0) 0 | pair(p) ]

    A "zero the last k columns" implementation would therefore be wrong for
    every order above 1, while looking right for order 1 and passing any test
    that only checked widths. The mapping below is written per particle block,
    and the accompanying test checks FUNCTION EQUALITY rather than column
    positions, so an index error cannot pass by agreeing with the same
    reasoning that produced it.

    Parameters other than each order's first layer are copied verbatim: only
    the first layer's input width differs, so every later layer already has
    matching shapes.
    """

    added = target.particle_input_channels - source.particle_input_channels
    if added < 0:
        raise ValueError(
            "transplant target must be at least as wide as the source: "
            f"{target.particle_input_channels} < {source.particle_input_channels}"
        )
    if source.pair_input_channels:
        raise ValueError(
            "transplant source must consume no pair channel; it is the raw-feature "
            f"cell, but it declares pair_input_channels={source.pair_input_channels}"
        )
    if source.max_order != target.max_order:
        raise ValueError(
            f"transplant needs equal max_order, got {source.max_order} and {target.max_order}"
        )

    source_width = source.particle_input_channels
    target_width = target.particle_input_channels

    with torch.no_grad():
        for order_key, source_mlp in source.order_mlps.items():
            target_mlp = target.order_mlps[order_key]
            order = int(order_key)

            source_params = dict(source_mlp.named_parameters())
            for name, target_param in target_mlp.named_parameters():
                source_param = source_params[name]
                if source_param.shape == target_param.shape:
                    target_param.copy_(source_param)
                    continue

                # The only legitimate shape difference is the first layer's
                # input width. Anything else means these are not the same
                # architecture and silently proceeding would produce a model
                # that trains but is not the one anybody described.
                if target_param.ndim != 2 or source_param.shape[0] != target_param.shape[0]:
                    raise ValueError(
                        f"cannot transplant parameter {name!r} of order {order}: "
                        f"{tuple(source_param.shape)} vs {tuple(target_param.shape)}"
                    )

                target_param.zero_()
                for particle in range(order):
                    source_start = particle * source_width
                    target_start = particle * target_width
                    target_param[:, target_start : target_start + source_width].copy_(
                        source_param[:, source_start : source_start + source_width]
                    )
