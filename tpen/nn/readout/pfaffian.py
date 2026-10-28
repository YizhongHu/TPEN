"""Differentiable Pfaffian readout for real tuple features.

All readouts in the new TPEN core consume :class:`tpen.data.real.Feature`.
Readout-specific Fourier transforms should happen inside a component readout
before it contributes to the final wavefunction.
"""

from __future__ import annotations

import math

from tpen.data.batch import ElectronBatch, WavefunctionOutput
from tpen.data.partition import Partition
from tpen.data.real import Feature
from tpen.dependencies import require_torch, require_torch_nn
from tpen.nn.initialization import TorchInitializer

torch = require_torch(feature="Pfaffian readout")
nn = require_torch_nn(feature="Pfaffian readout")

_ODD_PADDING_IRREP = Partition((1,))
_READOUT_INITIALIZATIONS = {"equal", "signed_l1", "random_direction_l2"}


def _pfaffian_single(matrix: torch.Tensor) -> torch.Tensor:
    """Compute one Pfaffian by recursive expansion along the first row.

    Slow, unbatched reference implementation. It is deliberately retained as the
    correctness oracle for the batched routine below (CLAUDE.md, "Implement slow
    reference versions first": vectorized code must be tested against the slow
    reference). Production code paths call :func:`pfaffian`, never this.

    Parameters
    ----------
    matrix : torch.Tensor
        One skew-symmetric matrix with shape ``[n, n]``.

    Returns
    -------
    torch.Tensor
        Zero-dimensional Pfaffian.
    """

    n = matrix.shape[-1]
    if n == 0:
        return matrix.new_tensor(1.0)
    if n == 2:
        return matrix[0, 1]
    if n % 2 == 1:
        return matrix.new_tensor(0.0)
    total = matrix.new_tensor(0.0)
    remaining = torch.arange(n, device=matrix.device)
    for col in range(1, n):
        sign = 1.0 if col % 2 == 1 else -1.0
        idx = remaining[(remaining != 0) & (remaining != col)]
        submatrix = matrix.index_select(0, idx).index_select(1, idx)
        total = total + sign * matrix[0, col] * _pfaffian_single(submatrix)
    return total


def _pfaffian_batched(matrix: torch.Tensor) -> torch.Tensor:
    """Compute Pfaffians for a whole leading batch by recursive expansion.

    Term-for-term the same first-row expansion as :func:`_pfaffian_single`, with
    the same column order and the same ``(-1)**(col + 1)`` signs, but every
    arithmetic operation is applied to the full leading batch at once. The
    number of Python iterations therefore depends only on ``n`` (it is the
    double factorial ``(n - 1)!!`` of leaf terms), never on the batch size.

    At ``n == 2`` the expansion bottoms out immediately, so the whole routine is
    the single gather ``matrix[..., 0, 1]`` — the same element the reference
    returns, hence bitwise identical rather than merely close.

    Parameters
    ----------
    matrix : torch.Tensor
        Skew-symmetric matrices with shape ``[..., n, n]``. Any number of
        leading dimensions is accepted, including none.

    Returns
    -------
    torch.Tensor
        Pfaffians with the leading shape ``matrix.shape[:-2]``.
    """

    n = matrix.shape[-1]
    # An empty matrix has Pfaffian 1 by the empty-product convention, and an
    # odd-dimensional skew-symmetric matrix has Pfaffian 0. Both mirror the
    # reference exactly, broadcast over the leading batch shape.
    if n == 0:
        return matrix.new_ones(matrix.shape[:-2])
    if n == 2:
        return matrix[..., 0, 1]
    if n % 2 == 1:
        return matrix.new_zeros(matrix.shape[:-2])
    remaining = torch.arange(n, device=matrix.device)
    total = matrix.new_zeros(matrix.shape[:-2])
    for col in range(1, n):
        sign = 1.0 if col % 2 == 1 else -1.0
        idx = remaining[(remaining != 0) & (remaining != col)]
        # Drop row/column 0 and row/column ``col`` from every batch member at
        # once; ``index_select`` on the trailing axes leaves the batch intact.
        submatrix = matrix.index_select(-2, idx).index_select(-1, idx)
        total = total + sign * matrix[..., 0, col] * _pfaffian_batched(submatrix)
    return total


def pfaffian(matrix: torch.Tensor) -> torch.Tensor:
    """Compute Pfaffians for skew-symmetric matrices.

    The routine is batched over every leading dimension, so the caller does not
    pay one Python iteration per matrix. This matters because the readout runs
    inside the autodiff Laplacian's double backward, where a per-matrix Python
    loop over ``batch * channels`` matrices would be replayed by every backward
    pass.

    Parameters
    ----------
    matrix : torch.Tensor
        Skew-symmetric matrices with shape ``[..., n, n]``: a single ``[n, n]``
        matrix, a batch ``[batch, n, n]``, or any higher-rank batch such as
        ``[batch, channels, n, n]``.

    Returns
    -------
    torch.Tensor
        Pfaffians with shape ``matrix.shape[:-2]``: zero-dimensional for an
        unbatched input, ``[batch]`` for a batched input.
    """

    if matrix.ndim < 2:
        raise ValueError(f"Expected matrix rank at least 2, got shape {tuple(matrix.shape)}")
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("Pfaffian matrices must be square")
    return _pfaffian_batched(matrix)


class PfaffianReadout(nn.Module):
    """Per-channel Pfaffian readout: ``Psi = sum_c w_c Pf[K_c]``.

    Each order-2 channel is antisymmetrized into its own skew kernel
    ``K_c = 0.5 * (x_c - x_c^T)``. Odd-electron systems pad every channel
    kernel with that channel's order-1 ``(1)`` irrep block as border. The
    readout is the weighted sum of per-channel Pfaffians, not one Pfaffian
    of the channel-mixed kernel: Pf is degree-``n/2`` and nonlinear, so the
    two forms are different function classes (MIG-TPEN-000 section 2.2,
    decision B1, gate T6).

    Parameters
    ----------
    eps : float, optional
        Positive floor for signed-log conversion.
    channels, pair_channels : int
        Number of order-2 real feature channels read out. `channels` is a
        shorthand for `pair_channels`. Odd-electron padding requires the
        order-1 block to carry the same channel count.
    trainable : bool, optional
        Whether the per-channel readout weights ``w_c`` are trainable. The
        default keeps them as fixed buffers for scaffold determinism.
    initialization : {"equal", "signed_l1", "random_direction_l2"}, optional
        A12 readout-weight parameterization from design section 2.4. ``equal``
        keeps the control weights ``1 / C2``. ``signed_l1`` uses
        ``w = z / sum(abs(z))`` and ``random_direction_l2`` uses
        ``w = z / (sqrt(C2) * norm(z))`` for a float64 standard-normal vector
        ``z`` from the supplied named stream.
    initializer : TorchInitializer, optional
        Explicitly seeded named stream used by the two random initializations.
        It is required for those modes and is ignored by ``equal``.
    """

    def __init__(
        self,
        *,
        eps: float = 1.0e-12,
        channels: int | None = None,
        pair_channels: int | None = None,
        trainable: bool = False,
        initialization: str = "equal",
        initializer: TorchInitializer | None = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.trainable = bool(trainable)
        if initialization not in _READOUT_INITIALIZATIONS:
            raise ValueError(
                f"Unknown PfaffianReadout initialization {initialization!r}; "
                f"expected one of {sorted(_READOUT_INITIALIZATIONS)}"
            )
        self.initialization = initialization
        pair_channels = channels if pair_channels is None else pair_channels
        if pair_channels is None:
            raise ValueError("PfaffianReadout requires pair_channels or channels for eager initialization")
        self.pair_channels = _positive_int(pair_channels, "pair_channels")
        initial = _initial_channel_weights(
            self.pair_channels,
            initialization=initialization,
            initializer=initializer,
        )
        if self.trainable:
            self.channel_weights = nn.Parameter(initial)
            self.register_buffer("channel_weight_buffer", None, persistent=False)
        else:
            self.register_parameter("channel_weights", None)
            self.register_buffer("channel_weight_buffer", initial, persistent=False)

    def _weights(self) -> torch.Tensor:
        weight = self.channel_weights if self.trainable else self.channel_weight_buffer
        if weight is None:
            raise RuntimeError("PfaffianReadout channel weights were not eagerly initialized")
        return weight

    def build_skew_kernel(self, features: Feature, batch: ElectronBatch | None = None) -> torch.Tensor:
        """Construct the per-channel skew kernels consumed by the Pfaffians.

        Parameters
        ----------
        features : Feature
            Real feature state containing an order-2 block with shape
            ``[batch, channels, n, n]``.
        batch : ElectronBatch or None, optional
            Optional batch used only for shape checks.

        Returns
        -------
        torch.Tensor
            Skew kernels with shape ``[batch, channels, n, n]``, or the
            per-channel bordered shape ``[batch, channels, n + 1, n + 1]``
            for odd electron counts.
        """

        if 2 not in features:
            raise KeyError("PfaffianReadout requires an order-2 Feature block")
        pair = features.blocks[2]
        if pair.ndim != 4:
            raise ValueError(f"Order-2 block must have shape [batch, channels, n, n], got {tuple(pair.shape)}")
        if batch is not None and pair.shape[0] != batch.batch_size:
            raise ValueError("Feature batch size disagrees with ElectronBatch")
        if pair.shape[1] != self.pair_channels:
            raise ValueError(
                f"Order-2 block has {pair.shape[1]} channels, expected pair_channels={self.pair_channels}"
            )
        kernel = 0.5 * (pair - pair.transpose(-1, -2))
        if kernel.shape[-1] % 2 == 1:
            one_body = _odd_padding_block(features, kernel)
            bordered = kernel.new_zeros(kernel.shape[0], kernel.shape[1], kernel.shape[2] + 1, kernel.shape[3] + 1)
            bordered[..., :-1, :-1] = kernel
            bordered[..., :-1, -1] = one_body
            bordered[..., -1, :-1] = -one_body
            kernel = bordered
        return kernel

    def forward(self, features: Feature, batch: ElectronBatch) -> WavefunctionOutput:
        """Return the signed-log weighted sum of per-channel Pfaffians."""

        kernel = self.build_skew_kernel(features, batch)
        # `pfaffian` batches over every leading dimension, so the walker and
        # channel axes are passed through directly. The old flatten-to-rank-3
        # round trip existed only to feed a per-matrix Python loop.
        channel_pfaffians = pfaffian(kernel)
        psi = (channel_pfaffians * self._weights().reshape(1, -1)).sum(dim=1)
        sign = torch.sign(psi)
        logabs = torch.where(sign == 0, torch.full_like(psi, -torch.inf), 0.5 * torch.log(psi.square().clamp_min(self.eps)))
        return WavefunctionOutput(
            logabs=logabs,
            sign=sign,
            aux={"K": kernel, "channel_pfaffians": channel_pfaffians, "pfaffian": psi},
        )


def _odd_padding_block(features: Feature, kernel: torch.Tensor) -> torch.Tensor:
    if _ODD_PADDING_IRREP.order not in features:
        raise KeyError("Odd-electron Pfaffian padding requires the order-1 Feature block for irrep (1)")
    one_body = features.blocks[_ODD_PADDING_IRREP.order]
    if one_body.shape[0] != kernel.shape[0] or one_body.shape[-1] != kernel.shape[-1]:
        raise ValueError("Odd-electron (1) padding block must match order-2 batch and particle axes")
    if one_body.shape[1] == 0:
        raise KeyError("Odd-electron Pfaffian padding requires a nonempty order-1 Feature block for irrep (1)")
    if one_body.shape[1] != kernel.shape[1]:
        raise ValueError(
            "Per-channel odd-electron padding requires the order-1 block to match pair channels, "
            f"got {one_body.shape[1]} order-1 channels for {kernel.shape[1]} pair channels"
        )
    return one_body


def _positive_int(value: int, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _initial_channel_weights(
    pair_channels: int,
    *,
    initialization: str,
    initializer: TorchInitializer | None,
) -> torch.Tensor:
    """Construct readout weights for the A12 initialization coordinate."""

    if initialization == "equal":
        # Preserve the historical control tensor construction byte-for-byte.
        return torch.full((pair_channels,), 1.0 / pair_channels)
    if initializer is None or initializer.seed is None:
        raise ValueError(
            f"PfaffianReadout initialization={initialization!r} requires an explicitly seeded TorchInitializer"
        )
    z = torch.empty(pair_channels, dtype=torch.float64)
    initializer.normal_(z)
    if initialization == "signed_l1":
        weights = z / z.abs().sum()
    else:
        weights = z / (math.sqrt(pair_channels) * torch.linalg.vector_norm(z))
    return weights.to(dtype=torch.get_default_dtype())


__all__ = ["PfaffianReadout", "pfaffian"]
