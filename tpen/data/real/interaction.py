"""Path-resolved real tuple interaction state."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch

from tpen.data.equivariant_state import compare_tensor_blocks
from tpen.data.indices import common_particle_count, permute_tuple_axes
from tpen.data.permutation import Permutation
from tpen.data.paths import PathLayout
from tpen.data.real.base import _normalize_real_blocks, _validate_real_blocks


@dataclass(frozen=True)
class Interaction:
    """Store path-resolved real-space interaction blocks.

    Parameters
    ----------
    blocks : sequence of torch.Tensor
        Dense list of blocks indexed by target body order. Index 0 is reserved
        for the zero-order block and must have shape ``[batch, 0, paths]``.
        Positive-order blocks have shape
        ``[batch, channels, paths, indices...]``.

    Notes
    -----
    The interaction path/channel semantics are intentionally provisional in
    this scaffold. Validation enforces the shared batch axis, reserved
    zero-order block, and tuple-index geometry, but it does not assign meaning
    to the path axis beyond preserving it under particle permutations.
    """

    blocks: Sequence[torch.Tensor] = field(default_factory=list)
    layout: PathLayout | None = None

    def __post_init__(self) -> None:
        normalized = _normalize_real_blocks(self.blocks, prefix_ndim=3, name=type(self).__name__)
        object.__setattr__(self, "blocks", normalized)
        self.validate()

    def validate(self) -> "Interaction":
        """Validate tensor types, ranks, tuple axes, and batch consistency."""

        _validate_real_blocks(self.blocks, prefix_ndim=3, name=type(self).__name__, strict_zero_channels=True)
        if self.layout is not None:
            for order in range(1, len(self.blocks)):
                expected = self.layout.count_for_order(order)
                actual = int(self.blocks[order].shape[2])
                if actual != expected:
                    raise ValueError(
                        f"Interaction order-{order} path count {actual} does not match layout {expected}"
                    )
        return self

    @property
    def fingerprint(self) -> str | None:
        """Return the bound static layout fingerprint, if present."""

        return None if self.layout is None else self.layout.fingerprint

    @property
    def n_particles(self) -> int | None:
        """Return the shared particle count if positive-order blocks exist."""

        return common_particle_count(self.blocks, tuple_axis_start=3)

    @property
    def batch_size(self) -> int | None:
        """Return the leading batch size if any block is stored."""

        return None if not self.blocks else int(self.blocks[0].shape[0])

    def __getitem__(self, order: int) -> torch.Tensor:
        """Return one interaction block by body order."""

        return self.blocks[order]

    def __iter__(self) -> Iterator[int]:
        """Iterate over stored body orders."""

        return iter(range(len(self.blocks)))

    def items(self) -> Iterator[tuple[int, torch.Tensor]]:
        """Return ``(order, tensor)`` interaction block pairs."""

        return enumerate(self.blocks)

    def clone(self) -> "Interaction":
        """Clone every tensor block."""

        return type(self)([tensor.clone() for tensor in self.blocks], self.layout)

    def to(self, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> "Interaction":
        """Move every block to a new device or dtype."""

        return type(self)([tensor.to(device=device, dtype=dtype) for tensor in self.blocks], self.layout)

    def permute(self, permutation: Permutation) -> "Interaction":
        """Return a copy transformed by an active particle permutation."""

        return type(self)(
            [
                permute_tuple_axes(tensor, permutation, axis_start=3, order=order)
                for order, tensor in self.items()
            ], self.layout
        )

    def compare(self, other: "Interaction", *, atol: float = 1.0e-6, rtol: float = 1.0e-6) -> tuple[bool, dict[str, float]]:
        """Compare block-by-block; return ``(is_close, max_abs_error)``."""

        if type(self) is not type(other):
            return False, {"max_abs_error": float("inf")}
        if self.fingerprint != other.fingerprint:
            return False, {"max_abs_error": float("inf")}
        return compare_tensor_blocks(self.blocks, other.blocks, atol=atol, rtol=rtol)


__all__ = ["Interaction"]
