"""Static composition of unary and tensor-product interaction producers."""

from __future__ import annotations

from collections.abc import Sequence
from collections import OrderedDict

from tpen.data.paths import PathLayout
from tpen.data.real import Feature, Interaction, common_real_batch_size, common_real_dtype
from tpen.dependencies import require_torch, require_torch_nn
from tpen.equivariance import EquivariantMap
from tpen.nn.equivariant_mixing import EquivariantMixing
from tpen.nn.linear_equivariant_mixing import LinearEquivariantMixing

torch = require_torch(feature="TPEN composite mixing")
nn = require_torch_nn(feature="TPEN composite mixing")


class CompositeMixing(EquivariantMap):
    """Concatenate ordered producer families under one common activation.

    Parameters
    ----------
    layout : PathLayout
        Immutable union layout shared with :class:`PathAggregation`.
    producers : sequence of LinearEquivariantMixing or EquivariantMixing
        Ordered concrete producers. Linear paths must precede tensor-product
        paths, and all producer parameters already exist at construction.
    activation : torch.nn.Module, callable, or None, optional
        The one common pointwise ``Gamma`` applied after concatenation.
    """

    def __init__(
        self,
        *,
        layout: PathLayout,
        producers: Sequence[LinearEquivariantMixing | EquivariantMixing],
        activation=None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if not layout.family_slices:
            raise ValueError("CompositeMixing requires family slices in PathLayout")
        self.layout = layout
        self.activation = activation
        self.producers = nn.ModuleList(tuple(producers))
        if not self.producers:
            raise ValueError("CompositeMixing requires at least one producer")
        self._validate_producers()

    def forward_impl(self, x: Feature) -> Interaction:
        """Return the statically laid out union interaction."""

        x.validate()
        outputs = [producer.forward_pre_activation(x) for producer in self.producers]

        batch_size = common_real_batch_size(*outputs)
        dtype = common_real_dtype(*outputs)
        for output in outputs:
            zero = output.blocks[0]
            if zero.numel() != 0 or int(zero.shape[1]) != 0:
                raise ValueError("Composite producer order-0 blocks must be empty")
        # Interaction validation reserves order 0 as an empty zero-channel
        # block, so every valid producer agrees structurally. It carries no
        # values and is safe to retain from the first producer.
        blocks = [outputs[0].blocks[0]]
        for order in self.layout.output_orders.values:
            family_blocks = [result.blocks[order] for result in outputs]
            block = torch.cat(family_blocks, dim=2)
            if self.activation is not None:
                block = self.activation(block)
            blocks.append(block)
        interaction = Interaction(blocks, self.layout)
        if interaction.batch_size != batch_size or interaction.blocks[0].dtype != dtype:
            raise RuntimeError("Composite producer outputs disagree on batch or dtype")
        return interaction

    def _validate_producers(self) -> None:
        expected_families = tuple(slice_.family for slice_ in self.layout.family_slices)
        actual_families: list[str] = []
        for producer in self.producers:
            if isinstance(producer, LinearEquivariantMixing):
                actual_families.append("linear")
                paths = tuple(producer.paths)
            elif isinstance(producer, EquivariantMixing):
                actual_families.append("tensor_product")
                paths = tuple(producer.paths)
                if producer.activation is not None:
                    raise ValueError(
                        "CompositeMixing producers must be pre-Gamma; move TP activation to composite"
                    )
            else:
                raise TypeError("CompositeMixing producers must use concrete TPEN producer types")
            expected = _family_paths(self.layout, actual_families[-1])
            expected_paths = tuple(
                path
                for output in expected
                for path in output.paths
            )
            if expected_paths != paths:
                raise ValueError("producer paths do not match the static PathLayout")
        if tuple(actual_families) != expected_families:
            raise ValueError("producer order must match layout family order")

    def state_dict(self, *args, **kwargs):
        """Preserve direct TP state keys when the composition has one TP producer."""

        state = super().state_dict(*args, **kwargs)
        if len(self.producers) == 1 and isinstance(self.producers[0], EquivariantMixing):
            remapped = OrderedDict()
            for key, value in state.items():
                remapped[key.replace("producers.0.", "", 1)] = value
            return remapped
        return state

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load direct TP keys into the registered ModuleList producer."""

        if len(self.producers) == 1 and isinstance(self.producers[0], EquivariantMixing):
            state_dict = OrderedDict(
                (key.replace("weights.", "producers.0.weights.", 1) if key.startswith("weights.") else key, value)
                for key, value in state_dict.items()
            )
        return super().load_state_dict(state_dict, *args, **kwargs)


def _family_paths(layout: PathLayout, family: str) -> tuple[object, ...]:
    """Return a family's output slices without reflective dispatch."""

    for slice_ in layout.family_slices:
        if slice_.family == family:
            return slice_.outputs
    raise ValueError(f"layout has no {family} family")


__all__ = ["CompositeMixing"]
