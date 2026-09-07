"""Raw semantic-module versus Accelerate-prepared-wrapper access split.

WHERE DS-N's QUESTION IS ILL-POSED HERE, stated rather than silently replaced.
DS-N0's gate N-E2 required that "no caller reaches ``.module``". Accelerate's
SUPPORTED affordance for raw-module access is ``accelerator.unwrap_model``, so
forbidding the underlying attribute is not the same requirement. The DS-A
analogue A-E2 therefore asks three different things:

1. does raw access go through the supported affordance,
2. does that affordance return the ORIGINAL module identity, and
3. does any caller reach past it to ``.module`` anyway.

(2) is the one worth stating aloud: if ``unwrap_model`` returned a copy rather
than the same object, every parameter-identity assumption in the step -- reading
``.grad`` off the raw model after a wrapped backward -- would silently read the
wrong tensors. It is asserted at construction, not trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


class SemanticWavefunction(nn.Module):
    """Minimal differentiable scalar wavefunction used by the spike.

    Identical in form and initial parameters to the native spike's module, so the
    two prototypes are scored on the same function as well as the same fixtures.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.25], dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor([-0.10], dtype=torch.float64))

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Return one log-amplitude per leading sample."""

        if coordinates.ndim != 2:
            raise ValueError("SemanticWavefunction expects [sample, coordinate] input")
        return coordinates.sum(dim=1) * self.weight + self.bias


@dataclass(frozen=True)
class ModelAccess:
    """Typed handles separating raw semantic access from the prepared wrapper."""

    raw_model: SemanticWavefunction
    prepared_model: nn.Module
    accelerator: object
    forward_counts: dict[str, int] = field(
        default_factory=lambda: {"prepared": 0, "raw": 0}
    )
    gradient_counts: dict[str, int] = field(default_factory=lambda: {"parameter": 0})
    provenance: dict[str, object] = field(default_factory=dict)

    @classmethod
    def create(cls, raw_model: SemanticWavefunction, accelerator) -> "ModelAccess":
        """Prepare exactly the raw model once for score-function backward."""

        prepared = accelerator.prepare(raw_model)
        unwrapped = accelerator.unwrap_model(prepared)

        # See the module docstring, point (2). Identity, not equality: reading
        # ``.grad`` off a different object after a wrapped backward would give
        # stale or absent gradients while every assertion still passed.
        if unwrapped is not raw_model:
            raise RuntimeError(
                "accelerator.unwrap_model did not return the original module identity; "
                "every parameter-identity assumption in the step would be unsound"
            )

        provenance: dict[str, object] = {
            "prepared_type": type(prepared).__name__,
            "prepared_module_path": type(prepared).__module__,
            "unwrap_returns_original_identity": True,
            # Recorded because A-E4 has to name which wrapper Accelerate actually
            # installed, and because "prepared" not being DDP at world_size 1 is a
            # real behaviour a reader should not have to infer.
            "is_distributed_data_parallel": isinstance(
                prepared, torch.nn.parallel.DistributedDataParallel
            ),
            "gradient_accumulation_steps": int(
                getattr(accelerator, "gradient_accumulation_steps", 1)
            ),
        }

        access = cls(
            raw_model=raw_model,
            prepared_model=prepared,
            accelerator=accelerator,
            provenance=provenance,
        )
        access.raw_model.register_forward_pre_hook(access._count_raw_forward)
        if prepared is not raw_model:
            access.prepared_model.register_forward_pre_hook(access._count_prepared_forward)
        for parameter in access.raw_model.parameters():
            parameter.register_hook(access._count_parameter_gradient)
        return access

    def _count_raw_forward(self, _module, _inputs) -> None:
        self.forward_counts["raw"] += 1

    def _count_prepared_forward(self, _module, _inputs) -> None:
        self.forward_counts["prepared"] += 1

    def _count_parameter_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        """Observe autograd's parameter-gradient event without changing it."""

        self.gradient_counts["parameter"] += 1
        return gradient

    def coordinate_forward(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run coordinate autograd on the raw semantic module only.

        MCMC and kinetic-derivative work must never touch the prepared wrapper: a
        wrapped forward would enlist each of those passes in gradient
        synchronization, which is precisely what A-G6's communication-count
        invariant forbids.
        """

        coordinates = coordinates.detach().clone().requires_grad_(True)
        logabs = self.raw_model(coordinates)
        coordinate_gradient = torch.autograd.grad(
            logabs.sum(), coordinates, create_graph=True, retain_graph=False
        )[0]
        return logabs, coordinate_gradient

    def score_forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        """Run the sole prepared forward used by the parameter update."""

        return self.prepared_model(coordinates)


__all__ = ["ModelAccess", "SemanticWavefunction"]
