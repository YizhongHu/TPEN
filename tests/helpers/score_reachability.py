"""Autograd reachability probe for trainable parameters.

Owned here rather than duplicated per test module, and deliberately built from
nothing but one ordinary ``torch.autograd.grad`` call. The score seam in
`tpen.nn.tpen_wave_function` substitutes exact zeros for parameters autograd
cannot reach and reports which ones it substituted; a probe that consulted that
report would let the seam vouch for itself, so this one re-derives the fact
independently.
"""

from __future__ import annotations

import torch

from tpen.data.batch import ElectronBatch
from tpen.nn import TPENWaveFunction


def disconnected_parameter_names(
    model: TPENWaveFunction,
    batch: ElectronBatch,
) -> frozenset[str]:
    """Name every trainable parameter with no autograd path into ``logabs``.

    Parameters
    ----------
    model : TPENWaveFunction
        Model whose bound trainable parameters are probed.
    batch : ElectronBatch
        Batch to evaluate; the answer is batch-dependent, because TPEN's
        order-1 output weights reach ``logabs`` only through the odd-electron
        Pfaffian padding block and so are unreachable at an even count.

    Returns
    -------
    frozenset of str
        ``named_parameters()`` names whose gradient came back ``None``.

    Notes
    -----
    Identity is the join key between the ordered binding and the named
    parameters. A lookup by value would be ambiguous between tied or
    numerically equal parameters.
    """

    parameters = model.parameter_binding.parameters
    names_by_identity = {id(parameter): name for name, parameter in model.named_parameters()}
    with torch.enable_grad():
        gradients = torch.autograd.grad(model(batch).logabs.sum(), parameters, allow_unused=True)
    return frozenset(
        names_by_identity[id(parameter)]
        for parameter, gradient in zip(parameters, gradients, strict=True)
        if gradient is None
    )
