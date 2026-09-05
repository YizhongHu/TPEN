"""Independent execution-bound probes for the DS-N round-three review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from tests.spikes.native_ddp.model_access import SemanticWavefunction


def main() -> int:
    """Run the reviewed worker while recording actual backward activity."""

    models: list[SemanticWavefunction] = []
    original_init = SemanticWavefunction.__init__
    original_backward = torch.Tensor.backward
    original_forward = SemanticWavefunction.forward
    counts = {"backward_calls": 0, "parameter_gradient_events": 0}
    forward_calls = 0

    def tracked_init(model, *args, **kwargs):
        original_init(model, *args, **kwargs)
        models.append(model)

    def tracked_backward(tensor, *args, **kwargs):
        counts["backward_calls"] += 1
        result = original_backward(tensor, *args, **kwargs)
        if any(parameter.grad is not None for model in models for parameter in model.parameters()):
            counts["parameter_gradient_events"] += 1
        return result

    def tracked_forward(model, coordinates):
        nonlocal forward_calls
        forward_calls += 1
        return original_forward(model, coordinates)

    SemanticWavefunction.__init__ = tracked_init
    torch.Tensor.backward = tracked_backward
    SemanticWavefunction.forward = tracked_forward
    try:
        from tests.spikes.native_ddp import worker

        result = worker.main()
    finally:
        SemanticWavefunction.__init__ = original_init
        torch.Tensor.backward = original_backward
        SemanticWavefunction.forward = original_forward

    state_arg = sys.argv[sys.argv.index("--state-path") + 1]
    state_path = Path(state_arg)
    state = json.loads(state_path.read_text())
    state["review_backward_calls"] = counts["backward_calls"]
    state["review_parameter_gradient_events"] = counts["parameter_gradient_events"]
    state["review_semantic_model_forward_calls"] = forward_calls
    state_path.write_text(json.dumps(state, sort_keys=True))
    return result


if __name__ == "__main__":
    sys.exit(main())
