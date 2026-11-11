"""Side-effect-free initializers for TPEN-owned neural modules."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from tpen.dependencies import require_torch, require_torch_nn

torch = require_torch(feature="TPEN neural initializers")
nn = require_torch_nn(feature="TPEN neural initializers")

_MAX_TORCH_SEED = 2**63 - 1


@dataclass(frozen=True)
class TorchInitializer:
    """Generator-backed initializer that never mutates process-global RNG state.

    Parameters
    ----------
    seed : int or None, optional
        Base seed for deterministic initialization. ``None`` uses PyTorch's
        default independent generator state and still avoids global RNG
        mutation.
    stream : str, optional
        Stable logical stream name mixed into the base seed. Use
        :meth:`spawn` to derive child streams for nested modules.
    """

    seed: int | None = None
    stream: str = "default"

    def spawn(self, name: str) -> "TorchInitializer":
        """Return a child initializer for a stable nested stream."""

        return TorchInitializer(seed=self.seed, stream=f"{self.stream}/{name}")

    def generator(self, *, device: torch.device | str | None = None) -> torch.Generator:
        """Return a fresh generator for this stream and device."""

        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        generator = torch.Generator(**kwargs)
        if self.seed is not None:
            generator.manual_seed(_stream_seed(int(self.seed), self.stream))
        return generator

    def uniform_(self, tensor: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Fill ``tensor`` uniformly without touching global RNG state."""

        if tensor.numel() == 0:
            return tensor
        values = torch.rand(
            tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
            generator=self.generator(device=tensor.device),
        )
        with torch.no_grad():
            tensor.copy_(values.mul_(float(high) - float(low)).add_(float(low)))
        return tensor

    def normal_(self, tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
        """Fill ``tensor`` with normal values without touching global RNG state."""

        if tensor.numel() == 0:
            return tensor
        values = torch.randn(
            tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
            generator=self.generator(device=tensor.device),
        )
        with torch.no_grad():
            tensor.copy_(values.mul_(float(std)).add_(float(mean)))
        return tensor

    def xavier_uniform_(self, tensor: torch.Tensor, *, gain: float = 1.0) -> torch.Tensor:
        """Fill ``tensor`` with Xavier uniform initialization."""

        fan_in, fan_out = _fan_in_and_fan_out(tensor)
        bound = float(gain) * math.sqrt(6.0 / float(fan_in + fan_out))
        return self.uniform_(tensor, -bound, bound)

    def linear_kaiming_uniform_(self, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        """Initialize a linear layer like ``torch.nn.Linear`` using this stream."""

        fan_in, _fan_out = _fan_in_and_fan_out(weight)
        gain = math.sqrt(2.0 / (1.0 + math.sqrt(5.0) ** 2))
        bound = math.sqrt(3.0) * gain / math.sqrt(float(fan_in))
        self.uniform_(weight, -bound, bound)
        if bias is not None:
            bias_bound = 1.0 / math.sqrt(float(fan_in)) if fan_in > 0 else 0.0
            self.spawn("bias").uniform_(bias, -bias_bound, bias_bound)


class SeededLinear(nn.Module):
    """Linear layer initialized by an explicit :class:`TorchInitializer`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        initializer: TorchInitializer,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        if self.in_features <= 0:
            raise ValueError("SeededLinear in_features must be positive")
        if self.out_features <= 0:
            raise ValueError("SeededLinear out_features must be positive")
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        initializer.linear_kaiming_uniform_(self.weight, self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the affine transform."""

        return torch.nn.functional.linear(inputs, self.weight, self.bias)

    def extra_repr(self) -> str:
        """Return the standard linear-layer representation."""

        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def _stream_seed(seed: int, stream: str) -> int:
    payload = f"{seed}:{stream}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % _MAX_TORCH_SEED


def _fan_in_and_fan_out(tensor: torch.Tensor) -> tuple[int, int]:
    if tensor.ndim < 2:
        raise ValueError("fan_in and fan_out require a tensor with at least 2 dimensions")
    receptive_field_size = 1
    if tensor.ndim > 2:
        receptive_field_size = math.prod(int(dim) for dim in tensor.shape[2:])
    fan_in = int(tensor.shape[1]) * receptive_field_size
    fan_out = int(tensor.shape[0]) * receptive_field_size
    return fan_in, fan_out


__all__ = ["SeededLinear", "TorchInitializer"]
