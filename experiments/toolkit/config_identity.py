"""Study-side access to core's public resolved-config identity contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tpen.inspection import resolved_config_identity as _core_resolved_config_identity


def expected_resolved_config_identity(
    config: Any, *, run_values: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """Compute the expected identity from an intended unresolved config."""

    return _core_resolved_config_identity(config, run_values=run_values)


__all__ = ["expected_resolved_config_identity"]
