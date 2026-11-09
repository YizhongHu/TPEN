"""Study-side access to core's public resolved-config identity contract."""

from __future__ import annotations

from typing import Any

from tpen.inspection import resolved_config_identity as _core_resolved_config_identity


def expected_resolved_config_identity(config: Any) -> dict[str, str]:
    """Compute the expected identity from an intended resolved config."""

    return _core_resolved_config_identity(config)


__all__ = ["expected_resolved_config_identity"]
