"""Configured logical executor tasks."""

from __future__ import annotations

import os
from typing import ClassVar, Sequence

from .runners import TPENCommand
from ._records import frozen_record


@frozen_record
class ConfiguredRun:
    """Hydra-constructible command configuration for one logical task.

    Parameters
    ----------
    id : str
        Stable logical task identifier.
    config : str
        Absolute Hydra configuration path.
    overrides : sequence of str, optional
        Dot-list overrides passed unchanged to the child process.
    """

    schema: ClassVar[str] = "tpen-executor/configured-run-v1"
    id: str
    config: str
    overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id.strip():
            raise TypeError("id must be a non-empty string")
        if type(self.config) is not str or not os.path.isabs(self.config):
            raise ValueError("config must be an absolute path")
        if isinstance(self.overrides, (str, bytes, bytearray)) or not isinstance(self.overrides, Sequence) or any(type(item) is not str for item in self.overrides):
            raise TypeError("overrides must be a sequence of strings")
        object.__setattr__(self, "overrides", tuple(self.overrides))

    def command(self) -> TPENCommand:
        """Return the executor command represented by this task."""
        return TPENCommand(self.config, self.overrides)


__all__ = ["ConfiguredRun"]
