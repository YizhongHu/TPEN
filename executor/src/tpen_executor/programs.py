"""Process-local materialization of staged executor programs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf

from ._codec import thaw
from .contracts import ProgramSpecRef


def _materialize(ref: ProgramSpecRef, env: Mapping[str, Any] | None) -> Any:
    """Resolve and instantiate one captured specification in the owner process."""

    raw = thaw(ref.config_ref)
    config = OmegaConf.create(raw)
    if env:
        # Environment values are deliberately merged only at materialization;
        # the submitter never needs to know or resolve these values.
        config = OmegaConf.merge(config, OmegaConf.create(dict(env)))
    return instantiate(config)


@dataclass(frozen=True, slots=True)
class WorkerProgram:
    """Thin holder for an unresolved worker-process program specification."""

    spec: ProgramSpecRef

    def materialize(self, env: Mapping[str, Any] | None = None) -> Any:
        """Resolve and construct the worker program in its owning process."""

        return _materialize(self.spec, env)


@dataclass(frozen=True, slots=True)
class AllocationProgram:
    """Thin holder for an unresolved allocation-process program specification."""

    spec: ProgramSpecRef

    def materialize(self, env: Mapping[str, Any] | None = None) -> Any:
        """Resolve and construct the allocation program in its owning process."""

        return _materialize(self.spec, env)


class DummyAllocation:
    """Minimal packaged target used by the configuration composition example."""

    def __init__(self, workers: Any = None, **kwargs: Any) -> None:
        self.workers = workers
        self.options = kwargs


class DummyWorker:
    """Minimal packaged worker target used by the configuration example."""

    def __init__(self, command: str = "worker", **kwargs: Any) -> None:
        self.command = command
        self.options = kwargs


__all__ = ["AllocationProgram", "DummyAllocation", "DummyWorker", "WorkerProgram"]
