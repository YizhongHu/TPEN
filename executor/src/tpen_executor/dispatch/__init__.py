"""Dispatch implementations for owner-process executor allocations."""

from .local import LocalDispatcher
from .parsl import ParslDispatcher, ParslUnavailableError

__all__ = ["LocalDispatcher", "ParslDispatcher", "ParslUnavailableError"]
