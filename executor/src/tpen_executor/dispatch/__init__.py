"""Dispatch implementations for owner-process executor allocations."""

from .local import LocalDispatcher

__all__ = ["LocalDispatcher"]
