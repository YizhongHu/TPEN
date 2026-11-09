"""Typed events owned by the checkpoint domain."""

from __future__ import annotations

from dataclasses import dataclass

from tpen.events import Event

from .restore import RestoreReport


@dataclass(frozen=True)
class CheckpointRestored(Event):
    """A checkpoint restore finished and reported what it loaded.

    This event lives with the checkpoint domain rather than with either runner
    because both `tpen.runner.Train` and `tpen.runner.Evaluate` reach the same
    moment, and `RestoreReport` is defined here.

    Historically this had no callback subscriber: the legacy
    ``checkpoint_restored`` string has two emitters and zero handlers
    repo-wide. ``tpen.callback.InitialCheckpoint`` is the first typed
    subscriber, through a stateless, class-owned group -- it needs to know a
    run was restored at all, independent of what the restored counters say,
    to refuse a resumed lineage whose counters happen to read zero. It was
    typed in the first place because its consumer is the durable record. The
    human's D3 ruling put restored
    checkpoint identity on the typed evaluation lifecycle rather than smuggling
    it into a metric key or a `tpen.logging.LogRecord` field, and that identity
    reaches ``occurrences.jsonl`` only through this event. Field-wise
    serialization of a nested non-`Event` dataclass is the prerequisite: without
    it the report would serialize as a bare type marker and ``completed_updates``
    -- the whole point of carrying it -- would be silently dropped.

    Parameters
    ----------
    report : RestoreReport
        What the restore resolved and loaded, including the resume cursor and
        the applied-update counter that identify the restored model version.
    """

    report: RestoreReport


@dataclass(frozen=True)
class LoadStarted(Event):
    """A configured checkpoint restore was requested."""

    path: str
    mode: str
    strict: bool


@dataclass(frozen=True)
class LoadFailed(Event):
    """A checkpoint restore failed after being requested."""

    path: str
    mode: str
    exception_type: str
    message: str


@dataclass(frozen=True)
class LoadSucceeded(Event):
    """A checkpoint restore completed and reports its typed result."""

    path: str
    report: RestoreReport


__all__ = ["CheckpointRestored", "LoadFailed", "LoadStarted", "LoadSucceeded"]
