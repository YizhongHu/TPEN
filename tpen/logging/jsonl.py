"""JSON lines metric logger."""

from __future__ import annotations

import json
from pathlib import Path

from tpen.durable_append import append_record

from .base import LogRecord, Logger


class JSONL(Logger):
    """Append metric records as JSON lines.

    Parameters
    ----------
    path : str or pathlib.Path
        Output JSONL path.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def log(self, record: LogRecord) -> None:
        """Append one JSON object."""

        payload = {
            "step": record.step,
            "namespace": record.namespace,
            "metrics": record.metrics,
        }
        append_record(
            self.path, json.dumps(payload, sort_keys=True, allow_nan=False)
        )



__all__ = ["JSONL"]
