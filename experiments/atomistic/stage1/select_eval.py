"""Select unchanged evaluation rows from a hashed phase-2 manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.atomistic.stage1.protocol import MANIFEST_SCHEMA, manifest_hash


def select_rows(document: Mapping[str, Any], *, system: str | None = None,
                lineage: int | None = None, checkpoint: int | None = None,
                stream: int | None = None) -> list[dict[str, Any]]:
    """Select evaluation rows, preserving their content and manifest order.

    Parameters
    ----------
    document
        Hashed phase-2 manifest.
    system, lineage, checkpoint, stream
        Optional equality filters, combined with AND.

    Returns
    -------
    list of dict
        Original row mappings suitable for the evaluation launcher.
    """
    rows = document.get("rows")
    if (document.get("schema") != MANIFEST_SCHEMA or document.get("phase") != 2
            or not isinstance(rows, list)
            or any(not isinstance(row, dict) for row in rows)):
        raise ValueError("expected a phase-2 manifest")
    if document.get("manifest_hash") != manifest_hash(rows):
        raise ValueError("phase-2 manifest_hash mismatch")
    selected = [row for row in rows if row.get("kind") == "eval"
                and (system is None or row.get("system") == system)
                and (lineage is None or row.get("lineage") == lineage)
                and (checkpoint is None or row.get("checkpoint") == checkpoint)
                and (stream is None or row.get("stream") == stream)]
    if not selected:
        raise ValueError("no matching eval rows")
    return selected


def main() -> int:
    """Emit one row object or all matching rows as JSON Lines."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system")
    parser.add_argument("--lineage", type=int)
    parser.add_argument("--checkpoint", type=int)
    parser.add_argument("--stream", type=int)
    parser.add_argument("--all", action="store_true", help="emit matching rows as JSON Lines")
    args = parser.parse_args()
    rows = select_rows(json.loads(args.manifest.read_text(encoding="utf-8")),
                       system=args.system, lineage=args.lineage, checkpoint=args.checkpoint,
                       stream=args.stream)
    if not args.all and len(rows) != 1:
        parser.error("selection is not unique; supply filters or --all")
    # A single selection is the bare object read by the batch wrapper.
    text = ("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
            if args.all else json.dumps(rows[0], indent=2, sort_keys=True) + "\n")
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
