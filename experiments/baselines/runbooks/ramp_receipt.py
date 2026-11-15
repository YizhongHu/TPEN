"""Write and gate mechanical BL-R1 ramp-row receipts.

The helper deliberately makes no scientific judgement.  It records a producer
process outcome and rejects a ramp marker unless every named row wrote one
successful receipt.  It is independent of scheduler state, so its logic is
covered by a small offline test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def record(path: Path, *, row: str, returncode: int) -> None:
    """Persist one unambiguous row outcome, refusing a collision."""

    if not row or "/" in row or "\\" in row:
        raise ValueError("row must be a non-empty leaf name")
    if path.exists():
        raise FileExistsError(f"receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(
        json.dumps({"schema": "bl-r1-ramp-row/v1", "row": row, "returncode": returncode}) + "\n",
        encoding="utf-8",
    )


def gate(root: Path, expected_rows: Sequence[str]) -> dict[str, object]:
    """Return the complete receipt account or raise on any failed/missing row."""

    rows = list(expected_rows)
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("expected rows must be non-empty and unique")
    observed: list[dict[str, object]] = []
    failures: list[str] = []
    for row in rows:
        path = root / row / "receipt.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{row}: no readable receipt ({exc})")
            continue
        observed.append(payload)
        if payload.get("schema") != "bl-r1-ramp-row/v1" or payload.get("row") != row:
            failures.append(f"{row}: receipt identity mismatch")
        if payload.get("returncode") != 0:
            failures.append(f"{row}: producer returncode={payload.get('returncode')!r}")
    report: dict[str, object] = {"schema": "bl-r1-ramp-gate/v1", "rows": observed, "failures": failures}
    if failures:
        raise RuntimeError(json.dumps(report, sort_keys=True))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the receipt writer or gate."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--path", type=Path, required=True)
    record_parser.add_argument("--row", required=True)
    record_parser.add_argument("--returncode", type=int, required=True)
    gate_parser = subparsers.add_parser("gate")
    gate_parser.add_argument("--root", type=Path, required=True)
    gate_parser.add_argument("--row", action="append", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            record(args.path, row=args.row, returncode=args.returncode)
        else:
            print(json.dumps(gate(args.root, args.row), sort_keys=True))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
