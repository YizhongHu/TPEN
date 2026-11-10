#!/usr/bin/env python3
"""Apply and restore narrowly specified source mutants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def spec(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("mutants")
    if not isinstance(rows, list) or not 3 <= len(rows) <= 5:
        raise ValueError("mutant spec must contain 3-5 mutants")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not {"id", "target", "old", "new"} <= row.keys():
            raise ValueError("each mutant needs id, target, old, and new")
        item = {key: str(row[key]) for key in ("id", "target", "old", "new")}
        if not item["id"] or item["id"] in seen or Path(item["target"]).is_absolute() or ".." in Path(item["target"]).parts:
            raise ValueError(f"invalid mutant: {item['id']}")
        seen.add(item["id"])
        result.append(item)
    return result


def find(rows: list[dict[str, str]], mutant_id: str) -> dict[str, str]:
    for row in rows:
        if row["id"] == mutant_id:
            return row
    raise ValueError(f"unknown mutant id: {mutant_id}")


def apply_mutant(spec_path: Path, source: Path, backup: Path, mutant_id: str) -> int:
    row = find(spec(spec_path), mutant_id)
    target = source / row["target"]
    if not under(target, source) or not target.is_file():
        print(json.dumps({"edit_rc": 1, "reason": "target_missing", "target": str(target)}))
        return 1
    original = target.read_bytes()
    old = row["old"].encode()
    changed = original.replace(old, row["new"].encode(), 1)
    count = original.count(old)
    if count != 1 or changed == original:
        print(json.dumps({"edit_rc": 1, "needle_count": count, "target": str(target)}))
        return 1
    if not under(backup, source.parent):
        raise ValueError("backup must remain beside the source clone")
    backup.write_bytes(original)
    target.write_bytes(changed)
    print(json.dumps({"edit_rc": 0, "needle_count": count, "target": str(target)}))
    return 0


def restore(source: Path, backup: Path, target: str) -> int:
    target_path = source / target
    if not under(target_path, source) or not backup.is_file():
        return 1
    target_path.write_bytes(backup.read_bytes())
    backup.unlink()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "apply", "restore"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--target")
    args = parser.parse_args()
    if args.action == "validate":
        print(json.dumps(spec(args.spec), sort_keys=True))
        return 0
    if not args.backup or (args.action == "apply" and not args.id) or (args.action == "restore" and not args.target):
        parser.error("apply requires --id and --backup; restore requires --target and --backup")
    if args.action == "apply":
        return apply_mutant(args.spec, args.source, args.backup, args.id)
    return restore(args.source, args.backup, args.target)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"mutant helper error: {exc}", file=sys.stderr)
        raise SystemExit(2)
