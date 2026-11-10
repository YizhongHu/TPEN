#!/usr/bin/env python3
"""Apply and restore narrowly specified source mutants."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


REQUIRED_KEYS = ("id", "target", "old", "new", "killer")


def spec(path: Path) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("mutants")
    if not isinstance(rows, list) or not 3 <= len(rows) <= 5:
        raise ValueError("mutant spec must contain 3-5 mutants")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not set(REQUIRED_KEYS) <= row.keys():
            raise ValueError("each mutant needs id, target, old, new, and killer")
        item = {key: str(row[key]) for key in REQUIRED_KEYS}
        if (
            not item["id"]
            or item["id"] in seen
            or Path(item["target"]).is_absolute()
            or ".." in Path(item["target"]).parts
            or not item["killer"]
        ):
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


def _junit_testcases(xml_path: Path) -> list[tuple[str, str]]:
    """Return (nodeid, status) for every testcase in a pytest --junit-xml report.

    nodeid is reconstructed from classname/name (classname dots -> path
    separators, trailing ".py::name"); good enough for function-style tests,
    which is all this runner's killers use.
    """
    root = ET.parse(xml_path).getroot()
    cases: list[tuple[str, str]] = []
    for testcase in root.iter("testcase"):
        classname = testcase.get("classname", "")
        name = testcase.get("name", "")
        nodeid = f"{classname.replace('.', '/')}.py::{name}" if classname else name
        if testcase.find("failure") is not None:
            status = "failed"
        elif testcase.find("error") is not None:
            status = "error"
        elif testcase.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        cases.append((nodeid, status))
    return cases


def _matches_killer(nodeid: str, killer: str) -> bool:
    return nodeid == killer or nodeid.startswith(killer + "[")


def verdict(baseline_xml: Path, mutant_xml: Path, killer: str) -> dict[str, str]:
    if not baseline_xml.is_file() or not mutant_xml.is_file():
        return {"verdict": "HARNESS_ERROR", "reason": "missing junit xml report"}
    try:
        baseline_cases = [c for c in _junit_testcases(baseline_xml) if _matches_killer(c[0], killer)]
        mutant_cases = [c for c in _junit_testcases(mutant_xml) if _matches_killer(c[0], killer)]
    except ET.ParseError as exc:
        return {"verdict": "HARNESS_ERROR", "reason": f"unparseable junit xml: {exc}"}
    if not baseline_cases:
        return {"verdict": "HARNESS_ERROR", "reason": "killer not collected in baseline"}
    if any(status != "passed" for _, status in baseline_cases):
        return {"verdict": "HARNESS_ERROR", "reason": "killer did not pass in baseline"}
    if not mutant_cases:
        return {"verdict": "HARNESS_ERROR", "reason": "killer not collected under mutant"}
    baseline_ids = {nodeid for nodeid, _ in baseline_cases}
    mutant_ids = {nodeid for nodeid, _ in mutant_cases}
    if baseline_ids != mutant_ids:
        return {"verdict": "HARNESS_ERROR", "reason": "killer parametrization set changed under mutant"}
    if any(status in ("failed", "error") for _, status in mutant_cases):
        return {"verdict": "KILLED"}
    return {"verdict": "SURVIVED"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("validate", "apply", "restore", "verdict"))
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--id")
    parser.add_argument("--target")
    parser.add_argument("--baseline-junit", type=Path)
    parser.add_argument("--mutant-junit", type=Path)
    args = parser.parse_args()
    if args.action == "validate":
        print(json.dumps(spec(args.spec), sort_keys=True))
        return 0
    if args.action == "verdict":
        if not args.id or not args.baseline_junit or not args.mutant_junit:
            parser.error("verdict requires --id, --baseline-junit, and --mutant-junit")
        killer = find(spec(args.spec), args.id)["killer"]
        print(json.dumps(verdict(args.baseline_junit, args.mutant_junit, killer), sort_keys=True))
        return 0
    if not args.source:
        parser.error("apply and restore require --source")
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
